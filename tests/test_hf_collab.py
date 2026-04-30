"""Tests for HF collab tools, hub_remotes plugin family, and HfCollabConfig.

Exercises:
  - HfCollabConfig defaults + yaml parsing
  - HfError type + hierarchy
  - hub_remotes registry (git + hf_dataset registered as builtins)
  - Git hub_remote roundtrip (push -> list -> pull)
  - hf_writer error wrapping (huggingface_hub missing or call failures)
  - All 5 hf_* MCP tools — disabled-state guards + happy path with mocked HF SDK
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from crucible.core.config import HfCollabConfig, ProjectConfig, load_config
from crucible.core.errors import HfError, HubError


# ---------------------------------------------------------------------------
# Config + error
# ---------------------------------------------------------------------------


class TestHfCollabConfig:
    def test_defaults_disabled_and_private(self):
        c = HfCollabConfig()
        assert c.enabled is False
        assert c.private is True
        # `org` field intentionally absent — only {project} substitution is supported.
        assert not hasattr(c, "org")
        assert c.leaderboard_repo == ""
        assert c.findings_repo == ""
        assert c.recipes_repo == ""
        assert c.artifacts_repo == ""

    def test_yaml_load(self, tmp_path: Path):
        config_yaml = tmp_path / "crucible.yaml"
        config_yaml.write_text(yaml.safe_dump({
            "name": "demo",
            "hf_collab": {
                "enabled": True,
                "leaderboard_repo": "demo-org/lb",
                "findings_repo": "demo-org/find",
                "recipes_repo": "demo-org/rec",
                "artifacts_repo": "demo-org/art-{project}",
                "private": False,
            },
        }))
        c = load_config(config_yaml)
        assert c.hf_collab.enabled is True
        assert c.hf_collab.leaderboard_repo == "demo-org/lb"
        assert c.hf_collab.private is False
        assert c.hf_collab.artifacts_repo == "demo-org/art-{project}"

    def test_unknown_yaml_keys_dropped(self, tmp_path: Path):
        # The dropped 'org' field must not crash load_config — yaml builder
        # only forwards known keys, so unknown keys silently disappear.
        config_yaml = tmp_path / "crucible.yaml"
        config_yaml.write_text(yaml.safe_dump({
            "name": "demo",
            "hf_collab": {"enabled": True, "org": "ignored"},
        }))
        c = load_config(config_yaml)
        assert c.hf_collab.enabled is True
        assert not hasattr(c.hf_collab, "org")


class TestHfError:
    def test_subclass_of_hub(self):
        assert issubclass(HfError, HubError)

    def test_raise_and_catch(self):
        with pytest.raises(HubError):
            raise HfError("boom")


# ---------------------------------------------------------------------------
# hub_remotes plugin family
# ---------------------------------------------------------------------------


class TestHubRemotesRegistry:
    def test_builtins_registered(self):
        from crucible.core.hub_remotes import list_hub_remotes

        names = list_hub_remotes()
        assert "git" in names
        assert "hf_dataset" in names

    def test_unknown_remote_raises(self):
        from crucible.core.errors import PluginError
        from crucible.core.hub_remotes import build_hub_remote

        with pytest.raises(PluginError):
            build_hub_remote("nonexistent")


class TestGitHubRemoteRoundtrip:
    def test_push_list_pull(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", str(tmp_path / "hub"))

        from crucible.core.hub_remotes import build_hub_remote

        remote = build_hub_remote("git")
        src = tmp_path / "src"
        src.mkdir()
        (src / "leaderboard.jsonl").write_text('{"rank":1}\n')
        (src / "README.md").write_text("# test\n")

        ref = remote.push(str(src), "leaderboards/demo")
        assert ref.endswith("leaderboards/demo")

        listed = remote.list_remote("leaderboards/demo")
        assert sorted(listed) == ["README.md", "leaderboard.jsonl"]

        dst = tmp_path / "dst"
        out = remote.pull("leaderboards/demo", str(dst))
        assert (out / "README.md").read_text() == "# test\n"
        assert (out / "leaderboard.jsonl").read_text() == '{"rank":1}\n'

    def test_invalid_repo_id_rejected(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", str(tmp_path / "hub"))
        from crucible.core.hub_remotes import build_hub_remote

        remote = build_hub_remote("git")
        with pytest.raises(HubError):
            remote.push(str(tmp_path), "../escape")

    def test_absolute_repo_id_rejected(self, tmp_path: Path, monkeypatch):
        # Path("hub") / "/abs" returns "/abs" — the hub root is silently
        # discarded. Make sure we reject before resolution.
        monkeypatch.setenv("CRUCIBLE_HUB_DIR", str(tmp_path / "hub"))
        from crucible.core.hub_remotes import build_hub_remote

        remote = build_hub_remote("git")
        with pytest.raises(HubError, match="must be relative"):
            remote.push(str(tmp_path), "/etc/passwd-target")
        with pytest.raises(HubError, match="must be relative"):
            remote.pull("/abs/repo", str(tmp_path / "out"))


class TestHubRemoteIdempotentRegistration:
    def test_reimport_does_not_raise(self):
        # Re-running the module-level _register_builtins() must not crash
        # even though the registry already holds 'git' and 'hf_dataset'.
        from crucible.core import hub_remotes

        hub_remotes._register_builtins()
        hub_remotes._register_builtins()
        names = hub_remotes.list_hub_remotes()
        assert names.count("git") == 1
        assert names.count("hf_dataset") == 1


# ---------------------------------------------------------------------------
# hf_writer — exercised with a fake huggingface_hub module
# ---------------------------------------------------------------------------


class FakeHfModule(types.ModuleType):
    """Fake `huggingface_hub` that records every interaction.

    Tests assert against ``self.uploads`` (upload_folder/upload_file calls),
    ``self.created`` (HfApi.create_repo calls), ``self.snapshots`` and
    ``self.list_calls`` for the read paths.
    """
    def __init__(self):
        super().__init__("huggingface_hub")
        self.uploads: list[dict[str, Any]] = []
        self.created: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        module = self

        class _FakeHfApi:
            def __init__(self, token=None):
                self.token = token

            def create_repo(self, repo_id, repo_type, private, exist_ok):
                module.created.append({
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "private": private,
                    "exist_ok": exist_ok,
                    "token": self.token,
                })

        self.HfApi = _FakeHfApi

    def upload_folder(self, **kw):
        self.uploads.append({"kind": "folder", **kw})
        return f"https://huggingface.co/{kw['repo_id']}"

    def upload_file(self, **kw):
        self.uploads.append({"kind": "file", **kw})
        return f"https://huggingface.co/{kw['repo_id']}/blob/main/{kw['path_in_repo']}"

    def hf_hub_download(self, **kw):
        # Materialize a small file in local_dir (or a fake path)
        local_dir = kw.get("local_dir") or "/tmp/hf-fake"
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        out = Path(local_dir) / kw["filename"]
        out.write_text("fake")
        return str(out)

    def snapshot_download(self, **kw):
        self.snapshots.append(dict(kw))
        local_dir = Path(kw["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "fake.txt").write_text("fake")
        # Also drop a nested file to exercise rglob in hf_pull_artifact's
        # file listing.
        nested = local_dir / "subdir"
        nested.mkdir(exist_ok=True)
        (nested / "weights.bin").write_text("w")
        return str(local_dir)

    def list_repo_files(self, **kw):
        self.list_calls.append(dict(kw))
        return ["README.md", "leaderboard.jsonl"]


@pytest.fixture()
def fake_hf(monkeypatch):
    """Inject a fake huggingface_hub module into sys.modules."""
    fake = FakeHfModule()
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return fake


class TestHfWriter:
    def test_resolve_token_env(self, monkeypatch):
        from crucible.core import hf_writer

        monkeypatch.setenv("HF_TOKEN", "abc")
        assert hf_writer.resolve_token() == "abc"
        assert hf_writer.resolve_token("explicit") == "explicit"
        monkeypatch.delenv("HF_TOKEN", raising=False)
        assert hf_writer.resolve_token() is None

    def test_import_failure_raises_hferror(self, monkeypatch):
        from crucible.core import hf_writer

        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        with pytest.raises(HfError):
            hf_writer.list_files("any/repo")

    def test_push_folder_happy_path(self, fake_hf, tmp_path: Path):
        from crucible.core import hf_writer

        d = tmp_path / "stage"
        d.mkdir()
        (d / "x.txt").write_text("y")
        url = hf_writer.push_folder(d, "demo/repo", commit_message="m")
        assert "demo/repo" in url
        assert fake_hf.uploads[0]["kind"] == "folder"
        assert fake_hf.uploads[0]["folder_path"] == str(d)

    def test_push_file_missing_raises(self, fake_hf, tmp_path: Path):
        from crucible.core import hf_writer

        with pytest.raises(HfError):
            hf_writer.push_file(tmp_path / "nope.txt", "demo/repo")

    def test_ensure_repo_records(self, fake_hf):
        from crucible.core import hf_writer

        hf_writer.ensure_repo("demo/repo", repo_type="dataset", private=True, token="t")
        assert len(fake_hf.created) == 1
        rec = fake_hf.created[0]
        assert rec["repo_id"] == "demo/repo"
        assert rec["repo_type"] == "dataset"
        assert rec["private"] is True
        assert rec["exist_ok"] is True
        assert rec["token"] == "t"


class TestHfDatasetRemote:
    def test_push_calls_ensure_then_upload(self, fake_hf, tmp_path: Path):
        from crucible.core.hub_remotes import build_hub_remote

        d = tmp_path / "stage"
        d.mkdir()
        (d / "f.json").write_text("{}")
        remote = build_hub_remote("hf_dataset", repo_type="dataset", private=True, token="t")
        url = remote.push(str(d), "demo/findings")
        assert "demo/findings" in url
        # ensure_repo must run BEFORE upload_folder, with the same repo_id
        # and the configured private flag + token.
        assert len(fake_hf.created) == 1
        assert fake_hf.created[0]["repo_id"] == "demo/findings"
        assert fake_hf.created[0]["private"] is True
        assert fake_hf.created[0]["token"] == "t"
        assert len(fake_hf.uploads) == 1
        assert fake_hf.uploads[0]["repo_id"] == "demo/findings"
        assert fake_hf.uploads[0]["repo_type"] == "dataset"
        assert fake_hf.uploads[0]["token"] == "t"

    def test_push_skips_ensure_when_disabled(self, fake_hf, tmp_path: Path):
        from crucible.core.hub_remotes import build_hub_remote

        d = tmp_path / "stage"
        d.mkdir()
        (d / "f.json").write_text("{}")
        remote = build_hub_remote("hf_dataset", repo_type="dataset", ensure=False)
        remote.push(str(d), "demo/findings")
        assert fake_hf.created == []
        assert len(fake_hf.uploads) == 1

    def test_pull_calls_snapshot(self, fake_hf, tmp_path: Path):
        from crucible.core.hub_remotes import build_hub_remote

        remote = build_hub_remote("hf_dataset", repo_type="dataset")
        out = remote.pull("demo/findings", str(tmp_path / "dest"))
        assert out.exists()
        assert (out / "fake.txt").read_text() == "fake"


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@pytest.fixture()
def hf_enabled_config(tmp_path: Path, monkeypatch):
    """A ProjectConfig with hf_collab.enabled=true and template repos."""
    project = tmp_path / "proj"
    project.mkdir()
    cfg = ProjectConfig()
    cfg.project_root = project
    cfg.store_dir = ".crucible"
    cfg.name = "demo"
    cfg.hf_collab = HfCollabConfig(
        enabled=True,
        leaderboard_repo="demo-org/lb",
        findings_repo="demo-org/find",
        recipes_repo="demo-org/rec",
        artifacts_repo="demo-org/art-{project}",
        private=True,
    )
    monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: cfg)
    return cfg


@pytest.fixture()
def hf_disabled_config(tmp_path: Path, monkeypatch):
    project = tmp_path / "proj"
    project.mkdir()
    cfg = ProjectConfig()
    cfg.project_root = project
    cfg.store_dir = ".crucible"
    monkeypatch.setattr("crucible.mcp.tools._get_config", lambda: cfg)
    return cfg


class TestDisabledGuards:
    def test_push_artifact_disabled(self, hf_disabled_config, tmp_path: Path):
        from crucible.mcp.tools import hf_push_artifact

        out = hf_push_artifact({"local_dir": str(tmp_path)})
        assert "disabled" in out["error"]

    def test_pull_artifact_disabled(self, hf_disabled_config):
        from crucible.mcp.tools import hf_pull_artifact

        out = hf_pull_artifact({"repo_id": "x/y"})
        assert "disabled" in out["error"]

    def test_publish_leaderboard_disabled(self, hf_disabled_config):
        from crucible.mcp.tools import hf_publish_leaderboard

        out = hf_publish_leaderboard({})
        assert "disabled" in out["error"]

    def test_publish_findings_disabled(self, hf_disabled_config):
        from crucible.mcp.tools import hf_publish_findings

        out = hf_publish_findings({})
        assert "disabled" in out["error"]

    def test_publish_recipes_disabled(self, hf_disabled_config):
        from crucible.mcp.tools import hf_publish_recipes

        out = hf_publish_recipes({})
        assert "disabled" in out["error"]


class TestHfPushArtifact:
    def test_happy_path(self, hf_enabled_config, fake_hf, tmp_path: Path):
        from crucible.mcp.tools import hf_push_artifact

        artifact = tmp_path / "ckpt"
        artifact.mkdir()
        (artifact / "model.pt").write_text("weights")
        out = hf_push_artifact({
            "local_dir": str(artifact),
            "run_id": "run-42",
        })
        assert out["ok"] is True
        # template substitutes {project}=demo
        assert out["repo_id"] == "demo-org/art-demo"
        assert out["run_id"] == "run-42"

    def test_explicit_repo_id_wins(self, hf_enabled_config, fake_hf, tmp_path: Path):
        from crucible.mcp.tools import hf_push_artifact

        artifact = tmp_path / "ckpt"
        artifact.mkdir()
        out = hf_push_artifact({
            "local_dir": str(artifact),
            "repo_id": "other/explicit",
        })
        assert out["repo_id"] == "other/explicit"

    def test_missing_local_dir(self, hf_enabled_config, fake_hf):
        from crucible.mcp.tools import hf_push_artifact

        out = hf_push_artifact({"local_dir": "/no/such/path"})
        assert "not found" in out["error"]


class TestHfPullArtifact:
    def test_happy_path(self, hf_enabled_config, fake_hf, tmp_path: Path):
        from crucible.mcp.tools import hf_pull_artifact

        out = hf_pull_artifact({"repo_id": "demo-org/art-demo"})
        assert out["ok"] is True
        assert "fake.txt" in out["files"]


class TestHfPublishLeaderboard:
    def test_happy_path_empty_results(self, hf_enabled_config, fake_hf, monkeypatch):
        # No experiments → empty leaderboard, but publish should still succeed
        # AND the staged folder must contain leaderboard.jsonl + README.md
        # with the right project name baked in.
        from crucible.mcp.tools import hf_publish_leaderboard

        # Capture the staged folder before the tempdir is cleaned up by
        # asserting on what upload_folder saw.
        out = hf_publish_leaderboard({"top_n": 10})
        assert out["ok"] is True
        assert out["repo_id"] == "demo-org/lb"
        assert out["rows"] == 0

        assert len(fake_hf.uploads) == 1
        upload = fake_hf.uploads[0]
        assert upload["kind"] == "folder"
        assert upload["repo_id"] == "demo-org/lb"
        assert upload["repo_type"] == "dataset"
        # ensure_repo ran first.
        assert any(c["repo_id"] == "demo-org/lb" for c in fake_hf.created)

    def test_published_rows_carry_challenge_field(self, hf_enabled_config, fake_hf, tmp_path: Path):
        # Cross-phase contract: hf_publish_leaderboard must emit a stable
        # `challenge` field on every row so fetch_prior_runs(challenge_id=...)
        # in Phase 2 can filter peer rows reliably. Default = project name.
        from crucible.mcp.tools import hf_publish_leaderboard

        # Seed one fake completed experiment so leaderboard is non-empty.
        store = hf_enabled_config.project_root / hf_enabled_config.store_dir
        store.mkdir(parents=True, exist_ok=True)
        (hf_enabled_config.project_root / "experiments.jsonl").write_text(
            '{"name": "test-run", "status": "completed", '
            '"result": {"val_loss": 1.5, "steps_completed": 1000}, '
            '"model_bytes": 12345, "contract_status": "ok"}\n'
        )

        captured: dict = {}
        original_upload = fake_hf.upload_folder

        def capture(**kw):
            staged = Path(kw["folder_path"])
            jsonl = (staged / "leaderboard.jsonl").read_text()
            captured["rows"] = [
                __import__("json").loads(line)
                for line in jsonl.splitlines() if line.strip()
            ]
            return original_upload(**kw)

        fake_hf.upload_folder = capture  # type: ignore[assignment]
        out = hf_publish_leaderboard({"top_n": 5})
        assert out["ok"] is True
        for row in captured["rows"]:
            assert "challenge" in row, "every row must have a challenge field"
            assert row["challenge"] == hf_enabled_config.name
            assert row["project"] == hf_enabled_config.name

    def test_explicit_challenge_overrides_project_name(self, hf_enabled_config, fake_hf):
        from crucible.mcp.tools import hf_publish_leaderboard

        (hf_enabled_config.project_root / "experiments.jsonl").write_text(
            '{"name": "r1", "status": "completed", '
            '"result": {"val_loss": 1.0}, "model_bytes": 1, "contract_status": "ok"}\n'
        )

        captured: dict = {}
        original_upload = fake_hf.upload_folder

        def capture(**kw):
            staged = Path(kw["folder_path"])
            jsonl = (staged / "leaderboard.jsonl").read_text()
            captured["rows"] = [
                __import__("json").loads(line)
                for line in jsonl.splitlines() if line.strip()
            ]
            return original_upload(**kw)

        fake_hf.upload_folder = capture  # type: ignore[assignment]
        out = hf_publish_leaderboard({"top_n": 5, "challenge": "parameter-golf-2026"})
        assert out["ok"] is True
        for row in captured["rows"]:
            assert row["challenge"] == "parameter-golf-2026"

    def test_staged_files_recorded(self, hf_enabled_config, fake_hf, monkeypatch):
        # Hook upload_folder so we can inspect the staged contents before
        # the tempdir disappears.
        from crucible.mcp.tools import hf_publish_leaderboard

        captured: dict[str, Any] = {}

        original_upload = fake_hf.upload_folder

        def capture(**kw):
            staged = Path(kw["folder_path"])
            captured["files"] = sorted(p.name for p in staged.iterdir())
            captured["readme"] = (staged / "README.md").read_text()
            captured["jsonl"] = (staged / "leaderboard.jsonl").read_text()
            return original_upload(**kw)

        fake_hf.upload_folder = capture  # type: ignore[assignment]
        out = hf_publish_leaderboard({"top_n": 5})
        assert out["ok"] is True
        assert "README.md" in captured["files"]
        assert "leaderboard.jsonl" in captured["files"]
        assert "demo" in captured["readme"]  # project name
        assert "primary_metric" in captured["readme"]

    def test_bad_repo_template_returns_error(self, hf_enabled_config, fake_hf):
        from crucible.mcp.tools import hf_publish_leaderboard

        # `{org}` placeholder not supported — only `{project}` is.
        out = hf_publish_leaderboard({"repo_id": "demo-org/{org}-lb"})
        assert "ValueError" in out["error"]
        # No HF call should have happened.
        assert fake_hf.uploads == []
        assert fake_hf.created == []


class TestHfPublishFindings:
    def test_project_scope_no_state(self, hf_enabled_config, fake_hf):
        from crucible.mcp.tools import hf_publish_findings

        out = hf_publish_findings({"scope": "project"})
        assert out["ok"] is True
        assert out["scope"] == "project"
        assert out["count"] == 0


class TestHfPublishRecipes:
    def test_publishes_all_recipes(self, hf_enabled_config, fake_hf):
        # Create a recipe on disk
        recipes_dir = hf_enabled_config.project_root / hf_enabled_config.store_dir / "recipes"
        recipes_dir.mkdir(parents=True)
        (recipes_dir / "demo.yaml").write_text("name: demo\nsteps: []\n")

        from crucible.mcp.tools import hf_publish_recipes

        out = hf_publish_recipes({})
        assert out["ok"] is True
        assert "demo" in out["names"]
        assert out["count"] == 1

    def test_no_recipes_dir(self, hf_enabled_config, fake_hf):
        from crucible.mcp.tools import hf_publish_recipes

        out = hf_publish_recipes({})
        assert "No recipes directory" in out["error"]

    def test_names_filter(self, hf_enabled_config, fake_hf):
        recipes_dir = hf_enabled_config.project_root / hf_enabled_config.store_dir / "recipes"
        recipes_dir.mkdir(parents=True)
        (recipes_dir / "a.yaml").write_text("name: a\nsteps: []\n")
        (recipes_dir / "b.yaml").write_text("name: b\nsteps: []\n")

        from crucible.mcp.tools import hf_publish_recipes

        out = hf_publish_recipes({"names": ["b"]})
        assert out["names"] == ["b"]
        assert out["count"] == 1

    def test_symlinks_skipped(self, hf_enabled_config, fake_hf, tmp_path: Path):
        # A malicious symlink in .crucible/recipes/ pointing at a secret file
        # outside the recipes dir must NOT be published. Our guard skips it
        # and reports it under skipped_symlinks.
        recipes_dir = hf_enabled_config.project_root / hf_enabled_config.store_dir / "recipes"
        recipes_dir.mkdir(parents=True)
        (recipes_dir / "real.yaml").write_text("name: real\nsteps: []\n")

        secret = tmp_path / "secret.yaml"
        secret.write_text("sensitive: true\n")
        (recipes_dir / "evil.yaml").symlink_to(secret)

        from crucible.mcp.tools import hf_publish_recipes

        out = hf_publish_recipes({})
        assert out["ok"] is True
        assert out["names"] == ["real"]
        assert "evil" in out["skipped_symlinks"][0]
        # The upload payload must not contain evil.yaml.
        upload = fake_hf.uploads[0]
        staged = Path(upload["folder_path"]) / "recipes"
        if staged.exists():
            assert "evil.yaml" not in [p.name for p in staged.iterdir()]

    def test_bad_repo_template_returns_error(self, hf_enabled_config, fake_hf):
        recipes_dir = hf_enabled_config.project_root / hf_enabled_config.store_dir / "recipes"
        recipes_dir.mkdir(parents=True)
        (recipes_dir / "r.yaml").write_text("name: r\nsteps: []\n")

        from crucible.mcp.tools import hf_publish_recipes

        out = hf_publish_recipes({"repo_id": "demo-org/{nope}"})
        assert "ValueError" in out["error"]
        assert fake_hf.uploads == []


class TestHfPullArtifactNested:
    def test_files_list_includes_nested(self, hf_enabled_config, fake_hf):
        from crucible.mcp.tools import hf_pull_artifact

        out = hf_pull_artifact({"repo_id": "demo-org/art-demo"})
        assert out["ok"] is True
        # snapshot_download fake creates fake.txt + subdir/weights.bin —
        # both must show up via the new rglob listing.
        assert "fake.txt" in out["files"]
        assert any(f.endswith("weights.bin") for f in out["files"])
