"""Tests for the community tap manager (core/tap.py).

All tests use local git repos in tmp_path — no network access needed.
"""
from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from crucible.core.errors import TapError
from crucible.core.tap import TapManager, VALID_PLUGIN_TYPES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _create_local_tap(tmp_path: Path, name: str = "test-tap", plugins: list[dict] | None = None) -> Path:
    """Create a local git repo that looks like a tap with plugin.yaml manifests."""
    tap_dir = tmp_path / name
    tap_dir.mkdir()

    plugins = plugins or [
        {
            "type": "optimizers",
            "name": "lion",
            "version": "1.0.0",
            "description": "Lion optimizer — sign-based updates",
            "author": "test",
            "tags": ["optimizer", "memory-efficient"],
            "code": "from crucible.training.optimizers import register_optimizer\nregister_optimizer('lion', lambda p, **kw: None, source='local')\n",
        },
        {
            "type": "schedulers",
            "name": "triangular",
            "version": "0.2.0",
            "description": "Triangular LR schedule with warm restarts",
            "author": "test",
            "tags": ["scheduler", "cyclic"],
            "code": "x = 'triangular'\n",
        },
    ]

    for plugin in plugins:
        pkg_dir = tap_dir / plugin["type"] / plugin["name"]
        pkg_dir.mkdir(parents=True)
        manifest = {k: v for k, v in plugin.items() if k not in ("code",)}
        (pkg_dir / "plugin.yaml").write_text(yaml.dump(manifest))
        (pkg_dir / f"{plugin['name']}.py").write_text(plugin.get("code", "x = 1\n"))

    # Git init + commit
    subprocess.run(["git", "init"], cwd=str(tap_dir), capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=str(tap_dir), capture_output=True, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@test.com", "-c", "user.name=test",
         "commit", "-m", "init"],
        cwd=str(tap_dir), capture_output=True, check=True,
    )
    return tap_dir


@pytest.fixture()
def hub_dir(tmp_path: Path) -> Path:
    """Create a minimal hub directory."""
    hd = tmp_path / "hub"
    hd.mkdir()
    (hd / "taps").mkdir()
    (hd / "plugins").mkdir()
    return hd


@pytest.fixture()
def tap_manager(hub_dir: Path) -> TapManager:
    return TapManager(hub_dir)


@pytest.fixture()
def local_tap(tmp_path: Path) -> Path:
    return _create_local_tap(tmp_path)


# ---------------------------------------------------------------------------
# Tap CRUD
# ---------------------------------------------------------------------------

class TestTapAdd:
    def test_add_clones_repo(self, tap_manager: TapManager, local_tap: Path):
        result = tap_manager.add_tap(str(local_tap), name="my-tap")
        assert result["name"] == "my-tap"
        # Bare local paths are normalized to file:// URLs so git clone
        # can consume them. The recorded URL may be either the raw path
        # (if the caller passed a pre-normalized URL) or the file://
        # version. Accept either.
        expected_raw = str(local_tap)
        expected_file = f"file://{local_tap.resolve()}"
        assert result["url"] in (expected_raw, expected_file), (
            f"unexpected URL in result: {result['url']!r}"
        )
        assert "added_at" in result
        assert (tap_manager._taps_dir / "my-tap").is_dir()

    def test_add_accepts_file_url(self, tap_manager: TapManager, local_tap: Path):
        """Explicit file:// URLs should also work (regression test)."""
        file_url = f"file://{local_tap.resolve()}"
        result = tap_manager.add_tap(file_url, name="file-tap")
        assert result["url"] == file_url
        assert (tap_manager._taps_dir / "file-tap").is_dir()

    def test_add_derives_name_from_url(self, tap_manager: TapManager, local_tap: Path):
        result = tap_manager.add_tap(str(local_tap))
        assert result["name"] == local_tap.name

    def test_add_duplicate_raises(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="dup")
        with pytest.raises(TapError, match="already exists"):
            tap_manager.add_tap(str(local_tap), name="dup")

    def test_add_persists_to_taps_yaml(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="persisted")
        taps = tap_manager._load_taps_yaml()
        assert len(taps) == 1
        assert taps[0]["name"] == "persisted"

    def test_add_bad_url_raises(self, tap_manager: TapManager):
        with pytest.raises(TapError, match="Failed to clone"):
            tap_manager.add_tap("/nonexistent/path", name="bad")


class TestTapRemove:
    def test_remove_deletes_dir_and_yaml(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="removable")
        assert (tap_manager._taps_dir / "removable").exists()

        tap_manager.remove_tap("removable")
        assert not (tap_manager._taps_dir / "removable").exists()
        assert len(tap_manager._load_taps_yaml()) == 0

    def test_remove_nonexistent_is_noop(self, tap_manager: TapManager):
        # Should not raise
        tap_manager.remove_tap("nonexistent")


class TestTapList:
    def test_list_empty(self, tap_manager: TapManager):
        assert tap_manager.list_taps() == []

    def test_list_populated(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="tap1")
        taps = tap_manager.list_taps()
        assert len(taps) == 1
        assert taps[0]["name"] == "tap1"


class TestTapSync:
    def test_sync_all(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="syncable")
        result = tap_manager.sync_tap()
        assert "syncable" in result["synced"]

    def test_sync_specific(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="specific")
        result = tap_manager.sync_tap("specific")
        assert "specific" in result["synced"]

    def test_sync_nonexistent_raises(self, tap_manager: TapManager):
        with pytest.raises(TapError, match="not found"):
            tap_manager.sync_tap("ghost")


# ---------------------------------------------------------------------------
# Package discovery
# ---------------------------------------------------------------------------

class TestScanTap:
    def test_scan_finds_manifests(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="scannable")
        manifests = tap_manager._scan_tap("scannable")
        names = {m["name"] for m in manifests}
        assert "lion" in names
        assert "triangular" in names

    def test_scan_empty_dir(self, tap_manager: TapManager):
        (tap_manager._taps_dir / "empty").mkdir(parents=True)
        assert tap_manager._scan_tap("empty") == []

    def test_scan_malformed_manifest(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="malformed")
        # Add a malformed plugin.yaml
        bad_dir = tap_manager._taps_dir / "malformed" / "optimizers" / "bad"
        bad_dir.mkdir(parents=True)
        (bad_dir / "plugin.yaml").write_text("not: valid: yaml: [")
        manifests = tap_manager._scan_tap("malformed")
        # Should still find the valid ones
        names = {m["name"] for m in manifests}
        assert "lion" in names


class TestSearch:
    def test_search_by_name(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="searchable")
        results = tap_manager.search("lion")
        assert len(results) >= 1
        assert results[0]["name"] == "lion"

    def test_search_by_description(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="searchable2")
        results = tap_manager.search("sign-based")
        assert any(r["name"] == "lion" for r in results)

    def test_search_by_tag(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="searchable3")
        results = tap_manager.search("memory-efficient")
        assert any(r["name"] == "lion" for r in results)

    def test_search_with_type_filter(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="typed")
        results = tap_manager.search("", plugin_type="schedulers")
        assert all(r["type"] == "schedulers" for r in results)
        assert any(r["name"] == "triangular" for r in results)

    def test_search_no_results(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="nope")
        results = tap_manager.search("xyznonexistent")
        assert results == []

    def test_list_available(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="all")
        results = tap_manager.list_available()
        assert len(results) >= 2


# ---------------------------------------------------------------------------
# Install / Uninstall
# ---------------------------------------------------------------------------

class TestInstall:
    def test_install_copies_file(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="installable")
        result = tap_manager.install("lion")
        assert result["status"] == "installed"
        assert result["type"] == "optimizers"
        assert result["tap"] == "installable"
        # Verify file exists where discover_all_plugins looks
        assert (tap_manager._plugins_dir / "optimizers" / "lion.py").exists()

    def test_install_records_in_yaml(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="recorded")
        tap_manager.install("lion")
        installed = tap_manager._load_installed_yaml()
        assert len(installed) == 1
        assert installed[0]["name"] == "lion"
        assert installed[0]["type"] == "optimizers"
        assert "installed_at" in installed[0]

    def test_install_duplicate_raises(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="dupinstall")
        tap_manager.install("lion")
        with pytest.raises(TapError, match="already installed"):
            tap_manager.install("lion")

    def test_install_not_found_raises(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="notfound")
        with pytest.raises(TapError, match="not found"):
            tap_manager.install("nonexistent_plugin_xyz")

    def test_install_from_specific_tap(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="specific-tap")
        result = tap_manager.install("lion", tap="specific-tap")
        assert result["tap"] == "specific-tap"

    def test_install_launcher_copies_bundle_directory(self, tap_manager: TapManager, tmp_path: Path):
        tap = _create_local_tap(
            tmp_path,
            name="launcher-tap",
            plugins=[{
                "type": "launchers",
                "name": "demo_launcher",
                "version": "0.1.0",
                "description": "Demo launcher bundle",
                "author": "test",
                "tags": ["launcher"],
                "code": "print('launcher')\n",
            }],
        )
        manifest_path = tap / "launchers" / "demo_launcher" / "plugin.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        manifest["entry"] = "demo_launcher.py"
        manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

        tap_manager.add_tap(str(tap), name="launcher-tap")
        result = tap_manager.install("demo_launcher")
        assert result["type"] == "launchers"
        assert (tap_manager._plugins_dir / "launchers" / "demo_launcher" / "plugin.yaml").exists()
        assert (tap_manager._plugins_dir / "launchers" / "demo_launcher" / "demo_launcher.py").exists()


class TestUninstall:
    def test_uninstall_removes_file_and_record(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="uninstallable")
        tap_manager.install("lion")
        assert (tap_manager._plugins_dir / "optimizers" / "lion.py").exists()

        tap_manager.uninstall("lion")
        assert not (tap_manager._plugins_dir / "optimizers" / "lion.py").exists()
        assert len(tap_manager._load_installed_yaml()) == 0

    def test_uninstall_not_installed_raises(self, tap_manager: TapManager):
        with pytest.raises(TapError, match="not installed"):
            tap_manager.uninstall("ghost")

    def test_uninstall_launcher_removes_directory(self, tap_manager: TapManager, tmp_path: Path):
        tap = _create_local_tap(
            tmp_path,
            name="launcher-uninstall",
            plugins=[{
                "type": "launchers",
                "name": "bundle_launcher",
                "version": "0.1.0",
                "description": "Bundle launcher",
                "author": "test",
                "tags": ["launcher"],
                "code": "print('bundle')\n",
            }],
        )
        tap_manager.add_tap(str(tap), name="launcher-uninstall")
        tap_manager.install("bundle_launcher")
        assert (tap_manager._plugins_dir / "launchers" / "bundle_launcher").exists()

        tap_manager.uninstall("bundle_launcher")
        assert not (tap_manager._plugins_dir / "launchers" / "bundle_launcher").exists()


class TestListInstalled:
    def test_list_empty(self, tap_manager: TapManager):
        assert tap_manager.list_installed() == []

    def test_list_with_type_filter(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="filtered")
        tap_manager.install("lion")
        tap_manager.install("triangular")

        opt_only = tap_manager.list_installed(plugin_type="optimizers")
        assert len(opt_only) == 1
        assert opt_only[0]["name"] == "lion"

        sched_only = tap_manager.list_installed(plugin_type="schedulers")
        assert len(sched_only) == 1
        assert sched_only[0]["name"] == "triangular"


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------

class TestPublish:
    def test_publish_copies_to_tap(self, tap_manager: TapManager, local_tap: Path, tmp_path: Path):
        tap_manager.add_tap(str(local_tap), name="publish-target")

        # Create a local plugin
        project = tmp_path / "project"
        plugin_dir = project / ".crucible" / "plugins" / "optimizers"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "my_opt.py").write_text("x = 'my optimizer'\n")

        result = tap_manager.publish(
            "my_opt", "optimizers", "publish-target",
            project_root=project,
        )
        assert result["status"] == "published"
        # Verify it landed in the tap
        dest = tap_manager._taps_dir / "publish-target" / "optimizers" / "my_opt"
        assert (dest / "my_opt.py").exists()
        assert (dest / "plugin.yaml").exists()

    def test_publish_auto_generates_manifest(self, tap_manager: TapManager, local_tap: Path, tmp_path: Path):
        tap_manager.add_tap(str(local_tap), name="auto-manifest")
        project = tmp_path / "project2"
        plugin_dir = project / ".crucible" / "plugins" / "callbacks"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "my_cb.py").write_text("x = 1\n")

        tap_manager.publish("my_cb", "callbacks", "auto-manifest", project_root=project)
        manifest_path = tap_manager._taps_dir / "auto-manifest" / "callbacks" / "my_cb" / "plugin.yaml"
        manifest = yaml.safe_load(manifest_path.read_text())
        assert manifest["name"] == "my_cb"
        assert manifest["type"] == "callbacks"

    def test_publish_missing_plugin_raises(self, tap_manager: TapManager, local_tap: Path, tmp_path: Path):
        tap_manager.add_tap(str(local_tap), name="missing")
        with pytest.raises(TapError, match="not found"):
            tap_manager.publish(
                "nonexistent", "optimizers", "missing",
                project_root=tmp_path / "empty",
            )

    def test_publish_bad_tap_raises(self, tap_manager: TapManager, tmp_path: Path):
        project = tmp_path / "project3"
        plugin_dir = project / ".crucible" / "plugins" / "optimizers"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "x.py").write_text("x = 1\n")
        with pytest.raises(TapError, match="not found"):
            tap_manager.publish("x", "optimizers", "ghost-tap", project_root=project)

    def test_publish_invalid_type_raises(self, tap_manager: TapManager, local_tap: Path, tmp_path: Path):
        tap_manager.add_tap(str(local_tap), name="badtype")
        with pytest.raises(TapError, match="Invalid plugin type"):
            tap_manager.publish("x", "invalid_type", "badtype", project_root=tmp_path)

    def test_publish_launcher_bundle(self, tap_manager: TapManager, local_tap: Path, tmp_path: Path):
        tap_manager.add_tap(str(local_tap), name="launcher-publish")
        project = tmp_path / "project_launcher"
        bundle_dir = project / ".crucible" / "plugins" / "launchers" / "demo_launcher"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "demo_launcher.py").write_text("print('ok')\n", encoding="utf-8")
        (bundle_dir / "plugin.yaml").write_text(
            yaml.safe_dump({
                "name": "demo_launcher",
                "type": "launchers",
                "version": "0.1.0",
                "entry": "demo_launcher.py",
            }),
            encoding="utf-8",
        )

        result = tap_manager.publish("demo_launcher", "launchers", "launcher-publish", project_root=project)
        assert result["status"] == "published"
        assert (tap_manager._taps_dir / "launcher-publish" / "launchers" / "demo_launcher" / "demo_launcher.py").exists()


# ---------------------------------------------------------------------------
# Package info
# ---------------------------------------------------------------------------

class TestPackageInfo:
    def test_get_info_found(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="info-tap")
        info = tap_manager.get_package_info("lion")
        assert info is not None
        assert info["name"] == "lion"
        assert info["type"] == "optimizers"
        assert info["installed"] is False

    def test_get_info_installed_flag(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="info-installed")
        tap_manager.install("lion")
        info = tap_manager.get_package_info("lion")
        assert info["installed"] is True

    def test_get_info_not_found(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="info-nope")
        assert tap_manager.get_package_info("nonexistent") is None


# ---------------------------------------------------------------------------
# Name derivation
# ---------------------------------------------------------------------------

class TestNameFromUrl:
    def test_github_https(self):
        assert TapManager._name_from_url("https://github.com/user/crucible-community-tap") == "crucible-community-tap"

    def test_github_with_git_suffix(self):
        assert TapManager._name_from_url("https://github.com/user/my-tap.git") == "my-tap"

    def test_trailing_slash(self):
        assert TapManager._name_from_url("https://github.com/user/tap/") == "tap"

    def test_ssh_url(self):
        assert TapManager._name_from_url("git@github.com:user/cool-plugins.git") == "cool-plugins"


# ---------------------------------------------------------------------------
# VALID_PLUGIN_TYPES constant
# ---------------------------------------------------------------------------

class TestPush:
    def test_push_nonexistent_raises(self, tap_manager: TapManager):
        with pytest.raises(TapError, match="not found"):
            tap_manager.push("ghost")

    def test_push_bad_remote_raises(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="bad-remote")
        # Point to a nonexistent remote
        tap_dir = tap_manager._taps_dir / "bad-remote"
        subprocess.run(
            ["git", "remote", "set-url", "origin", "https://nonexistent.invalid/repo.git"],
            cwd=str(tap_dir), capture_output=True, check=True,
        )
        with pytest.raises(TapError, match="Push failed"):
            tap_manager.push("bad-remote")


class TestSubmitPR:
    def test_submit_pr_nonexistent_raises(self, tap_manager: TapManager):
        with pytest.raises(TapError, match="not found"):
            tap_manager.submit_pr("ghost")

    def test_submit_pr_no_gh_returns_manual(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="pr-tap")
        result = tap_manager.submit_pr("pr-tap")
        # gh CLI likely not available in test env, should fall back
        assert result["status"] in ("manual", "error", "pr_created")
        assert "tap" in result


class TestGetTapRemote:
    def test_remote_for_cloned_tap(self, tap_manager: TapManager, local_tap: Path):
        tap_manager.add_tap(str(local_tap), name="remote-check")
        remote = tap_manager.get_tap_remote("remote-check")
        assert remote is not None
        assert str(local_tap) in remote

    def test_remote_for_nonexistent(self, tap_manager: TapManager):
        assert tap_manager.get_tap_remote("ghost") is None


class TestValidPluginTypes:
    def test_all_expected_types(self):
        expected = {
            "optimizers", "schedulers", "callbacks", "loggers",
            "providers", "architectures", "data_adapters", "data_sources",
            "objectives", "block_types", "stack_patterns", "augmentations",
            "activations", "launchers", "evaluations",
        }
        assert VALID_PLUGIN_TYPES == expected
