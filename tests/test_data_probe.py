"""Tests for config-driven data probe (Phase 2d).

Removes the hardcoded fineweb paths from fleet bootstrap in favor of a
`data.probe` section in crucible.yaml. Non-LM projects can now ship
their own probe paths / download command, or omit them entirely for a
no-op data bootstrap.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from crucible.core.config import (
    DataConfig,
    DataProbeConfig,
    ProjectConfig,
    load_config,
)
from crucible.fleet.bootstrap import (
    _build_data_probe_command,
    _legacy_fineweb_download_command,
    _generate_paths_probe,
)


# ---------------------------------------------------------------------------
# DataProbeConfig parsing
# ---------------------------------------------------------------------------


class TestDataProbeConfigParsing:
    def test_empty_probe_section_gives_empty_defaults(self, tmp_path: Path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text("name: test\ndata:\n  source: huggingface\n")
        cfg = load_config(yaml_path)
        assert isinstance(cfg.data.probe, DataProbeConfig)
        assert cfg.data.probe.paths == []
        assert cfg.data.probe.script == ""
        assert cfg.data.probe.download_command == ""

    def test_populated_probe_section(self, tmp_path: Path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "name": "test",
            "data": {
                "source": "huggingface",
                "probe": {
                    "paths": ["data/train.h5", "data/val.h5"],
                    "download_command": "python3 download.py",
                },
            },
        }))
        cfg = load_config(yaml_path)
        assert cfg.data.probe.paths == ["data/train.h5", "data/val.h5"]
        assert cfg.data.probe.download_command == "python3 download.py"


# ---------------------------------------------------------------------------
# _build_data_probe_command
# ---------------------------------------------------------------------------


def _make_cfg(
    *,
    paths: list[str] | None = None,
    script: str = "",
    download_command: str = "",
    source: str = "huggingface",
    variant: str = "fineweb10B_sp1024",
) -> ProjectConfig:
    data = DataConfig(
        source=source,
        variant=variant,
        probe=DataProbeConfig(
            paths=list(paths or []),
            script=script,
            download_command=download_command,
        ),
    )
    return ProjectConfig(data=data)


class TestBuildDataProbeCommand:
    def test_returns_none_without_probe_or_legacy(self):
        cfg = _make_cfg(source="local_files", variant="")
        assert _build_data_probe_command(cfg, "/ws", "python3") is None

    def test_uses_script_when_set(self):
        cfg = _make_cfg(script="scripts/probe.py")
        cmd = _build_data_probe_command(cfg, "/ws", "python3")
        assert cmd is not None
        assert "scripts/probe.py" in cmd
        assert "/ws" in cmd
        assert "python3" in cmd

    def test_script_path_is_shell_quoted(self):
        cfg = _make_cfg(script="scripts/my probe.py")
        cmd = _build_data_probe_command(cfg, "/ws", "python3")
        # shlex.quote escapes the space
        assert "'scripts/my probe.py'" in cmd

    def test_uses_paths_when_set_and_no_script(self):
        cfg = _make_cfg(paths=["data/train.h5"])
        cmd = _build_data_probe_command(cfg, "/ws", "python3")
        assert cmd is not None
        assert "data/train.h5" in cmd
        # Uses the heredoc Python pattern
        assert "<<'PY'" in cmd

    def test_script_wins_over_paths(self):
        cfg = _make_cfg(script="probe.py", paths=["a", "b"])
        cmd = _build_data_probe_command(cfg, "/ws", "python3")
        assert "probe.py" in cmd
        # paths list should not be embedded
        assert "'a'" not in cmd

    def test_legacy_fineweb_fallback_fires_only_for_pg_config(self):
        # PG-style: huggingface + fineweb variant
        cfg = _make_cfg(source="huggingface", variant="fineweb10B_sp1024")
        cmd = _build_data_probe_command(cfg, "/ws", "python3")
        assert cmd is not None
        assert "fineweb10B_sp1024" in cmd
        assert "fineweb_1024_bpe.model" in cmd

    def test_non_fineweb_no_probe_returns_none(self):
        cfg = _make_cfg(source="huggingface", variant="c4-en")
        assert _build_data_probe_command(cfg, "/ws", "python3") is None

    def test_wandb_artifact_source_no_legacy_fallback(self):
        cfg = _make_cfg(source="wandb_artifact", variant="fineweb10B_sp1024")
        assert _build_data_probe_command(cfg, "/ws", "python3") is None


# ---------------------------------------------------------------------------
# _generate_paths_probe
# ---------------------------------------------------------------------------


class TestGeneratePathsProbe:
    def test_embeds_all_paths(self):
        probe = _generate_paths_probe("/ws", "python3", ["a.bin", "b.bin"])
        assert "a.bin" in probe
        assert "b.bin" in probe
        assert "print(1 if ok else 0)" in probe

    def test_handles_glob_patterns(self):
        probe = _generate_paths_probe("/ws", "python3", ["data/*.h5"])
        # Glob branch uses Path.glob
        assert "data/*.h5" in probe
        assert "glob" in probe

    def test_directory_paths_checked_for_non_empty(self):
        probe = _generate_paths_probe("/ws", "python3", ["data/datasets/"])
        assert "data/datasets/" in probe
        assert "is_dir()" in probe
        assert "iterdir()" in probe

    def test_wraps_in_heredoc_command(self):
        probe = _generate_paths_probe("/ws", "python3", ["f.txt"])
        assert probe.startswith("cd /ws && python3 - <<'PY'")
        assert probe.endswith("PY")


# ---------------------------------------------------------------------------
# _legacy_fineweb_download_command
# ---------------------------------------------------------------------------


class TestLegacyFinewebDownloadCommand:
    def test_returns_command_for_pg_config(self):
        cfg = _make_cfg(source="huggingface", variant="fineweb10B_sp1024")
        cmd = _legacy_fineweb_download_command(cfg, "/ws", "python3", 4)
        assert "cached_challenge_fineweb.py" in cmd
        assert "sp1024" in cmd
        assert "--train-shards 4" in cmd

    def test_empty_for_non_fineweb_configs(self):
        cfg = _make_cfg(source="huggingface", variant="c4")
        assert _legacy_fineweb_download_command(cfg, "/ws", "python3", 4) == ""

    def test_empty_for_wandb_artifact(self):
        cfg = _make_cfg(source="wandb_artifact", variant="fineweb10B_sp1024")
        assert _legacy_fineweb_download_command(cfg, "/ws", "python3", 4) == ""

    def test_handles_missing_data(self):
        cfg = ProjectConfig()  # default has huggingface + fineweb10B_sp1024
        cmd = _legacy_fineweb_download_command(cfg, "/ws", "python3", 1)
        # default variant is fineweb-like, so this should fire
        assert "fineweb" in cmd


# ---------------------------------------------------------------------------
# Integration sanity: three modality configurations
# ---------------------------------------------------------------------------


class TestThreeModalityConfigs:
    """Sanity-check that each of the main modality shapes produces the
    expected probe behavior end-to-end."""

    def test_lm_fineweb_style_gets_probe(self, tmp_path: Path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "name": "lm-test",
            "data": {
                "source": "huggingface",
                "variant": "fineweb10B_sp1024",
                "probe": {
                    "paths": ["data/tokenizers/fineweb_1024_bpe.model"],
                    "download_command": "python3 download_fineweb.py",
                },
            },
        }))
        cfg = load_config(yaml_path)
        probe_cmd = _build_data_probe_command(cfg, "/ws", "python3")
        assert probe_cmd is not None
        assert "fineweb_1024_bpe.model" in probe_cmd

    def test_generic_no_probe_is_noop(self, tmp_path: Path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "name": "generic-test",
            "data": {
                "source": "local_files",
                "path": "/tmp",
            },
        }))
        cfg = load_config(yaml_path)
        # No probe.paths, not a fineweb variant → no-op
        assert _build_data_probe_command(cfg, "/ws", "python3") is None

    def test_custom_probe_script(self, tmp_path: Path):
        yaml_path = tmp_path / "crucible.yaml"
        yaml_path.write_text(yaml.safe_dump({
            "name": "custom-test",
            "data": {
                "source": "huggingface",
                "probe": {
                    "script": "scripts/check_data.py",
                    "download_command": "python3 scripts/download.py",
                },
            },
        }))
        cfg = load_config(yaml_path)
        probe_cmd = _build_data_probe_command(cfg, "/ws", "python3")
        assert probe_cmd is not None
        assert "scripts/check_data.py" in probe_cmd
        assert cfg.data.probe.download_command == "python3 scripts/download.py"
