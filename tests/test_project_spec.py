"""Tests for ProjectSpec loading and listing."""
from pathlib import Path

import pytest
import yaml

from crucible.core.config import (
    PodOverrides,
    ProjectMetrics,
    ProjectSpec,
    load_project_spec,
    list_project_specs,
)


def _write_spec(tmp_path, name, content):
    d = tmp_path / ".crucible" / "projects"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(content), encoding="utf-8")
    return p


class TestLoadProjectSpec:
    def test_load_full_spec(self, tmp_path):
        _write_spec(tmp_path, "myproj", {
            "name": "myproj",
            "repo": "https://github.com/user/repo.git",
            "branch": "dev",
            "shallow": False,
            "workspace": "/workspace/myproj",
            "python": "3.11",
            "install": ["numpy", "torch"],
            "install_torch": "torch==2.6.0",
            "install_flags": "--index-url https://example.com",
            "system_packages": ["ffmpeg", "git"],
            "setup": ["echo hello"],
            "train": "python train.py",
            "timeout": 7200,
            "launch_timeout": 480,
            "env_forward": ["WANDB_API_KEY"],
            "env_set": {"FOO": "bar"},
            "pod": {"gpu_type": "A100", "container_disk": 40, "interruptible": False},
            "metrics": {"source": "wandb", "primary": "val_loss"},
        })
        spec = load_project_spec("myproj", tmp_path)
        assert spec.name == "myproj"
        assert spec.repo == "https://github.com/user/repo.git"
        assert spec.branch == "dev"
        assert spec.shallow is False
        assert spec.python == "3.11"
        assert spec.install == ["numpy", "torch"]
        assert spec.install_torch == "torch==2.6.0"
        assert spec.system_packages == ["ffmpeg", "git"]
        assert spec.train == "python train.py"
        assert spec.timeout == 7200
        assert spec.launch_timeout == 480
        assert spec.env_forward == ["WANDB_API_KEY"]
        assert spec.env_set == {"FOO": "bar"}
        assert spec.pod.gpu_type == "A100"
        assert spec.pod.container_disk == 40
        assert spec.pod.interruptible is False
        assert spec.metrics.source == "wandb"
        assert spec.metrics.primary == "val_loss"

    def test_load_minimal_spec(self, tmp_path):
        _write_spec(tmp_path, "minimal", {"name": "minimal", "repo": "https://example.com/r.git"})
        spec = load_project_spec("minimal", tmp_path)
        assert spec.name == "minimal"
        assert spec.branch == "main"
        assert spec.shallow is True
        assert spec.install == []
        assert spec.system_packages == []
        assert spec.pod.image == ""
        assert spec.pod.interruptible is None
        assert spec.launch_timeout == 300
        assert spec.metrics.direction == "minimize"

    def test_load_missing_spec(self, tmp_path):
        # After the C11 fix, the error message includes the searched paths
        # (local, hub, and any configured taps). Match the new format that
        # lists the missing spec name + "not found".
        with pytest.raises(FileNotFoundError, match=r"Project spec .* not found"):
            load_project_spec("nonexistent", tmp_path)

    def test_defaults_name_from_filename(self, tmp_path):
        _write_spec(tmp_path, "fromfile", {"repo": "https://example.com/r.git"})
        spec = load_project_spec("fromfile", tmp_path)
        assert spec.name == "fromfile"


class TestListProjectSpecs:
    def test_empty_dir(self, tmp_path):
        assert list_project_specs(tmp_path) == []

    def test_lists_multiple(self, tmp_path):
        _write_spec(tmp_path, "a", {"name": "a", "repo": "r1", "train": "t1"})
        _write_spec(tmp_path, "b", {"name": "b", "repo": "r2", "train": "t2"})
        specs = list_project_specs(tmp_path)
        assert len(specs) == 2
        names = {s["name"] for s in specs}
        assert names == {"a", "b"}

    def test_returns_summary_fields(self, tmp_path):
        _write_spec(tmp_path, "x", {
            "name": "x", "repo": "https://r.git", "train": "python t.py",
            "metrics": {"primary": "loss"},
        })
        specs = list_project_specs(tmp_path)
        assert specs[0]["repo"] == "https://r.git"
        assert specs[0]["train"] == "python t.py"
        assert specs[0]["metrics_primary"] == "loss"


class TestDefaults:
    def test_pod_overrides_defaults(self):
        p = PodOverrides()
        assert p.image == ""
        assert p.gpu_type == ""
        assert p.container_disk == 0
        assert p.interruptible is None

    def test_project_metrics_defaults(self):
        m = ProjectMetrics()
        assert m.source == "wandb"
        assert m.primary == "val_loss"
        assert m.direction == "minimize"
