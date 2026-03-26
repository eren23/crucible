"""Project-level integration tests for packaged example projects."""
from __future__ import annotations

import os
import shutil
import textwrap
from pathlib import Path

from crucible.core.config import load_config
from crucible.runner.experiment import run_experiment


def _copy_example_project(tmp_path: Path, name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    example_root = repo_root / "examples" / name
    project_root = tmp_path / name
    project_root.mkdir()

    for filename in ("crucible.yaml", "model.py", "data_adapter.py"):
        shutil.copy2(example_root / filename, project_root / filename)

    (project_root / "train_generic.py").write_text(
        textwrap.dedent(
            f"""
            from __future__ import annotations

            import sys
            from pathlib import Path


            def _bootstrap_import_paths() -> None:
                repo_root = Path({str(repo_root)!r})
                for path in (repo_root, repo_root / "src"):
                    path_str = str(path)
                    if path_str not in sys.path:
                        sys.path.insert(0, path_str)


            def main() -> None:
                _bootstrap_import_paths()

                from data_adapter import register as register_adapter
                from model import register as register_model
                from crucible.training.generic_backend import run_generic_training

                register_model()
                register_adapter()
                run_generic_training()


            if __name__ == "__main__":
                main()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return project_root


def _run_project(project_root: Path, *, preset: str = "smoke", overrides: dict[str, str]) -> dict:
    cfg = load_config(project_root / "crucible.yaml")
    return run_experiment(
        config=overrides,
        name=f"{project_root.name}-integration",
        backend="generic",
        preset=preset,
        project_root=project_root,
        project_config=cfg,
        stream_output=False,
        timeout_seconds=45,
        results_file=project_root / "tmp_results.jsonl",
    )


def _write_torchvision_stub(tmp_path: Path) -> Path:
    stub_root = tmp_path / "stub"
    package_dir = stub_root / "torchvision"
    package_dir.mkdir(parents=True)

    (package_dir / "__init__.py").write_text(
        "from . import datasets, transforms\n",
        encoding="utf-8",
    )
    (package_dir / "datasets.py").write_text(
        textwrap.dedent(
            """
            import torch


            class MNIST:
                def __init__(self, root, train=True, download=True, transform=None):
                    self.transform = transform
                    self.samples = [
                        torch.zeros(1, 28, 28),
                        torch.ones(1, 28, 28),
                    ]

                def __len__(self):
                    return len(self.samples)

                def __getitem__(self, idx):
                    image = self.samples[idx % len(self.samples)].clone()
                    if self.transform is not None:
                        image = self.transform(image)
                    return image, 0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    (package_dir / "transforms.py").write_text(
        textwrap.dedent(
            """
            class Compose:
                def __init__(self, transforms):
                    self.transforms = transforms

                def __call__(self, value):
                    for transform in self.transforms:
                        value = transform(value)
                    return value


            class ToTensor:
                def __call__(self, value):
                    return value


            class Normalize:
                def __init__(self, mean, std):
                    self.mean = mean[0]
                    self.std = std[0]

                def __call__(self, value):
                    return (value - self.mean) / self.std
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return stub_root


def test_world_model_example_runs_as_project(tmp_path: Path):
    project_root = _copy_example_project(tmp_path, "world_model")

    result = _run_project(
        project_root,
        overrides={
            "ITERATIONS": "1",
            "VAL_INTERVAL": "1",
            "LOG_INTERVAL": "1",
            "BATCH_SIZE": "2",
            "NUM_FRAMES": "3",
            "IMAGE_SIZE": "16",
            "MODEL_DIM": "16",
            "BASE_CHANNELS": "8",
            "PREDICTOR_HIDDEN": "32",
            "MAX_WALLCLOCK_SECONDS": "20",
        },
    )

    assert result["status"] == "completed"
    assert result["result"] is not None
    assert "pred_loss" in result["result"]


def test_diffusion_example_runs_as_project_with_stubbed_torchvision(tmp_path: Path):
    project_root = _copy_example_project(tmp_path, "diffusion")
    stub_root = _write_torchvision_stub(tmp_path)
    repo_src = Path(__file__).resolve().parents[1] / "src"

    result = _run_project(
        project_root,
        overrides={
            "PYTHONPATH": os.pathsep.join([str(stub_root), str(repo_src)]),
            "ITERATIONS": "1",
            "VAL_INTERVAL": "1",
            "LOG_INTERVAL": "1",
            "BATCH_SIZE": "2",
            "MODEL_DIM": "16",
            "DIFFUSION_STEPS": "10",
            "MAX_WALLCLOCK_SECONDS": "20",
        },
    )

    assert result["status"] == "completed"
    assert result["result"] is not None
    assert "noise_mse" in result["result"]
