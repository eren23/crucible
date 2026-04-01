"""Precompute PushT HDF5 dataset into a HuggingFace dataset with transforms baked in.

Runs inside the upstream ``lucas-maes/le-wm`` workspace (same as the launcher).
Loads the HDF5 data via ``stable_worldmodel``, applies the same transforms
(image resize + column normalization) that the launcher applies at training
time, and saves the result as a HuggingFace ``datasets.Dataset`` with
train / validation splits.

Env vars
--------
HF_PUSH_REPO : str, optional
    If set, push the resulting dataset to this HuggingFace Hub repo
    (e.g. ``eren23/pusht-lewm-precomputed``).
HF_SAVE_DIR : str, optional
    Local directory to save the dataset (default: ``/root/stable-wm-data/precomputed``).

Usage (from the le-wm workspace root)::

    python precompute_pusht.py data=pusht
    # or with hub push:
    HF_PUSH_REPO=eren23/pusht-lewm-precomputed python precompute_pusht.py data=pusht
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path.cwd()))


# ---------------------------------------------------------------------------
# Hydra config loading (identical to the launcher)
# ---------------------------------------------------------------------------

def _load_config(dotlist: list[str]) -> Any:
    from hydra import compose, initialize_config_dir

    config_dir = Path.cwd() / "config" / "train"
    if not config_dir.exists():
        raise FileNotFoundError(f"Expected upstream LE-WM config directory at {config_dir}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="lewm", overrides=dotlist)


# ---------------------------------------------------------------------------
# Main precompute logic
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    import torch
    import stable_pretraining as spt
    import stable_worldmodel as swm
    from datasets import Dataset, DatasetDict, Features, Array3D, Array2D, Value

    from utils import get_column_normalizer, get_img_preprocessor

    args = list(sys.argv[1:] if argv is None else argv)
    cfg = _load_config(args)

    save_dir = os.environ.get("HF_SAVE_DIR", "/root/stable-wm-data/precomputed")
    push_repo = os.environ.get("HF_PUSH_REPO", "")

    # -----------------------------------------------------------------------
    # 1. Load HDF5 dataset (no transform yet -- we apply manually per sample)
    # -----------------------------------------------------------------------
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)

    # -----------------------------------------------------------------------
    # 2. Build the same transforms the launcher uses
    # -----------------------------------------------------------------------
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]

    normalizer_stats: dict[str, dict[str, list[float]]] = {}

    from omegaconf import open_dict
    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue
            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.wm, f"{col}_dim", dataset.get_dim(col))

            # Extract normalizer statistics for metadata
            if hasattr(normalizer, "mean") and hasattr(normalizer, "std"):
                normalizer_stats[col] = {
                    "mean": normalizer.mean.numpy().tolist() if hasattr(normalizer.mean, "numpy") else list(normalizer.mean),
                    "std": normalizer.std.numpy().tolist() if hasattr(normalizer.std, "numpy") else list(normalizer.std),
                }
            elif hasattr(normalizer, "transform"):
                inner = normalizer.transform
                if hasattr(inner, "mean") and hasattr(inner, "std"):
                    m = inner.mean
                    s = inner.std
                    normalizer_stats[col] = {
                        "mean": m.numpy().tolist() if hasattr(m, "numpy") else list(m),
                        "std": s.numpy().tolist() if hasattr(s, "numpy") else list(s),
                    }

    compose_transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = compose_transform

    # -----------------------------------------------------------------------
    # 3. Split with the same seed as the launcher
    # -----------------------------------------------------------------------
    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    # -----------------------------------------------------------------------
    # 4. Stream via DataLoader → HF Dataset.from_generator (no OOM)
    # -----------------------------------------------------------------------
    metadata = {
        "img_size": int(cfg.img_size),
        "seed": int(cfg.seed),
        "train_split": float(cfg.train_split),
        "normalizer_stats": normalizer_stats,
        "source": "precompute_pusht.py",
        "keys_to_load": list(cfg.data.dataset.keys_to_load),
    }

    def _make_generator(split_dataset: Any, split_name: str):
        """Yields one sample at a time — never accumulates in RAM."""
        loader = torch.utils.data.DataLoader(
            split_dataset, batch_size=1, shuffle=False, num_workers=4,
            persistent_workers=True,
        )
        n = len(split_dataset)
        print(f"Processing {split_name}: {n} samples", flush=True)
        for i, batch in enumerate(loader):
            if i % 5000 == 0:
                print(f"  {split_name}: {i}/{n}", flush=True)
            yield {
                "pixels": batch["pixels"].squeeze(0).numpy().astype(np.float32),
                "action": batch["action"].squeeze(0).numpy().astype(np.float32),
            }
        print(f"  {split_name}: done ({n} samples)", flush=True)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    print("Building train split...", flush=True)
    train_ds = Dataset.from_generator(
        lambda: _make_generator(train_set, "train"),
        cache_dir=str(save_path / "_cache"),
    )
    train_ds.info.description = json.dumps(metadata, indent=2)

    print("Building validation split...", flush=True)
    val_ds = Dataset.from_generator(
        lambda: _make_generator(val_set, "validation"),
        cache_dir=str(save_path / "_cache"),
    )
    val_ds.info.description = json.dumps(metadata, indent=2)

    ds_dict = DatasetDict({"train": train_ds, "validation": val_ds})

    # -----------------------------------------------------------------------
    # 6. Save locally
    # -----------------------------------------------------------------------
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(save_path))
    print(f"Saved precomputed dataset to {save_path}")

    # Also save normalizer stats as a standalone JSON for easy access
    stats_path = save_path / "normalizer_stats.json"
    stats_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved normalizer stats to {stats_path}")

    # -----------------------------------------------------------------------
    # 7. Optionally push to HF Hub
    # -----------------------------------------------------------------------
    if push_repo:
        print(f"Pushing to HuggingFace Hub: {push_repo}")
        ds_dict.push_to_hub(push_repo)
        print(f"Pushed to {push_repo}")


if __name__ == "__main__":
    main()
