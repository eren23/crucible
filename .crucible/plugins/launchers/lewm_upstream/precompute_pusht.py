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
    (e.g. ``eren23/lewm-pusht-processed``).
HF_SAVE_DIR : str, optional
    Local directory to save the dataset (default: ``/root/stable-wm-data/precomputed``).
HF_TOKEN : str, optional
    HuggingFace API token for pushing to private repos.  Falls back to
    the cached token from ``huggingface-cli login`` if not set.
MAX_SAMPLES : int, optional
    If set, cap total samples before splitting (e.g. 10000 for a 10k subset).
    Useful to keep disk usage manageable on small pods.

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
    print("step:0/4 precompute:loading_hdf5", flush=True)
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
    # 3b. Optionally cap total samples (keeps disk manageable on small pods)
    # -----------------------------------------------------------------------
    max_samples_raw = os.environ.get("MAX_SAMPLES", "")
    max_samples = int(max_samples_raw) if max_samples_raw else 0
    if max_samples > 0:
        total = len(train_set) + len(val_set)
        if max_samples < total:
            train_cap = int(max_samples * cfg.train_split)
            val_cap = max_samples - train_cap
            train_set = torch.utils.data.Subset(train_set, range(min(train_cap, len(train_set))))
            val_set = torch.utils.data.Subset(val_set, range(min(val_cap, len(val_set))))
            print(f"Capped to {max_samples} samples: {len(train_set)} train, {len(val_set)} val", flush=True)

    # -----------------------------------------------------------------------
    # 4. Stream via DataLoader → HF Dataset.from_generator (no OOM)
    # -----------------------------------------------------------------------
    metadata = {
        "img_size": int(cfg.img_size),
        "seed": int(cfg.seed),
        "train_split": float(cfg.train_split),
        "max_samples": max_samples or None,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
        "normalizer_stats": normalizer_stats,
        "source": "precompute_pusht.py",
        "keys_to_load": list(cfg.data.dataset.keys_to_load),
    }

    columns = list(cfg.data.dataset.keys_to_load)
    print(f"Columns to store: {columns}", flush=True)

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
            sample = {}
            for col in columns:
                if col in batch:
                    sample[col] = batch[col].squeeze(0).numpy().astype(np.float32)
            yield sample
        print(f"  {split_name}: done ({n} samples)", flush=True)

    import shutil

    cache_path = Path(save_dir) / "_cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None

    # Build train split, push immediately, then free disk
    print("step:1/4 precompute:building_train_split", flush=True)
    train_ds = Dataset.from_generator(
        lambda: _make_generator(train_set, "train"),
        cache_dir=str(cache_path),
    )
    train_ds.info.description = json.dumps(metadata, indent=2)

    if push_repo:
        print("step:2/4 precompute:pushing_train_split", flush=True)
        train_ds.push_to_hub(push_repo, split="train", private=True, token=token)
        print(f"Pushed train split ({len(train_ds)} samples) to {push_repo}")
    del train_ds
    # Free cache to reclaim disk for val split
    shutil.rmtree(cache_path, ignore_errors=True)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Build val split, push, then free disk
    print("step:3/4 precompute:building_val_split", flush=True)
    val_ds = Dataset.from_generator(
        lambda: _make_generator(val_set, "validation"),
        cache_dir=str(cache_path),
    )
    val_ds.info.description = json.dumps(metadata, indent=2)

    if push_repo:
        print("Pushing validation split...", flush=True)
        val_ds.push_to_hub(push_repo, split="validation", private=True, token=token)
        print(f"Pushed validation split ({len(val_ds)} samples) to {push_repo}")
    del val_ds
    shutil.rmtree(cache_path, ignore_errors=True)

    # Save normalizer stats as standalone JSON (tiny file)
    stats_path = Path(save_dir) / "normalizer_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved normalizer stats to {stats_path}")

    print("step:4/4 precompute:done", flush=True)


if __name__ == "__main__":
    main()
