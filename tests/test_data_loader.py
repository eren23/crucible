"""Tests for crucible.training.data_loader."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from crucible.training.data_loader import count_shard_tokens


def _write_fw_shard(path: Path, magic: int, token_count: int) -> None:
    """Write a minimal FineWeb binary shard: 256-int32 header + token_count uint16 tokens."""
    with open(path, "wb") as fh:
        header = np.zeros(256, dtype="<i4")
        header[0] = magic
        header[2] = token_count
        header.tofile(fh)
        np.zeros(token_count, dtype="<u2").tofile(fh)


class TestCountShardTokens:
    def test_happy_path_multiple_shards(self, tmp_path: Path) -> None:
        _write_fw_shard(tmp_path / "shard_00.bin", 20240520, 1000)
        _write_fw_shard(tmp_path / "shard_01.bin", 20240520, 2000)
        assert count_shard_tokens(str(tmp_path) + "/shard_" + "*.bin") == 3000

    def test_wrong_magic_ignored(self, tmp_path: Path) -> None:
        _write_fw_shard(tmp_path / "bad.bin", 999999, 1000)
        assert count_shard_tokens(str(tmp_path / "bad.bin")) == 0

    def test_shard_limit_respected(self, tmp_path: Path) -> None:
        _write_fw_shard(tmp_path / "a.bin", 20240520, 100)
        _write_fw_shard(tmp_path / "b.bin", 20240520, 200)
        assert count_shard_tokens(str(tmp_path) + "/*.bin", shard_limit=1) == 100

    def test_no_matching_files_returns_zero(self, tmp_path: Path) -> None:
        assert count_shard_tokens(str(tmp_path / "nonexistent_*.bin")) == 0

    def test_truncated_header_returns_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.bin"
        path.write_bytes(b"\x00\x00\x00\x00")
        assert count_shard_tokens(str(path)) == 0
