"""End-to-end tests for architecture plugin tap workflow.

Tests the full round-trip: create tap with architecture plugins -> add tap ->
search -> install -> verify registration -> uninstall -> verify removal.

Uses local git repos in tmp_path — no network access needed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from crucible.core.tap import TapManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arch_tap(tmp_path: Path) -> Path:
    """Create a local tap repo with 4 architecture plugins matching the community tap."""
    tap_dir = tmp_path / "community-tap"
    tap_dir.mkdir()

    plugins = [
        {
            "name": "test_moe",
            "type": "architectures",
            "version": "1.0.0",
            "description": "MoE architecture test plugin",
            "author": "test",
            "tags": ["moe", "architecture"],
            "code": (
                "# MoE test plugin\n"
                "def register(): pass\n"
            ),
        },
        {
            "name": "test_partial_rope",
            "type": "architectures",
            "version": "1.0.0",
            "description": "Partial RoPE test plugin for SOTA experiments",
            "author": "test",
            "tags": ["partial-rope", "sota"],
            "code": (
                "# Partial RoPE test plugin\n"
                "def register(): pass\n"
            ),
        },
        {
            "name": "test_looped",
            "type": "architectures",
            "version": "1.0.0",
            "description": "Looped augmented test plugin (declarative)",
            "author": "test",
            "tags": ["looped", "declarative"],
            "code": (
                "# Looped augmented test wrapper\n"
                "def register(): pass\n"
            ),
        },
        {
            "name": "test_sota_v1",
            "type": "architectures",
            "version": "1.0.0",
            "description": "SOTA inspired v1 test plugin (declarative)",
            "author": "test",
            "tags": ["sota", "competition"],
            "code": (
                "# SOTA v1 test wrapper\n"
                "def register(): pass\n"
            ),
        },
    ]

    for p in plugins:
        pkg_dir = tap_dir / p["type"] / p["name"]
        pkg_dir.mkdir(parents=True)
        manifest = {k: v for k, v in p.items() if k != "code"}
        (pkg_dir / "plugin.yaml").write_text(yaml.dump(manifest))
        (pkg_dir / f"{p['name']}.py").write_text(p["code"])

    subprocess.run(["git", "init"], cwd=str(tap_dir), capture_output=True, check=True)
    subprocess.run(["git", "add", "."], cwd=str(tap_dir), capture_output=True, check=True)
    subprocess.run(
        ["git", "-c", "user.email=test@test.com", "-c", "user.name=test",
         "commit", "-m", "init 4 architecture plugins"],
        cwd=str(tap_dir), capture_output=True, check=True,
    )
    return tap_dir


@pytest.fixture()
def hub(tmp_path: Path) -> Path:
    hd = tmp_path / "hub"
    hd.mkdir()
    (hd / "taps").mkdir()
    (hd / "plugins").mkdir()
    return hd


@pytest.fixture()
def tm(hub: Path) -> TapManager:
    return TapManager(hub)


@pytest.fixture()
def arch_tap(tmp_path: Path) -> Path:
    return _make_arch_tap(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestArchitectureTapDiscovery:
    """Tap can discover all architecture plugins."""

    def test_search_finds_all_four(self, tm: TapManager, arch_tap: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        results = tm.search()
        assert len(results) == 4
        names = {r["name"] for r in results}
        assert names == {"test_moe", "test_partial_rope", "test_looped", "test_sota_v1"}

    def test_search_by_type_filters(self, tm: TapManager, arch_tap: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        results = tm.search("", plugin_type="architectures")
        assert len(results) == 4
        results = tm.search("", plugin_type="optimizers")
        assert len(results) == 0

    def test_search_by_keyword(self, tm: TapManager, arch_tap: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        assert len(tm.search("moe")) == 1
        assert tm.search("moe")[0]["name"] == "test_moe"
        assert len(tm.search("sota")) == 2  # test_sota_v1 + test_partial_rope (has 'sota' tag)
        assert len(tm.search("looped")) == 1

    def test_search_by_tag(self, tm: TapManager, arch_tap: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        results = tm.search("competition")
        assert len(results) == 1
        assert results[0]["name"] == "test_sota_v1"


class TestArchitectureTapInstall:
    """Install/uninstall round-trip for architecture plugins."""

    def test_install_creates_file(self, tm: TapManager, arch_tap: Path, hub: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        result = tm.install("test_moe")
        assert result["status"] == "installed"
        assert result["type"] == "architectures"
        dest = hub / "plugins" / "architectures" / "test_moe.py"
        assert dest.exists()
        assert "MoE test plugin" in dest.read_text()

    def test_install_all_four(self, tm: TapManager, arch_tap: Path, hub: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        for name in ["test_moe", "test_partial_rope", "test_looped", "test_sota_v1"]:
            result = tm.install(name)
            assert result["status"] == "installed"
        assert len(tm.list_installed()) == 4
        assert len(list((hub / "plugins" / "architectures").glob("*.py"))) == 4

    def test_uninstall_removes_file(self, tm: TapManager, arch_tap: Path, hub: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        tm.install("test_moe")
        dest = hub / "plugins" / "architectures" / "test_moe.py"
        assert dest.exists()
        tm.uninstall("test_moe")
        assert not dest.exists()
        assert len(tm.list_installed()) == 0

    def test_install_uninstall_roundtrip(self, tm: TapManager, arch_tap: Path, hub: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        names = ["test_moe", "test_partial_rope", "test_looped", "test_sota_v1"]
        # Install all
        for name in names:
            tm.install(name)
        assert len(tm.list_installed()) == 4
        # Uninstall all
        for name in names:
            tm.uninstall(name)
        assert len(tm.list_installed()) == 0
        # Reinstall all
        for name in names:
            tm.install(name)
        assert len(tm.list_installed()) == 4

    def test_duplicate_install_raises(self, tm: TapManager, arch_tap: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        tm.install("test_moe")
        from crucible.core.errors import TapError
        with pytest.raises(TapError, match="already installed"):
            tm.install("test_moe")


class TestArchitectureTapPackageInfo:
    """Package info correctly reports install status."""

    def test_info_before_install(self, tm: TapManager, arch_tap: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        info = tm.get_package_info("test_moe")
        assert info is not None
        assert info["name"] == "test_moe"
        assert info["installed"] is False

    def test_info_after_install(self, tm: TapManager, arch_tap: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        tm.install("test_moe")
        info = tm.get_package_info("test_moe")
        assert info["installed"] is True
        assert info["version"] == "1.0.0"
        assert "moe" in info["tags"]

    def test_info_after_uninstall(self, tm: TapManager, arch_tap: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        tm.install("test_moe")
        tm.uninstall("test_moe")
        info = tm.get_package_info("test_moe")
        assert info["installed"] is False


class TestArchitectureTapSync:
    """Tap sync pulls latest changes."""

    def test_sync_updates_tap(self, tm: TapManager, arch_tap: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")

        # Add a new plugin to the source
        new_dir = arch_tap / "architectures" / "test_new_arch"
        new_dir.mkdir(parents=True)
        (new_dir / "plugin.yaml").write_text(yaml.dump({
            "name": "test_new_arch", "type": "architectures",
            "version": "0.1.0", "description": "Brand new arch",
        }))
        (new_dir / "test_new_arch.py").write_text("# new\n")
        subprocess.run(["git", "add", "."], cwd=str(arch_tap), capture_output=True, check=True)
        subprocess.run(
            ["git", "-c", "user.email=test@test.com", "-c", "user.name=test",
             "commit", "-m", "add test_new_arch"],
            cwd=str(arch_tap), capture_output=True, check=True,
        )

        # Sync
        result = tm.sync_tap("arch-tap")
        assert "arch-tap" in result["synced"]

        # New plugin should be discoverable
        results = tm.search("test_new_arch")
        assert len(results) == 1
        assert results[0]["name"] == "test_new_arch"

    def test_sync_all_taps(self, tm: TapManager, arch_tap: Path):
        tm.add_tap(str(arch_tap), name="arch-tap")
        result = tm.sync_tap()
        assert "arch-tap" in result["synced"]
        assert len(result["errors"]) == 0
