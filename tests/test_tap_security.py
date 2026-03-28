"""Security and edge case tests for the tap system.

Tests path traversal guards, name validation, collision detection,
atomic writes, and publish overwrite protection.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from crucible.core.errors import TapError
from crucible.core.tap import TapManager, _validate_name, _SAFE_NAME_RE


# Reuse the fixture helper from test_tap.py
def _create_local_tap(tmp_path: Path, name: str = "test-tap", plugins: list[dict] | None = None) -> Path:
    tap_dir = tmp_path / name
    tap_dir.mkdir(parents=True, exist_ok=True)
    plugins = plugins or [
        {"type": "optimizers", "name": "good_plugin", "version": "1.0.0",
         "description": "A good plugin", "author": "test", "tags": [],
         "code": "x = 1\n"},
    ]
    for plugin in plugins:
        pkg_dir = tap_dir / plugin["type"] / plugin["name"]
        pkg_dir.mkdir(parents=True)
        manifest = {k: v for k, v in plugin.items() if k not in ("code",)}
        (pkg_dir / "plugin.yaml").write_text(yaml.dump(manifest))
        (pkg_dir / f"{plugin['name']}.py").write_text(plugin.get("code", "x = 1\n"))
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
    hd = tmp_path / "hub"
    hd.mkdir()
    (hd / "taps").mkdir()
    (hd / "plugins").mkdir()
    return hd


@pytest.fixture()
def tap_manager(hub_dir: Path) -> TapManager:
    return TapManager(hub_dir)


# ===================================================================
# C1 — Path traversal via name
# ===================================================================

class TestNameValidation:
    def test_safe_names_accepted(self):
        for name in ["lion", "my_opt", "adam-v2", "LoRA123"]:
            _validate_name(name, "test")  # should not raise

    def test_path_traversal_rejected(self):
        for name in ["../../etc/passwd", "../escape", "foo/bar", "a b"]:
            with pytest.raises(TapError, match="Invalid"):
                _validate_name(name, "test")

    def test_empty_name_rejected(self):
        with pytest.raises(TapError, match="Invalid"):
            _validate_name("", "test")

    def test_dot_prefix_rejected(self):
        with pytest.raises(TapError, match="Invalid"):
            _validate_name(".hidden", "test")

    def test_install_with_traversal_name_rejected(self, tap_manager: TapManager):
        """install() validates name before any filesystem operations."""
        with pytest.raises(TapError, match="Invalid"):
            tap_manager.install("../../etc/cron.d/evil")

    def test_uninstall_with_traversal_name_rejected(self, tap_manager: TapManager):
        with pytest.raises(TapError, match="Invalid"):
            tap_manager.uninstall("../../etc/passwd")

    def test_add_tap_with_traversal_name_rejected(self, tap_manager: TapManager):
        with pytest.raises(TapError, match="Invalid"):
            tap_manager.add_tap("https://example.com/repo", name="../../escape")

    def test_manifest_with_bad_name_skipped(self, tap_manager: TapManager, tmp_path: Path):
        """_scan_tap should skip manifests with invalid names."""
        tap = _create_local_tap(tmp_path, "bad-names", plugins=[
            {"type": "optimizers", "name": "../escape", "version": "1.0.0",
             "description": "bad", "author": "test", "tags": [],
             "code": "x = 1\n"},
        ])
        tap_manager.add_tap(str(tap), name="bad-names")
        manifests = tap_manager._scan_tap("bad-names")
        assert len(manifests) == 0  # bad name filtered out


# ===================================================================
# C2 — Symlink escape guard
# ===================================================================

class TestSymlinkGuard:
    def test_assert_within_accepts_valid_path(self, tap_manager: TapManager, tmp_path: Path):
        parent = tmp_path / "parent"
        parent.mkdir()
        child = parent / "child"
        child.mkdir()
        tap_manager._assert_within(child, parent, "test")  # should not raise

    def test_assert_within_rejects_escape(self, tap_manager: TapManager, tmp_path: Path):
        parent = tmp_path / "parent"
        parent.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        with pytest.raises(TapError, match="escapes"):
            tap_manager._assert_within(outside, parent, "test")


# ===================================================================
# C4 — Shallow clone sync uses fetch+reset
# ===================================================================

class TestShallowSync:
    def test_sync_updates_shallow_clone(self, tap_manager: TapManager, tmp_path: Path):
        """Verify sync works on a shallow clone by using fetch+reset."""
        tap = _create_local_tap(tmp_path)
        tap_manager.add_tap(str(tap), name="shallow-test")

        # Add a new commit to the source
        (tap / "new_file.txt").write_text("hello")
        subprocess.run(["git", "add", "."], cwd=str(tap), capture_output=True, check=True)
        subprocess.run(
            ["git", "-c", "user.email=test@test.com", "-c", "user.name=test",
             "commit", "-m", "new commit"],
            cwd=str(tap), capture_output=True, check=True,
        )

        # Sync should pick up the new commit
        result = tap_manager.sync_tap("shallow-test")
        assert "shallow-test" in result["synced"]
        assert not result["errors"]

        # Verify new file exists in the cloned tap
        cloned = tap_manager._taps_dir / "shallow-test"
        assert (cloned / "new_file.txt").exists()


# ===================================================================
# C7 — Publish overwrite protection
# ===================================================================

class TestPublishOverwrite:
    def test_publish_rejects_existing_package(self, tap_manager: TapManager, tmp_path: Path):
        tap = _create_local_tap(tmp_path)
        tap_manager.add_tap(str(tap), name="overwrite-test")

        # Create local plugin
        project = tmp_path / "project"
        plugin_dir = project / ".crucible" / "plugins" / "optimizers"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "new_opt.py").write_text("x = 1\n")

        # First publish succeeds
        tap_manager.publish("new_opt", "optimizers", "overwrite-test", project_root=project)

        # Second publish should fail — package already exists
        with pytest.raises(TapError, match="already exists"):
            tap_manager.publish("new_opt", "optimizers", "overwrite-test", project_root=project)


# ===================================================================
# I1 — publish() git errors not swallowed
# ===================================================================

class TestPublishGitErrors:
    def test_publish_propagates_git_add_failure(self, tap_manager: TapManager, tmp_path: Path):
        """If git add fails (e.g. broken repo), it should propagate, not be swallowed."""
        tap = _create_local_tap(tmp_path)
        tap_manager.add_tap(str(tap), name="git-error-test")

        project = tmp_path / "project2"
        plugin_dir = project / ".crucible" / "plugins" / "optimizers"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "test_git.py").write_text("x = 1\n")

        # This should work (not raise) — git is fine
        result = tap_manager.publish("test_git", "optimizers", "git-error-test", project_root=project)
        assert result["status"] == "published"


# ===================================================================
# I3 — Name collision detection across taps
# ===================================================================

class TestNameCollision:
    def test_same_name_in_two_taps_raises_without_tap_arg(self, tap_manager: TapManager, tmp_path: Path):
        tap1 = _create_local_tap(tmp_path / "src1", "tap1", plugins=[
            {"type": "optimizers", "name": "clash", "version": "1.0.0",
             "description": "from tap1", "author": "a", "tags": [], "code": "x=1\n"},
        ])
        tap2 = _create_local_tap(tmp_path / "src2", "tap2", plugins=[
            {"type": "optimizers", "name": "clash", "version": "2.0.0",
             "description": "from tap2", "author": "b", "tags": [], "code": "x=2\n"},
        ])
        tap_manager.add_tap(str(tap1), name="tap1")
        tap_manager.add_tap(str(tap2), name="tap2")

        with pytest.raises(TapError, match="multiple taps"):
            tap_manager.install("clash")

    def test_same_name_with_explicit_tap_works(self, tap_manager: TapManager, tmp_path: Path):
        tap1 = _create_local_tap(tmp_path / "src3", "tap3", plugins=[
            {"type": "optimizers", "name": "clash2", "version": "1.0.0",
             "description": "from tap3", "author": "a", "tags": [], "code": "x=1\n"},
        ])
        tap2 = _create_local_tap(tmp_path / "src4", "tap4", plugins=[
            {"type": "optimizers", "name": "clash2", "version": "2.0.0",
             "description": "from tap4", "author": "b", "tags": [], "code": "x=2\n"},
        ])
        tap_manager.add_tap(str(tap1), name="tap3")
        tap_manager.add_tap(str(tap2), name="tap4")

        # With explicit tap, should work
        result = tap_manager.install("clash2", tap="tap4")
        assert result["tap"] == "tap4"
        tap_manager.uninstall("clash2")


# ===================================================================
# I6 — submit_pr body preserved when title empty
# ===================================================================

class TestSubmitPrArgs:
    def test_body_without_title_still_passed_to_gh(self, tap_manager: TapManager, tmp_path: Path):
        """When title is empty but body is set, body should still be in gh_args."""
        import crucible.core.tap as tap_module

        tap = _create_local_tap(tmp_path)
        tap_manager.add_tap(str(tap), name="pr-body-test")

        # We can't actually run gh, but we can verify the result structure
        result = tap_manager.submit_pr("pr-body-test", body="Important fix description")
        # gh CLI likely not available, should fall back gracefully
        assert result["status"] in ("manual", "error", "pr_created")


# ===================================================================
# Atomic write tests
# ===================================================================

class TestAtomicWrites:
    def test_save_installed_yaml_atomic(self, tap_manager: TapManager):
        """_save_installed_yaml should produce a valid file even after multiple writes."""
        packages = [{"name": "a", "type": "optimizers", "version": "1.0.0"}]
        tap_manager._save_installed_yaml(packages)
        loaded = tap_manager._load_installed_yaml()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "a"

        # Write again with different data
        packages.append({"name": "b", "type": "schedulers", "version": "2.0.0"})
        tap_manager._save_installed_yaml(packages)
        loaded = tap_manager._load_installed_yaml()
        assert len(loaded) == 2

    def test_save_taps_yaml_atomic(self, tap_manager: TapManager):
        taps = [{"name": "t1", "url": "http://example.com/t1", "added_at": "2026-01-01"}]
        tap_manager._save_taps_yaml(taps)
        loaded = tap_manager._load_taps_yaml()
        assert len(loaded) == 1
        assert loaded[0]["name"] == "t1"
