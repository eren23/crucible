"""Community plugin tap manager — Homebrew-style git-based package sharing.

A "tap" is a git repository containing plugins organized by type, each with
a ``plugin.yaml`` manifest.  The :class:`TapManager` handles cloning taps,
searching for packages, installing plugins into the hub's global ``plugins/``
directory (where ``discover_all_plugins`` already scans), and publishing
local plugins to a tap repo.

Usage::

    from crucible.core.tap import TapManager
    tm = TapManager(hub_dir=Path("~/.crucible-hub"))
    tm.add_tap("https://github.com/user/crucible-community-tap")
    results = tm.search("lion")
    tm.install("lion")
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

from crucible.core.errors import TapError
from crucible.core.log import utc_now_iso

VALID_PLUGIN_TYPES = frozenset({
    "optimizers", "schedulers", "callbacks", "loggers",
    "providers", "architectures", "data_adapters", "data_sources",
    "objectives", "block_types", "stack_patterns", "augmentations",
    "activations", "launchers", "evaluations",
})

# Plugin types that ship as multi-file directory bundles (copied whole)
# rather than single `.py` files. Keep in sync with the install/uninstall
# branches below.
BUNDLE_PLUGIN_TYPES = frozenset({"launchers", "evaluations"})

_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")

_DEFAULT_HUB_DIR = Path.home() / ".crucible-hub"


def _validate_name(name: str, context: str = "plugin") -> None:
    """Raise TapError if *name* contains unsafe characters (path traversal guard)."""
    if not name or not _SAFE_NAME_RE.match(name):
        raise TapError(
            f"Invalid {context} name {name!r}: must match [a-zA-Z0-9][a-zA-Z0-9_-]*"
        )


class TapManager:
    """Manages community plugin taps and package installation."""

    def __init__(self, hub_dir: Path | None = None) -> None:
        self.hub_dir = Path(hub_dir or _DEFAULT_HUB_DIR)
        self._taps_dir = self.hub_dir / "taps"
        self._plugins_dir = self.hub_dir / "plugins"
        self._taps_file = self.hub_dir / "taps.yaml"
        self._installed_file = self.hub_dir / "installed.yaml"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_taps_yaml(self) -> list[dict[str, Any]]:
        if not self._taps_file.exists():
            return []
        data = yaml.safe_load(self._taps_file.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _save_taps_yaml(self, taps: list[dict[str, Any]]) -> None:
        """Atomic write of taps.yaml (write-then-rename)."""
        content = yaml.dump(taps, default_flow_style=False, sort_keys=False)
        fd, tmp = tempfile.mkstemp(dir=str(self.hub_dir), suffix=".tmp")
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            os.replace(tmp, str(self._taps_file))
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _load_installed_yaml(self) -> list[dict[str, Any]]:
        if not self._installed_file.exists():
            return []
        data = yaml.safe_load(self._installed_file.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []

    def _save_installed_yaml(self, packages: list[dict[str, Any]]) -> None:
        """Atomic write of installed.yaml (write-then-rename)."""
        content = yaml.dump(packages, default_flow_style=False, sort_keys=False)
        fd, tmp = tempfile.mkstemp(dir=str(self.hub_dir), suffix=".tmp")
        try:
            os.write(fd, content.encode("utf-8"))
            os.close(fd)
            os.replace(tmp, str(self._installed_file))
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    def _git_run(self, *args: str, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=str(cwd or self.hub_dir),
            capture_output=True,
            text=True,
            check=check,
        )

    @staticmethod
    def _name_from_url(url: str) -> str:
        """Derive a tap name from a git URL."""
        slug = url.rstrip("/").split("/")[-1]
        if slug.endswith(".git"):
            slug = slug[:-4]
        return slug

    def _get_tap_sha(self, tap_name: str) -> str:
        """Get the current HEAD SHA of a tap repo."""
        tap_dir = self._taps_dir / tap_name
        try:
            result = self._git_run("rev-parse", "HEAD", cwd=tap_dir, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def _assert_within(self, path: Path, parent: Path, context: str = "path") -> None:
        """Ensure *path* resolves under *parent* (symlink escape guard)."""
        try:
            path.resolve().relative_to(parent.resolve())
        except ValueError:
            raise TapError(f"{context} escapes allowed directory: {path}")

    # ------------------------------------------------------------------
    # Tap CRUD
    # ------------------------------------------------------------------

    def add_tap(self, url: str, *, name: str = "") -> dict[str, Any]:
        """Clone a tap repository.

        Returns tap metadata dict.
        Raises :class:`TapError` if the tap already exists.
        """
        name = name or self._name_from_url(url)
        _validate_name(name, "tap")
        tap_dir = self._taps_dir / name

        if tap_dir.exists():
            raise TapError(f"Tap {name!r} already exists at {tap_dir}")

        self._taps_dir.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(tap_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise TapError(f"Failed to clone {url}: {exc.stderr.strip()}") from exc

        now = utc_now_iso()
        tap_info: dict[str, Any] = {
            "name": name,
            "url": url,
            "added_at": now,
            "last_synced": now,
        }

        taps = self._load_taps_yaml()
        taps.append(tap_info)
        self._save_taps_yaml(taps)

        return tap_info

    def remove_tap(self, name: str) -> None:
        """Remove a tap and its cloned repo."""
        tap_dir = self._taps_dir / name
        if tap_dir.exists():
            shutil.rmtree(tap_dir)

        taps = self._load_taps_yaml()
        taps = [t for t in taps if t.get("name") != name]
        self._save_taps_yaml(taps)

    def list_taps(self) -> list[dict[str, Any]]:
        """Return all configured taps."""
        return self._load_taps_yaml()

    def sync_tap(self, name: str = "") -> dict[str, Any]:
        """Pull latest from one or all taps.

        If *name* is empty, syncs all taps.
        Returns ``{synced: [names], errors: [...]}``.
        """
        taps = self._load_taps_yaml()
        if name:
            targets = [t for t in taps if t.get("name") == name]
            if not targets:
                raise TapError(f"Tap {name!r} not found")
        else:
            targets = taps

        synced: list[str] = []
        errors: list[str] = []

        for tap in targets:
            tap_name = tap["name"]
            tap_dir = self._taps_dir / tap_name
            if not tap_dir.exists():
                errors.append(f"{tap_name}: directory missing")
                continue
            try:
                # Correct way to update a shallow clone:
                # fetch latest shallow snapshot, then reset to it.
                self._git_run("fetch", "--depth", "1", "origin", cwd=tap_dir, check=True)
                self._git_run("reset", "--hard", "FETCH_HEAD", cwd=tap_dir, check=True)
                synced.append(tap_name)
                tap["last_synced"] = utc_now_iso()
            except subprocess.CalledProcessError as exc:
                errors.append(f"{tap_name}: {exc.stderr.strip()}")

        # Persist last_synced — single read/write, no TOCTOU
        synced_times = {t["name"]: t.get("last_synced", "") for t in targets if t["name"] in set(synced)}
        for t in taps:
            if t["name"] in synced_times:
                t["last_synced"] = synced_times[t["name"]]
        self._save_taps_yaml(taps)

        return {"synced": synced, "errors": errors}

    # ------------------------------------------------------------------
    # Package discovery
    # ------------------------------------------------------------------

    def _scan_tap(self, tap_name: str) -> list[dict[str, Any]]:
        """Walk a tap directory for plugin.yaml manifests."""
        tap_dir = self._taps_dir / tap_name
        if not tap_dir.is_dir():
            return []

        manifests: list[dict[str, Any]] = []
        for manifest_path in tap_dir.rglob("plugin.yaml"):
            try:
                # Symlink escape guard
                self._assert_within(manifest_path, tap_dir, "manifest")
                data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict) or "name" not in data:
                    continue
                # Validate name from manifest (path traversal guard)
                pkg_name = data["name"]
                if not _SAFE_NAME_RE.match(pkg_name):
                    continue
                data["_tap"] = tap_name
                data["_dir"] = str(manifest_path.parent)
                manifests.append(data)
            except TapError:
                continue  # skip symlink-escaped manifests
            except Exception:
                continue  # skip malformed manifests
        return manifests

    def search(self, query: str = "", *, plugin_type: str = "") -> list[dict[str, Any]]:
        """Search for plugins across all taps.

        Matches *query* against name, description, tags, and author
        (case-insensitive substring). Filters by *plugin_type* if given.
        """
        query_lower = query.lower()
        results: list[dict[str, Any]] = []

        taps = self._load_taps_yaml()
        for tap in taps:
            for manifest in self._scan_tap(tap["name"]):
                if plugin_type and manifest.get("type", "") != plugin_type:
                    continue
                if query_lower:
                    searchable = " ".join([
                        manifest.get("name", ""),
                        manifest.get("description", ""),
                        manifest.get("author", ""),
                        " ".join(manifest.get("tags", [])),
                    ]).lower()
                    if query_lower not in searchable:
                        continue
                results.append({
                    "name": manifest.get("name", ""),
                    "type": manifest.get("type", ""),
                    "version": manifest.get("version", ""),
                    "description": manifest.get("description", ""),
                    "author": manifest.get("author", ""),
                    "tap": manifest.get("_tap", ""),
                    "tags": manifest.get("tags", []),
                })

        # Sort: name matches first, then alphabetical
        results.sort(key=lambda r: (query_lower not in r["name"].lower(), r["name"]))
        return results

    def list_available(self, *, plugin_type: str = "") -> list[dict[str, Any]]:
        """List all available packages across all taps."""
        return self.search("", plugin_type=plugin_type)

    def get_package_info(self, name: str) -> dict[str, Any] | None:
        """Get detailed info for a specific package, including install status."""
        taps = self._load_taps_yaml()
        for tap in taps:
            for manifest in self._scan_tap(tap["name"]):
                if manifest.get("name") == name:
                    installed = self._load_installed_yaml()
                    is_installed = any(p.get("name") == name for p in installed)
                    return {
                        **manifest,
                        "installed": is_installed,
                    }
        return None

    # ------------------------------------------------------------------
    # Install / Uninstall
    # ------------------------------------------------------------------

    def install(
        self,
        name: str,
        *,
        tap: str = "",
        plugin_type: str = "",
    ) -> dict[str, Any]:
        """Install a plugin from a tap into the hub's plugins directory.

        Copies the plugin's ``.py`` file to ``~/.crucible-hub/plugins/{type}/{name}.py``
        where ``discover_all_plugins`` already scans.

        When the same ``name`` exists in multiple plugin types within the same
        tap (e.g. ``code_wm`` as both ``architectures`` and ``evaluations``),
        pass ``plugin_type`` to disambiguate.

        Raises :class:`TapError` if the package is not found, is ambiguous,
        or is already installed.
        """
        _validate_name(name, "package")

        # Check if already installed (name + type combination — allows
        # installing the same name across different types without collision).
        installed = self._load_installed_yaml()
        already_installed = [
            p for p in installed
            if p.get("name") == name
            and (not plugin_type or p.get("type") == plugin_type)
        ]
        if already_installed:
            existing = already_installed[0]
            raise TapError(
                f"Package {name!r} (type={existing.get('type', '?')!r}) "
                f"is already installed. Use uninstall first."
            )

        # Find the package — detect collisions across taps AND types
        taps = self._load_taps_yaml()
        if tap:
            taps = [t for t in taps if t.get("name") == tap]

        candidates: list[dict[str, Any]] = []
        for t in taps:
            for m in self._scan_tap(t["name"]):
                if m.get("name") != name:
                    continue
                if plugin_type and m.get("type") != plugin_type:
                    continue
                candidates.append(m)

        if not candidates:
            if plugin_type:
                raise TapError(
                    f"Package {name!r} of type {plugin_type!r} not found in any tap"
                )
            raise TapError(f"Package {name!r} not found in any tap")

        if len(candidates) > 1:
            # Check if disambiguation-by-type is sufficient (all in one tap but
            # different types) or if we need tap-level disambiguation too.
            types_seen = {c.get("type", "?") for c in candidates}
            taps_seen = {c.get("_tap", "?") for c in candidates}
            if len(types_seen) > 1 and not plugin_type:
                raise TapError(
                    f"Package {name!r} found with multiple plugin types: "
                    f"{sorted(types_seen)}. Specify plugin_type= to disambiguate."
                )
            if len(taps_seen) > 1 and not tap:
                raise TapError(
                    f"Package {name!r} found in multiple taps: {sorted(taps_seen)}. "
                    f"Specify tap= to disambiguate."
                )

        manifest = candidates[0]
        plugin_type = manifest.get("type", "")
        if plugin_type not in VALID_PLUGIN_TYPES:
            raise TapError(f"Invalid plugin type {plugin_type!r} in manifest for {name!r}")

        # Find the package payload — with symlink escape guard
        pkg_dir = Path(manifest["_dir"])
        tap_dir = self._taps_dir / manifest.get("_tap", "")
        self._assert_within(pkg_dir, tap_dir, "package directory")
        dest_root = self._plugins_dir / plugin_type
        dest_root.mkdir(parents=True, exist_ok=True)

        if plugin_type in BUNDLE_PLUGIN_TYPES:
            # launchers/, evaluations/: multi-file directory bundles —
            # copy the whole package directory.
            dest_path = dest_root / name
            if dest_path.exists():
                raise TapError(
                    f"{plugin_type[:-1].capitalize()} package {name!r} is already "
                    f"installed at {dest_path}"
                )
            shutil.copytree(pkg_dir, dest_path)
        else:
            py_files = list(pkg_dir.glob("*.py"))
            if not py_files:
                raise TapError(f"No .py files found in {pkg_dir}")

            source_file = None
            for pf in py_files:
                if pf.stem == name:
                    source_file = pf
                    break
            if source_file is None:
                source_file = py_files[0]

            dest_path = dest_root / f"{name}.py"
            shutil.copy2(source_file, dest_path)

        # Record installation
        tap_name = manifest.get("_tap", "")
        sha = self._get_tap_sha(tap_name)
        record: dict[str, Any] = {
            "name": name,
            "type": plugin_type,
            "version": manifest.get("version", ""),
            "tap": tap_name,
            "installed_at": utc_now_iso(),
            "sha": sha,
            "path": str(dest_path),
        }
        installed.append(record)
        self._save_installed_yaml(installed)

        return {
            "status": "installed",
            "name": name,
            "type": plugin_type,
            "version": manifest.get("version", ""),
            "tap": tap_name,
            "path": str(dest_path),
        }

    def uninstall(self, name: str, *, plugin_type: str = "") -> None:
        """Remove an installed tap plugin.

        When multiple plugins share a name across types (e.g. architectures/code_wm
        and evaluations/code_wm), pass ``plugin_type`` to disambiguate.
        """
        _validate_name(name, "package")

        installed = self._load_installed_yaml()
        matches = [
            p for p in installed
            if p.get("name") == name
            and (not plugin_type or p.get("type") == plugin_type)
        ]

        if not matches:
            if plugin_type:
                raise TapError(
                    f"Package {name!r} of type {plugin_type!r} is not installed"
                )
            raise TapError(f"Package {name!r} is not installed")

        if len(matches) > 1 and not plugin_type:
            types_seen = sorted({m.get("type", "?") for m in matches})
            raise TapError(
                f"Package {name!r} is installed with multiple types: {types_seen}. "
                f"Specify plugin_type= to disambiguate."
            )

        record = matches[0]
        matched_type = record.get("type", "")

        # Remove the installed payload (name validated, no traversal)
        if matched_type in BUNDLE_PLUGIN_TYPES:
            plugin_path = self._plugins_dir / matched_type / name
            if plugin_path.exists():
                shutil.rmtree(plugin_path)
        else:
            plugin_path = self._plugins_dir / matched_type / f"{name}.py"
            if plugin_path.exists():
                plugin_path.unlink()

        # Update ledger — remove ONLY the matched (name, type) record,
        # not every record with this name.
        installed = [
            p for p in installed
            if not (p.get("name") == name and p.get("type") == matched_type)
        ]
        self._save_installed_yaml(installed)

    def list_installed(self, *, plugin_type: str = "") -> list[dict[str, Any]]:
        """Return all installed packages, optionally filtered by type."""
        installed = self._load_installed_yaml()
        if plugin_type:
            installed = [p for p in installed if p.get("type") == plugin_type]
        return installed

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(
        self,
        name: str,
        plugin_type: str,
        tap: str,
        *,
        project_root: Path | None = None,
        store_dir: str = ".crucible",
        plugins_subdir: str = "plugins",
    ) -> dict[str, Any]:
        """Publish a local plugin to a tap repository.

        Copies the plugin file and its metadata (if present) to the tap,
        stages and commits. The user must push manually.

        Raises :class:`TapError` if the local plugin or tap is not found.
        """
        _validate_name(name, "plugin")

        if plugin_type not in VALID_PLUGIN_TYPES:
            raise TapError(f"Invalid plugin type: {plugin_type!r}")

        tap_dir = self._taps_dir / tap
        if not tap_dir.exists():
            raise TapError(f"Tap {tap!r} not found at {tap_dir}")

        # Find the local plugin
        if project_root is None:
            project_root = Path.cwd()
        local_root = project_root / store_dir / plugins_subdir / plugin_type
        if plugin_type == "launchers":
            local_bundle = local_root / name
            if not local_bundle.is_dir():
                raise TapError(f"Local plugin not found: {local_bundle}")
        else:
            local_plugin = local_root / f"{name}.py"
            if not local_plugin.exists():
                raise TapError(f"Local plugin not found: {local_plugin}")

        # Check for existing package in tap
        dest_dir = tap_dir / plugin_type / name
        dest_file = dest_dir / f"{name}.py"
        if dest_dir.exists():
            raise TapError(
                f"Package {name!r} already exists in tap {tap!r} at {dest_dir}. "
                f"Remove it manually or update the existing version."
            )

        if plugin_type == "launchers":
            shutil.copytree(local_bundle, dest_dir)
        else:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_plugin, dest_file)

        # Copy or create plugin.yaml
        manifest_path = dest_dir / "plugin.yaml"
        if not manifest_path.exists():
            if plugin_type == "launchers":
                local_meta = local_bundle / "plugin.yaml"
            else:
                local_meta = local_plugin.with_suffix(".yaml")
            if local_meta.exists():
                shutil.copy2(local_meta, manifest_path)
            else:
                manifest = {
                    "name": name,
                    "type": plugin_type,
                    "version": "0.1.0",
                    "description": f"{name} plugin",
                    "author": "",
                    "tags": [plugin_type],
                }
                manifest_path.write_text(
                    yaml.dump(manifest, default_flow_style=False, sort_keys=False),
                    encoding="utf-8",
                )

        # Git add + commit in tap repo
        self._git_run("add", ".", cwd=tap_dir)
        self._git_run(
            "commit", "-m", f"Add {plugin_type}/{name}",
            cwd=tap_dir, check=False,  # ok if nothing changed
        )

        return {
            "status": "published",
            "name": name,
            "type": plugin_type,
            "tap": tap,
            "path": str(dest_dir),
            "next_steps": [
                f"Push:  cd {tap_dir} && git push",
                "Or open a PR if you forked the tap (see 'crucible tap submit-pr').",
            ],
        }

    # ------------------------------------------------------------------
    # Push & PR
    # ------------------------------------------------------------------

    def push(self, tap: str) -> dict[str, Any]:
        """Push a tap repo to its git remote.

        Raises :class:`TapError` if the tap doesn't exist or push fails.
        """
        tap_dir = self._taps_dir / tap
        if not tap_dir.exists():
            raise TapError(f"Tap {tap!r} not found")

        try:
            self._git_run("push", cwd=tap_dir, check=True)
            return {"status": "pushed", "tap": tap}
        except subprocess.CalledProcessError as exc:
            raise TapError(f"Push failed for {tap!r}: {exc.stderr.strip()}") from exc

    def get_tap_remote(self, tap: str) -> str | None:
        """Return the remote URL for a tap, or None."""
        tap_dir = self._taps_dir / tap
        if not tap_dir.exists():
            return None
        try:
            result = self._git_run("remote", "get-url", "origin", cwd=tap_dir, check=True)
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return None

    def submit_pr(self, tap: str, *, title: str = "", body: str = "") -> dict[str, Any]:
        """Open a GitHub PR from the tap's current branch to its upstream.

        Requires the ``gh`` CLI to be installed and authenticated.
        Falls back to returning a URL the user can open manually.

        Raises :class:`TapError` if the tap doesn't exist.
        """
        tap_dir = self._taps_dir / tap
        if not tap_dir.exists():
            raise TapError(f"Tap {tap!r} not found")

        remote_url = self.get_tap_remote(tap) or ""

        # Try gh CLI first — build args incrementally
        try:
            gh_args = ["gh", "pr", "create", "--fill"]
            if title:
                gh_args += ["--title", title]
            if body:
                gh_args += ["--body", body]
            result = subprocess.run(
                gh_args,
                cwd=str(tap_dir),
                capture_output=True,
                text=True,
                check=True,
            )
            pr_url = result.stdout.strip()
            return {"status": "pr_created", "tap": tap, "pr_url": pr_url}
        except FileNotFoundError:
            return {
                "status": "manual",
                "tap": tap,
                "remote": remote_url,
                "instructions": (
                    f"gh CLI not found. Push your changes first, then open a PR manually:\n"
                    f"  1. cd {tap_dir} && git push\n"
                    f"  2. Open {remote_url} in a browser and create a PR"
                ),
            }
        except subprocess.CalledProcessError as exc:
            return {
                "status": "error",
                "tap": tap,
                "error": exc.stderr.strip(),
                "instructions": (
                    f"Push first, then open a PR manually:\n"
                    f"  1. cd {tap_dir} && git push\n"
                    f"  2. Open {remote_url} in a browser and create a PR"
                ),
            }
