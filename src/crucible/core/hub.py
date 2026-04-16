"""Crucible Hub — cross-project knowledge management and research tracks.

The hub lives outside any single project (default ``~/.crucible-hub/``) and
provides:
  - A registry of linked Crucible projects
  - Named research tracks that group related experiments
  - Hub-scoped findings (track-level and global)
  - Git-backed sync for sharing across machines

Disk layout::

    ~/.crucible-hub/
      hub.yaml                    # name, created_at, default_track, version
      registry.jsonl              # linked projects
      tracks/
        {track-slug}/
          track.yaml
          findings.jsonl
          findings/{slug}/
            v1.yaml
            current.yaml
      global/
        findings.jsonl
        findings/{slug}/
          v1.yaml
          current.yaml
      .git/
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from crucible.core.errors import HubError
from crucible.core.finding import (
    can_promote,
    make_finding_id,
    validate_finding,
)
from crucible.core.io import append_jsonl, read_jsonl, read_yaml, write_jsonl, write_yaml
from crucible.core.log import utc_now_iso


_DEFAULT_HUB_DIR = Path.home() / ".crucible-hub"
_HUB_VERSION = "1"


def _slugify(name: str) -> str:
    """Convert a name to a filesystem-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower().strip()).strip("-")
    if not slug:
        raise HubError(f"Cannot slugify empty name: {name!r}")
    return slug


class HubStore:
    """Cross-project hub for research tracks and findings."""

    def __init__(self, hub_dir: Path | None = None) -> None:
        # Resolve: explicit arg > CRUCIBLE_HUB_DIR env > ~/.crucible-hub
        if hub_dir is not None:
            self.hub_dir = Path(hub_dir)
        else:
            env = os.environ.get("CRUCIBLE_HUB_DIR")
            if env:
                self.hub_dir = Path(env)
            else:
                self.hub_dir = _DEFAULT_HUB_DIR

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _hub_yaml_path(self) -> Path:
        return self.hub_dir / "hub.yaml"

    @property
    def _registry_path(self) -> Path:
        return self.hub_dir / "registry.jsonl"

    @property
    def _tracks_dir(self) -> Path:
        return self.hub_dir / "tracks"

    @property
    def _global_dir(self) -> Path:
        return self.hub_dir / "global"

    @property
    def _architectures_dir(self) -> Path:
        return self.hub_dir / "architectures"

    @property
    def _arch_plugins_dir(self) -> Path:
        return self._architectures_dir / "plugins"

    @property
    def _arch_specs_dir(self) -> Path:
        return self._architectures_dir / "specs"

    @property
    def _arch_registry_path(self) -> Path:
        return self._architectures_dir / "registry.jsonl"

    @property
    def initialized(self) -> bool:
        return self._hub_yaml_path.exists()

    def _require_init(self) -> None:
        if not self.initialized:
            raise HubError(
                f"Hub not initialized at {self.hub_dir}. Run HubStore.init() first."
            )

    # ------------------------------------------------------------------
    # Discovery & initialization
    # ------------------------------------------------------------------

    @classmethod
    def resolve_hub_dir(
        cls,
        *,
        explicit: str | Path | None = None,
        config_hub_dir: str | None = None,
    ) -> Path:
        """Resolve the hub directory from various sources.

        Priority: explicit arg > CRUCIBLE_HUB_DIR env > config.hub_dir > ~/.crucible-hub
        """
        if explicit:
            return Path(explicit)
        env = os.environ.get("CRUCIBLE_HUB_DIR")
        if env:
            return Path(env)
        if config_hub_dir:
            return Path(config_hub_dir).expanduser()
        return _DEFAULT_HUB_DIR

    @staticmethod
    def discover(
        *,
        explicit: str | Path | None = None,
        config_hub_dir: str | None = None,
    ) -> Path | None:
        """Find hub dir. Returns None if not initialized."""
        candidates = [
            explicit,
            os.environ.get("CRUCIBLE_HUB_DIR"),
            config_hub_dir,
            _DEFAULT_HUB_DIR,
        ]
        for candidate in candidates:
            if not candidate:
                continue
            p = Path(candidate).expanduser()
            if (p / "hub.yaml").exists():
                return p
        return None

    @staticmethod
    def init(hub_dir: Path | None = None, name: str = "") -> "HubStore":
        """Create hub directory, git init, write hub.yaml."""
        if hub_dir is None:
            env = os.environ.get("CRUCIBLE_HUB_DIR")
            if env:
                hub_dir = Path(env)
            else:
                hub_dir = _DEFAULT_HUB_DIR

        hub_dir = Path(hub_dir)
        hub_yaml = hub_dir / "hub.yaml"

        if hub_yaml.exists():
            raise HubError(f"Hub already initialized at {hub_dir}")

        hub_dir.mkdir(parents=True, exist_ok=True)
        (hub_dir / "tracks").mkdir(exist_ok=True)
        (hub_dir / "global").mkdir(exist_ok=True)
        (hub_dir / "architectures" / "plugins").mkdir(parents=True, exist_ok=True)
        (hub_dir / "architectures" / "specs").mkdir(parents=True, exist_ok=True)
        (hub_dir / "taps").mkdir(exist_ok=True)

        config = {
            "name": name or hub_dir.name,
            "created_at": utc_now_iso(),
            "default_track": "",
            "version": _HUB_VERSION,
        }
        write_yaml(hub_yaml, config)

        # Git init (best-effort)
        try:
            subprocess.run(
                ["git", "init"],
                cwd=str(hub_dir),
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "add", "."],
                cwd=str(hub_dir),
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "commit", "-m", "crucible-hub: init"],
                cwd=str(hub_dir),
                capture_output=True,
                check=True,
            )
        except FileNotFoundError:
            from crucible.core.log import log_warn
            log_warn("git not installed; hub will not support git sync")
        except subprocess.CalledProcessError as exc:
            from crucible.core.log import log_warn
            log_warn(f"git init failed: {exc.stderr.strip() if exc.stderr else exc}")

        store = HubStore(hub_dir)
        return store

    def _read_hub_yaml(self) -> dict[str, Any]:
        self._require_init()
        raw = read_yaml(self._hub_yaml_path)
        if not isinstance(raw, dict):
            raise HubError(f"Invalid hub.yaml at {self._hub_yaml_path}")
        return raw

    def _write_hub_yaml(self, data: dict[str, Any]) -> None:
        write_yaml(self._hub_yaml_path, data)

    # ------------------------------------------------------------------
    # Project registry
    # ------------------------------------------------------------------

    def link_project(self, name: str, path: Path) -> dict[str, Any]:
        """Register a project with this hub. Returns the registry entry."""
        self._require_init()
        path = Path(path).resolve()
        if not path.exists():
            raise HubError(f"Project path does not exist: {path}")

        # Check for duplicates
        existing = read_jsonl(self._registry_path)
        for entry in existing:
            if entry.get("name") == name:
                raise HubError(f"Project '{name}' already linked.")

        record: dict[str, Any] = {
            "name": name,
            "path": str(path),
            "linked_at": utc_now_iso(),
        }
        append_jsonl(self._registry_path, record)
        return record

    def unlink_project(self, name: str) -> bool:
        """Remove a project from the registry. Returns True if found and removed."""
        self._require_init()
        existing = read_jsonl(self._registry_path)
        updated = [e for e in existing if e.get("name") != name]
        if len(updated) == len(existing):
            return False
        write_jsonl(self._registry_path, updated)
        return True

    def list_projects(self) -> list[dict[str, Any]]:
        """List all linked projects."""
        self._require_init()
        return read_jsonl(self._registry_path)

    # ------------------------------------------------------------------
    # Tracks
    # ------------------------------------------------------------------

    def _track_dir(self, name: str) -> Path:
        return self._tracks_dir / _slugify(name)

    def _track_yaml_path(self, name: str) -> Path:
        return self._track_dir(name) / "track.yaml"

    def _read_track_yaml(self, name: str) -> dict[str, Any]:
        path = self._track_yaml_path(name)
        if not path.exists():
            raise HubError(f"Track '{name}' not found.")
        raw = read_yaml(path)
        if not isinstance(raw, dict):
            raise HubError(f"Invalid track.yaml for '{name}'")
        return raw

    def _write_track_yaml(self, name: str, data: dict[str, Any]) -> None:
        path = self._track_yaml_path(name)
        write_yaml(path, data)

    def create_track(
        self,
        name: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a new research track. Returns the track metadata."""
        self._require_init()
        track_dir = self._track_dir(name)
        if track_dir.exists():
            raise HubError(f"Track '{name}' already exists.")

        track_dir.mkdir(parents=True, exist_ok=True)
        (track_dir / "findings").mkdir(exist_ok=True)

        track_data: dict[str, Any] = {
            "name": name,
            "description": description,
            "tags": tags or [],
            "linked_projects": [],
            "created_at": utc_now_iso(),
            "active": True,
        }
        self._write_track_yaml(name, track_data)
        return track_data

    def get_track(self, name: str) -> dict[str, Any] | None:
        """Get track metadata by name. Returns None if not found."""
        self._require_init()
        try:
            return self._read_track_yaml(name)
        except HubError:
            return None

    def list_tracks(self) -> list[dict[str, Any]]:
        """List all tracks."""
        self._require_init()
        tracks: list[dict[str, Any]] = []
        if not self._tracks_dir.exists():
            return tracks
        for child in sorted(self._tracks_dir.iterdir()):
            track_yaml = child / "track.yaml"
            if track_yaml.exists():
                try:
                    raw = read_yaml(track_yaml)
                    if isinstance(raw, dict):
                        tracks.append(raw)
                    else:
                        from crucible.core.log import log_warn
                        log_warn(f"Malformed track.yaml in {child.name}: expected dict")
                except Exception as exc:
                    from crucible.core.log import log_warn
                    log_warn(f"Failed to parse track.yaml in {child.name}: {exc}")
        return tracks

    def activate_track(self, name: str) -> None:
        """Set a track as the default/active track."""
        self._require_init()
        track = self.get_track(name)
        if track is None:
            raise HubError(f"Track '{name}' not found.")

        hub_data = self._read_hub_yaml()
        hub_data["default_track"] = name
        self._write_hub_yaml(hub_data)

    def get_active_track(self) -> str | None:
        """Get the active track name, or None if none set."""
        self._require_init()
        hub_data = self._read_hub_yaml()
        track = hub_data.get("default_track", "")
        return track if track else None

    def link_project_to_track(self, track_name: str, project_name: str) -> None:
        """Associate a linked project with a track."""
        self._require_init()
        track = self._read_track_yaml(track_name)

        # Verify project is linked to hub
        projects = self.list_projects()
        project_names = {p["name"] for p in projects}
        if project_name not in project_names:
            raise HubError(
                f"Project '{project_name}' is not linked to the hub. Link it first."
            )

        linked = list(track.get("linked_projects", []))
        if project_name not in linked:
            linked.append(project_name)
            track["linked_projects"] = linked
            self._write_track_yaml(track_name, track)

    # ------------------------------------------------------------------
    # Finding storage helpers
    # ------------------------------------------------------------------

    def _scope_dir(self, scope: str, track: str | None = None) -> Path:
        """Resolve the directory for a given scope."""
        if scope == "global":
            return self._global_dir
        elif scope == "track":
            if not track:
                raise HubError("Track name required for track-scoped findings.")
            return self._track_dir(track)
        else:
            raise HubError(f"Invalid scope '{scope}'. Use 'track' or 'global'.")

    def _findings_ledger(self, scope: str, track: str | None = None) -> Path:
        return self._scope_dir(scope, track) / "findings.jsonl"

    def _finding_dir(
        self, finding_id: str, scope: str, track: str | None = None
    ) -> Path:
        return self._scope_dir(scope, track) / "findings" / finding_id

    def _next_finding_version(
        self, finding_id: str, scope: str, track: str | None = None
    ) -> int:
        """Determine the next version number for a finding."""
        fdir = self._finding_dir(finding_id, scope, track)
        if not fdir.exists():
            return 1
        existing = sorted(fdir.glob("v*.yaml"))
        if not existing:
            return 1
        # Extract version numbers
        versions = []
        for p in existing:
            m = re.match(r"v(\d+)\.yaml", p.name)
            if m:
                versions.append(int(m.group(1)))
        return (max(versions) + 1) if versions else 1

    def _write_finding_yaml(
        self,
        finding: dict[str, Any],
        finding_id: str,
        version: int,
        scope: str,
        track: str | None = None,
    ) -> None:
        """Write a finding as versioned YAML files."""
        fdir = self._finding_dir(finding_id, scope, track)
        fdir.mkdir(parents=True, exist_ok=True)

        version_path = fdir / f"v{version}.yaml"
        write_yaml(version_path, finding)

        current_path = fdir / "current.yaml"
        shutil.copy2(version_path, current_path)

    def _read_finding_yaml(
        self,
        finding_id: str,
        scope: str,
        track: str | None = None,
        version: int | None = None,
    ) -> dict[str, Any] | None:
        """Read a finding from YAML. Returns None if not found."""
        fdir = self._finding_dir(finding_id, scope, track)
        if version is not None:
            path = fdir / f"v{version}.yaml"
        else:
            path = fdir / "current.yaml"
        raw = read_yaml(path)
        return raw if isinstance(raw, dict) else None

    # ------------------------------------------------------------------
    # Findings CRUD
    # ------------------------------------------------------------------

    def store_finding(
        self,
        finding: dict[str, Any],
        scope: str,
        track: str | None = None,
    ) -> dict[str, Any]:
        """Store a new finding. Returns the enriched finding dict."""
        self._require_init()

        # Validate
        errors = validate_finding(finding)
        if errors:
            raise HubError(f"Invalid finding: {'; '.join(errors)}")

        # Generate ID and version
        finding_id = finding.get("id") or make_finding_id(
            finding["title"], scope, track
        )
        version = self._next_finding_version(finding_id, scope, track)

        # Enrich the finding
        enriched = dict(finding)
        enriched.setdefault("id", finding_id)
        enriched.setdefault("status", "active")
        enriched.setdefault("category", "observation")
        enriched.setdefault("confidence", 0.7)
        enriched.setdefault("tags", [])
        enriched.setdefault("source_experiments", [])
        enriched.setdefault("created_at", utc_now_iso())
        enriched["version"] = version
        enriched["scope"] = scope
        if track:
            enriched["track"] = track

        # Write YAML version files
        self._write_finding_yaml(enriched, finding_id, version, scope, track)

        # Append to ledger
        ledger_entry = {
            "kind": "finding",
            "ts": enriched["created_at"],
            "finding_id": finding_id,
            "version": version,
            "title": enriched["title"],
            "status": enriched.get("status", "active"),
            "scope": scope,
        }
        if track:
            ledger_entry["track"] = track
        append_jsonl(self._findings_ledger(scope, track), ledger_entry)

        return enriched

    def get_finding(
        self,
        finding_id: str,
        scope: str,
        track: str | None = None,
    ) -> dict[str, Any] | None:
        """Get the current version of a finding."""
        self._require_init()
        return self._read_finding_yaml(finding_id, scope, track)

    def list_findings(
        self,
        scope: str,
        track: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List findings from the ledger, optionally filtered."""
        self._require_init()
        ledger = read_jsonl(self._findings_ledger(scope, track))

        # Deduplicate: keep only the latest entry per finding_id
        latest: dict[str, dict[str, Any]] = {}
        for entry in ledger:
            if entry.get("kind") != "finding":
                continue
            fid = entry.get("finding_id", "")
            if fid:
                latest[fid] = entry

        results: list[dict[str, Any]] = []
        for fid, entry in latest.items():
            # Apply status filter
            if status and entry.get("status") != status:
                continue

            # Load full YAML to check tags
            full = self._read_finding_yaml(fid, scope, track)
            if full is None:
                continue

            if tags:
                finding_tags = set(full.get("tags", []))
                if not set(tags) & finding_tags:
                    continue

            results.append(full)

        # Sort by created_at
        results.sort(key=lambda f: f.get("created_at", ""))
        return results

    def supersede_finding(
        self,
        finding_id: str,
        new_finding: dict[str, Any],
        scope: str,
        track: str | None = None,
    ) -> dict[str, Any]:
        """Supersede an existing finding with a new version.

        Marks the old finding as 'superseded' and creates the new one
        with the same finding_id.
        """
        self._require_init()

        old = self.get_finding(finding_id, scope, track)
        if old is None:
            raise HubError(f"Finding '{finding_id}' not found in scope '{scope}'.")

        # Mark old as superseded
        old["status"] = "superseded"
        old_version = old.get("version", 1)
        self._write_finding_yaml(old, finding_id, old_version, scope, track)

        # Create new version
        version = self._next_finding_version(finding_id, scope, track)
        enriched = dict(new_finding)
        enriched["id"] = finding_id
        enriched.setdefault("status", "active")
        enriched.setdefault("category", old.get("category", "observation"))
        enriched.setdefault("confidence", old.get("confidence", 0.7))
        enriched.setdefault("tags", old.get("tags", []))
        enriched.setdefault("source_experiments", old.get("source_experiments", []))
        enriched["created_at"] = utc_now_iso()
        enriched["version"] = version
        enriched["scope"] = scope
        enriched["supersedes_version"] = old_version
        if track:
            enriched["track"] = track

        self._write_finding_yaml(enriched, finding_id, version, scope, track)

        # Ledger entry
        ledger_entry = {
            "kind": "finding",
            "ts": enriched["created_at"],
            "finding_id": finding_id,
            "version": version,
            "title": enriched.get("title", ""),
            "status": enriched.get("status", "active"),
            "scope": scope,
            "supersedes_version": old_version,
        }
        if track:
            ledger_entry["track"] = track
        append_jsonl(self._findings_ledger(scope, track), ledger_entry)

        return enriched

    def promote_finding(
        self,
        finding_id: str,
        from_scope: str,
        to_scope: str,
        from_track: str | None = None,
        to_track: str | None = None,
    ) -> dict[str, Any]:
        """Promote a finding from one scope to another.

        Valid promotions: track -> global.
        """
        self._require_init()

        if not can_promote(from_scope, to_scope):
            raise HubError(
                f"Cannot promote from '{from_scope}' to '{to_scope}'. "
                f"Valid: track -> global."
            )

        source = self.get_finding(finding_id, from_scope, from_track)
        if source is None:
            raise HubError(
                f"Finding '{finding_id}' not found in scope '{from_scope}'."
            )

        # Mark source as promoted
        source["status"] = "promoted"
        source_version = source.get("version", 1)
        self._write_finding_yaml(source, finding_id, source_version, from_scope, from_track)

        # Update source ledger
        ledger_entry = {
            "kind": "finding",
            "ts": utc_now_iso(),
            "finding_id": finding_id,
            "version": source_version,
            "title": source.get("title", ""),
            "status": "promoted",
            "scope": from_scope,
            "promoted_to": to_scope,
        }
        if from_track:
            ledger_entry["track"] = from_track
        append_jsonl(self._findings_ledger(from_scope, from_track), ledger_entry)

        # Store in destination scope
        promoted = dict(source)
        promoted["status"] = "active"
        promoted["scope"] = to_scope
        promoted["promoted_from"] = {
            "scope": from_scope,
            "track": from_track,
            "version": source_version,
        }
        promoted.pop("version", None)
        promoted.pop("track", None)
        if to_track:
            promoted["track"] = to_track

        return self.store_finding(promoted, to_scope, to_track)

    # ------------------------------------------------------------------
    # Architecture storage
    # ------------------------------------------------------------------

    def _normalize_architecture_record(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Return a backward-compatible architecture metadata record."""
        record = dict(entry)
        kind = record.get("kind", "code")
        if kind not in {"code", "spec"}:
            raise HubError(f"Unsupported architecture kind: {kind!r}")
        record["kind"] = kind
        if "relative_path" not in record:
            suffix = ".py" if kind == "code" else ".yaml"
            directory = "plugins" if kind == "code" else "specs"
            record["relative_path"] = str(Path("architectures") / directory / f"{record['name']}{suffix}")
        return record

    def _architecture_path_from_record(self, record: dict[str, Any]) -> Path:
        normalized = self._normalize_architecture_record(record)
        return self.hub_dir / normalized["relative_path"]

    def _read_architecture_registry(self) -> list[dict[str, Any]]:
        return [self._normalize_architecture_record(entry) for entry in read_jsonl(self._arch_registry_path)]

    def store_architecture(
        self,
        name: str,
        code: str,
        source_project: str = "",
        tags: list[str] | None = None,
        *,
        kind: str = "code",
    ) -> dict[str, Any]:
        """Store a code plugin or YAML spec in the hub for cross-project reuse."""
        self._require_init()

        if not name.isidentifier():
            raise HubError(
                f"Architecture name must be a valid Python identifier: {name!r}"
            )

        if kind not in {"code", "spec"}:
            raise HubError(f"Architecture kind must be 'code' or 'spec', got {kind!r}")

        if kind == "code" and "register_model" not in code:
            raise HubError(
                f"Architecture code must contain a register_model call: {name!r}"
            )

        # Check for duplicates in the registry ledger
        existing = self._read_architecture_registry()
        for entry in existing:
            if entry.get("name") == name:
                raise HubError(f"Architecture '{name}' already exists in the hub.")

        suffix = ".py" if kind == "code" else ".yaml"
        target_dir = self._arch_plugins_dir if kind == "code" else self._arch_specs_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        plugin_path = target_dir / f"{name}{suffix}"
        plugin_path.write_text(code, encoding="utf-8")

        # Build metadata record
        record: dict[str, Any] = {
            "name": name,
            "kind": kind,
            "relative_path": str(plugin_path.relative_to(self.hub_dir)),
            "added_at": utc_now_iso(),
            "source_project": source_project,
            "tags": tags or [],
        }

        # Append to the registry ledger
        append_jsonl(self._arch_registry_path, record)

        return record

    def get_architecture(self, name: str) -> dict[str, Any] | None:
        """Get architecture metadata by name from the registry ledger.

        Returns None if no architecture with that name exists.
        """
        self._require_init()
        entries = self._read_architecture_registry()
        for entry in entries:
            if entry.get("name") == name:
                return entry
        return None

    def get_architecture_code(self, name: str) -> str | None:
        """Read the plugin source code for an architecture.

        Returns the file contents, or None if the plugin file doesn't exist.
        """
        self._require_init()
        record = self.get_architecture(name)
        if record is not None and record.get("kind") != "code":
            return None
        plugin_path = self._architecture_path_from_record(record or {"name": name, "kind": "code"})
        if not plugin_path.exists():
            return None
        return plugin_path.read_text(encoding="utf-8")

    def get_architecture_content(self, name: str) -> str | None:
        """Read the stored source or YAML content for an architecture."""
        self._require_init()
        record = self.get_architecture(name)
        if record is None:
            return None
        path = self._architecture_path_from_record(record)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_architectures(self) -> list[dict[str, Any]]:
        """List all architecture entries from the registry ledger.

        Returns an empty list if the ledger doesn't exist.
        """
        self._require_init()
        return self._read_architecture_registry()

    def remove_architecture(self, name: str) -> bool:
        """Remove an architecture from the hub.

        Deletes the plugin file and removes the entry from the registry
        ledger. Returns True if the architecture was found and removed,
        False otherwise.
        """
        self._require_init()

        # Remove from ledger
        existing = self._read_architecture_registry()
        target = next((e for e in existing if e.get("name") == name), None)
        updated = [e for e in existing if e.get("name") != name]
        if len(updated) == len(existing):
            return False
        write_jsonl(self._arch_registry_path, updated)

        # Delete the stored asset file
        plugin_path = self._architecture_path_from_record(target or {"name": name, "kind": "code"})
        if plugin_path.exists():
            plugin_path.unlink()

        return True

    # ------------------------------------------------------------------
    # Context loading
    # ------------------------------------------------------------------

    def load_context_for_track(
        self,
        track_name: str,
        include_global: bool = True,
        max_findings: int = 50,
    ) -> list[dict[str, Any]]:
        """Load findings for a track, optionally including global findings.

        Returns findings sorted by created_at (newest first), limited to
        *max_findings* total.
        """
        self._require_init()

        findings: list[dict[str, Any]] = []

        # Track-scoped findings
        track_findings = self.list_findings("track", track=track_name, status="active")
        for f in track_findings:
            f["_source_scope"] = "track"
        findings.extend(track_findings)

        # Global findings
        if include_global:
            global_findings = self.list_findings("global", status="active")
            for f in global_findings:
                f["_source_scope"] = "global"
            findings.extend(global_findings)

        # Sort newest first
        findings.sort(key=lambda f: f.get("created_at", ""), reverse=True)

        return findings[:max_findings]

    # ------------------------------------------------------------------
    # Git sync
    # ------------------------------------------------------------------

    def _git_run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        """Run a git command in the hub directory."""
        return subprocess.run(
            ["git", *args],
            cwd=str(self.hub_dir),
            capture_output=True,
            text=True,
            check=check,
        )

    def set_remote(self, remote_url: str) -> None:
        """Set or update the git remote origin."""
        self._require_init()
        try:
            # Try to add first; if already exists, update
            result = self._git_run("remote", "get-url", "origin", check=False)
            if result.returncode == 0:
                self._git_run("remote", "set-url", "origin", remote_url)
            else:
                self._git_run("remote", "add", "origin", remote_url)
        except Exception as exc:
            raise HubError(f"Failed to set remote: {exc}") from exc

    def sync(self, remote: str | None = None) -> dict[str, Any]:
        """Sync hub state with git remote.

        Stages all changes, commits if dirty, then pulls and pushes.
        Returns a dict with sync status info.
        """
        self._require_init()

        result: dict[str, Any] = {
            "committed": False,
            "pushed": False,
            "pulled": False,
            "errors": [],
        }

        remote_name = remote or "origin"

        try:
            # Stage all changes
            self._git_run("add", "-A")

            # Check if there are changes to commit
            status = self._git_run("status", "--porcelain", check=False)
            if status.stdout.strip():
                self._git_run(
                    "commit", "-m",
                    f"crucible-hub: sync {utc_now_iso()}"
                )
                result["committed"] = True

            # Pull (with rebase to keep history linear)
            try:
                pull = self._git_run("pull", "--rebase", remote_name, "HEAD", check=False)
                if pull.returncode == 0:
                    result["pulled"] = True
                else:
                    # Remote may not exist yet — not an error
                    if "No remote" not in pull.stderr and "does not appear" not in pull.stderr:
                        result["errors"].append(f"pull: {pull.stderr.strip()}")
            except Exception as exc:
                result["errors"].append(f"pull: {exc}")

            # Push
            try:
                push = self._git_run("push", remote_name, "HEAD", check=False)
                if push.returncode == 0:
                    result["pushed"] = True
                else:
                    if "No remote" not in push.stderr and "does not appear" not in push.stderr:
                        result["errors"].append(f"push: {push.stderr.strip()}")
            except Exception as exc:
                result["errors"].append(f"push: {exc}")

        except Exception as exc:
            result["errors"].append(str(exc))

        return result
