from __future__ import annotations


class CrucibleError(Exception):
    """Base error for all Crucible operations."""

class ConfigError(CrucibleError):
    """Bad YAML, missing config, invalid settings."""

class FleetError(CrucibleError):
    """Provider, SSH, provisioning, bootstrap failures."""

class PartialProvisionError(FleetError):
    """Provision loop completed with some successes AND some failures.

    Carries the partial state so the caller can persist whatever was created
    before re-raising. Attributes:

    - ``created``: list of successfully-created node records (already on the
      provider side; need to be committed to inventory by the caller)
    - ``failed``: list of ``{"name": str, "error": str}`` entries for pods
      that could not be created after exhausting cloud-type fallbacks
    """

    def __init__(self, message: str, created: list | None = None, failed: list | None = None):
        super().__init__(message)
        self.created = created or []
        self.failed = failed or []

class RunnerError(CrucibleError):
    """Experiment execution failures."""

class ResearcherError(CrucibleError):
    """LLM, hypothesis generation, research loop failures."""

class DataError(CrucibleError):
    """Data download, manifest, sync failures."""

class StoreError(CrucibleError):
    """Version store read/write/integrity failures."""

class HubError(CrucibleError):
    """Hub init, sync, track, finding promotion failures."""

class ApiError(CrucibleError):
    """API server startup, auth, or request handling failures."""

class ComposerError(CrucibleError):
    """Architecture composition spec loading, resolution, or build failures."""

class SearchTreeError(CrucibleError):
    """Search tree creation, expansion, pruning, or persistence failures."""

class RecipeError(CrucibleError):
    """Recipe save, retrieval, or validation failures."""

class PluginError(CrucibleError):
    """Plugin registration, discovery, or build failures."""

class TapError(CrucibleError):
    """Tap clone, sync, install, search, or publish failures."""

class ProjectTemplateError(ConfigError):
    """Project template loading, variable substitution, or inheritance failures."""
