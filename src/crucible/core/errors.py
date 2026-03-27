class CrucibleError(Exception):
    """Base error for all Crucible operations."""

class ConfigError(CrucibleError):
    """Bad YAML, missing config, invalid settings."""

class FleetError(CrucibleError):
    """Provider, SSH, provisioning, bootstrap failures."""

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
