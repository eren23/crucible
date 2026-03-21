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
