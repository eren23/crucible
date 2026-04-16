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


class SshNotReadyError(FleetError):
    """SSH connection failed but the target is probably still starting up.

    Transient — caller should back off and retry. Raised by
    ``wait_for_ssh_ready`` when the pod is booting and refusing
    connections (typical during the first ~30-90s of a new RunPod pod).
    """


class SshTimeoutError(FleetError):
    """SSH connection timed out after exhausting the retry budget.

    Caller should treat the target as permanently unreachable for this
    attempt and either replace the node or surface the error to the user.
    """


class SshAuthError(FleetError):
    """SSH authentication failed — wrong key or pod-side auth misconfig.

    Fatal: do not retry. The key needs to be fixed before the node can be
    used. Distinct from SshNotReadyError / SshTimeoutError because no
    amount of waiting will fix it.
    """

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

class ResearchDAGError(CrucibleError):
    """Research DAG bridge sync, mapping, or Spider Chat communication failures."""

class ProjectTemplateError(ConfigError):
    """Project template loading, variable substitution, or inheritance failures."""

class DomainSpecError(CrucibleError):
    """Domain spec loading, validation, or missing-field failures."""

class CandidateValidationError(CrucibleError):
    """Harness candidate failed validation (syntax, interface, constraints)."""

class HarnessOptimizerError(CrucibleError):
    """Harness optimizer orchestration failures."""
