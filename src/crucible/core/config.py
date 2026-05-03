"""Load and validate crucible.yaml project configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from crucible import __version__ as CRUCIBLE_VERSION


@dataclass
class ProviderConfig:
    type: str = "runpod"  # "runpod" | "ssh"
    ssh_key: str = "~/.ssh/id_ed25519_runpod"
    image: str = ""
    gpu_types: list[str] = field(default_factory=list)
    gpu_count: int = 1  # Number of GPUs per node (multi-GPU support)
    interruptible: bool = True
    defaults: dict[str, Any] = field(default_factory=dict)
    network_volume_id: str = ""  # RunPod network volume ID for shared storage
    template_id: str = ""        # RunPod template ID for standardized provisioning


@dataclass
class DataProbeConfig:
    """Config-driven remote data-readiness probe.

    When any of these fields are set, fleet bootstrap generates a Python
    probe script that checks the remote node for the listed paths. If
    the probe reports missing data, the optional ``download_command`` is
    run. If all fields are empty (default), data bootstrap is a no-op
    rather than silently running a hardcoded fineweb check.

    - ``paths``: list of remote paths that must all exist for data to be
      considered ready. Resolved relative to the node's workspace_path.
      Paths ending with ``/`` are treated as directories (must be
      non-empty); other paths must exist as files. Globs are supported
      via Path.glob for the first segment containing ``*``.
    - ``script``: optional remote Python file that prints ``1`` if data
      is ready, ``0`` otherwise. Overrides ``paths`` when set. Useful
      for bespoke readiness logic (e.g., checksum verification).
    - ``download_command``: shell command run on the remote when the
      probe reports missing data. Runs in the workspace dir. Empty
      string means "don't attempt to download; just warn".
    """
    paths: list[str] = field(default_factory=list)
    script: str = ""
    download_command: str = ""


@dataclass
class DataConfig:
    source: str = "huggingface"
    repo_id: str = ""
    remote_prefix: str = "datasets"
    local_root: str = "./data"
    manifest: str = "manifest.json"
    variant: str = "fineweb10B_sp1024"
    path: str = ""  # root path when source is local_files
    wandb_entity: str = ""
    wandb_project: str = ""
    wandb_artifact: str = ""
    wandb_artifact_type: str = "dataset"
    probe: DataProbeConfig = field(default_factory=DataProbeConfig)


@dataclass
class TrainingConfig:
    script: str = "train.py"
    backend: str = "torch"  # "torch" | "mlx" | custom
    python: str = "python3"
    modality: str = "lm"    # "lm" | "vision" | "diffusion" | "rl" | "generic"
    optimizer: str = ""          # default optimizer (env var OPTIMIZER overrides)
    lr_schedule: str = ""        # default LR schedule (env var LR_SCHEDULE overrides)
    logging_backends: str = ""   # comma-separated logger names (env var LOGGING_BACKEND overrides)
    callbacks: str = ""          # comma-separated callback names (env var CALLBACKS overrides)


@dataclass
class MetricsConfig:
    primary: str = "val_loss"        # metric to rank by (from result dict)
    secondary: str = ""              # optional secondary display metric
    size: str = "model_bytes"        # for Pareto frontier
    direction: str = "minimize"      # "minimize" | "maximize"


@dataclass
class ResearcherConfig:
    model: str = "claude-sonnet-4-6-20250514"
    max_tokens: int = 4096
    budget_hours: float = 10.0
    max_iterations: int = 20
    program_file: str = "program.md"


@dataclass
class WandbConfig:
    required: bool = True
    project: str = ""
    entity: str = ""
    mode: str = "online"


@dataclass
class HfCollabConfig:
    """HuggingFace Hub collaboration backbone (opt-in).

    When ``enabled=True`` the new ``hf_*`` MCP tools and the ``hf_dataset``
    hub_remote can publish leaderboard / findings / recipes / artifacts to
    HuggingFace so external agents (e.g. ml-intern) can read them. All paths
    stay no-op when ``enabled=False`` — Crucible never auto-pushes.

    Repo names are templates: ``{project}`` is substituted from
    ``ProjectConfig.name`` at call time.

    Auth: ``HF_TOKEN`` env var (read at write time, never persisted).
    """
    enabled: bool = False
    leaderboard_repo: str = ""
    findings_repo: str = ""
    recipes_repo: str = ""
    artifacts_repo: str = ""   # supports {project} placeholder substitution
    private: bool = True   # default to private repos to avoid accidental leak
    briefing_auto_pull: bool = False
    """Auto-pull peer leaderboard rows into get_research_briefing.

    Off by default even when ``enabled=True`` — every briefing call hits HF
    over HTTP and adds 1-30s latency. Opt in explicitly when you want
    briefings to surface peer-agent prior runs without an extra tool call.
    """


@dataclass
class FleetSSHInitialConnectConfig:
    """Exponential-backoff params for the first SSH connect attempt.

    Used by ``wait_for_ssh_ready`` before bootstrap. Typical RunPod pod
    boot time is 30-90s; these defaults give a 5-10-20-40-80s backoff
    schedule and a 180s total budget. Bumping ``max_wait`` helps on
    slow cloud types; reducing it fails faster when a pod is truly dead.
    """
    max_attempts: int = 6
    backoff_base: int = 5       # seconds; doubles each attempt
    max_wait: int = 180         # total budget in seconds for initial connect


@dataclass
class FleetSSHConfig:
    """SSH-level reliability knobs for the fleet layer.

    - ``initial_connect``: how to wait for a freshly-provisioned pod to
      accept SSH traffic. Exponential backoff, not one giant timeout.
    - ``step_timeouts``: per-step timeout map for bootstrap operations.
      Keys are step names (e.g. ``pip_install``, ``data_download``);
      lookups fall back to the ``default`` key if a step isn't listed.
    """
    initial_connect: FleetSSHInitialConnectConfig = field(
        default_factory=FleetSSHInitialConnectConfig
    )
    step_timeouts: dict[str, int] = field(default_factory=lambda: {
        "default": 300,          # most bootstrap steps
        "sync_repo": 600,        # rsync of the whole repo can be slow
        "pip_install": 900,      # long on fresh images
        "data_download": 1800,   # very long
        "data_probe": 30,        # quick
        "python_version": 30,    # quick
        "torch_import": 60,      # quick
    })


@dataclass
class FleetConfig:
    """Top-level fleet configuration (SSH, retries, etc.)."""
    ssh: FleetSSHConfig = field(default_factory=FleetSSHConfig)


@dataclass
class ExecutionPolicyConfig:
    require_remote: bool = True
    required_provider: str = "runpod"
    allow_local_dev: bool = False


@dataclass
class PluginsConfig:
    discover: bool = True           # auto-discover plugins from .crucible/plugins/ and hub
    local_dir: str = "plugins"      # subdirectory under store_dir for local plugins
    hub_discover: bool = True       # also discover from ~/.crucible-hub/plugins/


@dataclass
class ProjectConfig:
    name: str = "crucible-project"
    version: str = CRUCIBLE_VERSION
    project_root: Path = field(default_factory=Path.cwd)
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: list[TrainingConfig] = field(default_factory=list)
    presets: dict[str, dict[str, str]] = field(default_factory=dict)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    researcher: ResearcherConfig = field(default_factory=ResearcherConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    hf_collab: HfCollabConfig = field(default_factory=HfCollabConfig)
    execution_policy: ExecutionPolicyConfig = field(default_factory=ExecutionPolicyConfig)
    fleet: FleetConfig = field(default_factory=FleetConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    store_dir: str = ".crucible"
    compose_builtin_specs: bool = False
    auto_commit_versions: bool = False
    research_state_file: str = "research_state.jsonl"
    sync_excludes: list[str] = field(default_factory=lambda: [
        ".git", ".venv", "__pycache__",
        ".crucible/designs", ".crucible/context", ".crucible/notes",
        ".crucible/store.jsonl", ".crucible/notes.jsonl",
        "logs", "data/datasets", "data/tokenizers",
    ])
    results_file: str = "experiments.jsonl"
    fleet_results_file: str = "experiments_fleet.jsonl"
    logs_dir: str = "logs"
    nodes_file: str = "nodes.json"
    runner_script: str = "src/crucible/runner/run_remote.py"
    remote_results_file: str = "experiments.jsonl"
    timeout_map: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "torch": {"smoke": 120, "proxy": 2400, "medium": 4800, "promotion": 9600, "overnight": 5400},
        "mlx": {"smoke": 180, "proxy": 3600, "medium": 7200, "promotion": 14400, "overnight": 7200},
    })
    hub_dir: str = ""              # override hub directory (default: ~/.crucible-hub)
    active_track: str = ""         # currently active research track


def _build_provider(raw: dict[str, Any]) -> ProviderConfig:
    return ProviderConfig(
        type=raw.get("type", "ssh"),
        ssh_key=raw.get("ssh_key", "~/.ssh/id_ed25519"),
        image=raw.get("image", ""),
        gpu_types=raw.get("gpu_types", []),
        gpu_count=raw.get("gpu_count", 1),
        interruptible=raw.get("interruptible", True),
        defaults=raw.get("defaults", {}),
        network_volume_id=raw.get("network_volume_id", ""),
        template_id=raw.get("template_id", ""),
    )


def _build_data_probe(raw: dict[str, Any]) -> DataProbeConfig:
    if not raw:
        return DataProbeConfig()
    return DataProbeConfig(
        paths=list(raw.get("paths", []) or []),
        script=str(raw.get("script", "") or ""),
        download_command=str(raw.get("download_command", "") or ""),
    )


def _build_data(raw: dict[str, Any]) -> DataConfig:
    return DataConfig(
        source=raw.get("source", "huggingface"),
        repo_id=raw.get("repo_id", ""),
        remote_prefix=raw.get("remote_prefix", "datasets"),
        local_root=raw.get("local_root", "./data"),
        manifest=raw.get("manifest", "manifest.json"),
        variant=raw.get("variant", "fineweb10B_sp1024"),
        path=raw.get("path", ""),
        wandb_entity=raw.get("wandb_entity", ""),
        wandb_project=raw.get("wandb_project", ""),
        wandb_artifact=raw.get("wandb_artifact", ""),
        wandb_artifact_type=raw.get("wandb_artifact_type", "dataset"),
        probe=_build_data_probe(raw.get("probe", {}) or {}),
    )


def _build_training(raw: list[dict[str, Any]]) -> list[TrainingConfig]:
    return [
        TrainingConfig(
            script=t.get("script", "train.py"),
            backend=t.get("backend", "torch"),
            python=t.get("python", "python3"),
            modality=t.get("modality", "lm"),
            optimizer=t.get("optimizer", ""),
            lr_schedule=t.get("lr_schedule", ""),
            logging_backends=t.get("logging_backends", ""),
            callbacks=t.get("callbacks", ""),
        )
        for t in raw
    ]


def _build_plugins(raw: dict[str, Any]) -> PluginsConfig:
    return PluginsConfig(
        discover=raw.get("discover", True),
        local_dir=raw.get("local_dir", "plugins"),
        hub_discover=raw.get("hub_discover", True),
    )


def _build_metrics(raw: dict[str, Any]) -> MetricsConfig:
    return MetricsConfig(
        primary=raw.get("primary", "val_loss"),
        secondary=raw.get("secondary", ""),
        size=raw.get("size", "model_bytes"),
        direction=raw.get("direction", "minimize"),
    )


def _build_researcher(raw: dict[str, Any]) -> ResearcherConfig:
    return ResearcherConfig(
        model=raw.get("model", "claude-sonnet-4-6-20250514"),
        max_tokens=raw.get("max_tokens", 4096),
        budget_hours=raw.get("budget_hours", 10.0),
        max_iterations=raw.get("max_iterations", 20),
        program_file=raw.get("program_file", "program.md"),
    )


def _build_wandb(raw: dict[str, Any]) -> WandbConfig:
    return WandbConfig(
        required=raw.get("required", True),
        project=raw.get("project", ""),
        entity=raw.get("entity", ""),
        mode=raw.get("mode", "online"),
    )


def _build_hf_collab(raw: dict[str, Any]) -> HfCollabConfig:
    if not raw:
        return HfCollabConfig()
    return HfCollabConfig(
        enabled=bool(raw.get("enabled", False)),
        leaderboard_repo=str(raw.get("leaderboard_repo", "") or ""),
        findings_repo=str(raw.get("findings_repo", "") or ""),
        recipes_repo=str(raw.get("recipes_repo", "") or ""),
        artifacts_repo=str(raw.get("artifacts_repo", "") or ""),
        private=bool(raw.get("private", True)),
        briefing_auto_pull=bool(raw.get("briefing_auto_pull", False)),
    )


def _build_execution_policy(raw: dict[str, Any]) -> ExecutionPolicyConfig:
    return ExecutionPolicyConfig(
        require_remote=raw.get("require_remote", True),
        required_provider=raw.get("required_provider", "runpod"),
        allow_local_dev=raw.get("allow_local_dev", False),
    )


def _build_fleet(raw: dict[str, Any]) -> FleetConfig:
    """Parse the optional ``fleet:`` section of crucible.yaml."""
    if not raw:
        return FleetConfig()
    ssh_raw = raw.get("ssh", {}) or {}
    initial_raw = ssh_raw.get("initial_connect", {}) or {}
    initial = FleetSSHInitialConnectConfig(
        max_attempts=int(initial_raw.get("max_attempts", 6)),
        backoff_base=int(initial_raw.get("backoff_base", 5)),
        max_wait=int(initial_raw.get("max_wait", 180)),
    )
    # Start from defaults, then merge user overrides. User keys win.
    default_timeouts = FleetSSHConfig().step_timeouts
    step_timeouts = dict(default_timeouts)
    for k, v in (ssh_raw.get("step_timeouts", {}) or {}).items():
        step_timeouts[str(k)] = int(v)
    return FleetConfig(
        ssh=FleetSSHConfig(
            initial_connect=initial,
            step_timeouts=step_timeouts,
        ),
    )


def load_config(path: Path | None = None) -> ProjectConfig:
    """Load a crucible.yaml file and return a ProjectConfig."""
    if path is None:
        path = find_config()
    if path is None:
        from crucible.core.log import log_warn
        log_warn("No crucible.yaml found, using default config")
        return ProjectConfig()

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    project_root = path.parent.resolve()

    return ProjectConfig(
        name=raw.get("name", "crucible-project"),
        version=raw.get("version", CRUCIBLE_VERSION),
        project_root=project_root,
        provider=_build_provider(raw.get("provider", {})),
        data=_build_data(raw.get("data", {})),
        training=_build_training(raw.get("training", [])),
        presets=raw.get("presets", {}),
        metrics=_build_metrics(raw.get("metrics", {})),
        researcher=_build_researcher(raw.get("researcher", {})),
        wandb=_build_wandb(raw.get("wandb", {})),
        hf_collab=_build_hf_collab(raw.get("hf_collab", {})),
        execution_policy=_build_execution_policy(raw.get("execution_policy", {})),
        fleet=_build_fleet(raw.get("fleet", {})),
        plugins=_build_plugins(raw.get("plugins", {})),
        store_dir=raw.get("store_dir", ".crucible"),
        compose_builtin_specs=raw.get("compose_builtin_specs", False),
        auto_commit_versions=raw.get("auto_commit_versions", False),
        research_state_file=raw.get("research_state_file", "research_state.jsonl"),
        sync_excludes=raw.get("sync_excludes", [
            ".git", ".venv", "__pycache__",
            ".crucible/designs", ".crucible/context", ".crucible/notes",
            ".crucible/store.jsonl", ".crucible/notes.jsonl",
            "logs", "data/datasets", "data/tokenizers",
        ]),
        results_file=raw.get("results_file", "experiments.jsonl"),
        fleet_results_file=raw.get("fleet_results_file", "experiments_fleet.jsonl"),
        logs_dir=raw.get("logs_dir", "logs"),
        nodes_file=raw.get("nodes_file", "nodes.json"),
        runner_script=raw.get("runner_script", "src/crucible/runner/run_remote.py"),
        remote_results_file=raw.get("remote_results_file", "experiments.jsonl"),
        timeout_map=raw.get("timeout_map", {
            "torch": {"smoke": 120, "proxy": 2400, "medium": 4800, "promotion": 9600, "overnight": 5400},
            "mlx": {"smoke": 180, "proxy": 3600, "medium": 7200, "promotion": 14400, "overnight": 7200},
        }),
        hub_dir=raw.get("hub_dir", ""),
        active_track=raw.get("active_track", ""),
    )


# ---------------------------------------------------------------------------
# External project specs
# ---------------------------------------------------------------------------

@dataclass
class PodOverrides:
    """Pod config overrides for external projects."""
    image: str = ""
    gpu_type: str = ""
    gpu_count: int = 0
    container_disk: int = 0
    volume_disk: int = 0
    interruptible: bool | None = None


@dataclass
class ProjectMetrics:
    """Metrics config for external project result collection."""
    source: str = "wandb"       # "wandb" | "stdout"
    primary: str = "val_loss"
    direction: str = "minimize"


@dataclass
class ProjectSpec:
    """Spec for running an external project on fleet pods."""
    name: str
    repo: str
    branch: str = "main"
    shallow: bool = True
    workspace: str = "/workspace/project"
    python: str = ""                         # empty = use system python
    install: list[str] = field(default_factory=list)
    install_flags: str = ""                  # global pip flags (e.g. --index-url)
    install_torch: str = ""                  # separate torch install line
    launcher: str = ""                       # reusable launcher bundle name
    launcher_entry: str = ""                # entry script inside launcher bundle
    local_files: list[str] = field(default_factory=list)  # local paths to scp to workspace
    system_packages: list[str] = field(default_factory=list)  # extra OS packages for bootstrap
    setup: list[str] = field(default_factory=list)
    setup_timeout: int = 3600
    train: str = ""
    timeout: int = 0                         # training timeout, 0 = no limit
    launch_timeout: int = 300               # SSH timeout for detached launch
    env_forward: list[str] = field(default_factory=list)
    env_set: dict[str, str] = field(default_factory=dict)
    pod: PodOverrides = field(default_factory=PodOverrides)
    metrics: ProjectMetrics = field(default_factory=ProjectMetrics)
    # Named variant dicts. Each variant maps to {ENV_VAR: value} that
    # `run_project(variant=...)` merges into the launch overrides before
    # the caller's own `overrides` (which still win). Until the caller
    # passes a variant name, the dict is inert.
    variants: dict[str, dict[str, str]] = field(default_factory=dict)


def _build_pod_overrides(raw: dict[str, Any]) -> PodOverrides:
    if not raw:
        return PodOverrides()
    return PodOverrides(
        image=raw.get("image", ""),
        gpu_type=raw.get("gpu_type", ""),
        gpu_count=raw.get("gpu_count", 0),
        container_disk=raw.get("container_disk", 0),
        volume_disk=raw.get("volume_disk", 0),
        interruptible=raw.get("interruptible"),
    )


def _build_project_metrics(raw: dict[str, Any]) -> ProjectMetrics:
    if not raw:
        return ProjectMetrics()
    return ProjectMetrics(
        source=raw.get("source", "wandb"),
        primary=raw.get("primary", "val_loss"),
        direction=raw.get("direction", "minimize"),
    )


def load_project_spec(name: str, project_root: Path | None = None) -> ProjectSpec:
    """Load a project spec from ``.crucible/projects/<name>.yaml``.

    Resolution order mirrors ``resolve_launcher_bundle``:

    1. Project-local: ``<project_root>/.crucible/projects/<name>.yaml``
    2. Hub: ``~/.crucible-hub/projects/<name>.yaml`` (future-proofing; not yet
       written to by any CLI)
    3. Any configured tap: ``~/.crucible-hub/taps/*/projects/<name>.yaml``

    This lets ``crucible run_project``, ``provision_project`` etc. consume
    specs that ship directly from community taps without a manual ``cp`` step.
    """
    root = project_root or Path.cwd()
    local_spec_path = root / ".crucible" / "projects" / f"{name}.yaml"

    candidates: list[Path] = [local_spec_path]

    # Hub + tap resolution (only walked if the local candidate doesn't exist).
    if not local_spec_path.exists():
        try:
            from crucible.core.hub import HubStore
            hub_dir = HubStore.resolve_hub_dir()
        except (ImportError, OSError):
            hub_dir = None
        if hub_dir is not None:
            candidates.append(hub_dir / "projects" / f"{name}.yaml")
            taps_root = hub_dir / "taps"
            if taps_root.is_dir():
                for tap_dir in sorted(taps_root.iterdir()):
                    if tap_dir.is_dir():
                        candidates.append(tap_dir / "projects" / f"{name}.yaml")

    spec_path: Path | None = None
    for candidate in candidates:
        if candidate.exists():
            spec_path = candidate
            break

    if spec_path is None:
        searched = "\n  ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"Project spec {name!r} not found. Searched:\n  {searched}"
        )

    raw = yaml.safe_load(spec_path.read_text(encoding="utf-8")) or {}
    return ProjectSpec(
        name=raw.get("name", name),
        repo=raw.get("repo", ""),
        branch=raw.get("branch", "main"),
        shallow=raw.get("shallow", True),
        workspace=raw.get("workspace", "/workspace/project"),
        python=raw.get("python", ""),
        install=raw.get("install", []),
        install_flags=raw.get("install_flags", ""),
        install_torch=raw.get("install_torch", ""),
        launcher=raw.get("launcher", ""),
        launcher_entry=raw.get("launcher_entry", ""),
        local_files=raw.get("local_files", []),
        system_packages=raw.get("system_packages", []),
        setup=raw.get("setup", []),
        setup_timeout=raw.get("setup_timeout", 3600),
        train=raw.get("train", ""),
        timeout=raw.get("timeout", 0),
        launch_timeout=raw.get("launch_timeout", 300),
        env_forward=raw.get("env_forward", []),
        env_set={k: str(v) for k, v in (raw.get("env_set") or {}).items()},
        pod=_build_pod_overrides(raw.get("pod", {})),
        metrics=_build_project_metrics(raw.get("metrics", {})),
        variants=_build_variants(raw.get("variants", {})),
    )


def _build_variants(raw: Any) -> dict[str, dict[str, str]]:
    """Coerce the ``variants:`` yaml block into a ``{name: {ENV_VAR: str}}`` dict.

    Each variant's values are stringified so they can be exported as environment
    variables unchanged. Non-dict values (malformed yaml) are skipped with a
    warning. An empty or missing block returns an empty dict.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, str]] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            continue
        out[str(name)] = {str(k): str(v) for k, v in body.items()}
    return out


def list_project_specs(project_root: Path | None = None) -> list[dict[str, Any]]:
    """List all project specs in ``.crucible/projects/``."""
    root = project_root or Path.cwd()
    specs_dir = root / ".crucible" / "projects"
    if not specs_dir.is_dir():
        return []
    results = []
    for p in sorted(specs_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            results.append({
                "name": raw.get("name", p.stem),
                "repo": raw.get("repo", ""),
                "launcher": raw.get("launcher", ""),
                "train": raw.get("train", ""),
                "metrics_primary": raw.get("metrics", {}).get("primary", "val_loss"),
            })
        except (OSError, ValueError, KeyError, TypeError) as exc:
            from crucible.core.log import log_warn
            log_warn(f"Failed to parse project spec {p.name}: {exc}")
            continue
    return results


def find_config() -> Path | None:
    """Locate ``crucible.yaml`` for the current project.

    Resolution order:

    1. ``$CRUCIBLE_PROJECT_ROOT/crucible.yaml`` — explicit override. Set
       this when the cwd is ambiguous (e.g. running from a parent dir, a
       git worktree, an MCP server spawned from the wrong directory).
       If the env var is set but the file is missing, raises
       :class:`crucible.core.errors.ConfigError` — the operator clearly
       meant something specific, and silently falling back to defaults
       would re-introduce the multi-project safety hole this override
       exists to close.
    2. Walk up from ``Path.cwd()`` until a ``crucible.yaml`` is found.

    Returns ``None`` only when the walk-up itself produces no hit and no
    env var is set.
    """
    explicit = os.environ.get("CRUCIBLE_PROJECT_ROOT")
    if explicit:
        candidate = Path(explicit).expanduser().resolve() / "crucible.yaml"
        if candidate.exists():
            return candidate
        from crucible.core.errors import ConfigError
        raise ConfigError(
            f"CRUCIBLE_PROJECT_ROOT={explicit!r} is set but no crucible.yaml "
            f"exists at {candidate}. Either fix the path, unset the env var, "
            f"or run from inside the project directory."
        )

    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "crucible.yaml"
        if candidate.exists():
            return candidate
    return None


def generate_default_config() -> str:
    """Return a default crucible.yaml template as a string."""
    return """\
# Crucible project configuration
name: my-project
version: "{version}"

# Compute provider
provider:
  type: runpod                           # "runpod" | "ssh"
  ssh_key: ~/.ssh/id_ed25519_runpod
  # image: runpod/pytorch:...            # container image (RunPod only)
  # gpu_types: ["NVIDIA GeForce RTX 4090"]
  # interruptible: true                  # false = secure/on-demand RunPod pods

# Data pipeline
data:
  source: huggingface
  repo_id: ""                            # HuggingFace dataset repo
  local_root: ./data
  manifest: manifest.json

# Training backends
training:
  - backend: torch
    script: train.py

# Metrics configuration
metrics:
  primary: val_loss                        # metric to rank by
  # secondary: ""                          # optional secondary display metric
  # size: model_bytes                      # for Pareto frontier
  # direction: minimize                    # "minimize" | "maximize"

# Experiment presets
presets:
  smoke:
    MAX_WALLCLOCK_SECONDS: "60"
    ITERATIONS: "400"
  proxy:
    MAX_WALLCLOCK_SECONDS: "1800"
    ITERATIONS: "6000"

# Autonomous researcher
researcher:
  model: claude-sonnet-4-6-20250514
  budget_hours: 10.0
  program_file: program.md

# Weights & Biases
wandb:
  required: true
  project: ""                            # or set WANDB_PROJECT in env
  entity: ""                             # optional team/entity
  mode: online                           # "online" | "offline" | "disabled"

# Experiment execution policy
execution_policy:
  require_remote: true
  required_provider: runpod
  allow_local_dev: false

# Version store
# store_dir: .crucible                    # version store directory
# auto_commit_versions: false             # auto-commit versions to git

# Hub integration
# hub_dir: ~/.crucible-hub                 # cross-project knowledge store
# active_track: ""                          # current research track

# Sync exclusions
sync_excludes:
  - .git
  - .venv
  - __pycache__
  - logs
  - data/datasets

# Output paths
results_file: experiments.jsonl
logs_dir: logs
""".format(version=CRUCIBLE_VERSION)
