"""Load and validate crucible.yaml project configuration."""
from __future__ import annotations

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


@dataclass
class DataConfig:
    source: str = "huggingface"
    repo_id: str = ""
    remote_prefix: str = "datasets"
    local_root: str = "./data"
    manifest: str = "manifest.json"


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
    execution_policy: ExecutionPolicyConfig = field(default_factory=ExecutionPolicyConfig)
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
    )


def _build_data(raw: dict[str, Any]) -> DataConfig:
    return DataConfig(
        source=raw.get("source", "huggingface"),
        repo_id=raw.get("repo_id", ""),
        remote_prefix=raw.get("remote_prefix", "datasets"),
        local_root=raw.get("local_root", "./data"),
        manifest=raw.get("manifest", "manifest.json"),
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


def _build_execution_policy(raw: dict[str, Any]) -> ExecutionPolicyConfig:
    return ExecutionPolicyConfig(
        require_remote=raw.get("require_remote", True),
        required_provider=raw.get("required_provider", "runpod"),
        allow_local_dev=raw.get("allow_local_dev", False),
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
        execution_policy=_build_execution_policy(raw.get("execution_policy", {})),
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


def _build_pod_overrides(raw: dict[str, Any]) -> PodOverrides:
    if not raw:
        return PodOverrides()
    return PodOverrides(
        image=raw.get("image", ""),
        gpu_type=raw.get("gpu_type", ""),
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
    """Load a project spec from ``.crucible/projects/<name>.yaml``."""
    root = project_root or Path.cwd()
    spec_path = root / ".crucible" / "projects" / f"{name}.yaml"
    if not spec_path.exists():
        raise FileNotFoundError(f"Project spec not found: {spec_path}")
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
        env_set=raw.get("env_set", {}),
        pod=_build_pod_overrides(raw.get("pod", {})),
        metrics=_build_project_metrics(raw.get("metrics", {})),
    )


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
        except Exception as exc:
            from crucible.core.log import log_warn
            log_warn(f"Failed to parse project spec {p.name}: {exc}")
            continue
    return results


def find_config() -> Path | None:
    """Walk up from cwd looking for crucible.yaml."""
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
