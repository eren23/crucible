"""Node bootstrap: env setup, pip install, dataset download, git sync."""
from __future__ import annotations

import concurrent.futures
import json
import shlex
import threading
import time
from pathlib import Path
from typing import Any

from crucible.core.log import log_info, log_step, log_success, log_warn, utc_now_iso
from crucible.core.types import NodeRecord
from crucible.fleet.day_run import append_event
from crucible.fleet.inventory import (
    NODES_LOCK,
    count_bootstrapped_ready,
    load_nodes_if_exists,
    load_nodes_snapshot,
    merge_node_snapshots,
    ready_state,
    save_nodes,
    upsert_node_record,
)
from crucible.fleet.sync import (
    checked_remote_exec,
    local_git_sha,
    rsync_base,
    ssh_ok,
    _run,
    sync_env_file,
    sync_repo,
    sync_taps,
)

BOOTSTRAP_ATTEMPTS = 3
DEFAULT_PROJECT_SYSTEM_PACKAGES = ("git", "rsync", "curl")


def _remote_data_source_partial_probe_script(plugin_name: str, config_dict: dict[str, Any]) -> str:
    """Python source run on the node: print 1 if data source status is PARTIAL, else 0."""
    cfg_literal = repr(json.dumps(config_dict))
    pl_literal = repr(plugin_name)
    return (
        "import json\n"
        "import crucible.data_sources\n"
        "from crucible.core.data_sources import build_data_source, DataStatus\n"
        f"cfg = json.loads({cfg_literal})\n"
        f"st = build_data_source({pl_literal}, name='_bootstrap_ds', config=cfg).status()\n"
        "print(1 if st.status == DataStatus.PARTIAL else 0)\n"
    )


# ---------------------------------------------------------------------------
# Global plugin materialization
# ---------------------------------------------------------------------------

def _materialize_global_architectures(project_root: Path) -> None:
    """Mirror hub architectures into .crucible/architectures/_hub/ for pod sync."""
    import shutil
    try:
        from crucible.core.config import ProjectConfig, load_config
        from crucible.core.hub import HubStore
        cfg_path = project_root / "crucible.yaml"
        cfg = load_config(cfg_path) if cfg_path.exists() else ProjectConfig(project_root=project_root)
        hub_dir = HubStore.discover(config_hub_dir=getattr(cfg, "hub_dir", ""))
        if hub_dir is None:
            return
        plugins_dir = hub_dir / "architectures" / "plugins"
        specs_dir = hub_dir / "architectures" / "specs"
        if not plugins_dir.is_dir() and not specs_dir.is_dir():
            return
        arch_root = project_root / ".crucible" / "architectures"
        target_dir = arch_root / "_hub"
        target_dir.mkdir(parents=True, exist_ok=True)

        # Remove legacy top-level mirrors from the old _hub_<name>.py scheme.
        for legacy in arch_root.glob("_hub_*.py"):
            legacy.unlink(missing_ok=True)
        for legacy in arch_root.glob("_hub_*.yaml"):
            legacy.unlink(missing_ok=True)

        # Rebuild the mirror directory so deletions in the hub propagate cleanly.
        for existing in target_dir.glob("*"):
            if existing.is_file():
                existing.unlink()

        for source_dir, pattern in ((plugins_dir, "*.py"), (specs_dir, "*.yaml")):
            if not source_dir.is_dir():
                continue
            for src_file in source_dir.glob(pattern):
                if src_file.name.startswith("_"):
                    continue
                shutil.copy2(src_file, target_dir / src_file.name)
    except Exception as exc:
        log_warn(f"Hub architecture materialization failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Single-node bootstrap
# ---------------------------------------------------------------------------

def _resolve_step_timeout(label: str, explicit: int | None) -> int | None:
    """Pick a timeout for a bootstrap step.

    Precedence:
      1. explicit caller override (passed via ``timeout=...``)
      2. per-step entry from ``fleet.ssh.step_timeouts`` in crucible.yaml
      3. ``default`` entry from the same config map
      4. None (use checked_remote_exec's own default of 600s)
    """
    if explicit is not None:
        return explicit
    try:
        from crucible.core.config import load_config
        cfg = load_config()
        timeouts = cfg.fleet.ssh.step_timeouts
    except Exception:
        # Config missing or unreadable — fall back to the hard-coded default
        return None
    if label in timeouts:
        return int(timeouts[label])
    if "default" in timeouts:
        return int(timeouts["default"])
    return None


def bootstrap_step(
    node: NodeRecord,
    label: str,
    command: str,
    *,
    timeout: int | None = None,
) -> Any:
    """Run one labelled bootstrap command on a node.

    ``timeout`` defaults to the per-step value from
    ``fleet.ssh.step_timeouts`` (see FleetSSHConfig). Pass an explicit
    integer to override for a single call.
    """
    effective_timeout = _resolve_step_timeout(label, timeout)
    log_step(f"bootstrap {node['name']}: {label}")
    return checked_remote_exec(
        node, f"bootstrap:{label}", command, timeout=effective_timeout
    )


def _record_step(
    node: NodeRecord,
    step_name: str,
    fn: Any,
    *,
    required: bool = True,
) -> Any:
    """Run *fn* and record its outcome in ``node["bootstrap_steps"]``.

    Every bootstrap step goes through this helper so we can surface
    per-step status in get_fleet_status and avoid the "ready-but-broken"
    failure mode where a partially-bootstrapped node gets flagged ready
    because only the final state assignment was reached.

    :param node: the node record to mutate
    :param step_name: machine-readable step identifier (e.g. "sync_repo")
    :param fn: a thunk (no-arg callable) that runs the actual work
    :param required: if True, re-raises on failure so the caller / retry
        loop can see it; if False, logs at WARN and swallows, keeping the
        overall bootstrap progressing. Optional steps are things like
        auxiliary data probes and hub-plugin materialization — useful
        when they work, non-fatal when they don't.
    """
    import traceback
    steps = node.setdefault("bootstrap_steps", {})
    steps[step_name] = {
        "status": "running",
        "started_at": utc_now_iso(),
        "error": None,
        "required": required,
    }
    try:
        result = fn()
    except BaseException as thrown:
        steps[step_name] = {
            "status": "failed",
            "started_at": steps[step_name]["started_at"],
            "finished_at": utc_now_iso(),
            "error": str(thrown),
            "required": required,
        }
        if required:
            raise
        # Optional step — log the full traceback (previously swallowed)
        # so operators can see WHY the probe/mirror failed without
        # losing forward progress on the bootstrap.
        log_warn(
            f"{node['name']}: optional bootstrap step {step_name!r} "
            f"failed (non-fatal): {thrown}"
        )
        log_warn(
            f"{node['name']}: {step_name!r} traceback:\n{traceback.format_exc()}"
        )
        return None
    steps[step_name] = {
        "status": "ok",
        "started_at": steps[step_name]["started_at"],
        "finished_at": utc_now_iso(),
        "error": None,
        "required": required,
    }
    return result


def bootstrap_state_summary(node: NodeRecord) -> dict[str, Any]:
    """Summarize bootstrap progress for display in get_fleet_status.

    Returns a compact dict with:
    - total: number of tracked steps
    - ok: count of successful steps
    - failed: list of {step, error} for failures (required + optional)
    - required_failed: count of REQUIRED steps that failed
    - all_required_ok: bool — True iff no required step failed

    Returns an empty summary if the node has no bootstrap_steps yet.
    """
    steps = node.get("bootstrap_steps") or {}
    if not steps:
        return {
            "total": 0,
            "ok": 0,
            "failed": [],
            "required_failed": 0,
            "all_required_ok": True,
        }
    ok_count = 0
    failed: list[dict[str, Any]] = []
    required_failed = 0
    for name, info in steps.items():
        status = info.get("status")
        if status == "ok":
            ok_count += 1
        elif status == "failed":
            failed.append({
                "step": name,
                "error": info.get("error", ""),
                "required": bool(info.get("required", True)),
            })
            if info.get("required", True):
                required_failed += 1
    return {
        "total": len(steps),
        "ok": ok_count,
        "failed": failed,
        "required_failed": required_failed,
        "all_required_ok": required_failed == 0,
    }


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        item = str(value).strip()
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def ensure_project_system_tools(
    node: NodeRecord,
    spec: Any,
    *,
    timeout: int,
) -> None:
    """Install generic OS tools required for external-project bootstrap.

    Bare container images often miss ``git``, ``curl``, or ``rsync``. Those are
    needed for clone/update, uv installation, and launcher sync. Projects can
    request additional OS packages via ``system_packages`` in their spec.
    """
    packages = _dedupe_preserve_order(
        list(DEFAULT_PROJECT_SYSTEM_PACKAGES) + list(getattr(spec, "system_packages", []) or [])
    )
    package_args = " ".join(shlex.quote(pkg) for pkg in packages)
    core_checks = " && ".join(
        f"command -v {shlex.quote(tool)} >/dev/null 2>&1"
        for tool in DEFAULT_PROJECT_SYSTEM_PACKAGES
    )
    needs_install = "true" if getattr(spec, "system_packages", []) else f"! ( {core_checks} )"
    install_cmd = (
        f"if {needs_install}; then "
        "if command -v apt-get >/dev/null 2>&1; then "
        "export DEBIAN_FRONTEND=noninteractive && apt-get update && "
        f"apt-get install -y {package_args}; "
        "elif command -v apk >/dev/null 2>&1; then "
        f"apk add --no-cache {package_args}; "
        "elif command -v dnf >/dev/null 2>&1; then "
        f"dnf install -y {package_args}; "
        "elif command -v yum >/dev/null 2>&1; then "
        f"yum install -y {package_args}; "
        "else "
        "echo unsupported_package_manager >&2; exit 90; "
        "fi; "
        "else echo system_tools_ready; "
        "fi"
    )
    bootstrap_step(node, "system_tools", install_cmd, timeout=timeout)


def _build_data_probe_command(
    proj_cfg: Any,
    workspace: str,
    py: str,
) -> str | None:
    """Generate a remote probe command from the project's data config.

    Returns ``None`` if no probe is configured, in which case data
    bootstrap is a no-op. Priority order:

    1. ``data.probe.script``: run a custom Python file on the remote.
       The file is expected to print ``1`` if data is ready, ``0``
       otherwise.
    2. ``data.probe.paths``: generate a Python one-liner that checks
       every listed path exists (files must exist, directories must
       be non-empty).
    3. Legacy fineweb paths: only when ``data.source == "huggingface"``
       and ``data.variant`` starts with ``fineweb`` (preserves the
       Parameter Golf reference config's behavior without hardcoding).
    4. Otherwise: ``None``.
    """
    data_cfg = getattr(proj_cfg, "data", None)
    if data_cfg is None:
        return None
    probe = getattr(data_cfg, "probe", None)
    if probe is not None:
        script = (getattr(probe, "script", "") or "").strip()
        if script:
            return (
                f"cd {workspace} && {py} {shlex.quote(script)}"
            )
        paths = list(getattr(probe, "paths", []) or [])
        if paths:
            return _generate_paths_probe(workspace, py, paths)
    # Legacy fineweb fallback — fires only when this is clearly a
    # Parameter Golf-style LM setup. Every other project needs an
    # explicit ``data.probe.paths`` or ``data.probe.script``.
    source = str(getattr(data_cfg, "source", "") or "").lower()
    variant = str(getattr(data_cfg, "variant", "") or "").lower()
    if source == "huggingface" and variant.startswith("fineweb"):
        return (
            f"cd {workspace} && {py} - <<'PY'\n"
            "from pathlib import Path\n"
            "root = Path('data/datasets/fineweb10B_sp1024')\n"
            "tok = Path('data/tokenizers/fineweb_1024_bpe.model')\n"
            "train = list(root.glob('fineweb_train_*.bin')) if root.exists() else []\n"
            "val = list(root.glob('fineweb_val_*.bin')) if root.exists() else []\n"
            "print(int(tok.exists() and bool(train) and bool(val)))\n"
            "PY"
        )
    return None


def _generate_paths_probe(workspace: str, py: str, paths: list[str]) -> str:
    """Render a Python probe that verifies all *paths* exist on the remote.

    Files must exist; directory paths (ending with ``/``) must exist
    AND be non-empty. Glob patterns (containing ``*``) must match at
    least one path. Prints ``1`` if every path check succeeds, ``0``
    otherwise.
    """
    import json
    paths_literal = json.dumps(list(paths))
    body = (
        "from pathlib import Path\n"
        f"paths = {paths_literal}\n"
        "ok = True\n"
        "for p in paths:\n"
        "    if '*' in p:\n"
        "        matches = list(Path('.').glob(p))\n"
        "        if not matches:\n"
        "            ok = False; break\n"
        "        continue\n"
        "    pth = Path(p)\n"
        "    if p.endswith('/'):\n"
        "        if not pth.is_dir() or not any(pth.iterdir()):\n"
        "            ok = False; break\n"
        "    else:\n"
        "        if not pth.exists():\n"
        "            ok = False; break\n"
        "print(1 if ok else 0)\n"
    )
    return f"cd {workspace} && {py} - <<'PY'\n{body}PY"


def _legacy_fineweb_download_command(
    proj_cfg: Any,
    workspace: str,
    py: str,
    train_shards: int,
) -> str:
    """Return the legacy fineweb auto-download command for PG-style configs.

    Only fires when the project config looks like a Parameter Golf LM
    setup (source=huggingface, variant starts with 'fineweb'). For
    every other project the caller should supply an explicit
    ``data.probe.download_command``. Returns an empty string when no
    fallback is warranted.
    """
    data_cfg = getattr(proj_cfg, "data", None)
    if data_cfg is None:
        return ""
    source = str(getattr(data_cfg, "source", "") or "").lower()
    variant = str(getattr(data_cfg, "variant", "") or "").lower()
    if source == "huggingface" and variant.startswith("fineweb"):
        return (
            f"cd {workspace} && {py} data/cached_challenge_fineweb.py "
            f"--variant sp1024 --train-shards {train_shards}"
        )
    return ""


def bootstrap_node(
    node: NodeRecord,
    *,
    project_root: Path,
    sync_excludes: list[str],
    train_shards: int,
    skip_install: bool = False,
    skip_data: bool = False,
    data_download_cmd: str | None = None,
    data_source_name: str | None = None,
    data_source_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full bootstrap sequence for a single node.

    1. rsync the project
    2. rsync the .env file
    3. verify Python and CUDA
    4. pip install (unless *skip_install*)
    5. download dataset (unless *skip_data*)
    6. record git sha

    Data source pre-check (before auto-download) uses ``crucible.yaml`` ``data:`` when
    *data_source_name* is ``None``. It runs on the **remote** workspace so paths match
    the node. Optional *data_source_config* is shallow-merged onto the mapped config.
    """
    # Reset per-run step tracking so retries produce a clean record.
    node["bootstrap_steps"] = {}

    # Load the project config once up front — the data probe/download
    # step needs it, and loading it multiple times inside step closures
    # adds latency and confuses test mocks.
    from crucible.core.config import ProjectConfig, load_config
    cfg_path = project_root / "crucible.yaml"
    proj_cfg = (
        load_config(cfg_path)
        if cfg_path.is_file()
        else ProjectConfig(project_root=project_root.resolve())
    )

    _record_step(
        node, "materialize_plugins",
        lambda: _materialize_global_architectures(project_root),
        required=False,
    )
    _record_step(
        node, "sync_repo",
        lambda: sync_repo(node, project_root=project_root, sync_excludes=sync_excludes),
    )
    _record_step(
        node, "sync_env_file",
        lambda: sync_env_file(node, project_root=project_root),
    )
    # Promote sync_taps to required when the user has at least one tap
    # configured in ~/.crucible-hub/taps/. Silent rsync failures otherwise
    # cause late ImportError on pods (the tap's architecture / callback /
    # data-adapter plugins never land on disk and the training script dies
    # at module-load time). Users with zero taps are unaffected — sync_taps
    # becomes a no-op anyway.
    _taps_dir = Path.home() / ".crucible-hub" / "taps"
    _has_taps = _taps_dir.is_dir() and any(
        p.is_dir() for p in _taps_dir.iterdir() if not p.name.startswith(".")
    )
    _record_step(node, "sync_taps", lambda: sync_taps(node), required=_has_taps)

    ws_path = node.get("workspace_path", "/workspace/project")
    workspace = shlex.quote(ws_path)
    py = shlex.quote(node.get("python_bin", "python3"))
    py_src = shlex.quote(str(Path(ws_path) / "src"))

    _record_step(
        node, "python_version",
        lambda: bootstrap_step(node, "python_version", f"cd {workspace} && {py} --version"),
    )
    _record_step(
        node, "torch_import",
        lambda: bootstrap_step(
            node,
            "torch_import",
            f"cd {workspace} && {py} - <<'PY'\n"
            "import torch\n"
            "print(torch.__version__)\n"
            "print(torch.version.cuda)\n"
            "ok = torch.cuda.is_available()\n"
            "print(ok)\n"
            "if not ok:\n"
            "    raise SystemExit('cuda_unavailable')\n"
            "print(torch.cuda.device_count())\n"
            "PY",
        ),
    )

    if not skip_install:
        _record_step(
            node, "pip_install",
            lambda: bootstrap_step(
                node,
                "pip_install",
                f"cd {workspace} && "
                "grep -viE '^(torch|torchvision|torchaudio)([<>=].*)?$' requirements.txt "
                "> /tmp/crucible.requirements.txt && "
                f"{py} -m pip install --break-system-packages -r /tmp/crucible.requirements.txt",
            ),
        )

    # Pre-download check: PARTIAL status on the remote node skips auto-download.
    # The probe is best-effort — it's informational, not required — so wrapping
    # it in _record_step(required=False) logs any failure at WARN without
    # killing the bootstrap. Previously this used a bare `except: pass` which
    # silently swallowed all errors.
    _skip_auto_download = False
    if not skip_data:
        from crucible.core.data_sources import bootstrap_data_source_spec_from_data_config

        spec = bootstrap_data_source_spec_from_data_config(
            proj_cfg.data,
            plugin_name_override=data_source_name,
            config_override=data_source_config,
        )
        if spec is not None:
            plugin_name, ds_cfg = spec

            def _probe_data_source() -> None:
                nonlocal _skip_auto_download
                probe_body = _remote_data_source_partial_probe_script(plugin_name, ds_cfg)
                status_step = bootstrap_step(
                    node,
                    "data_source_partial_probe",
                    f"cd {workspace} && PYTHONPATH={py_src} {py} - <<'PY'\n{probe_body}\nPY\n",
                )
                out = (status_step.stdout or "").strip()
                if out == "1":
                    _skip_auto_download = True
                    log_warn(
                        f"{node['name']}: data source {plugin_name!r} is PARTIAL "
                        f"(incomplete vs manifest); skipping auto-download. "
                        f"Verify shard layout on the node."
                    )

            _record_step(
                node, "data_source_partial_probe",
                _probe_data_source,
                required=False,
            )

    # Build a probe command from project config. Returns None if no
    # probe is configured (in which case data bootstrap is a no-op —
    # previously this used hardcoded fineweb paths that silently
    # failed for non-LM projects).
    probe_command = _build_data_probe_command(proj_cfg, workspace, py)

    data_probe: Any = None
    if not skip_data and probe_command is not None:
        def _run_data_probe() -> Any:
            return bootstrap_step(node, "data_probe", probe_command)

        # The data probe is required: if we can't tell whether the dataset
        # is present, we can't safely mark the node ready for dispatch.
        data_probe = _record_step(node, "data_probe", _run_data_probe)

    if skip_data or probe_command is None:
        # Projects that download data at training time, or have no data
        # probe configured, don't need a fleet-level dataset. Mark as
        # ready so dispatch doesn't filter the node out with
        # "dataset_missing".
        node["dataset_ready"] = True

    if not skip_data and data_probe is not None:
        data_missing = data_probe.stdout.strip().endswith("0")
        if data_missing:
            if _skip_auto_download:
                log_warn(
                    f"{node['name']}: auto-download skipped due to partial/incomplete "
                    f"data source status"
                )
            else:
                # Resolve download command: runtime override > config > legacy
                # fineweb default (only when ``data.source=='huggingface'`` and
                # ``data.variant`` looks like a fineweb variant, i.e. when this
                # is actually the Parameter Golf workflow).
                download_cmd = (
                    data_download_cmd
                    or (proj_cfg.data.probe.download_command or "").strip()
                    or _legacy_fineweb_download_command(
                        proj_cfg, workspace, py, train_shards
                    )
                )
                if download_cmd:
                    cmd_for_step = download_cmd
                    _record_step(
                        node, "data_download",
                        lambda: bootstrap_step(node, "data_download", cmd_for_step),
                    )
                    node["dataset_ready"] = True
                else:
                    log_warn(
                        f"{node['name']}: data probe reports missing data but "
                        f"no download_command is configured — leaving "
                        f"dataset_ready=False so dispatch skips this node"
                    )
        else:
            # Data exists on remote, mark as ready
            node["dataset_ready"] = True

    node["git_sha"] = local_git_sha(project_root)
    if node["git_sha"] is None:
        log_warn(f"{node['name']}: local git SHA unavailable; recording null git_sha")

    # Final state flip — only reached if every REQUIRED step in
    # _record_step succeeded (required failures re-raise and unwind us
    # before we get here). Optional step failures are visible via
    # node["bootstrap_steps"] and bootstrap_state_summary() so the user
    # can inspect them via get_fleet_status.
    summary = bootstrap_state_summary(node)
    node["last_seen_at"] = utc_now_iso()
    node["env_ready"] = summary["all_required_ok"]
    node["state"] = "ready" if summary["all_required_ok"] else "bootstrap_failed"
    if not summary["all_required_ok"]:
        log_warn(
            f"{node['name']}: bootstrap completed with "
            f"{summary['required_failed']} required step failure(s); "
            f"state set to bootstrap_failed"
        )
    return node


# ---------------------------------------------------------------------------
# External project bootstrap
# ---------------------------------------------------------------------------

def bootstrap_project(
    node: NodeRecord,
    spec: Any,
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Bootstrap an external project on a node. Idempotent and safe to retry.

    Steps: clone repo, create venv, install deps, forward env, run setup, mark ready.
    All values passed to shell commands are quoted via shlex.quote.
    """
    import shlex as _shlex
    from crucible.fleet.sync import write_remote_env

    ws = _shlex.quote(spec.workspace)
    name = node["name"]
    branch_q = _shlex.quote(spec.branch)
    repo_q = _shlex.quote(spec.repo)
    project_step_timeout = max(int(getattr(spec, "setup_timeout", 600) or 600), 1)

    ensure_project_system_tools(node, spec, timeout=project_step_timeout)

    # 0. Kill any old training processes to free GPU memory
    log_step(f"{name}: cleaning up previous training processes")
    try:
        # Uses pkill to find python training scripts in the workspace
        cleanup_cmd = f"pkill -9 -f 'python.*train' 2>/dev/null; sleep 1; echo cleanup_done"
        remote_exec(node, cleanup_cmd, check=False, timeout=15)
    except Exception:
        pass  # Non-fatal: cleanup is best-effort

    # 1. Clone or update repo
    log_step(f"{name}: cloning {spec.repo} (branch={spec.branch})")
    clone_check = bootstrap_step(
        node, "repo_check", f"test -d {ws}/.git && echo exists || echo missing",
    )
    if clone_check and "exists" in (clone_check.stdout or ""):
        bootstrap_step(
            node, "repo_update",
            f"cd {ws} && git fetch origin && "
            f"git checkout {branch_q} && "
            f"git reset --hard origin/{branch_q}",
            timeout=project_step_timeout,
        )
    else:
        depth = "--depth 1" if spec.shallow else ""
        bootstrap_step(
            node, "repo_clone",
            f"git clone {depth} -b {branch_q} {repo_q} {ws}",
            timeout=project_step_timeout,
        )

    # 2. Create Python venv
    if spec.python:
        venv_check = bootstrap_step(
            node, "venv_check",
            f"test -f {ws}/.venv/bin/activate && echo exists || echo missing",
        )
        if venv_check and "missing" in (venv_check.stdout or ""):
            log_step(f"{name}: creating Python {spec.python} venv")
            bootstrap_step(
                node, "install_uv",
                "command -v uv >/dev/null 2>&1 || "
                "(curl -LsSf https://astral.sh/uv/install.sh | sh)",
                timeout=project_step_timeout,
            )
            python_q = _shlex.quote(spec.python)
            bootstrap_step(
                node, "create_venv",
                f'export PATH="$HOME/.local/bin:$PATH" && '
                f"cd {ws} && uv venv --python={python_q} .venv",
                timeout=project_step_timeout,
            )

    # Activation prefix for subsequent commands
    activate = f"cd {ws}"
    if spec.python:
        activate += " && source .venv/bin/activate"
    uv_pfx = 'export PATH="$HOME/.local/bin:$PATH" && '

    # 3. Install torch separately (with install_flags for index-url)
    if spec.install_torch:
        log_step(f"{name}: installing torch")
        flags = spec.install_flags or ""
        # Quote each package spec to prevent shell interpretation of < > chars
        torch_pkgs = " ".join(_shlex.quote(p) for p in spec.install_torch.split())
        bootstrap_step(
            node, "install_torch",
            f"{uv_pfx}{activate} && uv pip install {torch_pkgs} {flags}",
            timeout=project_step_timeout,
        )

    # 4. Install remaining deps
    if spec.install:
        log_step(f"{name}: installing {len(spec.install)} packages")
        for i, pkg in enumerate(spec.install):
            safe_label = pkg.split("[")[0].split(">")[0].split("=")[0].replace("-", "_")
            bootstrap_step(
                node, f"install_{safe_label}",
                f'{uv_pfx}{activate} && uv pip install "{pkg}"',
                timeout=project_step_timeout,
            )

    # 5. Materialize shared launcher bundles into the external workspace.
    if spec.launcher:
        from crucible.core.config import ProjectConfig, load_config
        from crucible.core.hub import HubStore
        from crucible.fleet.project_launchers import launcher_runtime_dir, resolve_launcher_bundle

        launcher_root = project_root or Path.cwd()
        cfg_path = launcher_root / "crucible.yaml"
        cfg = load_config(cfg_path) if cfg_path.exists() else ProjectConfig(project_root=launcher_root)
        hub_dir = HubStore.resolve_hub_dir(config_hub_dir=getattr(cfg, "hub_dir", ""))
        launcher = resolve_launcher_bundle(
            project_root=launcher_root,
            launcher_name=spec.launcher,
            hub_dir=hub_dir,
        )
        if launcher is None:
            raise FileNotFoundError(
                f"Launcher bundle {spec.launcher!r} not found in local plugins, hub installs, or taps."
            )

        remote_root = launcher_runtime_dir(spec.workspace, spec.launcher)
        bootstrap_step(node, "launcher_dir", f"mkdir -p {_shlex.quote(remote_root)}")
        user = node.get("user", "root")
        remote_dest = f"{user}@{node['ssh_host']}:{remote_root}/"
        log_step(f"{name}: syncing launcher {spec.launcher!r} from {launcher['source']}")
        _run(
            rsync_base(node) + [f"{str(launcher['path'])}/", remote_dest],
            check=True,
        )

    # 6. Copy local files to workspace
    if spec.local_files:
        from crucible.fleet.sync import scp_to_node
        log_step(f"{name}: copying {len(spec.local_files)} local files")
        for local_path in spec.local_files:
            p = Path(local_path).expanduser()
            if not p.exists():
                log_warn(f"{name}: local file not found: {p}")
                continue
            remote_path = f"{spec.workspace}/{p.name}"
            scp_to_node(node, str(p), remote_path)

    # 7. Forward env vars
    log_step(f"{name}: forwarding env vars")
    write_remote_env(
        node,
        env_forward=spec.env_forward,
        env_set=spec.env_set,
        workspace=spec.workspace,
    )

    # 8. Run setup commands
    for i, cmd in enumerate(spec.setup):
        log_step(f"{name}: setup command {i + 1}/{len(spec.setup)}")
        bootstrap_step(
            node, f"setup_{i}",
            f"{uv_pfx}{activate} && source {ws}/.env 2>/dev/null; {cmd}",
            timeout=project_step_timeout,
        )

    # 9. Mark ready
    node["env_ready"] = True
    node["state"] = "ready"
    node["project"] = spec.name
    node["last_seen_at"] = utc_now_iso()
    log_success(f"{name}: project {spec.name!r} bootstrap complete")
    return node


# ---------------------------------------------------------------------------
# Bootstrap with retries (worker)
# ---------------------------------------------------------------------------

def bootstrap_node_worker(
    node: NodeRecord,
    *,
    nodes_file: Path,
    project_root: Path,
    sync_excludes: list[str],
    train_shards: int,
    data_download_cmd: str | None = None,
    data_source_name: str | None = None,
    data_source_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bootstrap a node with automatic retries."""
    last_exc: BaseException | None = None
    for attempt in range(1, BOOTSTRAP_ATTEMPTS + 1):
        try:
            updated = bootstrap_node(
                node,
                project_root=project_root,
                sync_excludes=sync_excludes,
                train_shards=train_shards,
                skip_install=False,
                skip_data=False,
                data_download_cmd=data_download_cmd,
                data_source_name=data_source_name,
                data_source_config=data_source_config,
            )
            upsert_node_record(nodes_file, updated)
            return updated
        except BaseException as exc:
            last_exc = exc
            if attempt >= BOOTSTRAP_ATTEMPTS:
                raise
            log_warn(
                f"{node['name']}: bootstrap attempt {attempt}/{BOOTSTRAP_ATTEMPTS} "
                f"failed, retrying: {exc}"
            )
            time.sleep(min(5 * attempt, 15))
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Bootstrap supervisor (threaded)
# ---------------------------------------------------------------------------

def start_bootstrap_supervisor(
    *,
    day_dir: Path,
    nodes: list[NodeRecord],
    nodes_file: Path,
    project_root: Path,
    sync_excludes: list[str],
    train_shards: int,
    target_ready_count: int,
    min_ready_to_start: int,
    bootstrap_workers: int,
    replacement_budget: int,
    replace_fn: Any = None,
    refresh_fn: Any = None,
    data_download_cmd: str | None = None,
    data_source_name: str | None = None,
    data_source_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Launch a background thread that bootstraps all nodes in parallel.

    Returns a state dict with threading events and status counters.

    Parameters
    ----------
    replace_fn : callable(failed_name) -> node_dict, optional
        Called when a node needs to be replaced.  Should provision a new
        node and return its record.
    refresh_fn : callable(nodes) -> nodes, optional
        Called to refresh node records from the provider API.
    """
    state: dict[str, Any] = {
        "min_ready_event": threading.Event(),
        "done_event": threading.Event(),
        "blocking_error": None,
        "degraded_error": None,
        "replacements_used": 0,
        "min_ready_to_start": min_ready_to_start,
        "ready_count": 0,
        "wave1_started": False,
    }

    def _bootstrap_ready_count() -> int:
        return count_bootstrapped_ready(load_nodes_snapshot(nodes_file))

    def supervisor() -> None:
        future_to_node: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
        pending_since: dict[str, float] = {}
        running_names: set[str] = set()
        completed_names: set[str] = set()
        abandoned_names: set[str] = set()

        def refresh_ready_state(current_nodes: list[NodeRecord]) -> int:
            ready_names = {n["name"] for n in current_nodes if ready_state(n) == "ready"}
            completed_names.update(ready_names)
            ready_bootstrapped = len(ready_names)
            state["ready_count"] = ready_bootstrapped
            if ready_bootstrapped >= min_ready_to_start:
                state["min_ready_event"].set()
            return ready_bootstrapped

        def mark_bootstrap_abandoned(name: str, reason: str) -> None:
            abandoned_names.add(name)
            running_names.discard(name)
            pending_since.pop(name, None)
            nodes_now = load_nodes_snapshot(nodes_file)
            for n in nodes_now:
                if n.get("name") == name:
                    n["state"] = "boot_failed"
                    upsert_node_record(nodes_file, n)
                    break
            append_event(day_dir, "node_bootstrap_abandoned", node=name, reason=reason)

        def note_bootstrap_problem(message: str) -> bool:
            current_ready = _bootstrap_ready_count()
            state["ready_count"] = current_ready
            if current_ready >= min_ready_to_start or state["wave1_started"]:
                state["degraded_error"] = RuntimeError(message)
                log_warn(message)
                return False
            state["blocking_error"] = RuntimeError(message)
            return True

        try:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, bootstrap_workers))
            with pool as executor:
                while True:
                    seed_nodes = load_nodes_snapshot(nodes_file) or nodes
                    if refresh_fn is not None:
                        refreshed_nodes = refresh_fn(seed_nodes)
                    else:
                        refreshed_nodes = seed_nodes
                    with NODES_LOCK:
                        latest_nodes = load_nodes_if_exists(nodes_file)
                        current_nodes = merge_node_snapshots(
                            latest_nodes or seed_nodes, refreshed_nodes,
                        )
                        save_nodes(nodes_file, current_nodes)
                    ready_bootstrapped = refresh_ready_state(current_nodes)
                    if ready_bootstrapped >= target_ready_count and not future_to_node:
                        return

                    now = time.time()
                    for n in current_nodes:
                        name = n["name"]
                        if name in completed_names or name in running_names or name in abandoned_names:
                            pending_since.pop(name, None)
                            continue
                        pending_since.setdefault(name, now)
                        if n.get("ssh_host") and ssh_ok(n):
                            running_names.add(name)
                            future = executor.submit(
                                bootstrap_node_worker,
                                n,
                                nodes_file=nodes_file,
                                project_root=project_root,
                                sync_excludes=sync_excludes,
                                train_shards=train_shards,
                                data_download_cmd=data_download_cmd,
                                data_source_name=data_source_name,
                                data_source_config=data_source_config,
                            )
                            future_to_node[future] = n
                            append_event(day_dir, "node_bootstrap_started", node=name)
                            continue
                        if now - pending_since[name] < 120:
                            continue
                        # Node not reachable after 120s -- attempt replacement
                        if state["replacements_used"] >= replacement_budget:
                            if note_bootstrap_problem(
                                f"Bootstrap replacement budget exhausted while waiting for {name}"
                            ):
                                return
                            mark_bootstrap_abandoned(name, "replacement_budget_exhausted")
                            continue
                        if replace_fn is not None:
                            try:
                                replacement = replace_fn(name)
                            except BaseException as exc:
                                if note_bootstrap_problem(
                                    f"Bootstrap replacement failed for {name}: {exc}"
                                ):
                                    return
                                mark_bootstrap_abandoned(name, f"replacement_failed:{exc}")
                                continue
                            state["replacements_used"] += 1
                            append_event(
                                day_dir,
                                "replacement_bootstrapped",
                                node=replacement["name"],
                                replaced=name,
                            )
                            pending_since.pop(name, None)
                            pending_since[replacement["name"]] = time.time()

                    done: set[concurrent.futures.Future[dict[str, Any]]] = set()
                    if future_to_node:
                        done, _ = concurrent.futures.wait(
                            list(future_to_node.keys()),
                            timeout=5,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                    else:
                        time.sleep(5)
                    for future in done:
                        n = future_to_node.pop(future)
                        running_names.discard(n["name"])
                        try:
                            result = future.result()
                            completed_names.add(result["name"])
                            state["ready_count"] = max(
                                state["ready_count"], _bootstrap_ready_count(),
                            )
                            append_event(day_dir, "node_bootstrapped", node=result["name"])
                            if state["ready_count"] >= min_ready_to_start:
                                state["min_ready_event"].set()
                            log_success(f"{result['name']}: bootstrap complete")
                        except BaseException as exc:
                            append_event(
                                day_dir, "node_bootstrap_failed",
                                node=n["name"], error=str(exc),
                            )
                            log_warn(f"{n['name']}: bootstrap failed: {exc}")
                            if state["replacements_used"] >= replacement_budget:
                                if note_bootstrap_problem(
                                    f"Bootstrap replacement budget exhausted after "
                                    f"{n['name']}: {exc}"
                                ):
                                    return
                                mark_bootstrap_abandoned(
                                    n["name"],
                                    f"replacement_budget_exhausted:{exc}",
                                )
                                continue
                            if replace_fn is not None:
                                try:
                                    replacement = replace_fn(n["name"])
                                except BaseException as replacement_exc:
                                    if note_bootstrap_problem(
                                        f"Bootstrap replacement failed for "
                                        f"{n['name']}: {replacement_exc}"
                                    ):
                                        return
                                    mark_bootstrap_abandoned(
                                        n["name"],
                                        f"replacement_failed:{replacement_exc}",
                                    )
                                    continue
                                state["replacements_used"] += 1
                                append_event(
                                    day_dir,
                                    "replacement_bootstrapped",
                                    node=replacement["name"],
                                    replaced=n["name"],
                                )
                                pending_since[replacement["name"]] = time.time()

                    current_nodes = load_nodes_snapshot(nodes_file) or current_nodes
                    ready_bootstrapped = refresh_ready_state(current_nodes)
                    actionable_pending = [
                        n for n in current_nodes
                        if n["name"] not in completed_names
                        and n["name"] not in running_names
                        and n["name"] not in abandoned_names
                    ]
                    if not future_to_node and not actionable_pending:
                        if ready_bootstrapped >= min_ready_to_start or state["wave1_started"]:
                            return
                        if note_bootstrap_problem(
                            "Bootstrap ended before minimum ready capacity was reached"
                        ):
                            return
        except BaseException as exc:
            if note_bootstrap_problem(f"Bootstrap supervisor failed: {exc}"):
                state["blocking_error"] = RuntimeError(
                    f"Bootstrap supervisor failed: {exc}"
                )
            else:
                state["degraded_error"] = RuntimeError(
                    f"Bootstrap supervisor failed: {exc}"
                )
        finally:
            state["ready_count"] = _bootstrap_ready_count()
            if state["ready_count"] >= min_ready_to_start:
                state["min_ready_event"].set()
            state["done_event"].set()

    thread = threading.Thread(target=supervisor, name="bootstrap-supervisor", daemon=True)
    thread.start()
    state["thread"] = thread
    return state
