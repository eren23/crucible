"""MCP server exposing Crucible fleet operations as tools for Claude agents.

Run via stdio:
    crucible mcp serve
    python -m crucible.mcp.server
"""
from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from crucible.core.env import load_env_files
from crucible.core.types import JsonDict, JsonValue
from crucible.mcp.tools import TOOL_DISPATCH

if TYPE_CHECKING:
    from collections.abc import Callable
    from crucible.mcp.tracer import SessionTracer

# Load .env files so secrets (RUNPOD_API_KEY, WANDB_API_KEY) are available to tools
load_env_files(Path(__file__).resolve().parent.parent.parent.parent)

app = Server("crucible-fleet")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log = logging.getLogger("crucible.mcp")
_CRUCIBLE_DEBUG = os.environ.get("CRUCIBLE_DEBUG", "0") == "1"


def _setup_logging() -> None:
    """Configure structured logging based on CRUCIBLE_DEBUG."""
    if _CRUCIBLE_DEBUG:
        level = logging.DEBUG
    else:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


_setup_logging()


# ---------------------------------------------------------------------------
# Session tracer (enabled via --trace flag or CRUCIBLE_TRACE=1 env var)
# ---------------------------------------------------------------------------

_tracer: SessionTracer | None = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_text(obj: JsonValue) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(obj, indent=2, default=str))]


def _error_text(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}))]


def _format_error(exc: BaseException, tb: str | None = None) -> JsonDict:
    """Build a structured error response with full context.

    Returns a dict that will be JSON-serialized as the tool response.
    """
    exc_type = type(exc).__name__
    exc_module = type(exc).__module__
    tb_str = tb or traceback.format_exc()

    return {
        "error": str(exc),
        "error_type": exc_type,
        "error_module": exc_module,
        "traceback": tb_str,
    }


def _safe_call(fn: Callable[..., JsonValue], *args: Any, **kwargs: Any) -> tuple[list[TextContent], bool, BaseException | None]:
    """Call a tool handler and return (result, is_error, exception).

    Never raises for normal tool failures (Exception). KeyboardInterrupt, SystemExit,
    and asyncio.CancelledError propagate so the host can shut down or cancel cleanly.
    """
    try:
        result = fn(*args, **kwargs)
        return (_json_text(result), False, None)
    except Exception as exc:
        tb = traceback.format_exc()
        # Log at error level — appears in stderr for debugging without CRUCIBLE_DEBUG
        _log.error("Tool %s raised %s: %s\n%s", fn.__name__, type(exc).__name__, exc, tb)
        return (_json_text(_format_error(exc, tb)), True, exc)


def _extract_trace_identifiers(name: str, arguments: dict[str, Any], raw: JsonValue) -> dict[str, list[str]]:
    """Extract stable run/design identifiers for traceability across MCP calls."""
    identifiers: dict[str, list[str]] = {}
    if not isinstance(arguments, dict):
        arguments = {}
    raw_dict = raw if isinstance(raw, dict) else {}

    project_name = raw_dict.get("project") or raw_dict.get("project_name") or arguments.get("project_name")
    if project_name:
        identifiers["project_names"] = [str(project_name)]

    design_name = raw_dict.get("design_name") or arguments.get("design_name")
    if design_name:
        identifiers["design_names"] = [str(design_name)]

    wave_name = raw_dict.get("wave_name") or arguments.get("wave_name")
    if wave_name:
        identifiers["wave_names"] = [str(wave_name)]

    launch_id = raw_dict.get("launch_id") or arguments.get("launch_id")
    if launch_id:
        identifiers["launch_ids"] = [str(launch_id)]

    run_id = raw_dict.get("run_id") or raw_dict.get("id")
    if run_id:
        identifiers["run_ids"] = [str(run_id)]
    elif isinstance(raw_dict.get("runs"), list):
        run_ids = [
            str(row.get("run_id"))
            for row in raw_dict["runs"]
            if isinstance(row, dict) and row.get("run_id")
        ]
        if run_ids:
            identifiers["run_ids"] = sorted(set(run_ids))
    elif isinstance(raw_dict.get("nodes"), list):
        run_ids = [
            str(node.get("run_id"))
            for node in raw_dict["nodes"]
            if isinstance(node, dict) and node.get("run_id")
        ]
        if run_ids:
            identifiers["run_ids"] = sorted(set(run_ids))

    overrides = arguments.get("overrides", {}) if isinstance(arguments.get("overrides"), dict) else {}
    variant_name = (
        raw_dict.get("variant_name")
        or raw_dict.get("name")
        or overrides.get("CRUCIBLE_VARIANT_NAME")
    )
    if name == "run_project" and variant_name:
        identifiers["variant_names"] = [str(variant_name)]

    launcher_name = raw_dict.get("launcher") or raw_dict.get("launcher_name")
    if launcher_name:
        identifiers["launchers"] = [str(launcher_name)]

    node_names: list[str] = []
    if isinstance(arguments.get("node_names"), list):
        node_names.extend(str(n) for n in arguments["node_names"] if n)
    if isinstance(raw_dict.get("nodes"), list):
        node_names.extend(
            str(node.get("name"))
            for node in raw_dict["nodes"]
            if isinstance(node, dict) and node.get("name")
        )
    if node_names:
        identifiers["node_names"] = sorted(set(node_names))

    return identifiers


# ---------------------------------------------------------------------------
# Keepalive for long-running tool calls
# ---------------------------------------------------------------------------

# Tools known to be long-running (may take minutes).
# For these, the keepalive fires more aggressively.
_LONG_RUNNING_TOOLS: set[str] = {
    "bootstrap_nodes",
    "bootstrap_project",
    "collect_results",
    "collect_project_results",
    "data_prepare",
    "destroy_nodes",
    "dispatch_experiments",
    "fleet_refresh",
    "provision_nodes",
    "provision_project",
    "run_project",
    "run_project_chain",
    "start_nodes",
    "stop_nodes",
    "sync_code",
    "tree_enqueue_pending",
}

_KEEPALIVE_INTERVAL = 8.0  # seconds between keepalive pings


async def _run_with_keepalive(
    handler: Callable[..., Any],
    arguments: dict[str, Any],
    tool_name: str,
    session: object,  # mcp.server.Session — typed as object to avoid coupling to MCP internals
    request_id: object,  # MCP request identifier — opaque to Crucible
) -> tuple[list[TextContent], bool, BaseException | None]:  # type: ignore[type-arg]
    """Run a tool handler in a thread while sending periodic log messages.

    Long-running MCP tool calls can cause the stdio client to time out
    if no data flows on the pipe.  We send periodic ``notifications/message``
    (log-level "info") to keep the connection alive.

    Returns (result, is_error, exception) to match _safe_call signature.
    """
    done = asyncio.Event()
    result_box: list[tuple[list[TextContent], bool, BaseException | None] | None] = [None]
    exc_box: list[BaseException | None] = [None]

    async def _worker() -> None:
        try:
            # _safe_call returns a 3-tuple for tool failures; it only raises for
            # BaseException subclasses we intentionally do not catch (e.g. cancel).
            result_box[0] = await asyncio.to_thread(_safe_call, handler, arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # e.g. thread pool / serialization failure — return JSON instead of killing stdio
            tb = traceback.format_exc()
            _log.critical("Worker raised in %s: %s\n%s", tool_name, exc, tb)
            exc_box[0] = exc
        finally:
            done.set()

    async def _keepalive() -> None:
        elapsed = 0.0
        while not done.is_set():
            try:
                await asyncio.wait_for(done.wait(), timeout=_KEEPALIVE_INTERVAL)
                # done was set — exit cleanly
                return
            except asyncio.TimeoutError:
                pass
            elapsed += _KEEPALIVE_INTERVAL
            try:
                await session.send_log_message(
                    level="info",
                    data=f"[{tool_name}] still running ({elapsed:.0f}s elapsed)...",
                    logger="crucible.mcp.keepalive",
                    related_request_id=request_id,
                )
            except Exception as exc:
                # Log but never let keepalive errors break the tool
                _log.warning("Keepalive send failed for %s: %s", tool_name, exc)

    await asyncio.gather(_worker(), _keepalive())

    if exc_box[0] is not None:
        exc = exc_box[0]
        tb_str = "".join(traceback.format_exception(exc))
        return (_json_text(_format_error(exc, tb_str)), True, exc)

    # result_box[0] is the raw output from _safe_call: (list[TextContent], is_error, exc)
    raw_result = result_box[0]
    if raw_result is None:
        return (_error_text("Tool returned no result"), False, None)
    if isinstance(raw_result, tuple) and len(raw_result) == 3:
        return raw_result
    # Fallback: treat as raw TextContent list (should not happen)
    return (raw_result, False, None)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Tool catalogue
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="get_fleet_status",
        description=(
            "Node inventory, health summary, current assignments, and optional live GPU/memory/disk metrics.\n\n"
            "REQUIRES: Nothing (reads local inventory). include_metrics=true requires SSH access to nodes (slower, 5-15s).\n"
            "RETURNS: {summary, nodes: [{name, state, gpu, ssh_host, env_ready, dataset_ready}], metrics?: [...]}\n"
            "NEXT: fleet_refresh to update stale nodes, bootstrap_nodes for unready, dispatch_experiments for idle."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "include_metrics": {
                    "type": "boolean",
                    "description": "If true, SSH to each node to collect live GPU util, memory, disk metrics. Slower (5-15s).",
                    "default": False,
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_leaderboard",
        description=(
            "Top N experiment results sorted by primary metric (lower is better).\n\n"
            "REQUIRES: At least one completed experiment. Run collect_results first if experiments ran on fleet.\n"
            "RETURNS: {total_completed, primary_metric, top: [{rank, name, val_bpb, steps_completed, model_bytes}]}\n"
            "NEXT: get_experiment_result for details, design_compare_experiments for diffs, context_push_finding for insights."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "Number of results to return.", "default": 20},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_queue_status",
        description=(
            "Fleet queue state: counts of queued, running, and completed experiments.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {total, summary: {queue_total, queue_running, queue_queued, queue_finished}}\n"
            "NEXT: dispatch_experiments if queued>0 and nodes idle, collect_results if finished>0."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="enqueue_experiment",
        description=(
            "Add an experiment configuration to the fleet queue.\n\n"
            "REQUIRES: wandb.project set in crucible.yaml OR WANDB_PROJECT in env_set; "
            "WANDB_API_KEY exported (in .env*) when wandb.mode!=disabled. "
            "Contract validation rejects this call (ConfigError) when missing. "
            "See get_wandb_guide if you have not configured W&B for this project. "
            "Skips if same name+tier already queued.\n"
            "RETURNS: {status: 'enqueued'|'skipped', run_id}\n"
            "NEXT: dispatch_experiments to assign to nodes. Set CRUCIBLE_VARIANT_NAME (or WANDB_RUN_NAME) "
            "in config to give the W&B run a distinguishable name -- the default exp_id collides across variants."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Short experiment name (no spaces)."},
                "config": {
                    "type": "object",
                    "description": "Environment variable overrides.",
                    "additionalProperties": {"type": "string"},
                },
                "tier": {"type": "string", "description": "Experiment tier.", "default": "proxy"},
                "backend": {"type": "string", "description": "Training backend.", "default": "torch"},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags.", "default": []},
            },
            "required": ["name", "config"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_experiment_result",
        description=(
            "Get the result for a specific experiment run_id.\n\n"
            "REQUIRES: A run_id from enqueue_experiment or get_queue_status. Run collect_results for fleet experiments.\n"
            "RETURNS: {found: bool, result: {name, config, result: {val_loss, val_bpb, steps_completed}, status}}\n"
            "NEXT: design_compare_experiments for side-by-side, note_add for observations, get_run_logs for training output."
        ),
        inputSchema={
            "type": "object",
            "properties": {"run_id": {"type": "string", "description": "The unique run identifier."}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="provision_nodes",
        description=(
            "Create N new compute nodes via the configured provider (RunPod/SSH).\n\n"
            "REQUIRES: RUNPOD_API_KEY in .env (for RunPod provider).\n"
            "RETURNS: {created, new_nodes: [{name, node_id}]}\n"
            "NEXT: fleet_refresh (wait ~60s for SSH), then bootstrap_nodes. "
            "If you have not configured W&B for this project yet, call get_wandb_guide before enqueueing -- "
            "experiment runs require WANDB_API_KEY + WANDB_PROJECT or they will fail at startup."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of nodes to create.", "default": 2},
                "name_prefix": {"type": "string", "description": "Node name prefix.", "default": "crucible"},
                "interruptible": {"type": "boolean", "description": "Use spot/interruptible instances (cheaper, can be preempted). Default false.", "default": False},
                "network_volume_id": {"type": "string", "description": "RunPod network volume ID for shared storage."},
                "template_id": {"type": "string", "description": "RunPod template ID for standardized provisioning."},
            },
            "required": ["count"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="destroy_nodes",
        description=(
            "Tear down nodes. Supports names, pod IDs, or destroy-all.\n\n"
            "With no args: destroys ALL pods (inventory + orphans via RunPod API).\n"
            "With node_names: destroys matching nodes by name (inventory + orphan name match).\n"
            "With pod_ids: destroys specific pods by RunPod pod ID (direct API, no inventory needed).\n\n"
            "REQUIRES: RUNPOD_API_KEY for orphan/pod_id cleanup.\n"
            "RETURNS: {destroyed, status, orphan_pods_destroyed?}\n"
            "NEXT: provision_nodes to create new ones."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node names to destroy. If empty, destroys all.",
                    "default": [],
                },
                "pod_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "RunPod pod IDs to destroy directly (bypasses inventory).",
                    "default": [],
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="cleanup_orphans",
        description=(
            "List and optionally destroy pods on the provider that aren't in local inventory.\n\n"
            "An 'orphan' is any pod that exists on the provider side (e.g. RunPod) but has no\n"
            "entry in nodes.json — typically caused by a partially-failed provision batch, a\n"
            "crash during fleet operations, or pods created by another client.\n\n"
            "REQUIRES: Provider supports pod listing (RunPod ✓, SSH ✗). RUNPOD_API_KEY set.\n"
            "RETURNS: {orphans: [{name, pod_id}], destroyed: [pod_id, ...], total_orphans, status}\n"
            "NEXT: Review orphans with destroy=false first, then re-run with destroy=true if\n"
            "you want them gone. Alternatively, call fleet_refresh to 'adopt' them into\n"
            "inventory as reconciled_orphan nodes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "destroy": {
                    "type": "boolean",
                    "description": "If true, destroy the orphans via the provider API. Default: false (list only).",
                    "default": False,
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="stop_nodes",
        description=(
            "Stop running pods to save cost. Disk and bootstrap state are preserved.\n\n"
            "REQUIRES: Nodes in running/ready state.\n"
            "RETURNS: {stopped: [node_names], status}\n"
            "NEXT: start_nodes to resume. No re-bootstrapping needed."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node names to stop. If empty, stops all.",
                    "default": [],
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="start_nodes",
        description=(
            "Start stopped pods and wait for SSH readiness.\n\n"
            "REQUIRES: Nodes in 'stopped' state.\n"
            "RETURNS: {started: [node_names], status}\n"
            "NEXT: dispatch_experiments (bootstrap state is preserved from before stop)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node names to start. If empty, starts all stopped.",
                    "default": [],
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="runpod_list_volumes",
        description=(
            "List RunPod network volumes (persistent shared storage).\n\n"
            "REQUIRES: RUNPOD_API_KEY.\n"
            "RETURNS: {volumes: [{id, name, size, dataCenterId}], count}\n"
            "NEXT: provision_nodes with network_volume_id to attach."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="runpod_create_volume",
        description=(
            "Create a persistent RunPod network volume for shared data across pods.\n\n"
            "REQUIRES: RUNPOD_API_KEY.\n"
            "RETURNS: {volume: {id, name, size, dataCenterId}, status}\n"
            "NEXT: provision_nodes with network_volume_id=<id>."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Volume name."},
                "size_gb": {"type": "integer", "description": "Volume size in GB.", "default": 100},
                "datacenter_id": {"type": "string", "description": "Datacenter ID (e.g. US-GA-1).", "default": "US-GA-1"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="runpod_delete_volume",
        description=(
            "Delete a RunPod network volume.\n\n"
            "REQUIRES: RUNPOD_API_KEY. No pods attached to the volume.\n"
            "RETURNS: {deleted: volume_id, status}\n"
            "NEXT: runpod_list_volumes to confirm deletion."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "volume_id": {"type": "string", "description": "Volume ID to delete."},
            },
            "required": ["volume_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="runpod_gpu_availability",
        description=(
            "List available GPU types with spot and on-demand pricing.\n\n"
            "REQUIRES: RUNPOD_API_KEY.\n"
            "RETURNS: {gpu_types: [{gpuName, gpuTypeId, minimumBidPrice, uninterruptablePrice, minMemory, minVcpu}]}\n"
            "NEXT: provision_nodes with specific gpu_type_ids based on availability."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "gpu_count": {"type": "integer", "description": "GPUs per node.", "default": 1},
                "secure_cloud": {"type": "boolean", "description": "Filter to secure cloud only."},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="runpod_list_templates",
        description=(
            "List user's RunPod pod templates.\n\n"
            "REQUIRES: RUNPOD_API_KEY.\n"
            "RETURNS: {templates: [{id, name, imageName, ports}], count}\n"
            "NEXT: provision_nodes with template_id=<id> to create pods from a template."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="runpod_create_template",
        description=(
            "Create a reusable RunPod pod template.\n\n"
            "REQUIRES: RUNPOD_API_KEY.\n"
            "RETURNS: {template: {id, name}, status}\n"
            "NEXT: provision_nodes with template_id."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Template name."},
                "image": {"type": "string", "description": "Docker image name."},
                "container_disk_gb": {"type": "integer", "description": "Container disk size.", "default": 20},
                "volume_gb": {"type": "integer", "description": "Volume size.", "default": 40},
                "ports": {"type": "string", "description": "Ports spec.", "default": "22/tcp,8888/http"},
                "env": {
                    "type": "object",
                    "description": "Environment variables.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["name", "image"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="sync_code",
        description=(
            "Push local code to nodes via rsync. Long-running (1-3 min per node).\n\n"
            "REQUIRES: Nodes with SSH access (run fleet_refresh first).\n"
            "RETURNS: {synced: [node_names], errors: [...]}\n"
            "NEXT: Use after modifying code locally. bootstrap_nodes does this automatically."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "node_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Node names to sync. If empty, syncs to all.",
                    "default": [],
                },
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="fleet_refresh",
        description=(
            "Refresh node states from cloud provider API. Updates SSH hosts, GPU info, and node state.\n\n"
            "REQUIRES: Nodes exist (from provision_nodes).\n"
            "RETURNS: {refreshed, nodes: [{name, state, ssh_host, ssh_port, gpu, env_ready, dataset_ready}]}\n"
            "NEXT: bootstrap_nodes when nodes show state=running with ssh_host."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="bootstrap_nodes",
        description=(
            "Bootstrap fleet nodes: sync code, install deps, download data. Long-running (2-10 min).\n\n"
            "REQUIRES: Nodes with SSH hosts (run fleet_refresh first after provision_nodes). "
            "The env file pointed at by provider.defaults.env_source (default: .env.runpod.local) "
            "must contain WANDB_API_KEY when wandb.required=true; preflight on the pod exits with code 101 if it is missing.\n"
            "RETURNS: {total, bootstrapped, nodes: [{name, state, env_ready, dataset_ready}]}\n"
            "NEXT: enqueue_experiment or design_enqueue_batch, then dispatch_experiments. "
            "Call get_wandb_guide if uncertain about W&B configuration."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "train_shards": {"type": "integer", "description": "Number of data shards to download per node.", "default": 1},
                "skip_install": {"type": "boolean", "description": "Skip pip install step.", "default": False},
                "skip_data": {"type": "boolean", "description": "Skip data download step.", "default": False},
                "node_names": {"type": "array", "items": {"type": "string"}, "description": "Specific nodes to bootstrap. Empty = all."},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="dispatch_experiments",
        description=(
            "Dispatch queued experiments to idle bootstrapped nodes. One experiment per node. Long-running.\n\n"
            "REQUIRES: Bootstrapped nodes (env_ready=true) + queued experiments. "
            "Each pod must have WANDB_API_KEY in its sourced .env file when wandb.required=true; "
            "the runner raises RunnerError at startup if W&B init fails. See get_wandb_guide.\n"
            "RETURNS: {dispatched, assignments: [{node, experiment}]}\n"
            "NEXT: get_queue_status to monitor, collect_results when done. "
            "After collect_results, call wandb_get_url(run_id) to verify each run actually registered."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "max_assignments": {"type": "integer", "description": "Max experiments to dispatch.", "default": 8},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="collect_results",
        description=(
            "Collect experiment results from all fleet nodes via rsync and merge. Long-running (1-5 min).\n\n"
            "REQUIRES: Experiments dispatched and running/completed on nodes.\n"
            "RETURNS: {collected, total_results, completed}\n"
            "NEXT: get_leaderboard for rankings, get_experiment_result for details, get_run_logs for output."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="get_research_state",
        description=(
            "Current research state: hypotheses, beliefs, budget.\n\n"
            "REQUIRES: Nothing (returns available=false if no state file).\n"
            "RETURNS: {available, hypotheses_count, history_count, beliefs, budget_remaining}\n"
            "NEXT: design_generate_hypotheses to create new hypotheses, get_research_briefing for full orientation."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="get_sensitivity",
        description=(
            "Parameter sensitivity analysis across completed experiments.\n\n"
            "REQUIRES: Multiple completed experiments with varying configs.\n"
            "RETURNS: {parameters: {param_name: {values, metric_range, correlation}}}\n"
            "NEXT: design_generate_hypotheses to act on sensitivity insights."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    # -----------------------------------------------------------------------
    # Design tools
    # -----------------------------------------------------------------------
    Tool(
        name="design_browse_experiments",
        description=(
            "Browse completed experiments with filtering by name, family, tag, metric range, and config values.\n\n"
            "REQUIRES: Completed experiments (run collect_results first for fleet runs).\n"
            "RETURNS: {total_matched, experiments: [{name, config, val_loss, model_bytes, tags, status}]}\n"
            "NEXT: design_compare_experiments for side-by-side, context_push_finding for insights."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name_pattern": {"type": "string", "description": "Substring match on experiment names.", "default": ""},
                "family": {"type": "string", "description": "Filter by MODEL_FAMILY config value.", "default": ""},
                "tag": {"type": "string", "description": "Filter to experiments containing this tag.", "default": ""},
                "metric_below": {"type": "number", "description": "Only show experiments where primary metric is below this."},
                "metric_above": {"type": "number", "description": "Only show experiments where primary metric is above this."},
                "config_filter": {"type": "object", "description": "Filter by exact config key=value.", "additionalProperties": {"type": "string"}, "default": {}},
                "limit": {"type": "integer", "description": "Max results.", "default": 50},
                "sort_by": {"type": "string", "description": "Sort: 'metric', 'name', 'timestamp'.", "default": "metric"},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="design_compare_experiments",
        description=(
            "Side-by-side comparison of 2-5 experiments: config diffs and metric differences.\n\n"
            "REQUIRES: 2-5 experiment names from browse or leaderboard.\n"
            "RETURNS: {experiments, config_diff: {key: {exp1: val, exp2: val}}, metrics: {exp: {val_loss, model_bytes}}}\n"
            "NEXT: context_push_finding to record insights, design_generate_hypotheses to explore further."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "experiment_names": {"type": "array", "items": {"type": "string"}, "description": "Experiment names to compare.", "minItems": 2, "maxItems": 5},
            },
            "required": ["experiment_names"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="design_generate_hypotheses",
        description=(
            "Generate LLM-driven hypotheses from current results via Claude (config.researcher.model).\n\n"
            "REQUIRES: ANTHROPIC_API_KEY set. Completed experiments improve quality.\n"
            "RETURNS: {hypotheses: [{name, hypothesis, config, expected_impact, confidence, family, rationale}], total_generated}\n"
            "NEXT: design_batch_from_hypotheses to convert to configs, then design_enqueue_batch."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "extra_context": {"type": "string", "description": "Additional context for hypothesis generation.", "default": ""},
                "max_hypotheses": {"type": "integer", "description": "Maximum hypotheses to return.", "default": 5},
                "focus_family": {"type": "string", "description": "Model family to focus on.", "default": ""},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="design_batch_from_hypotheses",
        description=(
            "Convert hypotheses to an executable experiment batch with optional baseline control.\n\n"
            "REQUIRES: Hypotheses list (from design_generate_hypotheses or manually crafted).\n"
            "RETURNS: {batch: [{name, config, tier, backend, tags}], batch_size}\n"
            "NEXT: design_enqueue_batch to add to queue."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "hypothesis": {"type": "string"},
                            "config": {"type": "object", "additionalProperties": {"type": "string"}},
                            "expected_impact": {"type": "number"},
                            "confidence": {"type": "number"},
                            "family": {"type": "string"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["name", "config"],
                    },
                    "description": "Hypotheses to convert.",
                },
                "tier": {"type": "string", "description": "Experiment tier.", "default": "proxy"},
                "backend": {"type": "string", "description": "Training backend.", "default": "torch"},
                "include_baseline": {"type": "boolean", "description": "Include baseline control.", "default": True},
            },
            "required": ["hypotheses"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="design_enqueue_batch",
        description=(
            "Enqueue a batch of experiment configs to the fleet queue.\n\n"
            "REQUIRES: Batch of experiments (from design_batch_from_hypotheses or manually built); "
            "wandb.project + WANDB_API_KEY configured (contract validates each enqueue). "
            "Per-experiment config should set CRUCIBLE_VARIANT_NAME (or WANDB_RUN_NAME) for distinguishable W&B run names. "
            "See get_wandb_guide.\n"
            "RETURNS: {enqueued, wave_name, run_ids: [...]}\n"
            "NEXT: dispatch_experiments to assign to nodes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "batch": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "config": {"type": "object", "additionalProperties": {"type": "string"}},
                            "tier": {"type": "string"},
                            "backend": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "priority": {"type": "integer"},
                            "wave": {"type": "string"},
                        },
                        "required": ["name", "config"],
                    },
                    "description": "Experiments to enqueue.",
                },
                "wave_name": {"type": "string", "description": "Wave name for grouping.", "default": ""},
            },
            "required": ["batch"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Context tools
    # -----------------------------------------------------------------------
    Tool(
        name="context_get_analysis",
        description=(
            "Full structured analysis: leaderboard, family breakdown, sensitivity, beliefs, and research state.\n\n"
            "REQUIRES: Completed experiments.\n"
            "RETURNS: {leaderboard, sensitivity, findings, research_state}\n"
            "NEXT: design_generate_hypotheses to act on insights, context_push_finding to record observations."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="context_push_finding",
        description=(
            "Record a research finding or observation. Persists across sessions and informs hypothesis generation.\n\n"
            "REQUIRES: Nothing. Categories: belief, observation, constraint, rejected_hypothesis.\n"
            "RETURNS: {status, finding_index}\n"
            "NEXT: finding_promote to elevate to hub scope, annotate_run to link to specific experiments."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "finding": {"type": "string", "description": "The finding or observation to record."},
                "category": {"type": "string", "description": "Category: belief, observation, constraint, rejected_hypothesis.", "default": "observation"},
                "source_experiments": {"type": "array", "items": {"type": "string"}, "description": "Experiment names supporting this finding.", "default": []},
                "confidence": {"type": "number", "description": "Confidence in this finding (0-1).", "default": 0.7},
            },
            "required": ["finding"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="context_get_findings",
        description=(
            "Query accumulated research findings, optionally filtered by category.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {findings: [{finding, category, confidence, source_experiments}]}\n"
            "NEXT: finding_promote to share across projects, design_generate_hypotheses to build on findings."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Filter by category. Empty returns all.", "default": ""},
                "limit": {"type": "integer", "description": "Max findings to return.", "default": 50},
            },
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Version tools
    # -----------------------------------------------------------------------
    Tool(
        name="version_save_design",
        description=(
            "Save or update a versioned experiment design. Creates a new version each time.\n\n"
            "REQUIRES: Nothing. Use for long-lived designs you'll iterate on.\n"
            "RETURNS: {design_name, version, status}\n"
            "NEXT: version_run_design to execute, version_diff to compare versions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Design name (slug-style, no spaces)."},
                "config": {"type": "object", "description": "Env var overrides.", "additionalProperties": {"type": "string"}},
                "description": {"type": "string", "description": "What this experiment tests.", "default": ""},
                "hypothesis": {"type": "string", "description": "Expected outcome.", "default": ""},
                "base_preset": {"type": "string", "description": "Preset to merge on.", "default": "proxy"},
                "backend": {"type": "string", "description": "Training backend.", "default": "torch"},
                "family": {"type": "string", "description": "Model family.", "default": ""},
                "tags": {"type": "array", "items": {"type": "string"}, "default": []},
                "rationale": {"type": "string", "description": "Why this design exists.", "default": ""},
                "status": {"type": "string", "description": "Design status: draft, ready, running, completed, archived.", "default": "draft"},
                "parent_design": {"type": "string", "description": "Name of design this was forked from."},
                "summary": {"type": "string", "description": "Version summary.", "default": ""},
            },
            "required": ["name", "config"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="version_list_designs",
        description=(
            "List all versioned experiment designs with metadata and status.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {designs: [{name, current_version, status, family, tags}]}\n"
            "NEXT: version_get_design for full details, version_run_design to execute."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "status_filter": {"type": "string", "description": "Filter by design status (draft, ready, etc.)."},
                "tag_filter": {"type": "string", "description": "Filter by tag."},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="version_diff",
        description=(
            "Compare two versions of an experiment design showing what changed.\n\n"
            "REQUIRES: Design with at least 2 versions.\n"
            "RETURNS: {resource_name, version_a, version_b, changes: {added, removed, modified}}\n"
            "NEXT: version_get_design for full content at a specific version."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "resource_name": {"type": "string", "description": "Design name."},
                "version_a": {"type": "integer", "description": "First version number."},
                "version_b": {"type": "integer", "description": "Second version number."},
            },
            "required": ["resource_name", "version_a", "version_b"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="version_get_design",
        description=(
            "Get full content and metadata for a versioned experiment design.\n\n"
            "REQUIRES: Design name from version_list_designs.\n"
            "RETURNS: {name, version, config, hypothesis, status, linked_run_ids, ...}\n"
            "NEXT: version_run_design to execute, version_link_result after experiments complete."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "design_name": {"type": "string", "description": "Design name."},
                "version": {"type": "integer", "description": "Specific version number. Omit for latest."},
            },
            "required": ["design_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="version_run_design",
        description=(
            "Execute a versioned design: enqueues to fleet queue and updates status to running. Does NOT dispatch.\n\n"
            "REQUIRES: Design name from version_list_designs; wandb.project + WANDB_API_KEY configured "
            "(contract validates at enqueue time). See get_wandb_guide.\n"
            "RETURNS: {run_ids: [...], wave_name}\n"
            "NEXT: dispatch_experiments to assign to nodes, then collect_results, then version_link_result."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "design_name": {"type": "string", "description": "Design name to execute."},
                "tier": {"type": "string", "description": "Override the design's base_preset tier."},
                "backend": {"type": "string", "description": "Override the design's backend."},
            },
            "required": ["design_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="version_link_result",
        description=(
            "Link a completed experiment run_id back to a versioned design.\n\n"
            "REQUIRES: Design name + run_id from completed experiment.\n"
            "RETURNS: {status, design_name, run_id}\n"
            "NEXT: get_experiment_result to verify, version_get_design to see linked runs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "design_name": {"type": "string", "description": "Design name."},
                "run_id": {"type": "string", "description": "The experiment run_id to link."},
            },
            "required": ["design_name", "run_id"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Note tools
    # -----------------------------------------------------------------------
    Tool(
        name="note_add",
        description=(
            "Attach a markdown note to an experiment run (observations, analysis, hypotheses).\n\n"
            "REQUIRES: A valid run_id. Stages: pre-run, mid-run, post-run, analysis.\n"
            "RETURNS: {status, note_id}\n"
            "NEXT: note_get to retrieve, note_search to find across runs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The experiment run_id to attach the note to."},
                "text": {"type": "string", "description": "Markdown note body."},
                "stage": {"type": "string", "description": "When the note was created: pre-run, mid-run, post-run, analysis.", "default": ""},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for categorization.", "default": []},
                "confidence": {"type": "number", "description": "Confidence in any claims made (0-1)."},
                "created_by": {"type": "string", "description": "Identity of note creator.", "default": "mcp-agent"},
            },
            "required": ["run_id", "text"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="note_get",
        description=(
            "Get all notes for a specific experiment run, optionally filtered by stage.\n\n"
            "REQUIRES: A valid run_id.\n"
            "RETURNS: {run_id, notes: [{text, stage, tags, created_by, timestamp}]}\n"
            "NEXT: note_search for cross-run queries."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The experiment run_id."},
                "stage": {"type": "string", "description": "Filter by stage (pre-run, mid-run, post-run, analysis).", "default": ""},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="note_search",
        description=(
            "Search notes across all runs by text query, tags, stage, or run_id.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {results: [{run_id, text, stage, tags, timestamp}]}\n"
            "NEXT: note_get for full notes on a specific run."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text search across note metadata.", "default": ""},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter to notes containing all these tags."},
                "stage": {"type": "string", "description": "Filter by stage.", "default": ""},
                "run_id": {"type": "string", "description": "Filter by run_id.", "default": ""},
                "limit": {"type": "integer", "description": "Max results.", "default": 50},
            },
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # W&B tools
    # -----------------------------------------------------------------------
    Tool(
        name="wandb_log_image",
        description=(
            "Upload an image file to a W&B run.\n\n"
            "REQUIRES: Run must have a W&B URL in its status sidecar. WANDB_API_KEY set.\n"
            "RETURNS: {status, run_id}\n"
            "NEXT: wandb_get_url to view the dashboard."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Crucible run ID."},
                "image_path": {"type": "string", "description": "Path to image file to upload."},
                "caption": {"type": "string", "description": "Image caption.", "default": ""},
                "key": {"type": "string", "description": "W&B log key for the image.", "default": "image"},
            },
            "required": ["run_id", "image_path"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="wandb_get_url",
        description=(
            "Get the W&B dashboard URL for a Crucible experiment run.\n\n"
            "REQUIRES: Run must have been executed with W&B logging enabled.\n"
            "RETURNS: {run_id, wandb_url: str|null, reason}\n"
            "NEXT: wandb_annotate to push notes to W&B."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Crucible run ID."},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="wandb_annotate",
        description=(
            "Push a note or finding annotation to a W&B run summary.\n\n"
            "REQUIRES: Run with W&B URL. WANDB_API_KEY set.\n"
            "RETURNS: {status, run_id}\n"
            "NEXT: wandb_get_url to view in dashboard."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Crucible run ID."},
                "text": {"type": "string", "description": "The note or finding text."},
                "annotation_type": {"type": "string", "description": "Type: 'note' or 'finding'.", "default": "note"},
            },
            "required": ["run_id", "text"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Hub tools
    # -----------------------------------------------------------------------
    Tool(
        name="hub_status",
        description=(
            "Hub info: initialization state, active track, linked projects, and track summaries.\n\n"
            "REQUIRES: Nothing (reports uninit if hub not set up).\n"
            "RETURNS: {initialized, active_track, tracks, linked_projects}\n"
            "NEXT: track_create to start a track, hub_sync to push/pull."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="hub_sync",
        description=(
            "Git-sync the hub: stage, commit, pull, and push to remote.\n\n"
            "REQUIRES: Hub initialized (crucible hub init). Optional remote configured.\n"
            "RETURNS: {status, synced}\n"
            "NEXT: hub_status to verify."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "remote": {"type": "string", "description": "Git remote name (default: origin)."},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="track_create",
        description=(
            "Create a new research track for grouping related experiments across projects.\n\n"
            "REQUIRES: Hub initialized.\n"
            "RETURNS: {status, track_name}\n"
            "NEXT: track_switch to activate, hub_sync to persist."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Track name (human-readable, will be slugified)."},
                "description": {"type": "string", "description": "What this track is about.", "default": ""},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags for the track.", "default": []},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="track_list",
        description=(
            "List all research tracks with their metadata and active status.\n\n"
            "REQUIRES: Hub initialized.\n"
            "RETURNS: {tracks: [{name, description, active, tags}]}\n"
            "NEXT: track_switch to change active track."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="track_switch",
        description=(
            "Switch the active research track. The active track is used as default context.\n\n"
            "REQUIRES: Track name from track_list.\n"
            "RETURNS: {status, active_track}\n"
            "NEXT: get_research_briefing for orientation on the new track."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Track name to activate."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_findings_query",
        description=(
            "Query findings across hub scopes (track or global) with optional filters.\n\n"
            "REQUIRES: Hub initialized. Track name if scope='track'.\n"
            "RETURNS: {findings: [{id, finding, category, confidence, status, tags}]}\n"
            "NEXT: finding_promote to elevate scope."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "scope": {"type": "string", "description": "Scope: 'track' or 'global'.", "default": "global"},
                "track": {"type": "string", "description": "Track name (required when scope='track')."},
                "status": {"type": "string", "description": "Filter by status: active, superseded, archived, promoted."},
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Filter to findings with any of these tags."},
                "limit": {"type": "integer", "description": "Max findings to return.", "default": 50},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="finding_promote",
        description=(
            "Promote a finding across scopes: project→track, track→global, or track→track.\n\n"
            "REQUIRES:\n"
            "- project→*: finding_index (0-indexed into ResearchState.findings, from context_get_findings)\n"
            "- track→*: finding_id (from hub_findings_query)\n"
            "- to_track when to_scope='track'; from_track when from_scope='track'\n"
            "Confidence gates: project→track ≥0.6, track→global ≥0.8.\n"
            "RETURNS: {status: 'promoted', finding: {...}}\n"
            "NEXT: hub_sync to persist, hub_findings_query to verify."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "finding_id": {"type": "string", "description": "Finding ID (required when from_scope='track')."},
                "finding_index": {"type": "integer", "description": "Index into project ResearchState.findings (required when from_scope='project'). Use context_get_findings to list."},
                "from_scope": {"type": "string", "enum": ["project", "track"], "description": "Source scope."},
                "to_scope": {"type": "string", "enum": ["track", "global"], "description": "Destination scope."},
                "from_track": {"type": "string", "description": "Source track name (required when from_scope='track')."},
                "to_track": {"type": "string", "description": "Destination track name (required when to_scope='track')."},
            },
            "required": ["from_scope", "to_scope"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Briefing tools
    # -----------------------------------------------------------------------
    Tool(
        name="get_research_briefing",
        description=(
            "Session orientation: project state, leaderboard, hypotheses, findings, notes, next steps.\n\n"
            "REQUIRES: Nothing (best with completed experiments).\n"
            "RETURNS: {project_state, leaderboard, hypotheses, findings, notes, suggested_next_steps}\n"
            "NEXT: Follow the suggested_next_steps, or design_generate_hypotheses for new directions."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "track": {"type": "string", "description": "Track to brief on. Empty uses active track.", "default": ""},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="annotate_run",
        description=(
            "Bidirectional link: attach a finding to a run and record the run in the finding's source_experiments.\n\n"
            "REQUIRES: run_id + finding_index from context_get_findings.\n"
            "RETURNS: {status, run_id, finding_index}\n"
            "NEXT: context_get_findings to verify linkage."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The experiment run_id to annotate."},
                "finding_index": {"type": "integer", "description": "Index of the finding in the current findings list (0-indexed)."},
            },
            "required": ["run_id", "finding_index"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Model extensibility tools
    # -----------------------------------------------------------------------
    Tool(
        name="model_list_families",
        description=(
            "List all registered model architecture families.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {families: [str]} or {families: [{name, source, kind}]} if detailed=true.\n"
            "NEXT: model_fetch_architecture to read source, model_get_config_schema for parameters, get_architecture_guide for workflow."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "detailed": {"type": "boolean", "description": "If true, include source metadata for each family", "default": False},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="model_list_activations",
        description=(
            "List all available activation functions (e.g. relu_sq, gelu_sq, mish_sq).\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {activations: [str]}\n"
            "NEXT: model_add_activation to register new ones. Use ACTIVATION env var in experiment config."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="model_list_components",
        description=(
            "List all available model building-block components (attention, MLP, norm, etc.).\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {components: [{name, description}]}\n"
            "NEXT: model_list_block_types for composition blocks, model_compose to build architectures."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="model_get_config_schema",
        description=(
            "Get the accepted configuration parameters for a model family.\n\n"
            "REQUIRES: Family name from model_list_families.\n"
            "RETURNS: {family, schema: {param: {type, default, description}}}\n"
            "NEXT: model_validate_config to check a specific config, enqueue_experiment to run."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "family": {"type": "string", "description": "Model family name (e.g. 'baseline')."},
            },
            "required": ["family"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="model_validate_config",
        description=(
            "Pre-flight validation: checks MODEL_FAMILY and ACTIVATION are valid.\n\n"
            "REQUIRES: Config dict with at least MODEL_FAMILY.\n"
            "RETURNS: {valid: bool, errors: [str], warnings: [str]}\n"
            "NEXT: enqueue_experiment if valid."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "config": {"type": "object", "description": "Experiment config dict to validate.", "additionalProperties": {"type": "string"}},
            },
            "required": ["config"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="model_add_architecture",
        description=(
            "Write and register a new architecture family at runtime as a Python plugin.\n\n"
            "REQUIRES: Python code that calls register_model(name, factory). Use model_generate_template for boilerplate.\n"
            "RETURNS: {status: 'registered', scope, family, families: [all]}\n"
            "NEXT: model_validate_config to verify, enqueue_experiment to test."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Family name (snake_case)."},
                "code": {"type": "string", "description": "Full Python source that defines and registers the architecture."},
                "scope": {"type": "string", "description": "Where to store: 'local' (project) or 'global' (hub)", "default": "local", "enum": ["local", "global"]},
            },
            "required": ["name", "code"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="model_add_activation",
        description=(
            "Register a new activation function at runtime. Expression uses 'x' with torch/F available.\n\n"
            "REQUIRES: Python expression (e.g. 'torch.sigmoid(x) * x').\n"
            "RETURNS: {status, name, activations: [all]}\n"
            "NEXT: Use ACTIVATION={name} in experiment config."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Activation function name."},
                "code": {"type": "string", "description": "Python expression (e.g. 'torch.sigmoid(x) * x')."},
            },
            "required": ["name", "code"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="model_generate_template",
        description=(
            "Generate boilerplate Python code for a new architecture family.\n\n"
            "REQUIRES: A name for the new family. Optional base architecture to derive from.\n"
            "RETURNS: {name, code: str}\n"
            "NEXT: Edit the code, then model_add_architecture to register. Or model_fetch_architecture to read existing code."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Architecture family name (snake_case)."},
                "base": {"type": "string", "description": "Base architecture to derive from.", "default": "baseline"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Plugin promotion / import tools
    # -----------------------------------------------------------------------
    Tool(
        name="model_list_global_architectures",
        description=(
            "List architecture plugins stored in the global hub (~/.crucible-hub/).\n\n"
            "REQUIRES: Hub initialized.\n"
            "RETURNS: {architectures: [{name, kind}]}\n"
            "NEXT: model_import_architecture to use in project, model_promote_architecture to add from project."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="model_promote_architecture",
        description=(
            "Promote a project-local architecture plugin to the global hub for cross-project reuse.\n\n"
            "REQUIRES: Plugin exists locally (.crucible/architectures/).\n"
            "RETURNS: {status, name}\n"
            "NEXT: hub_sync to persist, model_list_global_architectures to verify."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Plugin family name (must exist in .crucible/architectures/)."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="model_import_architecture",
        description=(
            "Import a global hub architecture into the project's local architectures.\n\n"
            "REQUIRES: Architecture name from model_list_global_architectures.\n"
            "RETURNS: {status, name}\n"
            "NEXT: model_list_families to verify, enqueue_experiment to test."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Architecture name to import from the hub."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Composition tools
    # -----------------------------------------------------------------------
    Tool(
        name="model_compose",
        description=(
            "Create architecture from declarative YAML spec. No Python code needed.\n\n"
            "REQUIRES: Spec dict with block, stack config. See model_list_stack_patterns + model_list_block_types.\n"
            "RETURNS: {status, family, spec_path}\n"
            "NEXT: model_preview_spec to validate, enqueue_experiment to test."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Architecture family name (valid Python identifier, snake_case)."},
                "spec": {
                    "type": "object",
                    "description": "Architecture spec dict with block, stack, embedding, etc. See baseline.yaml for format.",
                    "properties": {
                        "version": {"type": "integer", "default": 1},
                        "base": {"type": "string", "default": "tied_embedding_lm"},
                        "embedding": {"type": "object", "description": "Embedding layer configuration."},
                        "block": {"type": "object", "description": "Block type and parameters (type, dim, params)."},
                        "stack": {"type": "object", "description": "Stack wiring pattern and layer config (pattern, num_layers, etc.)."},
                        "transform": {"type": "object", "description": "Optional pre/post stack transforms."},
                        "init": {"type": "object", "description": "Optional initialization config (e.g. ortho)."},
                        "augmentations": {"type": "object", "description": "Optional augmentations (smear_gate, bigram_hash, etc.)."},
                    },
                },
                "scope": {"type": "string", "description": "Where to store: 'local' (project) or 'global' (hub).", "default": "local", "enum": ["local", "global"]},
            },
            "required": ["name", "spec"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="model_from_template",
        description=(
            "Fork an existing spec-based architecture with overrides to create a new family.\n\n"
            "REQUIRES: Base family with YAML spec (use model_get_spec to check).\n"
            "RETURNS: {status, family, spec_path}\n"
            "NEXT: model_preview_spec to validate, enqueue_experiment to test."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "New architecture family name (valid Python identifier)."},
                "base": {"type": "string", "description": "Existing family name to fork from (must have a YAML spec)."},
                "overrides": {"type": "object", "description": "Dict of overrides to deep-merge into the base spec.", "default": {}},
            },
            "required": ["name", "base"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="model_list_stack_patterns",
        description=(
            "List available stack wiring patterns for declarative composition.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {patterns: [{name, description}]}\n"
            "NEXT: model_list_block_types for blocks, then model_compose to build."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="model_list_block_types",
        description=(
            "List available block types for declarative composition.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {block_types: [{name, description, params}]}\n"
            "NEXT: model_list_stack_patterns for wiring, then model_compose to build."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="model_preview_spec",
        description=(
            "Dry-run a spec: instantiate on CPU, return param count + layer structure. No GPU needed.\n\n"
            "REQUIRES: Spec dict (same format as model_compose). Optional config overrides.\n"
            "RETURNS: {param_count, layers: [{name, shape}], model_dim, num_layers}\n"
            "NEXT: model_compose to register if satisfied, adjust spec if param count too high/low."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "spec": {
                    "type": "object",
                    "description": "Architecture spec dict (same format as model_compose). Must include 'name'.",
                },
                "config": {
                    "type": "object",
                    "description": "Config overrides (MODEL_DIM, NUM_LAYERS, etc.) to resolve template variables.",
                    "additionalProperties": {"type": "string"},
                    "default": {},
                },
            },
            "required": ["spec"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="model_get_spec",
        description=(
            "Get the YAML spec for a model family. Returns null if code-defined (use model_fetch_architecture for code).\n\n"
            "REQUIRES: Family name from model_list_families.\n"
            "RETURNS: {family, spec: dict|null}\n"
            "NEXT: model_from_template to fork, model_fetch_architecture for Python source."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "family": {"type": "string", "description": "Model family name."},
            },
            "required": ["family"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Config tools
    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    # Queue management tools
    # -----------------------------------------------------------------------
    Tool(
        name="cancel_experiment",
        description=(
            "Cancel queued or running experiments by name, run_id, or wave.\n\n"
            "REQUIRES: At least one of: run_id, experiment_name, or wave.\n"
            "RETURNS: {cancelled, details}\n"
            "NEXT: get_queue_status to verify, clear_stale_queue for orphaned runs."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Cancel a specific run by ID."},
                "experiment_name": {"type": "string", "description": "Cancel all runs with this experiment name."},
                "wave": {"type": "string", "description": "Cancel all runs in this wave."},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="clear_stale_queue",
        description=(
            "Mark experiments as failed if assigned to nodes that no longer exist.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {cleared: [run_ids], count}\n"
            "NEXT: get_queue_status to verify, dispatch_experiments to retry with new nodes."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="purge_queue",
        description=(
            "Remove all completed/failed/finished items from the fleet queue.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {removed: int, remaining: int}\n"
            "NEXT: get_queue_status to verify, enqueue_experiment or dispatch_experiments."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    # -----------------------------------------------------------------------
    # Config tools
    # -----------------------------------------------------------------------
    Tool(
        name="config_get_presets",
        description=(
            "List all experiment presets with resolved config (built-in defaults merged with crucible.yaml).\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {presets: {name: {iterations, train_batch_tokens, ...}}}\n"
            "NEXT: Use preset name as tier in enqueue_experiment or version_save_design."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "preset_name": {"type": "string", "description": "Return only this preset. Empty returns all.", "default": ""},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="config_get_project",
        description=(
            "Get full configuration of a project spec including pod config, env vars, launcher, and metrics.\n\n"
            "REQUIRES: Project spec exists in .crucible/projects/.\n"
            "RETURNS: {name, repo, branch, workspace, launcher, pod: {gpu_type, ...}, env_set, env_forward, metrics, install, timeout}\n"
            "NEXT: provision_project to create nodes, run_project to launch training."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Name of the project spec (without .yaml)."},
            },
            "required": ["project_name"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Run logs tool
    # -----------------------------------------------------------------------
    Tool(
        name="get_run_logs",
        description=(
            "Fetch stdout/stderr logs for an experiment run. Checks local collected logs first, falls back to SSH.\n\n"
            "REQUIRES: A valid run_id from enqueue_experiment, get_queue_status, or get_leaderboard.\n"
            "RETURNS: {found, source: 'local'|'remote', log_text, lines_returned, total_lines}\n"
            "NEXT: Use get_experiment_result for structured metrics, note_add to annotate observations."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "The experiment run_id."},
                "tail_lines": {"type": "integer", "description": "Lines from end to return. 0 = all.", "default": 100},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Architecture fetch tool
    # -----------------------------------------------------------------------
    Tool(
        name="model_fetch_architecture",
        description=(
            "Fetch full source code (Python) or spec (YAML) for any registered architecture family.\n\n"
            "REQUIRES: A family name from model_list_families.\n"
            "RETURNS: {family, kind: 'code'|'spec', source: 'builtin'|'local'|'global', content, file_path}\n"
            "NEXT: Modify the content and use model_add_architecture or model_compose to register a variant."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "family": {"type": "string", "description": "Model family name (e.g., 'baseline', 'looped')."},
            },
            "required": ["family"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Architecture guide tool
    # -----------------------------------------------------------------------
    Tool(
        name="get_architecture_guide",
        description=(
            "Decision guide for creating architectures: when to use declarative composition vs Python plugins, with full workflows.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {decision_tree, workflows, tips}\n"
            "NEXT: model_list_families to see what exists, model_list_stack_patterns + model_list_block_types for composition."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="get_wandb_guide",
        description=(
            "Decision guide + checklist for wiring W&B correctly: how WANDB_PROJECT / WANDB_API_KEY / "
            "CRUCIBLE_VARIANT_NAME flow into a run, common silent-failure modes, and a reproducible workflow.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {decision_tree, checklist, common_failures, workflow, verification, tips, see_also}\n"
            "NEXT: config_get_project to inspect the active wandb config block; "
            "recipe_get(name='wandb-tracked-experiment') for the canonical recipe; "
            "wandb_get_url(run_id) to verify a registered run."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    # -----------------------------------------------------------------------
    # Tree search tools
    # -----------------------------------------------------------------------
    Tool(
        name="tree_create",
        description="Create a new search tree over experiments with root nodes and policies. REQUIRES: name, optional roots list. RETURNS: tree name, root_node_ids. NEXT: tree_expand_node or tree_enqueue_pending.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tree name (slug-style, no spaces)."},
                "description": {"type": "string", "description": "What this search tree explores.", "default": ""},
                "roots": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "config": {"type": "object", "additionalProperties": {"type": "string"}},
                            "hypothesis": {"type": "string"},
                            "rationale": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["name", "config"],
                    },
                    "description": "Root experiment nodes.",
                    "default": [],
                },
                "expansion_policy": {"type": "string", "description": "Node selection policy: agent_directed, ucb1, greedy, epsilon_greedy.", "default": "agent_directed"},
                "pruning_policy": {"type": "string", "description": "Pruning policy: agent_directed, threshold.", "default": "agent_directed"},
                "expansion_config": {"type": "object", "description": "Policy-specific config (e.g. ucb_c for ucb1).", "default": {}},
                "pruning_config": {"type": "object", "description": "Pruning policy config (e.g. metric_threshold).", "default": {}},
                "primary_metric": {"type": "string", "description": "Metric to optimize.", "default": "val_bpb"},
                "metric_direction": {"type": "string", "description": "minimize or maximize.", "default": "minimize"},
                "max_depth": {"type": "integer", "description": "Maximum tree depth.", "default": 10},
                "max_nodes": {"type": "integer", "description": "Maximum total nodes.", "default": 500},
                "max_expansions_per_node": {"type": "integer", "description": "Max children per node.", "default": 5},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="tree_get",
        description="Get search tree structure, summary, ASCII visualization, and best path. REQUIRES: name. RETURNS: summary, ascii_tree, best_path. NEXT: tree_expand_node, tree_prune, tree_enqueue_pending.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tree name."},
                "max_depth": {"type": "integer", "description": "Max depth for ASCII rendering."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="tree_expand_node",
        description="Add child experiment nodes to a completed node. Merges parent config with child overrides. REQUIRES: name, parent_node_id, children. RETURNS: new_node_ids. NEXT: tree_enqueue_pending.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tree name."},
                "parent_node_id": {"type": "string", "description": "Node ID to expand."},
                "children": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "config": {"type": "object", "additionalProperties": {"type": "string"}},
                            "hypothesis": {"type": "string"},
                            "rationale": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                            "generation_method": {"type": "string"},
                            "priority_score": {"type": "number"},
                        },
                        "required": ["name", "config"],
                    },
                    "description": "Child experiment specs.",
                },
            },
            "required": ["name", "parent_node_id", "children"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="tree_auto_expand",
        description="LLM-generate child experiments for a node using Claude. REQUIRES: name, node_id, ANTHROPIC_API_KEY. RETURNS: new_node_ids with hypotheses. NEXT: tree_enqueue_pending.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tree name."},
                "node_id": {"type": "string", "description": "Node ID to auto-expand."},
                "n_children": {"type": "integer", "description": "Number of children to generate.", "default": 3},
                "extra_context": {"type": "string", "description": "Additional context for LLM generation.", "default": ""},
            },
            "required": ["name", "node_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="tree_prune",
        description="Prune a node or entire branch in the search tree. REQUIRES: name, node_id. RETURNS: pruned count. NEXT: tree_get.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tree name."},
                "node_id": {"type": "string", "description": "Node ID to prune."},
                "reason": {"type": "string", "description": "Why this node/branch is being pruned.", "default": ""},
                "prune_branch": {"type": "boolean", "description": "If true, recursively prune all descendants.", "default": False},
            },
            "required": ["name", "node_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="tree_enqueue_pending",
        description="Move pending tree nodes to the fleet queue for execution. REQUIRES: name. RETURNS: enqueued run_ids. NEXT: dispatch_experiments.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tree name."},
                "node_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific node IDs to enqueue. Empty = all pending.",
                },
                "tier": {"type": "string", "description": "Experiment tier.", "default": "proxy"},
                "backend": {"type": "string", "description": "Training backend.", "default": "torch"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="tree_sync_results",
        description="Match completed fleet queue results back to tree nodes. Updates status and metrics. REQUIRES: name. RETURNS: synced nodes with metrics. NEXT: tree_get, tree_expand_node.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Tree name."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="tree_list",
        description="List all search trees with summary statistics. RETURNS: list of tree summaries. NEXT: tree_get.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    # -----------------------------------------------------------------------
    # Modalities tool
    # -----------------------------------------------------------------------
    Tool(
        name="config_get_modalities",
        description="List available training backends with their modality tags, data adapters, and training objectives.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    # -----------------------------------------------------------------------
    # External project tools
    # -----------------------------------------------------------------------
    Tool(
        name="list_projects",
        description="List all external project specs in .crucible/projects/.\n\nEach spec has its own WANDB_PROJECT in env_set. Best practice: use the same WANDB_PROJECT for related experiments (e.g., architecture variants) so results appear in one leaderboard.\n\nREQUIRES: Nothing.\nRETURNS: {projects: [{name, repo, train, metrics_primary}]}\nNEXT: config_get_project to inspect a spec, provision_project or bootstrap_project to launch.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="provision_project",
        description="Provision nodes for an external project, applying pod overrides (GPU, image, disk) from the project spec.\n\nREQUIRES: RUNPOD_API_KEY, project spec in .crucible/projects/. Use config_get_project to inspect pod config first.\nRETURNS: {created, new_nodes: [{name, node_id}]}\nNEXT: fleet_refresh (wait ~60s), then bootstrap_project.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Name of the project spec (without .yaml)."},
                "count": {"type": "integer", "default": 1, "description": "Number of nodes."},
                "interruptible": {"type": "boolean", "description": "Override spot/on-demand. If not set, uses project spec's pod.interruptible value."},
            },
            "required": ["project_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="bootstrap_project",
        description="Bootstrap an external project on fleet nodes: clone repo, create venv, install deps, forward env vars, run setup commands. Long-running (2-10 min).\n\nREQUIRES: Nodes with SSH (run fleet_refresh after provision_project).\nRETURNS: {total, bootstrapped, nodes: [{name, state, project}]}\nNEXT: run_project.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Name of the project spec."},
                "node_names": {"type": "array", "items": {"type": "string"}, "description": "Specific nodes. Empty = all."},
            },
            "required": ["project_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="run_project",
        description=(
            "Launch training for an external project as a detached process on bootstrapped nodes. "
            "Returns immediately with per-node run ids.\n\n"
            "VARIANTS: If the project spec has a `variants:` dict, pass `variant=<name>` to apply "
            "that variant's env-var overrides. Caller's `overrides` dict still wins over variant values, "
            "so you can tweak individual knobs (e.g. `variant='phase5_contrast_15k_high', overrides={'WM_SEED': '43'}`). "
            "Passing an unknown variant name is a loud error.\n\n"
            "W&B: WANDB_PROJECT comes from spec.env_set; WANDB_API_KEY must be in the env file synced to the pod "
            "(provider.defaults.env_source, default .env.runpod.local). The variant name becomes WANDB_RUN_NAME automatically. "
            "To consolidate experiments across different specs into one W&B project, pass WANDB_PROJECT in overrides. "
            "Contract enforcement is on (CRUCIBLE_ENFORCE_CONTRACT=1) -- runs fail at startup if W&B init fails. "
            "See get_wandb_guide.\n\n"
            "REQUIRES: Nodes bootstrapped via bootstrap_project.\n"
            "RETURNS: {launch_id, run_id?(single-node), nodes: [{run_id, name, pid, status}]}\n"
            "NEXT: get_project_run_status to monitor lifecycle, collect_project_results when done. "
            "wandb_get_url(run_id) verifies the run actually registered."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Name of the project spec."},
                "node_names": {"type": "array", "items": {"type": "string"}, "description": "Specific nodes. Empty = all ready."},
                "overrides": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Env var overrides for this run. Wins over variant values if the same key is in both."},
                "variant": {"type": "string", "description": "Name of a variant in spec.variants. Applies the variant's env dict before the caller's overrides."},
            },
            "required": ["project_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="run_project_chain",
        description=(
            "Run a sequence of project variants on the same node, auto-chaining. "
            "Launches the first variant, polls until completion, then launches the next. "
            "Long-running (minutes to hours). Runs in background thread.\n\n"
            "REQUIRES: Node bootstrapped via bootstrap_project. "
            "wandb.project + WANDB_API_KEY configured per the contract; each variant gets its own WANDB_RUN_NAME. "
            "See get_wandb_guide.\n"
            "RETURNS: {chain_id, variants_total, results: [{variant, run_id, status, duration_s}]}\n"
            "NEXT: get_project_run_status for individual runs, collect_project_results when chain completes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Name of the project spec."},
                "variants": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ordered list of variant names to run sequentially.",
                },
                "node_name": {"type": "string", "description": "Node to run on (single node for sequential chain)."},
                "overrides": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "Extra env overrides applied to ALL variants in the chain.",
                },
                "poll_interval": {
                    "type": "integer",
                    "default": 30,
                    "description": "Seconds between completion checks (default 30).",
                },
            },
            "required": ["project_name", "variants", "node_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="collect_project_results",
        description="Collect results from one external project run or an entire launch: rsync logs, parse metrics, fetch WandB data, and persist terminal status.\n\nFetches W&B metrics from the project specified in the spec's env_set.WANDB_PROJECT.\n\nREQUIRES: run_project has been called.\nRETURNS: {run_id|launch_id, status|summary, metrics|runs, log_tail}\nNEXT: get_leaderboard if completed.",
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Per-node run ID returned by run_project."},
                "launch_id": {"type": "string", "description": "Batch launch ID returned by run_project for multi-node launches."},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="get_project_run_status",
        description="Probe and reconcile the latest lifecycle state for an external project run, including recent lifecycle events.\n\nREQUIRES: run_project has been called. Use run_id from run_project response.\nRETURNS: {run_id, status, failure_class?, remote_node_state?, events:[{ts, event, detail}]}\nNEXT: collect_project_results when status is terminal (completed/failed/timeout), get_fleet_status(include_metrics=true) for broader fleet state, or get_run_logs for debugging failures.",
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Per-node run ID returned by run_project."},
                "event_limit": {"type": "integer", "description": "How many recent lifecycle events to include.", "default": 10},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    ),
    # Recipe tools
    Tool(
        name="recipe_save",
        description=(
            "Save a session recipe — a step-by-step reproduction guide capturing what MCP tools were called, "
            "what configs/versions worked, gotchas encountered, and final results. "
            "Other agents can follow this recipe to reproduce the session.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {saved, path, name}\n"
            "NEXT: recipe_list to see all recipes, recipe_get to retrieve one."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Slug name for the recipe (e.g., 'yolo11-nano-coco128')."},
                "title": {"type": "string", "description": "Human-readable title."},
                "goal": {"type": "string", "description": "What this session aimed to achieve."},
                "project_spec": {"type": "string", "description": "Name of the project spec used (from .crucible/projects/)."},
                "environment": {"type": "object", "description": "Runtime environment: gpu, torch version, python, provider, etc."},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "tool": {"type": "string"},
                            "args": {"type": "object"},
                            "note": {"type": "string"},
                        },
                    },
                    "description": "Ordered list of MCP tool calls with args and notes.",
                },
                "results": {"type": "object", "description": "Final metrics, W&B URLs, linked run IDs."},
                "gotchas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "issue": {"type": "string"},
                            "fix": {"type": "string"},
                        },
                    },
                    "description": "Problems encountered and how they were fixed.",
                },
                "tags": {"type": "array", "items": {"type": "string"}, "description": "Searchable tags."},
                "created_by": {"type": "string", "description": "Who created this recipe (default: mcp-agent)."},
            },
            "required": ["name", "steps"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="recipe_list",
        description=(
            "List all saved session recipes.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {recipes: [{name, title, created_at, tags, project_spec, goal}]}\n"
            "NEXT: recipe_get for full details."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "tag": {"type": "string", "description": "Filter recipes by tag."},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="recipe_get",
        description=(
            "Get a saved session recipe with full step-by-step details, configs, gotchas, and results. "
            "Pass this to an agent to reproduce the session.\n\n"
            "REQUIRES: Recipe exists.\n"
            "RETURNS: Full recipe with steps, environment, gotchas, results.\n"
            "NEXT: Follow the steps to reproduce."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Recipe name (from recipe_list)."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    # ---- Plugin registry tools ----
    Tool(
        name="plugin_list",
        description=(
            "List all registered plugins of a given type (builtin + global + local).\n\n"
            "REQUIRES: type — one of: optimizers, schedulers, providers, loggers, callbacks, "
            "block_types, stack_patterns, augmentations.\n"
            "RETURNS: {<type>: [{name, source}, ...]}\n"
            "NEXT: plugin_add to register a new plugin, plugin_get_schema for config details."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["optimizers", "schedulers", "providers", "loggers", "callbacks",
                             "block_types", "stack_patterns", "augmentations"],
                    "description": "Plugin type to list.",
                },
            },
            "required": ["type"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="plugin_add",
        description=(
            "Register a new plugin from Python code at runtime.\n\n"
            "The code should import the appropriate register function and call it:\n"
            "  optimizers → register_optimizer, schedulers → register_scheduler,\n"
            "  providers → register_provider, loggers → register_logger,\n"
            "  callbacks → register_callback, block_types/stack_patterns/augmentations → register in composer.\n\n"
            "REQUIRES: type, name, code.\n"
            "RETURNS: {status, name, plugin_type, path}\n"
            "NEXT: plugin_list to verify registration."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["optimizers", "schedulers", "providers", "loggers", "callbacks",
                             "block_types", "stack_patterns", "augmentations"],
                    "description": "Plugin type.",
                },
                "name": {"type": "string", "description": "Plugin name (e.g., 'lion', 'cosine_warm')."},
                "code": {"type": "string", "description": "Python source that registers the plugin."},
            },
            "required": ["type", "name", "code"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="plugin_get_schema",
        description=(
            "Get the config parameter schema for a named plugin.\n\n"
            "REQUIRES: type (optimizers or schedulers only), name.\n"
            "RETURNS: {type, name, schema}"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["optimizers", "schedulers"],
                    "description": "Plugin type (only optimizers and schedulers have schemas).",
                },
                "name": {"type": "string", "description": "Plugin name."},
            },
            "required": ["type", "name"],
            "additionalProperties": False,
        },
    ),
    # ---- Community tap tools ----
    Tool(
        name="hub_tap_add",
        description="Add a community plugin tap (git repo).\n\nREQUIRES: url.\nRETURNS: {name, url, added_at}\nNEXT: hub_tap_sync to pull latest, hub_search to find plugins.",
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Git URL of the tap repository."},
                "name": {"type": "string", "description": "Override tap name (derived from URL if omitted)."},
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_tap_remove",
        description="Remove a tap and its cloned repo.\n\nREQUIRES: name.\nRETURNS: {removed: true}",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Tap name to remove."}},
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_tap_list",
        description="List all configured taps.\n\nREQUIRES: nothing.\nRETURNS: {taps: [{name, url, added_at, last_synced}]}",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="hub_tap_sync",
        description="Pull latest from one or all taps.\n\nREQUIRES: nothing (optional name).\nRETURNS: {synced: [...], errors: [...]}\nNEXT: hub_search to find new plugins.",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Specific tap to sync. Syncs all if omitted."}},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_search",
        description="Search for plugins across all taps by name, description, or tags.\n\nREQUIRES: nothing (optional query, type).\nRETURNS: {results: [{name, type, version, description, author, tap}]}\nNEXT: hub_install to install a found plugin.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query (matches name, description, tags, author)."},
                "type": {"type": "string", "description": "Filter by plugin type (e.g. 'optimizers', 'callbacks')."},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_install",
        description="Install a community plugin from a tap into the hub's global plugins directory.\n\nREQUIRES: name. Hub initialized. At least one tap added.\nRETURNS: {status, name, type, version, tap, path}\nNEXT: optimizer_list_available (or similar) to verify the plugin loaded.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Plugin name to install."},
                "tap": {"type": "string", "description": "Specific tap to install from. Searches all if omitted."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_uninstall",
        description="Remove an installed tap plugin.\n\nREQUIRES: name.\nRETURNS: {removed: true}",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Plugin name to uninstall."}},
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_installed",
        description="List all installed tap plugins.\n\nREQUIRES: nothing (optional type filter).\nRETURNS: {packages: [{name, type, version, tap, installed_at}]}",
        inputSchema={
            "type": "object",
            "properties": {"type": {"type": "string", "description": "Filter by plugin type."}},
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_publish",
        description="Publish a local plugin to a tap repository. Commits to the tap repo; user pushes manually.\n\nREQUIRES: name, type, tap.\nRETURNS: {status, path, note}",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Local plugin name to publish."},
                "type": {"type": "string", "description": "Plugin type (e.g. 'optimizers')."},
                "tap": {"type": "string", "description": "Target tap name."},
            },
            "required": ["name", "type", "tap"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_tap_push",
        description="Push a tap repo to its git remote after publishing.\n\nREQUIRES: tap name.\nRETURNS: {status, tap}\nNEXT: hub_submit_pr to open a PR if this is a fork.",
        inputSchema={
            "type": "object",
            "properties": {"tap": {"type": "string", "description": "Tap name to push."}},
            "required": ["tap"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_submit_pr",
        description=(
            "Open a GitHub PR from a tap fork to its upstream. Uses gh CLI if available, "
            "falls back to manual instructions.\n\n"
            "REQUIRES: tap name. Optional: title, body.\n"
            "RETURNS: {status, pr_url} or {status: 'manual', instructions}"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "tap": {"type": "string", "description": "Tap name."},
                "title": {"type": "string", "description": "PR title."},
                "body": {"type": "string", "description": "PR body/description."},
            },
            "required": ["tap"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="hub_package_info",
        description="Get detailed info about a package including manifest and install status.\n\nREQUIRES: name.\nRETURNS: full manifest + {installed: bool}",
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Package name."}},
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    # Trace tools
    Tool(
        name="trace_list",
        description=(
            "List all session traces with metadata summaries.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {traces: [{session_id, started_at, ended_at, tool_calls, tool_counts, trace_file}]}\n"
            "NEXT: trace_get to view full entries for a session."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="trace_get",
        description=(
            "Get the full trace entries for a session.\n\n"
            "REQUIRES: session_id from trace_list.\n"
            "RETURNS: {session_id, entries: [{ts, seq, tool, arguments, result, duration_ms, status}], meta}"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID from trace_list."},
            },
            "required": ["session_id"],
            "additionalProperties": False,
        },
    ),
    # Data tools
    Tool(
        name="data_list",
        description=(
            "List all registered data sources.\n\n"
            "REQUIRES: Nothing\n"
            "RETURNS: {sources: [{name, type}]}"
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="data_status",
        description=(
            "Check data state for a source.\n\n"
            "REQUIRES: name\n"
            "RETURNS: {name, status, manifest, shard_count, last_prepared, issues}\n"
            "NEXT: data_prepare to refresh, data_validate to check integrity"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "config": {"type": "object"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="data_prepare",
        description=(
            "Prepare (download/cache) data. Long-running.\n\n"
            "REQUIRES: name\n"
            "RETURNS: {success, job_id, message, shards_downloaded}\n"
            "NEXT: data_status to check completion"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "config": {"type": "object"},
                "force": {"type": "boolean", "default": False},
                "background": {"type": "boolean", "default": False},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="data_validate",
        description=(
            "Validate data integrity.\n\n"
            "REQUIRES: name\n"
            "RETURNS: {valid, errors, warnings}"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string"},
                "config": {"type": "object"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="data_search",
        description=(
            "Search for available data.\n\n"
            "REQUIRES: query\n"
            "RETURNS: {results: [{name, source, description, shard_count}]}"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "type": {"type": "string"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="data_link",
        description=(
            "Link data source to experiment run for provenance.\n\n"
            "REQUIRES: run_id, data_name\n"
            "RETURNS: {linked, run_id, data_name}"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "data_name": {"type": "string"},
            },
            "required": ["run_id", "data_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="data_get_linked",
        description=(
            "Get data linked to an experiment run.\n\n"
            "REQUIRES: run_id\n"
            "RETURNS: {data_sources: [{name}]}"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    ),
    # ── Research DAG bridge tools ──────────────────────────────────
    Tool(
        name="research_dag_init",
        description=(
            "Initialize research DAG bridge for visual experiment tracking. "
            "Spider Chat is OPTIONAL — works in local-only mode without it (DAG state tracked locally). "
            "If Spider Chat is available, experiments sync to canvas as a visual DAG.\n\n"
            "REQUIRES: Optional flow_id (Spider Chat flow ID, omit for local-only). Optional: spiderchat_url, project_name. Token via SPIDERCHAT_TOKEN env var.\n"
            "RETURNS: {status, flow_id, project_name, total_mappings, mode: 'connected'|'local-only'}\n"
            "NEXT: research_dag_push_node or research_dag_sync"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "flow_id": {"type": "string", "description": "Spider Chat flow ID. Omit for local-only mode.", "default": ""},
                "project_name": {"type": "string", "description": "Human-readable project name.", "default": ""},
                "spiderchat_url": {"type": "string", "description": "Spider Chat backend URL. Token read from SPIDERCHAT_TOKEN env var.", "default": ""},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="research_dag_sync",
        description=(
            "Bidirectional sync between Crucible search tree and Spider Chat canvas (if available). "
            "Tracks DAG state locally always. If Spider Chat is connected, also pushes nodes to canvas and pulls manual nodes. "
            "Works fully without Spider Chat.\n\n"
            "REQUIRES: research_dag_init completed, optional tree_name.\n"
            "RETURNS: {mode, pushed, updated, pulled, skipped_canvas, manual_hypotheses, total_mappings}\n"
            "NEXT: dispatch_experiments (for pulled hypotheses) or continue research"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "tree_name": {"type": "string", "description": "Crucible search tree to sync. Omit to sync without tree data."},
                "flow_id": {"type": "string", "description": "Override flow ID (uses init default if omitted)."},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="research_dag_push_node",
        description=(
            "Push a single experiment or hypothesis to Spider Chat canvas as an information node.\n\n"
            "REQUIRES: research_dag_init completed, at least name or node_id.\n"
            "RETURNS: {canvas_node_id}\n"
            "NEXT: Create more nodes or research_dag_sync"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Experiment name."},
                "node_id": {"type": "string", "description": "Crucible node ID (defaults to name)."},
                "hypothesis": {"type": "string", "description": "Hypothesis text."},
                "rationale": {"type": "string", "description": "Why this experiment."},
                "config": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Experiment config (env vars)."},
                "status": {"type": "string", "description": "Experiment status.", "default": "pending"},
                "result": {"type": "object", "description": "Result dict if completed."},
                "result_metric": {"type": "number", "description": "Primary metric value."},
                "parent_canvas_ids": {"type": "array", "items": {"type": "string"}, "description": "Parent canvas node IDs for edges."},
                "flow_id": {"type": "string", "description": "Override flow ID."},
                "primary_metric": {"type": "string", "default": "val_bpb"},
                "best_metric": {"type": "number", "description": "Current best metric for highlighting."},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="research_dag_pull_manual",
        description=(
            "Import manually-created Spider Chat canvas nodes as Crucible hypotheses.\n\n"
            "REQUIRES: research_dag_init completed.\n"
            "RETURNS: {count, hypotheses: [{name, hypothesis, config, rationale, canvas_node_id}]}\n"
            "NEXT: Enrich hypotheses with configs, then enqueue_experiment"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "flow_id": {"type": "string", "description": "Override flow ID."},
            },
            "additionalProperties": False,
        },
    ),
    Tool(
        name="research_dag_status",
        description=(
            "Show current research DAG bridge status and mapping summary.\n\n"
            "REQUIRES: research_dag_init completed.\n"
            "RETURNS: {flow_id, project_name, total_mappings, status_breakdown, type_breakdown}"
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
    # Literature search
    Tool(
        name="research_literature_search",
        description=(
            "Search AI research papers on HuggingFace for literature relevant to "
            "current research direction. Supports multi-angle search: expands a "
            "query into cross-domain reformulations (synonyms, enabling mechanisms, "
            "adjacent fields) via LLM before searching.\n\n"
            "REQUIRES: Nothing (optional: active research state for auto mode, "
            "ANTHROPIC_API_KEY for multi-angle expansion).\n"
            "RETURNS: {papers: [{id, title, summary, upvotes, github_repo}], "
            "query_used, literature_context, count}\n"
            "NEXT: design_generate_hypotheses with extra_context containing relevant papers."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query for papers. If empty and auto=true, "
                        "queries are auto-generated from research state."
                    ),
                },
                "auto": {
                    "type": "boolean",
                    "description": (
                        "Auto-generate queries from research state "
                        "(beliefs, findings, program.md). Default false."
                    ),
                    "default": False,
                },
                "multi_angle": {
                    "type": "boolean",
                    "description": (
                        "Expand query into cross-domain search angles via LLM "
                        "before searching. Finds papers using different terminology "
                        "for the same concept. Default true for auto mode, false otherwise."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum papers to return. Default 10.",
                    "default": 10,
                },
            },
        },
    ),
    # ── Harness Optimization tools ─────────────────────────────────
    Tool(
        name="harness_init",
        description=(
            "Initialize a HarnessOptimizer for meta-harness-style evolutionary loops. "
            "Candidates are Python source files; metrics are tracked on an N-dimensional Pareto frontier.\n\n"
            "REQUIRES: domain_spec (path, name, or directory) and tree_name.\n"
            "RETURNS: {status, tree_name, domain_spec, metrics, tree_summary, frontier}\n"
            "NEXT: harness_propose, harness_iterate, harness_frontier"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "domain_spec": {
                    "type": "string",
                    "description": "Absolute path, project-relative path, or bare name under .crucible/domain_specs/",
                },
                "tree_name": {"type": "string", "description": "Search tree name (created on first use)."},
                "n_candidates": {"type": "integer", "description": "Candidates proposed per iteration.", "default": 3},
                "dry_run": {"type": "boolean", "description": "Skip LLM calls; use fixture candidates.", "default": False},
            },
            "required": ["domain_spec", "tree_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="harness_propose",
        description=(
            "Generate candidate harness implementations via the LLM proposer.\n\n"
            "REQUIRES: harness_init called for tree_name.\n"
            "RETURNS: {status, candidates: [{name, hypothesis, code, rationale, config}], count}\n"
            "NEXT: harness_validate or harness_iterate"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "tree_name": {"type": "string"},
                "n": {"type": "integer", "description": "Override n_candidates for this call."},
            },
            "required": ["tree_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="harness_validate",
        description=(
            "Validate candidate source code against the domain spec without dispatching.\n\n"
            "REQUIRES: harness_init called, candidates list with 'code' field each.\n"
            "RETURNS: {status, candidates (annotated with validation), valid_count, rejected_count}\n"
            "NEXT: harness_iterate"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "tree_name": {"type": "string"},
                "candidates": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of candidate dicts with 'code' (required) and optional 'config'.",
                },
            },
            "required": ["tree_name", "candidates"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="harness_iterate",
        description=(
            "Run one full propose→validate→benchmark cycle and append to the evolution log.\n\n"
            "REQUIRES: harness_init called.\n"
            "RETURNS: {status, iteration, proposed, validated, benchmarked_node_ids, frontier_summary, log_record}\n"
            "NEXT: harness_frontier, harness_evolution_log, or another harness_iterate"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "tree_name": {"type": "string"},
                "cost": {"type": "object", "description": "Optional {tokens, compute_hours, ...} for the log."},
                "notes": {"type": "string", "description": "Free-form notes for this iteration."},
            },
            "required": ["tree_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="harness_frontier",
        description=(
            "Return the current Pareto frontier snapshot for a harness tree.\n\n"
            "REQUIRES: tree_name (optimizer init is optional; reads directly from disk if absent).\n"
            "RETURNS: {status, frontier_node_ids, frontier_size, dominated_count, metrics, best_per_metric, hypervolume?}"
        ),
        inputSchema={
            "type": "object",
            "properties": {"tree_name": {"type": "string"}},
            "required": ["tree_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="harness_evolution_log",
        description=(
            "Return all iteration records from evolution_log.jsonl.\n\n"
            "REQUIRES: tree_name with a log file on disk.\n"
            "RETURNS: {status, records: [...], count}"
        ),
        inputSchema={
            "type": "object",
            "properties": {"tree_name": {"type": "string"}},
            "required": ["tree_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="tree_pareto",
        description=(
            "Return the Pareto frontier for any search tree (general-purpose; not harness-specific).\n\n"
            "REQUIRES: name of an existing tree under .crucible/search_trees/.\n"
            "RETURNS: {status, frontier_node_ids, frontier_size, dominated_count, metrics, best_per_metric, hypervolume?}"
        ),
        inputSchema={
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="eval_watch_start",
        description=(
            "Start the auto-eval daemon for a project. Polls running pods every "
            "`interval` seconds, SCPs new checkpoints from `remote_pattern`, and "
            "runs each script in the project's `eval_suite:` block on each new ckpt. "
            "Idempotent: same checkpoint+script never runs twice (SHA-tracked).\n\n"
            "REQUIRES: project YAML must contain an `eval_suite:` block (list of "
            "{script, args}). Pods must be reachable via the local SSH key.\n"
            "RETURNS: {status: 'started'|'already_running', state, suite_size}\n"
            "NEXT: eval_watch_status to inspect progress, eval_watch_stop to halt."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_name": {
                    "type": "string",
                    "description": "Project spec name (without .yaml).",
                },
                "interval": {
                    "type": "integer",
                    "description": "Seconds between polls. Default 300.",
                    "default": 300,
                },
                "remote_pattern": {
                    "type": "string",
                    "description": (
                        "Glob on the pod for checkpoints to pull. "
                        "Default `/workspace/project/checkpoints/*.pt`."
                    ),
                    "default": "/workspace/project/checkpoints/*.pt",
                },
                "env": {
                    "type": "object",
                    "description": "Extra env vars passed to every eval script.",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["project_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="eval_watch_stop",
        description=(
            "Stop the auto-eval daemon and wait for the in-flight poll to finish.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {status: 'stopped'|'not_running', state}"
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="eval_watch_status",
        description=(
            "Return the current state of the auto-eval daemon plus the most recent "
            "evaluation rows from `.crucible/eval_watch.jsonl`.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {state: {running, project, last_poll_at, total_runs, ...}, "
            "recent: [{label, script, ckpt_sha, ok, elapsed_s, result, ran_at}]}"
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "recent": {
                    "type": "integer",
                    "description": "How many recent rows to include. Default 10.",
                    "default": 10,
                },
            },
            "additionalProperties": False,
        },
    ),
    # ------------------------------------------------------------------
    # Plan tools (LLM-facing todo list)
    # ------------------------------------------------------------------
    Tool(
        name="plan_get",
        description=(
            "Return the current plan: a flat list of todo items with statuses "
            "'pending', 'in_progress', or 'completed'. Backed by .crucible/plan.json.\n\n"
            "REQUIRES: Nothing (empty plan if file missing).\n"
            "RETURNS: {items: [{id, description, status, created_at, updated_at}]}\n"
            "NEXT: plan_set to replace, plan_update_item to flip a single status."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="plan_set",
        description=(
            "Replace the entire plan with a new list of items. Use for work with "
            "3+ steps. Each call sends the COMPLETE updated list — never partial.\n\n"
            "Invariant: at most ONE item may be 'in_progress' at a time.\n\n"
            "REQUIRES: items list.\n"
            "RETURNS: {status: 'set', items: [...]}\n"
            "NEXT: plan_update_item to flip statuses as you make progress."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Full replacement plan.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "description": "Optional; auto-assigned if omitted."},
                            "description": {"type": "string", "description": "What needs doing."},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "default": "pending",
                            },
                        },
                        "required": ["description"],
                    },
                },
            },
            "required": ["items"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="plan_update_item",
        description=(
            "Flip one plan item's status. Enforces the one-in-progress rule.\n\n"
            "REQUIRES: id of existing item, new status.\n"
            "RETURNS: {status: 'updated', item: {...}}\n"
            "NEXT: plan_get to review, plan_update_item to flip the next item."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Plan item id."},
                "status": {
                    "type": "string",
                    "enum": ["pending", "in_progress", "completed"],
                },
            },
            "required": ["id", "status"],
            "additionalProperties": False,
        },
    ),
    # ------------------------------------------------------------------
    # HF ecosystem search (datasets / models / spaces / docs)
    # ------------------------------------------------------------------
    Tool(
        name="research_hf_search",
        description=(
            "Search HuggingFace datasets, models, spaces, or docs.\n\n"
            "REQUIRES: kind in {datasets, models, spaces, docs}, query. "
            "multi_angle=true triggers LLM-driven cross-domain query expansion (slower, broader).\n"
            "RETURNS: {kind, query, count, results: [...]}\n"
            "NEXT: research_hf_search with a different kind, or data_prepare / model_fetch_architecture."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["datasets", "models", "spaces", "docs"],
                    "description": "Which HF hub to search.",
                },
                "query": {"type": "string", "description": "Natural language or keyword query."},
                "limit": {"type": "integer", "default": 10},
                "multi_angle": {
                    "type": "boolean",
                    "default": False,
                    "description": "Expand the query via LLM and dedup across angles.",
                },
            },
            "required": ["kind", "query"],
            "additionalProperties": False,
        },
    ),
    # ------------------------------------------------------------------
    # GitHub search
    # ------------------------------------------------------------------
    Tool(
        name="research_github_code",
        description=(
            "Search GitHub code. Requires GITHUB_TOKEN env (unauthenticated /search/code is disabled).\n\n"
            "REQUIRES: GITHUB_TOKEN env var.\n"
            "RETURNS: {query, language, count, results: [{repo, path, url, sha, match_snippets}]}\n"
            "NEXT: research_github_read_file to fetch a matching file."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "GitHub code-search query."},
                "language": {"type": "string", "description": "Optional language filter (e.g. 'python')."},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="research_github_list_repos",
        description=(
            "Search GitHub repositories. Auth optional (but avoids rate limits).\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {query, count, results: [{full_name, description, stars, forks, language, url, updated_at}]}\n"
            "NEXT: research_github_read_file to inspect files."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Repository search query."},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="research_github_read_file",
        description=(
            "Fetch a single file from a GitHub repo.\n\n"
            "REQUIRES: repo ('owner/name'), path. Optional ref (default 'main').\n"
            "RETURNS: {path, ref, size, encoding, content, url}\n"
            "NEXT: model_add_architecture / paste snippet as hypothesis input."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "repo": {"type": "string", "description": "owner/name"},
                "path": {"type": "string", "description": "File path within the repo."},
                "ref": {"type": "string", "default": "main", "description": "Branch, tag, or sha."},
            },
            "required": ["repo", "path"],
            "additionalProperties": False,
        },
    ),
    # ------------------------------------------------------------------
    # Orchestrator-driven research loop (Crucible is infra; you are the LLM)
    # ------------------------------------------------------------------
    Tool(
        name="research_request_prompt",
        description=(
            "Build the orchestrator-facing prompt + JSON schema for the requested "
            "research stage. Crucible ships the prompts and context; you (the "
            "orchestrator) call your own LLM, parse per schema, then submit via "
            "research_submit. No API keys needed inside Crucible.\n\n"
            "Stages:\n"
            "- 'hypothesis': propose N experiment hypotheses given current state + literature.\n"
            "- 'reflection': digest recent results, update beliefs, pick promote/kill.\n"
            "- 'briefing': read-only markdown summary of project state (no submit needed).\n\n"
            "REQUIRES: Nothing. Works with empty state (returns baseline prompt).\n"
            "RETURNS: {stage, system, user, schema, state_snapshot}\n"
            "NEXT: call your LLM with {system, user} — expect JSON matching {schema}. Then research_submit."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "stage": {
                    "type": "string",
                    "enum": ["hypothesis", "reflection", "briefing"],
                    "description": "Which stage to build prompts for.",
                },
                "focus_family": {
                    "type": "string",
                    "description": "Optional — bias hypothesis generation toward a specific model family.",
                },
                "extra_context": {
                    "type": "string",
                    "description": "Extra free-form context appended to the analysis section.",
                },
                "literature_context": {
                    "type": "string",
                    "description": "Optional formatted literature section (see research_literature_search).",
                },
                "iteration": {"type": "integer", "default": 0},
            },
            "required": ["stage"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="research_submit",
        description=(
            "Apply an orchestrator-supplied response for the given stage. Response "
            "can be a parsed object (matching the schema from research_request_prompt) "
            "or a raw JSON string.\n\n"
            "Hypothesis: adds items to state.hypotheses (ready for design_batch_from_hypotheses).\n"
            "Reflection: updates state.beliefs + returns promote/kill lists the orchestrator "
            "can then apply via existing fleet tools.\n\n"
            "REQUIRES: stage + response matching the schema from research_request_prompt.\n"
            "RETURNS: {applied, summary, counts...}\n"
            "NEXT: (hypothesis) design_batch_from_hypotheses → design_enqueue_batch → dispatch_experiments.\n"
            "      (reflection) examine promote_names/kill_names; add promoted hypotheses if desired."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "stage": {
                    "type": "string",
                    "enum": ["hypothesis", "reflection"],
                    "description": "Which stage the response is for. (briefing has no submit.)",
                },
                "response": {
                    "description": "Parsed response object OR raw JSON string. Shape matches the schema from research_request_prompt.",
                },
                "iteration": {"type": "integer", "default": 0},
            },
            "required": ["stage", "response"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Notebook exporter
    # -----------------------------------------------------------------------
    Tool(
        name="notebook_export",
        description=(
            "Export a Crucible project spec as a standalone Colab-runnable notebook.\n\n"
            "REQUIRES: a project name that resolves via load_project_spec "
            "(local .crucible/projects/, hub, or tap).\n"
            "RETURNS: {ok, project, runtime, preset, variant, out_path, source_path, cells, size_bytes, open_in_colab_url}\n"
            "NEXT: commit the notebook to a repo and open the returned colab URL, "
            "or upload the .ipynb to Google Drive."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Project spec name."},
                "runtime": {
                    "type": "string",
                    "default": "colab-h100",
                    "description": "Runtime profile; list via notebook_list_runtimes.",
                },
                "preset": {"type": "string", "default": "smoke"},
                "variant": {"type": "string", "description": "Variant name from spec.variants (optional)."},
                "overrides": {
                    "type": "object",
                    "description": "Extra env var overrides; wins over variant + env_set.",
                    "additionalProperties": {"type": "string"},
                },
                "out_path": {"type": "string", "description": "Output path (.py or .ipynb). Default: <project>.ipynb."},
                "inline_plugins": {"type": "boolean", "default": False, "description": "Reserved — not yet implemented."},
                "crucible_install": {"type": "string", "description": "pip spec for Crucible itself (default: git+main)."},
            },
            "required": ["project"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="notebook_list_runtimes",
        description=(
            "List available notebook runtime profiles (Colab H100/A100/T4, local).\n\n"
            "REQUIRES: nothing.\n"
            "RETURNS: {ok, runtimes: [{name, description, gpu}]}\n"
            "NEXT: pick one and pass its name as notebook_export(runtime=...)."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:  # type: ignore[type-arg]
    handler = TOOL_DISPATCH.get(name)
    if handler is None:
        _log.warning("Unknown tool requested: %s", name)
        return _error_text(f"Unknown tool: {name}")

    t0 = time.monotonic()

    # For long-running tools, send periodic keepalive log messages to prevent
    # the stdio client from timing out on pipe reads.
    use_keepalive = name in _LONG_RUNNING_TOOLS
    session = None
    request_id = None
    if use_keepalive:
        try:
            ctx = app.request_context
            session = ctx.session
            request_id = ctx.request_id
        except LookupError:
            _log.debug("No request context available for keepalive, falling back to direct call")
            use_keepalive = False
        if session is None:
            _log.debug("No MCP session for keepalive, falling back to direct call")
            use_keepalive = False

    is_error = False
    exc: BaseException | None = None
    try:
        if use_keepalive:
            result, is_error, exc = await _run_with_keepalive(handler, arguments, name, session, request_id)
        else:
            result, is_error, exc = await asyncio.to_thread(_safe_call, handler, arguments)
    except asyncio.CancelledError:
        raise
    except Exception as call_exc:
        tb = traceback.format_exc()
        _log.critical("Unexpected exception in call_tool for %s: %s\n%s", name, call_exc, tb)
        result = _json_text(_format_error(call_exc, tb))
        is_error = True
        exc = call_exc

    duration_ms = (time.monotonic() - t0) * 1000

    if _tracer is not None:
        try:
            # Extract the raw result dict from TextContent for the trace
            raw = json.loads(result[0].text) if result else None
            error_has_error_key = isinstance(raw, dict) and "error" in raw
            # Trace errors with the full structured error dict
            trace_error = raw if error_has_error_key else (str(exc) if exc else None)
            _tracer.record(
                tool=name,
                arguments=arguments,
                result=raw,
                duration_ms=duration_ms,
                status="error" if (is_error or error_has_error_key) else "ok",
                error=trace_error,
                identifiers=_extract_trace_identifiers(name, arguments, raw),
            )
        except Exception as trace_exc:
            # Log but never let tracing break tool execution
            _log.warning("Tracing failed for %s: %s", name, trace_exc)

    # Log all errors to stderr for visibility in agent output
    if is_error:
        _log.error("Tool %s failed in %.0fms", name, duration_ms)

    return result


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _init_tracer() -> None:
    """Initialize the session tracer if CRUCIBLE_TRACE=1 is set."""
    global _tracer
    if os.environ.get("CRUCIBLE_TRACE") != "1":
        return

    try:
        from crucible.mcp.tracer import SessionTracer
    except Exception as exc:
        _log.warning("Failed to import SessionTracer (tracing disabled): %s", exc)
        return

    # Determine trace directory: .crucible/traces/ under the project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    trace_dir = project_root / ".crucible" / "traces"
    session_id = os.environ.get("CRUCIBLE_TRACE_ID") or None
    try:
        _tracer = SessionTracer(trace_dir, session_id=session_id)
        _log.info("Tracing enabled (session=%s, dir=%s)", session_id, trace_dir)
    except Exception as exc:
        _log.warning("Failed to initialize SessionTracer: %s", exc)
        return

    def _finalize_tracer() -> None:
        try:
            if _tracer is not None:
                _tracer.finalize()
        except Exception as exc:
            _log.warning("Failed to finalize tracer: %s", exc)

    atexit.register(_finalize_tracer)


async def _run_server() -> None:
    _init_tracer()

    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    except KeyboardInterrupt:
        _log.info("MCP server interrupted by user")
    except Exception as exc:
        tb = traceback.format_exc()
        _log.critical("MCP server crashed: %s\n%s", exc, tb)
        # Write crash info to a file for debugging
        _write_crash_report(exc, tb)
        raise


def _write_crash_report(exc: BaseException, tb: str) -> None:
    """Write crash information to a file so it can be retrieved later."""
    try:
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        crash_file = project_root / ".crucible" / "mcp_crash.log"
        crash_file.parent.mkdir(parents=True, exist_ok=True)
        import socket
        crash_file.write_text(
            f"MCP server crash at {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
            f"Hostname: {socket.gethostname()}\n"
            f"Exception: {type(exc).__name__}: {exc}\n"
            f"Module: {type(exc).__module__}\n"
            f"\nTraceback:\n{tb}\n",
            encoding="utf-8",
        )
    except Exception:
        pass  # Never let crash reporting itself crash


def main_cli() -> None:
    """Entry point for crucible-mcp console script."""
    try:
        asyncio.run(_run_server())
    except KeyboardInterrupt:
        pass  # Already handled above
    except Exception as exc:
        tb = traceback.format_exc()
        _log.critical("MCP server fatal error: %s\n%s", exc, tb)
        _write_crash_report(exc, tb)
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
