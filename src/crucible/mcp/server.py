"""MCP server exposing Crucible fleet operations as tools for Claude agents.

Run via stdio:
    crucible mcp serve
    python -m crucible.mcp.server
"""
from __future__ import annotations

import asyncio
import json
import traceback
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from crucible.core.env import load_env_files
from crucible.mcp.tools import TOOL_DISPATCH

# Load .env files so secrets (RUNPOD_API_KEY, WANDB_API_KEY) are available to tools
load_env_files(Path(__file__).resolve().parent.parent.parent.parent)

app = Server("crucible-fleet")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_text(obj: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(obj, indent=2, default=str))]


def _error_text(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}))]


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> list[TextContent]:
    try:
        result = fn(*args, **kwargs)
        return _json_text(result)
    except Exception:
        return _error_text(traceback.format_exc())


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
            "REQUIRES: Nothing. Skips if same name+tier already queued.\n"
            "RETURNS: {status: 'enqueued'|'skipped', run_id}\n"
            "NEXT: dispatch_experiments to assign to nodes."
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
            "NEXT: fleet_refresh (wait ~60s for SSH), then bootstrap_nodes."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "count": {"type": "integer", "description": "Number of nodes to create.", "default": 2},
                "name_prefix": {"type": "string", "description": "Node name prefix.", "default": "crucible"},
            },
            "required": ["count"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="destroy_nodes",
        description=(
            "Tear down tracked nodes. Optionally specify node names.\n\n"
            "REQUIRES: Nodes must exist in inventory.\n"
            "RETURNS: {destroyed, status}\n"
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
            },
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
            "REQUIRES: Nodes with SSH hosts (run fleet_refresh first after provision_nodes).\n"
            "RETURNS: {total, bootstrapped, nodes: [{name, state, env_ready, dataset_ready}]}\n"
            "NEXT: enqueue_experiment or design_enqueue_batch, then dispatch_experiments."
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
            "REQUIRES: Bootstrapped nodes (env_ready=true) + queued experiments.\n"
            "RETURNS: {dispatched, assignments: [{node, experiment}]}\n"
            "NEXT: get_queue_status to monitor, collect_results when done."
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
            "REQUIRES: Batch of experiments (from design_batch_from_hypotheses or manually built).\n"
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
            "REQUIRES: Design name from version_list_designs.\n"
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
            "Promote a finding from track scope to global scope (or track to track).\n\n"
            "REQUIRES: Finding ID from hub_findings_query.\n"
            "RETURNS: {status, finding_id, from_scope, to_scope}\n"
            "NEXT: hub_sync to persist, hub_findings_query to verify."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "finding_id": {"type": "string", "description": "The finding ID to promote."},
                "from_scope": {"type": "string", "description": "Source scope (e.g. 'track')."},
                "to_scope": {"type": "string", "description": "Destination scope (e.g. 'global')."},
                "from_track": {"type": "string", "description": "Source track name (if from_scope='track')."},
                "to_track": {"type": "string", "description": "Destination track name (if to_scope='track')."},
            },
            "required": ["finding_id", "from_scope", "to_scope"],
            "additionalProperties": False,
        },
    ),
    # -----------------------------------------------------------------------
    # Briefing tools
    # -----------------------------------------------------------------------
    Tool(
        name="get_research_briefing",
        description=(
            "Comprehensive session orientation: project state, leaderboard, hypotheses, findings, notes, next steps.\n\n"
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
                "name": {"type": "string", "description": "Plugin family name (must exist in user_architectures/)."},
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
            "Full resolved project configuration: metrics, researcher, training, presets, provider.\n\n"
            "REQUIRES: Nothing.\n"
            "RETURNS: {name, metrics, researcher, training, presets, provider, paths}\n"
            "NEXT: config_get_presets for preset details."
        ),
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
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
        description="List all external project specs in .crucible/projects/.\n\nREQUIRES: Nothing.\nRETURNS: {projects: [{name, repo, train, metrics_primary}]}\nNEXT: provision_project or bootstrap_project.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="provision_project",
        description="Provision nodes for an external project, applying pod overrides (GPU, image, disk) from the project spec.\n\nREQUIRES: RUNPOD_API_KEY, project spec in .crucible/projects/.\nRETURNS: {created, new_nodes: [{name, node_id}]}\nNEXT: fleet_refresh (wait ~60s), then bootstrap_project.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Name of the project spec (without .yaml)."},
                "count": {"type": "integer", "default": 1, "description": "Number of nodes."},
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
        description="Launch training for an external project as a detached process on bootstrapped nodes. Returns immediately with PID.\n\nREQUIRES: Nodes bootstrapped via bootstrap_project.\nRETURNS: {run_id, nodes: [{name, pid, status}]}\nNEXT: get_fleet_status(include_metrics=true) to monitor GPU, collect_project_results when done.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Name of the project spec."},
                "node_names": {"type": "array", "items": {"type": "string"}, "description": "Specific nodes. Empty = all ready."},
                "overrides": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Env var overrides for this run."},
            },
            "required": ["project_name"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="collect_project_results",
        description="Collect results from an external project run: rsync logs, parse metrics, fetch WandB data.\n\nREQUIRES: run_project has been called.\nRETURNS: {run_id, status, metrics, log_tail}\nNEXT: get_leaderboard if completed.",
        inputSchema={
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Run ID returned by run_project."},
            },
            "required": ["run_id"],
            "additionalProperties": False,
        },
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:  # type: ignore[type-arg]
    handler = TOOL_DISPATCH.get(name)
    if handler is None:
        return _error_text(f"Unknown tool: {name}")
    return await asyncio.to_thread(_safe_call, handler, arguments)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def _run_server() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main_cli() -> None:
    """Entry point for crucible-mcp console script."""
    asyncio.run(_run_server())


if __name__ == "__main__":
    main_cli()
