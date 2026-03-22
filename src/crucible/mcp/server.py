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
        description="Node inventory, health summary, and current assignments.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="get_leaderboard",
        description="Top N experiment results sorted by primary metric (lower is better).",
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
        description="Fleet queue state: counts of queued, running, and completed experiments.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="enqueue_experiment",
        description="Add an experiment configuration to the fleet queue.",
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
        description="Get the result for a specific experiment run_id.",
        inputSchema={
            "type": "object",
            "properties": {"run_id": {"type": "string", "description": "The unique run identifier."}},
            "required": ["run_id"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="provision_nodes",
        description="Create N new compute nodes.",
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
        description="Tear down tracked nodes. Optionally specify node names.",
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
        description="Push local code to nodes via rsync.",
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
        description="Refresh node states from cloud provider API. Updates SSH hosts, GPU info, and node state. Run after provision_nodes to get SSH connection info.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="bootstrap_nodes",
        description="Bootstrap fleet nodes: sync code, install dependencies, download training data. Run after provision_nodes + fleet_refresh.",
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
        description="Dispatch queued experiments to idle bootstrapped nodes. Assigns one experiment per available node. Run after bootstrap_nodes + enqueue_experiment.",
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
        description="Collect experiment results from all fleet nodes via rsync and merge into fleet results file. Run after experiments complete.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="get_research_state",
        description="Current research state: hypotheses, beliefs, budget.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="get_sensitivity",
        description="Parameter sensitivity analysis across completed experiments.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    # -----------------------------------------------------------------------
    # Design tools
    # -----------------------------------------------------------------------
    Tool(
        name="design_browse_experiments",
        description="Browse completed experiments with filtering by name, family, tag, metric range, and config values.",
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
        description="Side-by-side comparison of 2-5 experiments: config diffs and metric differences.",
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
        description="Generate LLM-driven experiment hypotheses from current results and optional agent context.",
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
        description="Convert hypotheses to an executable experiment batch with optional baseline control.",
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
        description="Enqueue a batch of experiment configs to the fleet queue. Returns enqueued run IDs.",
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
        description="Full structured analysis: leaderboard, family breakdown, sensitivity, beliefs, and research state.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="context_push_finding",
        description="Record a research finding or observation. Persists across sessions and informs future hypothesis generation.",
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
        description="Query accumulated research findings, optionally filtered by category.",
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
        description="Save or update a versioned experiment design. Creates a new version each time.",
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
        description="List all versioned experiment designs with metadata and status.",
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
        description="Compare two versions of an experiment design showing what changed.",
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
        description="Get full content and metadata for a versioned experiment design.",
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
        description="Execute a versioned design: converts to ExperimentConfig, enqueues to fleet, updates design status to running.",
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
        description="Link a completed experiment run_id back to a versioned design, updating its linked_run_ids.",
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
        description="Attach a freeform markdown note to an experiment run. Use for observations, analysis, or hypotheses about a specific run.",
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
        description="Get all notes for a specific experiment run, optionally filtered by stage.",
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
        description="Search notes across all runs by text query, tags, stage, or run_id.",
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
        description="Upload an image file to a W&B run. Requires the run to have a W&B URL in its status sidecar.",
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
        description="Get the W&B dashboard URL for a Crucible experiment run.",
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
        description="Push a note or finding annotation to a W&B run summary.",
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
        description="Hub info: initialization state, active track, linked projects, and track summaries.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="hub_sync",
        description="Git-sync the hub: stage, commit, pull, and push to remote.",
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
        description="Create a new research track for grouping related experiments across projects.",
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
        description="List all research tracks with their metadata and active status.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="track_switch",
        description="Switch the active research track. The active track is used as default context.",
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
        description="Query findings across hub scopes (track or global) with optional filters.",
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
        description="Promote a finding from track scope to global scope.",
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
        description="Comprehensive session orientation: project state, leaderboard, hypotheses, findings, notes, and suggested next steps.",
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
        description="Bidirectional link: attach a finding to a run and record the run in the finding's source_experiments.",
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
        description="List all registered model architecture families (e.g. baseline, looped, convloop, prefix_memory).",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="model_list_activations",
        description="List all available activation functions (e.g. relu_sq, gelu_sq, mish_sq).",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="model_list_components",
        description="List all available model building-block components (attention, MLP, norm, etc.).",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    Tool(
        name="model_get_config_schema",
        description="Get the accepted configuration parameters for a model family.",
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
        description="Pre-flight validation of an experiment config: checks MODEL_FAMILY and ACTIVATION are valid.",
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
        description="Write and register a new architecture family at runtime. Code must call register_model().",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Family name (snake_case)."},
                "code": {"type": "string", "description": "Full Python source that defines and registers the architecture."},
            },
            "required": ["name", "code"],
            "additionalProperties": False,
        },
    ),
    Tool(
        name="model_add_activation",
        description="Register a new activation function at runtime. Provide a Python expression using 'x' with torch/F available.",
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
        description="Generate boilerplate Python code for a new architecture family.",
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
    # Config tools
    # -----------------------------------------------------------------------
    Tool(
        name="config_get_presets",
        description="List all experiment presets with resolved configuration values (built-in defaults merged with crucible.yaml).",
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
        description="Full resolved project configuration: metrics, researcher settings, training backends, presets, provider.",
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
