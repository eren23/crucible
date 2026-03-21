"""MCP tool implementations for Crucible fleet operations."""
from __future__ import annotations

import json
from typing import Any

from crucible.core.config import ProjectConfig, load_config
from crucible.core.errors import CrucibleError
from crucible.core.io import read_jsonl
from crucible.core.log import utc_now_iso


def _get_config() -> ProjectConfig:
    return load_config()


def get_fleet_status(args: dict[str, Any]) -> dict[str, Any]:
    """Node inventory, health summary, and current assignments."""
    config = _get_config()
    try:
        from crucible.fleet.inventory import load_nodes, summarize_nodes

        nodes = load_nodes(config.project_root / config.nodes_file)
        summary = summarize_nodes(nodes)
        node_details = [
            {
                "name": n.get("name"),
                "node_id": n.get("node_id"),
                "state": n.get("state"),
                "gpu": n.get("gpu"),
                "ssh_host": n.get("ssh_host"),
                "env_ready": n.get("env_ready", False),
                "dataset_ready": n.get("dataset_ready", False),
            }
            for n in nodes
        ]
        return {"summary": summary, "nodes": node_details}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}", "nodes": []}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}", "nodes": []}


def get_leaderboard(args: dict[str, Any]) -> dict[str, Any]:
    """Top N experiment results sorted by primary metric."""
    config = _get_config()
    top_n = args.get("top_n", 20)
    primary = config.metrics.primary
    secondary = config.metrics.secondary or ""
    try:
        from crucible.analysis.leaderboard import leaderboard
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        top = leaderboard(results, top_n=top_n, cfg=config)
        entries = []
        for i, r in enumerate(top, 1):
            res = r.get("result", {})
            entry: dict[str, Any] = {
                "rank": i,
                "name": r.get("name", ""),
                "primary_metric": primary,
                primary: res.get(primary),
                "steps_completed": res.get("steps_completed"),
                "model_bytes": r.get("model_bytes"),
            }
            if secondary:
                entry[secondary] = res.get(secondary)
            entries.append(entry)
        return {"total_completed": len(results), "primary_metric": primary, "top": entries}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}", "top": []}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}", "top": []}


def get_queue_status(args: dict[str, Any]) -> dict[str, Any]:
    """Fleet queue state: counts of queued, running, and completed experiments."""
    config = _get_config()
    try:
        from crucible.fleet.queue import load_queue, summarize_queue

        rows = load_queue(config.project_root / config.fleet_results_file)
        summary = summarize_queue(rows)
        return {"total": len(rows), "summary": summary}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def enqueue_experiment(args: dict[str, Any]) -> dict[str, Any]:
    """Add an experiment configuration to the fleet queue."""
    config = _get_config()
    try:
        from crucible.fleet.queue import enqueue_experiments

        experiment = {
            "name": args["name"],
            "config": args["config"],
            "tier": args.get("tier", "proxy"),
            "backend": args.get("backend", "torch"),
            "tags": args.get("tags", []),
        }
        added = enqueue_experiments(
            config.project_root / "fleet_queue.jsonl",
            [experiment],
            limit=1,
        )
        if added:
            return {"status": "enqueued", "run_id": added[0]["run_id"], "item": added[0]}
        return {"status": "skipped", "reason": "Experiment with same name and tier already exists."}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def get_experiment_result(args: dict[str, Any]) -> dict[str, Any]:
    """Get the result for a specific experiment run_id."""
    config = _get_config()
    run_id = args["run_id"]
    try:
        from crucible.analysis.results import merged_results

        for row in merged_results(config):
            if row.get("id") == run_id or row.get("run_id") == run_id:
                return {"found": True, "result": row}
        return {"found": False, "run_id": run_id}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def provision_nodes(args: dict[str, Any]) -> dict[str, Any]:
    """Create N new compute nodes."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        new_nodes = fleet.provision(
            count=args.get("count", 2),
            name_prefix=args.get("name_prefix", "crucible"),
        )
        return {
            "created": len(new_nodes),
            "new_nodes": [{"name": n.get("name"), "node_id": n.get("node_id")} for n in new_nodes],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def destroy_nodes(args: dict[str, Any]) -> dict[str, Any]:
    """Tear down tracked nodes."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        node_names = args.get("node_names") or None
        fleet.destroy(node_names=node_names)
        return {"destroyed": node_names or "all", "status": "ok"}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def sync_code(args: dict[str, Any]) -> dict[str, Any]:
    """Push local code to nodes via rsync."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fleet = FleetManager(config)
        # Simple sync implementation
        from crucible.fleet.inventory import load_nodes
        from crucible.fleet.sync import sync_repo

        nodes = load_nodes(config.project_root / config.nodes_file)
        node_names = args.get("node_names")
        selected = set(node_names) if node_names else None
        synced = []
        errors = []
        for node in nodes:
            if selected and node["name"] not in selected:
                continue
            try:
                sync_repo(node, config.project_root, config.sync_excludes)
                synced.append(node["name"])
            except Exception as exc:
                errors.append({"node": node["name"], "error": str(exc)})
        return {"synced": synced, "errors": errors}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def get_research_state(args: dict[str, Any]) -> dict[str, Any]:
    """Current research state: hypotheses, beliefs, and budget info."""
    config = _get_config()
    state_path = config.project_root / config.research_state_file
    if not state_path.exists():
        return {"available": False}
    try:
        from crucible.researcher.state import ResearchState

        state = ResearchState(state_path)
        return {
            "available": True,
            "hypotheses_count": len(state.hypotheses),
            "history_count": len(state.history),
            "beliefs": state.beliefs,
            "budget_remaining": state.budget_remaining,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def get_sensitivity(args: dict[str, Any]) -> dict[str, Any]:
    """Parameter sensitivity analysis."""
    config = _get_config()
    try:
        from crucible.analysis.leaderboard import sensitivity_analysis
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        sens = sensitivity_analysis(results, cfg=config)
        return {"parameters": {k: v for k, v in list(sens.items())[:20]}}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Design tools
# ---------------------------------------------------------------------------


def _get_store():
    """Lazy-load a VersionStore from project config."""
    from crucible.core.store import VersionStore

    config = _get_config()
    store_dir = config.project_root / config.store_dir
    return VersionStore(store_dir)


def design_browse_experiments(args: dict[str, Any]) -> dict[str, Any]:
    """Browse completed experiments with filtering."""
    config = _get_config()
    try:
        from crucible.analysis.results import completed_results

        results = completed_results(config)
        primary = config.metrics.primary

        # Apply filters
        name_pattern = args.get("name_pattern", "")
        family = args.get("family", "")
        tag = args.get("tag", "")
        metric_below = args.get("metric_below")
        metric_above = args.get("metric_above")
        config_filter = args.get("config_filter", {})
        limit = args.get("limit", 50)
        sort_by = args.get("sort_by", "metric")

        filtered = []
        for r in results:
            if name_pattern and name_pattern not in r.get("name", ""):
                continue
            if family and r.get("config", {}).get("MODEL_FAMILY", "") != family:
                continue
            if tag and tag not in r.get("tags", []):
                continue
            metric_val = r.get("result", {}).get(primary)
            if metric_below is not None and isinstance(metric_val, (int, float)) and metric_val >= metric_below:
                continue
            if metric_above is not None and isinstance(metric_val, (int, float)) and metric_val <= metric_above:
                continue
            if config_filter:
                exp_config = r.get("config", {})
                if not all(exp_config.get(k) == v for k, v in config_filter.items()):
                    continue
            filtered.append(r)

        # Sort
        if sort_by == "metric":
            filtered.sort(key=lambda r: r.get("result", {}).get(primary, float("inf")))
        elif sort_by == "name":
            filtered.sort(key=lambda r: r.get("name", ""))
        elif sort_by == "timestamp":
            filtered.sort(key=lambda r: r.get("timestamp", ""), reverse=True)

        # Trim and format
        trimmed = []
        for r in filtered[:limit]:
            trimmed.append({
                "name": r.get("name"),
                "config": r.get("config", {}),
                primary: r.get("result", {}).get(primary),
                "model_bytes": r.get("model_bytes"),
                "tags": r.get("tags", []),
                "status": r.get("status"),
                "timestamp": r.get("timestamp"),
            })
        return {"total_matched": len(filtered), "experiments": trimmed}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def design_compare_experiments(args: dict[str, Any]) -> dict[str, Any]:
    """Side-by-side comparison of 2-5 experiments."""
    config = _get_config()
    names = args.get("experiment_names", [])
    if len(names) < 2 or len(names) > 5:
        return {"error": "Provide 2-5 experiment names."}
    try:
        from crucible.analysis.results import merged_results

        all_results = merged_results(config)
        by_name = {r["name"]: r for r in all_results if r.get("name") in names}
        missing = [n for n in names if n not in by_name]
        if missing:
            return {"error": f"Experiments not found: {missing}"}

        primary = config.metrics.primary
        experiments = [by_name[n] for n in names]

        # Config diffs: find keys that differ
        all_keys = set()
        for exp in experiments:
            all_keys.update(exp.get("config", {}).keys())

        config_diff = {}
        for key in sorted(all_keys):
            values = [exp.get("config", {}).get(key, "<not set>") for exp in experiments]
            if len(set(values)) > 1:
                config_diff[key] = dict(zip(names, values))

        # Metric comparison
        metric_comparison = {}
        for name in names:
            exp = by_name[name]
            res = exp.get("result", {})
            metric_comparison[name] = {
                primary: res.get(primary),
                "model_bytes": exp.get("model_bytes"),
                "status": exp.get("status"),
            }

        return {
            "experiments": names,
            "config_diff": config_diff,
            "config_same_keys": sorted(all_keys - set(config_diff.keys())),
            "metrics": metric_comparison,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def design_generate_hypotheses(args: dict[str, Any]) -> dict[str, Any]:
    """Generate LLM-driven experiment hypotheses with agent-provided context."""
    config = _get_config()
    try:
        from crucible.researcher.analysis import build_analysis
        from crucible.researcher.hypothesis import generate_hypotheses
        from crucible.researcher.llm_client import AnthropicClient
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path)

        analysis = build_analysis(config, state)
        extra_context = args.get("extra_context", "")
        if extra_context:
            analysis += f"\n\n## Additional Agent Context\n{extra_context}"

        focus = args.get("focus_family", "")
        if focus:
            analysis += f"\n\n## Focus Area\nFocus on the '{focus}' model family."

        program_path = config.project_root / config.researcher.program_file
        program_text = program_path.read_text(encoding="utf-8") if program_path.exists() else ""

        llm = AnthropicClient(model=config.researcher.model)
        hypotheses = generate_hypotheses(analysis, program_text, state, llm, iteration=0)
        state.save()

        max_hyp = args.get("max_hypotheses", 5)
        return {
            "hypotheses": hypotheses[:max_hyp],
            "total_generated": len(hypotheses),
            "analysis_summary": analysis[:500] + "..." if len(analysis) > 500 else analysis,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def design_batch_from_hypotheses(args: dict[str, Any]) -> dict[str, Any]:
    """Convert hypotheses to an executable experiment batch."""
    config = _get_config()
    try:
        from crucible.researcher.batch_design import design_batch
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path)

        hypotheses = args.get("hypotheses", [])
        tier = args.get("tier", "proxy")
        backend = args.get("backend", "torch")
        include_baseline = args.get("include_baseline", True)

        baseline_config = None
        if include_baseline:
            try:
                from crucible.analysis.leaderboard import leaderboard
                from crucible.analysis.results import completed_results

                results = completed_results(config)
                top = leaderboard(results, top_n=1, cfg=config)
                if top:
                    baseline_config = top[0].get("config", {})
            except ImportError:
                pass

        batch = design_batch(
            hypotheses, state, tier, backend, iteration=0,
            baseline_config=baseline_config if include_baseline else None,
        )
        return {"batch": batch, "batch_size": len(batch)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def design_enqueue_batch(args: dict[str, Any]) -> dict[str, Any]:
    """Enqueue a batch of experiment configs to the fleet queue."""
    config = _get_config()
    try:
        from crucible.fleet.queue import enqueue_experiments

        batch = args.get("batch", [])
        wave_name = args.get("wave_name", "")
        if not wave_name:
            wave_name = f"agent_{utc_now_iso()[:19].replace(':', '').replace('-', '')}"

        experiments = []
        for exp in batch:
            exp.setdefault("wave", wave_name)
            experiments.append(exp)

        added = enqueue_experiments(
            config.project_root / "fleet_queue.jsonl",
            experiments,
            limit=0,
        )
        return {
            "enqueued": len(added),
            "wave_name": wave_name,
            "run_ids": [item["run_id"] for item in added],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Context tools
# ---------------------------------------------------------------------------


def context_get_analysis(args: dict[str, Any]) -> dict[str, Any]:
    """Full structured analysis of experiment results."""
    config = _get_config()
    try:
        from crucible.researcher.analysis import build_analysis_structured
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path)
        return build_analysis_structured(config, state)
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def context_push_finding(args: dict[str, Any]) -> dict[str, Any]:
    """Record a research finding in the context store."""
    config = _get_config()
    try:
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path)

        entry = state.add_finding(
            finding=args["finding"],
            category=args.get("category", "observation"),
            source_experiments=args.get("source_experiments", []),
            confidence=args.get("confidence", 0.7),
            created_by=args.get("created_by", "mcp-agent"),
        )
        state.save()

        return {"status": "recorded", "entry": entry}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def context_get_findings(args: dict[str, Any]) -> dict[str, Any]:
    """Query accumulated research findings."""
    config = _get_config()
    try:
        from crucible.researcher.state import ResearchState

        state_path = config.project_root / config.research_state_file
        state = ResearchState(state_path)

        category = args.get("category", "")
        limit = args.get("limit", 50)
        findings = state.get_findings(category=category or None, limit=limit)
        return {"findings": findings, "total": len(findings)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Version tools
# ---------------------------------------------------------------------------


def version_save_design(args: dict[str, Any]) -> dict[str, Any]:
    """Save or update a versioned experiment design.

    Supports partial updates: if a design already exists, only fields
    present in args are overwritten. Unspecified fields keep their
    previous values.
    """
    try:
        import re

        store = _get_store()
        name = args["name"]

        # Validate name is a slug
        if not re.match(r'^[a-z0-9][a-z0-9_-]*$', name):
            return {"error": f"Invalid design name '{name}'. Use lowercase letters, numbers, hyphens, underscores. Must start with letter or number."}

        # Start from existing content if this is an update
        current = store.get_current("experiment_design", name)
        if current is not None:
            _, prev_content = current
            content = dict(prev_content)
        else:
            # Defaults for new designs
            content = {
                "name": name,
                "description": "",
                "hypothesis": "",
                "config": {},
                "base_preset": "proxy",
                "backend": "torch",
                "tags": [],
                "family": "",
                "status": "draft",
                "linked_run_ids": [],
                "parent_design": None,
                "rationale": "",
            }

        # Override only fields explicitly provided in args
        # (name is always set, linked_run_ids is managed internally)
        _UPDATABLE_FIELDS = [
            "description", "hypothesis", "config", "base_preset", "backend",
            "tags", "family", "status", "parent_design", "rationale",
        ]
        for field in _UPDATABLE_FIELDS:
            if field in args:
                content[field] = args[field]
        content["name"] = name  # always ensure name matches

        meta = store.create(
            "experiment_design",
            args["name"],
            content,
            summary=args.get("summary", f"Design: {args['name']}"),
            created_by=args.get("created_by", "mcp-agent"),
            tags=args.get("tags", []),
        )

        # Auto-commit if configured
        config = _get_config()
        if config.auto_commit_versions:
            store.git_commit_version(meta)

        return {"status": "saved", "version_meta": meta}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def version_list_designs(args: dict[str, Any]) -> dict[str, Any]:
    """List all versioned experiment designs."""
    try:
        store = _get_store()

        status_filter = args.get("status_filter")
        tag_filter = args.get("tag_filter")

        resources = store.list_resources(
            "experiment_design",
            status=status_filter,
            tag=tag_filter,
        )

        # Enrich with current content status
        designs = []
        for meta in resources:
            entry = dict(meta)
            current = store.get_current("experiment_design", meta["resource_name"])
            if current:
                _, content = current
                entry["design_status"] = content.get("status", "unknown")
                entry["family"] = content.get("family", "")
                entry["base_preset"] = content.get("base_preset", "")
            designs.append(entry)

        return {"designs": designs, "total": len(designs)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def version_diff(args: dict[str, Any]) -> dict[str, Any]:
    """Compare two versions of a design."""
    try:
        store = _get_store()

        resource_name = args["resource_name"]
        va = args["version_a"]
        vb = args["version_b"]

        result_a = store.get_version_number("experiment_design", resource_name, va)
        result_b = store.get_version_number("experiment_design", resource_name, vb)

        if result_a is None:
            return {"error": f"Version {va} not found for {resource_name}"}
        if result_b is None:
            return {"error": f"Version {vb} not found for {resource_name}"}

        meta_a, content_a = result_a
        meta_b, content_b = result_b

        # Compute diffs
        all_keys = set(content_a.keys()) | set(content_b.keys())
        changes = {}
        for key in sorted(all_keys):
            val_a = content_a.get(key)
            val_b = content_b.get(key)
            if val_a != val_b:
                changes[key] = {f"v{va}": val_a, f"v{vb}": val_b}

        return {
            "resource_name": resource_name,
            "version_a": {"version": va, "created_at": meta_a.get("created_at"), "summary": meta_a.get("summary")},
            "version_b": {"version": vb, "created_at": meta_b.get("created_at"), "summary": meta_b.get("summary")},
            "changes": changes,
            "unchanged_keys": sorted(all_keys - set(changes.keys())),
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def version_get_design(args: dict[str, Any]) -> dict[str, Any]:
    """Get full content and metadata for a versioned design."""
    try:
        store = _get_store()
        design_name = args["design_name"]
        version = args.get("version")

        if version is not None:
            result = store.get_version_number("experiment_design", design_name, version)
        else:
            result = store.get_current("experiment_design", design_name)

        if result is None:
            return {"error": f"Design '{design_name}' not found" + (f" at version {version}" if version else "")}

        meta, content = result
        return {"version_meta": meta, "design": content}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def version_run_design(args: dict[str, Any]) -> dict[str, Any]:
    """Execute a versioned design by converting it to an ExperimentConfig and enqueuing."""
    config = _get_config()
    try:
        from crucible.fleet.queue import enqueue_experiments
        from crucible.runner.design import design_to_experiment_config

        store = _get_store()
        design_name = args["design_name"]

        result = store.get_current("experiment_design", design_name)
        if result is None:
            return {"error": f"Design '{design_name}' not found"}

        meta, content = result

        # Apply tier/backend overrides if provided
        if "tier" in args:
            content["base_preset"] = args["tier"]
        if "backend" in args:
            content["backend"] = args["backend"]

        exp_config = design_to_experiment_config(content, meta)

        # Enqueue the experiment
        experiment = {
            "name": exp_config["name"],
            "config": exp_config["config"],
            "tier": exp_config.get("tier", "proxy"),
            "backend": exp_config.get("backend", "torch"),
            "tags": exp_config.get("tags", []),
        }
        added = enqueue_experiments(
            config.project_root / "fleet_queue.jsonl",
            [experiment],
            limit=1,
        )

        if not added:
            return {"error": "Experiment already enqueued (duplicate name+tier)."}

        run_id = added[0]["run_id"]

        # Update design status to "running" and link run_id
        content["status"] = "running"
        linked = list(content.get("linked_run_ids", []))
        linked.append(run_id)
        content["linked_run_ids"] = linked

        new_meta = store.create(
            "experiment_design", design_name, content,
            summary=f"Running as {run_id}",
            created_by=meta.get("created_by", "mcp-agent"),
            tags=meta.get("tags", []),
        )

        if config.auto_commit_versions:
            store.git_commit_version(new_meta)

        return {
            "status": "enqueued",
            "run_id": run_id,
            "version_meta": new_meta,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def version_link_result(args: dict[str, Any]) -> dict[str, Any]:
    """Link a completed experiment run_id back to a design."""
    try:
        from crucible.runner.design import link_result_to_design

        store = _get_store()
        design_name = args["design_name"]
        run_id = args["run_id"]

        new_meta = link_result_to_design(store, design_name, run_id)
        if new_meta is None:
            return {"error": f"Design '{design_name}' not found"}

        config = _get_config()
        if config.auto_commit_versions:
            store.git_commit_version(new_meta)

        return {"status": "linked", "version_meta": new_meta}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Config tools
# ---------------------------------------------------------------------------


def config_get_presets(args: dict[str, Any]) -> dict[str, Any]:
    """All presets with resolved config values."""
    config = _get_config()
    try:
        from crucible.runner.presets import get_preset, list_presets

        preset_name = args.get("preset_name", "")
        if preset_name:
            resolved = get_preset(preset_name, project_config=config)
            return {"preset": preset_name, "config": resolved}
        else:
            all_presets = {}
            for name in list_presets(project_config=config):
                all_presets[name] = get_preset(name, project_config=config)
            return {"presets": all_presets}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def config_get_project(args: dict[str, Any]) -> dict[str, Any]:
    """Full project configuration."""
    config = _get_config()
    try:
        return {
            "name": config.name,
            "version": config.version,
            "project_root": str(config.project_root),
            "provider": {
                "type": config.provider.type,
                "gpu_types": config.provider.gpu_types,
            },
            "metrics": {
                "primary": config.metrics.primary,
                "secondary": config.metrics.secondary,
                "size": config.metrics.size,
                "direction": config.metrics.direction,
            },
            "researcher": {
                "model": config.researcher.model,
                "budget_hours": config.researcher.budget_hours,
                "max_iterations": config.researcher.max_iterations,
            },
            "training": [
                {"backend": t.backend, "script": t.script}
                for t in config.training
            ],
            "data": {
                "source": config.data.source,
                "repo_id": config.data.repo_id,
                "local_root": config.data.local_root,
            },
            "store_dir": config.store_dir,
            "auto_commit_versions": config.auto_commit_versions,
            "results_file": config.results_file,
            "fleet_results_file": config.fleet_results_file,
            "logs_dir": config.logs_dir,
            "timeout_map": config.timeout_map,
            "research_state_file": config.research_state_file,
        }
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

TOOL_DISPATCH: dict[str, Any] = {
    # Existing tools
    "get_fleet_status": get_fleet_status,
    "get_leaderboard": get_leaderboard,
    "get_queue_status": get_queue_status,
    "enqueue_experiment": enqueue_experiment,
    "get_experiment_result": get_experiment_result,
    "provision_nodes": provision_nodes,
    "destroy_nodes": destroy_nodes,
    "sync_code": sync_code,
    "get_research_state": get_research_state,
    "get_sensitivity": get_sensitivity,
    # Design tools
    "design_browse_experiments": design_browse_experiments,
    "design_compare_experiments": design_compare_experiments,
    "design_generate_hypotheses": design_generate_hypotheses,
    "design_batch_from_hypotheses": design_batch_from_hypotheses,
    "design_enqueue_batch": design_enqueue_batch,
    # Context tools
    "context_get_analysis": context_get_analysis,
    "context_push_finding": context_push_finding,
    "context_get_findings": context_get_findings,
    # Version tools
    "version_save_design": version_save_design,
    "version_list_designs": version_list_designs,
    "version_diff": version_diff,
    "version_get_design": version_get_design,
    "version_run_design": version_run_design,
    "version_link_result": version_link_result,
    # Config tools
    "config_get_presets": config_get_presets,
    "config_get_project": config_get_project,
}
