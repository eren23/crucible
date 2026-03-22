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
        selected = set(node_names) if node_names else None
        fleet.destroy(selected_names=selected)
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


def fleet_refresh(args: dict[str, Any]) -> dict[str, Any]:
    """Refresh node states from the cloud provider API (updates SSH hosts, GPU info, state)."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(config)
        nodes = fm.refresh()
        return {
            "refreshed": len(nodes),
            "nodes": [
                {
                    "name": n.get("name"),
                    "state": n.get("state"),
                    "ssh_host": n.get("ssh_host"),
                    "ssh_port": n.get("ssh_port", 22),
                    "gpu": n.get("gpu"),
                    "env_ready": n.get("env_ready", False),
                    "dataset_ready": n.get("dataset_ready", False),
                }
                for n in nodes
            ],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def bootstrap_nodes(args: dict[str, Any]) -> dict[str, Any]:
    """Bootstrap fleet nodes: sync code, install deps, download data. Run after provision_nodes."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(config)
        train_shards = args.get("train_shards", 1)
        skip_install = args.get("skip_install", False)
        skip_data = args.get("skip_data", False)
        node_names = args.get("node_names")
        selected = set(node_names) if node_names else None

        nodes = fm.bootstrap(
            train_shards=train_shards,
            skip_install=skip_install,
            skip_data=skip_data,
            selected_names=selected,
        )
        bootstrapped = [n for n in nodes if n.get("state") == "bootstrapped"]
        return {
            "total": len(nodes),
            "bootstrapped": len(bootstrapped),
            "nodes": [
                {"name": n.get("name"), "state": n.get("state"), "env_ready": n.get("env_ready"), "dataset_ready": n.get("dataset_ready")}
                for n in nodes
            ],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def dispatch_experiments(args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch queued experiments to idle bootstrapped nodes. Run after bootstrap_nodes + enqueue."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager

        fm = FleetManager(config)
        max_assignments = args.get("max_assignments", 8)
        assignments = fm.dispatch(max_assignments=max_assignments)
        return {
            "dispatched": len(assignments),
            "assignments": [
                {"node": a.get("node_name", a.get("name", "")), "experiment": a.get("experiment_name", "")}
                for a in assignments
            ],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def collect_results(args: dict[str, Any]) -> dict[str, Any]:
    """Collect experiment results from all fleet nodes via rsync and merge into fleet results."""
    config = _get_config()
    try:
        from crucible.fleet.manager import FleetManager
        from crucible.analysis.results import merged_results

        fm = FleetManager(config)
        fm.collect()
        results = merged_results(config)
        completed = [r for r in results if r.get("status") == "completed"]
        return {
            "collected": True,
            "total_results": len(results),
            "completed": len(completed),
        }
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
        result = build_analysis_structured(config, state)

        # Inject hub findings if available
        try:
            from crucible.core.hub import HubStore

            hub = HubStore()
            if hub.initialized:
                active_track = hub.get_active_track() or config.active_track
                if active_track:
                    hub_findings = hub.load_context_for_track(
                        active_track, include_global=True, max_findings=20,
                    )
                    result["hub_findings"] = hub_findings
        except Exception:
            pass  # Hub is optional

        return result
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
# Note tools
# ---------------------------------------------------------------------------


def _get_note_store():
    """Lazy-load a NoteStore from project config."""
    from crucible.runner.notes import NoteStore

    config = _get_config()
    store_dir = config.project_root / config.store_dir
    return NoteStore(store_dir)


def note_add(args: dict[str, Any]) -> dict[str, Any]:
    """Attach a note to an experiment run."""
    try:
        store = _get_note_store()
        entry = store.add(
            run_id=args["run_id"],
            body=args["text"],
            stage=args.get("stage", ""),
            tags=args.get("tags", []),
            confidence=args.get("confidence"),
            created_by=args.get("created_by", "mcp-agent"),
        )
        return {"status": "added", "note": entry}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def note_get(args: dict[str, Any]) -> dict[str, Any]:
    """Get all notes for a run."""
    try:
        store = _get_note_store()
        run_id = args["run_id"]
        stage = args.get("stage", "")
        entries = store.get_for_run(run_id)
        if stage:
            entries = [e for e in entries if e.get("stage") == stage]
        return {"run_id": run_id, "notes": entries, "total": len(entries)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def note_search(args: dict[str, Any]) -> dict[str, Any]:
    """Search notes across runs."""
    try:
        store = _get_note_store()
        entries = store.search(
            query=args.get("query", ""),
            tags=args.get("tags"),
            stage=args.get("stage", ""),
            run_id=args.get("run_id", ""),
            limit=args.get("limit", 50),
        )
        return {"notes": entries, "total": len(entries)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# W&B tools
# ---------------------------------------------------------------------------


def wandb_log_image(args: dict[str, Any]) -> dict[str, Any]:
    """Log an image file to a W&B run."""
    config = _get_config()
    run_id = args["run_id"]
    image_path = args["image_path"]
    caption = args.get("caption", "")
    key = args.get("key", "image")

    try:
        from crucible.runner.wandb import _resolve_wandb_url

        wandb_url = _resolve_wandb_url(run_id, config)
        if not wandb_url:
            return {"error": f"No W&B URL found for run {run_id}"}

        try:
            import wandb  # type: ignore
        except ImportError:
            return {"error": "wandb not installed"}

        # Parse entity/project/run_id from URL
        parts = wandb_url.rstrip("/").split("/")
        runs_idx = parts.index("runs")
        wb_run_id = parts[runs_idx + 1]
        project = parts[runs_idx - 1]
        entity = parts[runs_idx - 2]

        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{wb_run_id}")
        run.upload_file(str(image_path))
        return {"status": "uploaded", "run_id": run_id, "wandb_url": wandb_url, "image_path": image_path}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def wandb_get_url(args: dict[str, Any]) -> dict[str, Any]:
    """Get W&B dashboard URL for a Crucible run."""
    config = _get_config()
    run_id = args["run_id"]
    try:
        from crucible.runner.wandb import _resolve_wandb_url

        url = _resolve_wandb_url(run_id, config)
        if url:
            return {"run_id": run_id, "wandb_url": url}
        return {"run_id": run_id, "wandb_url": None, "reason": "No W&B URL found in status sidecar or results."}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def wandb_annotate(args: dict[str, Any]) -> dict[str, Any]:
    """Push note/finding to W&B run summary."""
    config = _get_config()
    run_id = args["run_id"]
    text = args["text"]
    annotation_type = args.get("annotation_type", "note")

    try:
        from crucible.runner.wandb import _resolve_wandb_url, wandb_annotate_finished_run

        wandb_url = _resolve_wandb_url(run_id, config)
        if not wandb_url:
            return {"error": f"No W&B URL found for run {run_id}"}

        if annotation_type == "finding":
            ok = wandb_annotate_finished_run(wandb_url, findings=[text])
        else:
            ok = wandb_annotate_finished_run(wandb_url, notes=[text])

        if ok:
            return {"status": "annotated", "run_id": run_id, "annotation_type": annotation_type, "wandb_url": wandb_url}
        return {"error": f"Failed to annotate W&B run for {run_id}. wandb may not be installed or the run may be inaccessible."}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Hub tools
# ---------------------------------------------------------------------------


def _get_hub():
    """Lazy-load a HubStore, returning None if not initialized."""
    from crucible.core.hub import HubStore

    hub = HubStore()
    if not hub.initialized:
        return None
    return hub


def hub_status(args: dict[str, Any]) -> dict[str, Any]:
    """Hub info, active track, linked projects."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"initialized": False, "message": "Hub not initialized. Run hub init first."}

        active_track = hub.get_active_track()
        projects = hub.list_projects()
        tracks = hub.list_tracks()

        return {
            "initialized": True,
            "hub_dir": str(hub.hub_dir),
            "active_track": active_track,
            "projects": projects,
            "tracks_count": len(tracks),
            "tracks": [
                {"name": t.get("name"), "description": t.get("description", ""), "active": t.get("active", True)}
                for t in tracks
            ],
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def hub_sync(args: dict[str, Any]) -> dict[str, Any]:
    """Git sync the hub (push/pull/both)."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        remote = args.get("remote")
        result = hub.sync(remote=remote)
        return {"status": "synced", **result}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def track_create(args: dict[str, Any]) -> dict[str, Any]:
    """Create a new research track."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        name = args["name"]
        description = args.get("description", "")
        tags = args.get("tags", [])

        track = hub.create_track(name, description=description, tags=tags)
        return {"status": "created", "track": track}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def track_list(args: dict[str, Any]) -> dict[str, Any]:
    """List all research tracks."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        tracks = hub.list_tracks()
        active = hub.get_active_track()
        return {
            "active_track": active,
            "tracks": tracks,
            "total": len(tracks),
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def track_switch(args: dict[str, Any]) -> dict[str, Any]:
    """Switch the active research track."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        name = args["name"]
        hub.activate_track(name)
        return {"status": "switched", "active_track": name}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def hub_findings_query(args: dict[str, Any]) -> dict[str, Any]:
    """Query findings across hub scopes."""
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        scope = args.get("scope", "global")
        track = args.get("track")
        status = args.get("status")
        tags = args.get("tags")
        limit = args.get("limit", 50)

        findings = hub.list_findings(scope, track=track, status=status, tags=tags)
        return {"findings": findings[:limit], "total": len(findings)}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def _research_finding_to_hub_finding(finding: dict[str, Any], config: Any) -> dict[str, Any]:
    """Convert a ResearchState finding to hub-compatible Finding format."""
    from crucible.core.finding import make_finding_id

    return {
        "id": make_finding_id(finding.get("finding", "untitled")[:40], "project"),
        "title": finding.get("finding", "")[:80],
        "body": finding.get("finding", ""),
        "scope": "project",
        "status": "active",
        "confidence": finding.get("confidence", 0.5),
        "tags": [],
        "category": finding.get("category", "observation"),
        "source_project": config.name,
        "source_experiments": finding.get("source_experiments", []),
        "created_by": finding.get("created_by", "unknown"),
        "created_at": finding.get("ts", ""),
    }


def finding_promote(args: dict[str, Any]) -> dict[str, Any]:
    """Promote a finding from one scope to another. Supports project→track→global."""
    config = _get_config()
    try:
        hub = _get_hub()
        if hub is None:
            return {"error": "Hub not initialized."}

        from_scope = args["from_scope"]
        to_scope = args["to_scope"]
        from_track = args.get("from_track")
        to_track = args.get("to_track")

        # Confidence threshold check
        from crucible.core.finding import PROMOTION_RULES

        rule = PROMOTION_RULES.get((from_scope, to_scope), {})
        min_conf = rule.get("min_confidence", 0.0)

        if from_scope == "project":
            # Promote from ResearchState → Hub
            finding_index = args.get("finding_index")
            if finding_index is None:
                finding_id = args.get("finding_id", "")
                return {"error": "finding_index is required when promoting from project scope."}

            from crucible.researcher.state import ResearchState

            state_path = config.project_root / config.research_state_file
            state = ResearchState(state_path, budget_hours=config.researcher.budget_hours)

            if finding_index < 0 or finding_index >= len(state.findings):
                return {"error": f"Finding index {finding_index} out of range ({len(state.findings)} findings)."}

            finding = state.findings[finding_index]
            if finding.get("confidence", 0) < min_conf:
                return {"error": f"Confidence {finding.get('confidence', 0):.2f} below threshold {min_conf:.2f} for {from_scope}→{to_scope}."}

            hub_finding = _research_finding_to_hub_finding(finding, config)
            promoted = hub.store_finding(hub_finding, to_scope, track=to_track)
            return {"status": "promoted", "finding": promoted}
        else:
            # Hub-to-hub promotion (track→global)
            finding_id = args["finding_id"]
            existing = hub.get_finding(finding_id, from_scope, track=from_track)
            if existing is None:
                return {"error": f"Finding '{finding_id}' not found in {from_scope} scope."}

            if existing.get("confidence", 0) < min_conf:
                return {"error": f"Confidence {existing.get('confidence', 0):.2f} below threshold {min_conf:.2f} for {from_scope}→{to_scope}."}

            promoted = hub.promote_finding(
                finding_id, from_scope, to_scope,
                from_track=from_track, to_track=to_track,
            )
            return {"status": "promoted", "finding": promoted}
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Briefing tools
# ---------------------------------------------------------------------------


def get_research_briefing(args: dict[str, Any]) -> dict[str, Any]:
    """Comprehensive session orientation: project state, leaderboard, hypotheses, findings, notes, and suggested next steps."""
    config = _get_config()
    try:
        from crucible.researcher.briefing import build_briefing

        track = args.get("track")
        if track:
            original = config.active_track
            config.active_track = track
            try:
                return build_briefing(config)
            finally:
                config.active_track = original
        return build_briefing(config)
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


def annotate_run(args: dict[str, Any]) -> dict[str, Any]:
    """Bidirectional link: attach a finding to a run and record the run in the finding's source_experiments."""
    config = _get_config()
    run_id = args["run_id"]
    finding_index = args["finding_index"]

    try:
        from crucible.researcher.state import ResearchState
        from crucible.runner.notes import NoteStore

        state_path = config.project_root / config.research_state_file
        if not state_path.exists():
            return {"error": "No research state found. Record findings first with context_push_finding."}

        state = ResearchState(state_path, budget_hours=config.researcher.budget_hours)

        if finding_index < 0 or finding_index >= len(state.findings):
            return {
                "error": f"Finding index {finding_index} out of range. "
                f"There are {len(state.findings)} findings (0-indexed)."
            }

        finding = state.findings[finding_index]

        # Add run_id to the finding's source_experiments
        source_exps = finding.get("source_experiments", [])
        if run_id not in source_exps:
            source_exps.append(run_id)
            finding["source_experiments"] = source_exps
            state.save()

        # Add a note to the run referencing this finding
        store_dir = config.project_root / config.store_dir
        note_store = NoteStore(store_dir)
        finding_text = finding.get("finding", "")
        category = finding.get("category", "observation")
        note_body = (
            f"Linked to finding [{category}]: {finding_text}\n\n"
            f"(finding_index={finding_index}, confidence={finding.get('confidence', '?')})"
        )
        note_entry = note_store.add(
            run_id=run_id,
            body=note_body,
            stage="post-run",
            tags=["annotate_run", category],
            created_by="mcp-agent",
        )

        return {
            "status": "annotated",
            "run_id": run_id,
            "finding_index": finding_index,
            "finding_text": finding_text[:200],
            "note": note_entry,
        }
    except CrucibleError as exc:
        return {"error": f"[{type(exc).__name__}] {exc}"}
    except Exception as exc:
        return {"error": f"[unexpected] {exc}"}


# ---------------------------------------------------------------------------
# Model extensibility tools
# ---------------------------------------------------------------------------


def model_list_families(args: dict[str, Any]) -> dict[str, Any]:
    """List all registered model architecture families."""
    try:
        from crucible.models.registry import list_families
        return {"families": list_families()}
    except Exception as exc:
        return {"error": f"Failed to list families: {exc}"}


def model_list_activations(args: dict[str, Any]) -> dict[str, Any]:
    """List all available activation functions."""
    try:
        from crucible.models.components.mlp import ACTIVATIONS
        return {"activations": sorted(ACTIVATIONS.keys())}
    except ImportError:
        return {"error": "torch not installed"}


def model_list_components(args: dict[str, Any]) -> dict[str, Any]:
    """List all available model components."""
    try:
        from crucible.models import components
        return {"components": components.__all__}
    except ImportError:
        return {"error": "torch not installed"}


def model_get_config_schema(args: dict[str, Any]) -> dict[str, Any]:
    """Get accepted parameters for a model family."""
    family = args["family"]
    try:
        from crucible.models.registry import get_family_schema, list_families
        if family not in list_families():
            return {"error": f"Unknown family: {family}. Available: {list_families()}"}
        return {"family": family, "parameters": get_family_schema(family)}
    except Exception as exc:
        return {"error": f"Failed to get schema: {exc}"}


def model_validate_config(args: dict[str, Any]) -> dict[str, Any]:
    """Pre-flight validation of experiment config."""
    config = args.get("config", {})
    warnings: list[str] = []
    errors: list[str] = []
    family = config.get("MODEL_FAMILY", "baseline")
    try:
        from crucible.models.registry import list_families
        if family not in list_families():
            errors.append(f"Unknown MODEL_FAMILY: {family}")
    except Exception as exc:
        warnings.append(f"Could not validate MODEL_FAMILY: {exc}")
    activation = config.get("ACTIVATION", "relu_sq")
    try:
        from crucible.models.components.mlp import ACTIVATIONS
        if activation not in ACTIVATIONS:
            errors.append(f"Unknown ACTIVATION: {activation}. Available: {sorted(ACTIVATIONS.keys())}")
    except ImportError:
        warnings.append("torch not installed; cannot validate ACTIVATION")
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def model_add_architecture(args: dict[str, Any]) -> dict[str, Any]:
    """Write and register a new architecture family at runtime."""
    name = args["name"]
    code = args["code"]
    if "register_model" not in code:
        return {"error": "Code must call register_model() to register the family."}
    import importlib
    from pathlib import Path
    arch_dir = Path(__file__).parent.parent / "models" / "user_architectures"
    arch_dir.mkdir(parents=True, exist_ok=True)
    file_path = arch_dir / f"{name}.py"
    file_path.write_text(code, encoding="utf-8")
    try:
        importlib.import_module(f"crucible.models.user_architectures.{name}")
        from crucible.models.registry import list_families
        return {"status": "registered", "family": name, "families": list_families()}
    except Exception as e:
        file_path.unlink(missing_ok=True)
        return {"error": f"Failed to import: {e}"}


def model_add_activation(args: dict[str, Any]) -> dict[str, Any]:
    """Register a new activation function at runtime via restricted code expression."""
    name = args["name"]
    code = args["code"]
    try:
        import torch
        import torch.nn.functional as F  # noqa: N812
    except ImportError:
        return {"error": "torch not installed"}
    try:
        # Restricted namespace: only torch and F, no builtins for safety
        restricted_ns: dict[str, Any] = {"torch": torch, "F": F, "__builtins__": {}}
        compiled = compile(f"__result = lambda x: {code}", "<activation>", "exec")
        exec(compiled, restricted_ns)  # noqa: S102 — intentional: sandboxed activation builder
        activation_fn = restricted_ns["__result"]
        test = torch.randn(2, 3)
        result = activation_fn(test)
        assert result.shape == test.shape
    except Exception as e:
        return {"error": f"Invalid activation code: {e}"}
    from crucible.models.components.mlp import ACTIVATIONS
    ACTIVATIONS[name] = activation_fn
    return {"status": "registered", "name": name, "activations": sorted(ACTIVATIONS.keys())}


def model_generate_template(args: dict[str, Any]) -> dict[str, Any]:
    """Generate boilerplate code for a new architecture."""
    name = args["name"]
    cn = name.title().replace("_", "")
    lines = [
        f'"""User architecture: {name}."""',
        "from __future__ import annotations",
        "from typing import Any",
        "import torch",
        "from torch import Tensor, nn",
        "from crucible.models.base import TiedEmbeddingLM",
        "from crucible.models.registry import register_model",
        "from crucible.models.components.attention import Block",
        "",
        f"class {cn}GPT(TiedEmbeddingLM):",
        "    def __init__(self, vocab_size, num_layers, model_dim, num_heads, num_kv_heads,",
        '                 mlp_mult, tie_embeddings, tied_embed_init_std, logit_softcap,',
        '                 rope_base, qk_gain_init, activation="relu_sq", **kwargs):',
        "        super().__init__(vocab_size, model_dim, tie_embeddings, tied_embed_init_std, logit_softcap)",
        "        self.blocks = nn.ModuleList([",
        "            Block(model_dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,",
        "                  activation=activation) for _ in range(num_layers)])",
        "",
        "    def hidden(self, input_ids: Tensor, lora=None) -> Tensor:",
        "        x = self.embed_tokens(input_ids)",
        "        x0 = x",
        "        for block in self.blocks:",
        "            x = block(x, x0)",
        "        return x",
        "",
        f"def _build_{name}(args: Any) -> {cn}GPT:",
        f"    return {cn}GPT(",
        "        vocab_size=args.vocab_size, num_layers=args.num_layers,",
        "        model_dim=args.model_dim, num_heads=args.num_heads,",
        "        num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,",
        "        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,",
        "        logit_softcap=args.logit_softcap, rope_base=args.rope_base,",
        '        qk_gain_init=args.qk_gain_init, activation=getattr(args, "activation", "relu_sq"))',
        "",
        f'register_model("{name}", _build_{name})',
    ]
    return {"template": "\n".join(lines), "usage": f"Call model_add_architecture with name='{name}' and code=<this>"}


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
    "fleet_refresh": fleet_refresh,
    "bootstrap_nodes": bootstrap_nodes,
    "dispatch_experiments": dispatch_experiments,
    "collect_results": collect_results,
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
    # Note tools
    "note_add": note_add,
    "note_get": note_get,
    "note_search": note_search,
    # W&B tools
    "wandb_log_image": wandb_log_image,
    "wandb_get_url": wandb_get_url,
    "wandb_annotate": wandb_annotate,
    # Hub tools
    "hub_status": hub_status,
    "hub_sync": hub_sync,
    "track_create": track_create,
    "track_list": track_list,
    "track_switch": track_switch,
    "hub_findings_query": hub_findings_query,
    "finding_promote": finding_promote,
    # Briefing tools
    "get_research_briefing": get_research_briefing,
    "annotate_run": annotate_run,
    # Model extensibility tools
    "model_list_families": model_list_families,
    "model_list_activations": model_list_activations,
    "model_list_components": model_list_components,
    "model_get_config_schema": model_get_config_schema,
    "model_validate_config": model_validate_config,
    "model_add_architecture": model_add_architecture,
    "model_add_activation": model_add_activation,
    "model_generate_template": model_generate_template,
    # Config tools
    "config_get_presets": config_get_presets,
    "config_get_project": config_get_project,
}
