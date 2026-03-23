"""Hypothesis generation using LLM-driven analysis."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crucible.researcher.llm_client import LLMClient, parse_json_from_text
from crucible.researcher.state import ResearchState

if TYPE_CHECKING:
    from crucible.researcher.search_tree import SearchTree


def generate_hypotheses(
    context: str,
    program_text: str,
    state: ResearchState,
    llm: LLMClient,
    iteration: int,
) -> list[dict[str, Any]]:
    """Use an LLM to generate ranked experiment hypotheses from the current research state."""

    system_prompt = (
        "You are an autonomous ML research agent. "
        "Your goal is to minimize the primary metric (e.g., validation loss or bits-per-byte) "
        "for a model within the given constraints.\n\n"
        "Given the current research state and results, propose 3-5 experiment hypotheses "
        "ranked by expected impact. Each hypothesis should test ONE change at a time "
        "unless you have strong evidence a combination helps.\n\n"
        "Return a JSON object with a single key \"hypotheses\" containing a list of objects, "
        "each with:\n"
        "- \"hypothesis\": string describing what you expect\n"
        "- \"name\": short experiment name (no spaces, lowercase + underscores)\n"
        "- \"expected_impact\": float, expected improvement (positive = better)\n"
        "- \"confidence\": float 0-1, your confidence this will help\n"
        "- \"config\": dict of env var overrides (all values must be strings)\n"
        "- \"rationale\": 1-2 sentence explanation\n"
        "- \"family\": which model family this tests\n\n"
        "IMPORTANT: All config values must be strings. The config dict contains environment "
        "variable overrides like {\"MODEL_FAMILY\": \"looped\", \"RECURRENCE_STEPS\": \"12\"}.\n\n"
        "Focus on the research priorities in the program document. "
        "Mix exploitation (refine what works) with exploration (test new ideas)."
    )

    user_prompt = f"{program_text}\n\n---\n\n{context}"

    response_text = llm.complete(system_prompt, user_prompt)
    if response_text is None:
        return []

    hypotheses = _parse_hypotheses(response_text, iteration)
    for hyp in hypotheses:
        state.add_hypothesis(hyp)

    print(f"  Generated {len(hypotheses)} hypotheses:")
    for h in hypotheses:
        print(
            f"    - {h.get('name', '?')}: {h.get('hypothesis', '?')[:80]}  "
            f"(impact={h.get('expected_impact', 0):.4f}, conf={h.get('confidence', 0):.2f})"
        )

    return hypotheses


def _parse_hypotheses(text: str, iteration: int) -> list[dict[str, Any]]:
    """Extract hypotheses list from LLM response text."""
    parsed = parse_json_from_text(text)
    if parsed is None:
        return []
    hypotheses = parsed.get("hypotheses", [])
    if not isinstance(hypotheses, list):
        return []
    valid = []
    for h in hypotheses:
        if not isinstance(h, dict):
            continue
        if not h.get("config"):
            continue
        config = h.get("config", {})
        h["config"] = {k: str(v) for k, v in config.items()}
        h.setdefault("name", f"hyp_{iteration}_{len(valid)}")
        h.setdefault("hypothesis", h.get("rationale", "No hypothesis stated"))
        h.setdefault("expected_impact", h.get("expected_bpb_impact", 0.0))
        h.setdefault("confidence", 0.5)
        h.setdefault("family", config.get("MODEL_FAMILY", "unknown"))
        valid.append(h)
    return valid


# ---------------------------------------------------------------------------
# Tree-aware hypothesis generation
# ---------------------------------------------------------------------------


def _build_tree_context(tree: "SearchTree", node_id: str) -> str:
    """Build LLM context from a node's position in the search tree.

    Includes: the node's own result, ancestry path, sibling results,
    and a compact tree summary.
    """
    node = tree.get_node(node_id)
    if node is None:
        return "(node not found)"

    lines: list[str] = []

    # Tree summary
    summary = tree.get_tree_summary()
    lines.append("## Tree Summary")
    lines.append(f"Name: {summary['name']}")
    lines.append(f"Total nodes: {summary['total_nodes']}")
    lines.append(f"Completed: {summary['completed_nodes']}")
    lines.append(f"Best metric: {summary.get('best_metric', 'N/A')}")
    lines.append(f"Primary metric: {summary['primary_metric']} ({summary['metric_direction']})")
    lines.append("")

    # Ancestry path
    ancestry = tree.get_ancestry(node_id)
    if ancestry:
        lines.append("## Ancestry Path (root -> current)")
        for anc in ancestry:
            metric_str = f" {summary['primary_metric']}={anc['result_metric']:.4f}" if anc.get("result_metric") is not None else ""
            lines.append(
                f"  depth={anc['depth']} {anc['experiment_name']}{metric_str} "
                f"status={anc['status']}"
            )
            # Show key config diffs from parent
            if anc.get("config"):
                config_str = ", ".join(f"{k}={v}" for k, v in sorted(anc["config"].items()))
                lines.append(f"    config: {config_str}")
        lines.append("")

    # Current node details
    lines.append("## Current Node (to expand)")
    lines.append(f"Name: {node['experiment_name']}")
    lines.append(f"Status: {node['status']}")
    if node.get("result_metric") is not None:
        lines.append(f"Metric: {summary['primary_metric']}={node['result_metric']:.4f}")
    if node.get("hypothesis"):
        lines.append(f"Hypothesis: {node['hypothesis']}")
    if node.get("config"):
        config_str = ", ".join(f"{k}={v}" for k, v in sorted(node["config"].items()))
        lines.append(f"Config: {config_str}")
    lines.append("")

    # Sibling results
    siblings = tree.get_siblings(node_id)
    completed_siblings = [s for s in siblings if s["status"] == "completed"]
    if completed_siblings:
        lines.append("## Sibling Results")
        for sib in completed_siblings:
            metric_str = f"{summary['primary_metric']}={sib['result_metric']:.4f}" if sib.get("result_metric") is not None else "N/A"
            lines.append(f"  {sib['experiment_name']}: {metric_str}")
            if sib.get("hypothesis"):
                lines.append(f"    hypothesis: {sib['hypothesis']}")
        lines.append("")

    # Best path for reference
    best_path = tree.get_best_path()
    if best_path:
        lines.append("## Best Path So Far")
        for bp_node in best_path:
            metric_str = f" {summary['primary_metric']}={bp_node['result_metric']:.4f}" if bp_node.get("result_metric") is not None else ""
            lines.append(f"  {bp_node['experiment_name']}{metric_str}")
        lines.append("")

    return "\n".join(lines)


def _parse_tree_children(text: str, parent_name: str) -> list[dict[str, Any]]:
    """Parse LLM response into child specs for SearchTree.expand_node()."""
    parsed = parse_json_from_text(text)
    if parsed is None:
        return []

    children = parsed.get("children", parsed.get("hypotheses", []))
    if not isinstance(children, list):
        return []

    valid: list[dict[str, Any]] = []
    for i, child in enumerate(children):
        if not isinstance(child, dict):
            continue
        config = child.get("config", {})
        if not config:
            continue
        # Ensure all config values are strings
        config = {k: str(v) for k, v in config.items()}

        spec: dict[str, Any] = {
            "name": child.get("name", f"{parent_name}_child_{i}"),
            "config": config,
            "hypothesis": child.get("hypothesis", child.get("rationale", "")),
            "rationale": child.get("rationale", child.get("hypothesis", "")),
            "generation_method": "llm_tree",
            "priority_score": float(child.get("confidence", child.get("priority_score", 0.5))),
            "tags": child.get("tags", []),
        }
        valid.append(spec)

    return valid


def generate_tree_hypotheses(
    tree: "SearchTree",
    node_id: str,
    llm: LLMClient,
    n_children: int = 3,
) -> list[dict[str, Any]]:
    """Generate child experiment specs for a search tree node.

    Uses the node's ancestry, sibling results, and tree summary as LLM
    context to propose variations on the node's config.

    Returns a list of child specs compatible with
    :meth:`SearchTree.expand_node`.
    """
    node = tree.get_node(node_id)
    if node is None:
        return []

    context = _build_tree_context(tree, node_id)
    primary_metric = tree.meta.get("primary_metric", "val_bpb")
    direction = tree.meta.get("metric_direction", "minimize")

    system_prompt = (
        "You are an autonomous ML research agent exploring a tree of experiments. "
        f"Your goal is to {direction} the primary metric ({primary_metric}).\n\n"
        f"Given a parent experiment and its context in the search tree, propose {n_children} "
        "child experiments that are variations on the parent's config. Each child should "
        "test ONE meaningful change.\n\n"
        "Return a JSON object with a key \"children\" containing a list of objects, each with:\n"
        "- \"name\": short experiment name (lowercase, underscores, no spaces)\n"
        "- \"config\": dict of env var overrides (ALL values must be strings)\n"
        "- \"hypothesis\": what you expect this change to achieve\n"
        "- \"rationale\": 1-2 sentence explanation of why\n"
        "- \"confidence\": float 0-1\n\n"
        "IMPORTANT:\n"
        "- Config values override the parent's config (inherited automatically).\n"
        "- Only include keys you want to CHANGE from the parent.\n"
        "- All config values must be strings.\n"
        "- Mix exploitation (refine what works) with exploration (test new ideas).\n"
        "- Consider what siblings have already tried to avoid redundant experiments."
    )

    response_text = llm.complete(system_prompt, context)
    if response_text is None:
        return []

    children = _parse_tree_children(response_text, node["experiment_name"])

    # Limit to requested count
    children = children[:n_children]

    if children:
        print(f"  Generated {len(children)} tree children for '{node['experiment_name']}':")
        for c in children:
            print(f"    - {c['name']}: {c.get('hypothesis', '?')[:80]}")

    return children
