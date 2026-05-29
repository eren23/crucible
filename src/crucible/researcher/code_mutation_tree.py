"""Bridge between code-mutation policies and SearchTree.

Composes the two surfaces so one call expands a parent node with
N mutation children, sandboxes + scores each, and records results
into the tree. The orchestrator can then run UCB1 / GRPO / Pareto
selection over the mutation-generated branch just like any other
expansion policy's output.

Kept as a separate module so ``search_tree.py`` doesn't depend on
``code_mutation`` (and vice versa). Both can ship as taps with
zero churn to either side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from crucible.researcher.code_mutation import (
    CodeMutationPolicy,
    MutationProposal,
    MutationResult,
)
from crucible.researcher.search_tree import SearchTree


@dataclass
class CodeMutationNode:
    """Per-child summary the caller can use to drive next-step selection."""

    node_id: str
    proposal_name: str
    success: bool
    score: float | None
    error: str | None


def expand_tree_with_mutations(
    tree: SearchTree,
    parent_node_id: str,
    proposals: list[MutationProposal],
    policy: CodeMutationPolicy,
    *,
    metric_name: str | None = None,
) -> list[CodeMutationNode]:
    """Apply each mutation in ``proposals``, score it, and record on ``tree``.

    Children land under ``parent_node_id`` with
    ``generation_method="code_mutation"`` so the tree's selection
    policies can distinguish mutation-generated nodes from
    config-tweak nodes.

    ``metric_name`` defaults to the tree's primary metric. Each
    MutationResult.score is stored under that key (and under
    ``code_mutation_score`` for posterity).

    Returns one ``CodeMutationNode`` per proposal, in input order.
    Failed mutations still get a tree node (status: completed,
    score: None) so the run is auditable.
    """
    if metric_name is None:
        metric_name = tree.meta.get("primary_metric")
        if metric_name is None:
            raise ValueError(
                "tree.meta['primary_metric'] is unset; pass metric_name= "
                "explicitly to expand_tree_with_mutations()"
            )

    child_specs: list[dict[str, Any]] = []
    for proposal in proposals:
        child_specs.append({
            "name": proposal.name,
            "config": {
                "CRUCIBLE_VARIANT_NAME": proposal.name,
                "MUTATION_TARGET_FILE": proposal.target_file,
            },
            "hypothesis": proposal.hypothesis,
            "rationale": proposal.rationale,
            "generation_method": "code_mutation",
            "tags": ["code_mutation"],
        })

    new_ids = tree.expand_node(parent_node_id, child_specs)

    summaries: list[CodeMutationNode] = []
    for proposal, node_id in zip(proposals, new_ids):
        result: MutationResult = policy.apply(proposal)
        record_payload: dict[str, Any] = {"code_mutation_score": result.score}
        if result.score is not None:
            record_payload[metric_name] = result.score
        if not result.success:
            record_payload["error"] = result.error
        tree.record_result(node_id, record_payload)
        summaries.append(CodeMutationNode(
            node_id=node_id,
            proposal_name=proposal.name,
            success=result.success,
            score=result.score,
            error=result.error,
        ))
    return summaries
