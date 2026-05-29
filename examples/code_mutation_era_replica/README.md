# Code Mutation — ERA Replica Demo

A tiny end-to-end demo of Crucible Phase 5.1 code mutation, modelled
after the ERA (Harvard + DeepMind, *Nature*, May 2026) approach of
tree-searching over scientific software for a scorable empirical task.

This demo is intentionally minimal: no GPU, no datasets, ~1 second per
mutation. It exercises every Phase 5.1 surface:

- `MutationProposal` / `MutationResult` shapes
- `AstLocalEditPolicy` (function-level swap_literal / swap_identifier)
- `SandboxRunner` (rsync clone + subprocess scoring)
- `AstSafetyChecker` (rejects subprocess / network / dunder injection)
- `apply_unified_diff` (`git apply` in a clean workspace)
- `score_stdout` (parses the standard val_bpb pattern)
- Tree bridge via `code_mutation_tree.expand_tree_with_mutations`
  (not exercised in `run_demo.py`; see `tests/test_code_mutation_phase5.py::TestCodeMutationTreeBridge` for the API)

## What it scores

`baseline.py` runs a 1-hidden-layer regression on a noisy
`sin(2x) + 0.3·cos(5x)` target. The unmutated baseline uses
intentionally suboptimal hyperparameters — small `HIDDEN_DIM`,
aggressive `LEARNING_RATE`, plain `relu`. Three hand-picked mutations
try to improve it:

| Mutation | What it changes | Why |
|---|---|---|
| `gelu_activation` | `ACTIVATION = "relu"` → `"gelu"` | Smoother near zero, fewer dead neurons |
| `lr_0p2` | `LEARNING_RATE = 0.5` → `0.2` | Less late-epoch oscillation |
| `hidden_dim_16` | `HIDDEN_DIM = 4` → `16` | Underfit on the 2-frequency target |

## Run it

```bash
PYTHONPATH=src python3 examples/code_mutation_era_replica/run_demo.py
```

You'll see a baseline score, three mutation results, and a leaderboard
sorted by val_bpb. Expected outcome: `hidden_dim_16` wins by the
biggest margin; `lr_0p2` improves moderately; `gelu_activation` may or
may not improve depending on seed.

## Plugging in an LLM (real ERA loop)

Swap the hand-picked mutations for ones an LLM generates:

```python
from crucible.researcher.code_mutation import (
    LlmDiffPolicy,
    llm_diff_request_prompt,
    llm_diff_parse_response,
)

envelope = llm_diff_request_prompt(
    target_file="baseline.py",
    intent="lower val_bpb by improving the activation choice",
    project_root=PROJECT,
    mutation_scope=["baseline.py"],
)
# Your orchestrator runs the LLM with envelope["system"] / ["user"] / ["schema"]
llm_response = your_llm.complete(envelope["system"], envelope["user"], envelope["schema"])
proposal = llm_diff_parse_response(
    llm_response,
    target_file="baseline.py",
    mutation_scope=["baseline.py"],
)
policy = LlmDiffPolicy(project_root=PROJECT, scorer=scorer)
result = policy.apply(proposal)
```

Or call the same flow via MCP from any orchestrator:

```python
mcp.call("code_mutation_propose", target_file="baseline.py", intent="...")
# orchestrator runs its LLM
mcp.call("code_mutation_apply", policy="llm_diff", target_file="baseline.py",
         llm_response={...}, scorer={"cmd": ["python3", "scorer.py"]})
```

## Wiring into a search tree

`run_demo.py` is single-shot. For multi-iteration mutation-driven
search, drive the tree directly:

```python
from crucible.researcher.code_mutation_tree import expand_tree_with_mutations
from crucible.researcher.search_tree import SearchTree

tree = SearchTree.create(tree_dir=".crucible/search_trees/era_replica",
                         name="era_replica", primary_metric="val_bpb",
                         metric_direction="minimize")
# expand root with this iteration's mutations
expand_tree_with_mutations(tree, root_id, proposals, policy)
# pick top-K via UCB1 / Pareto, generate next-round proposals, repeat
```

`tests/test_code_mutation_phase5.py::TestCodeMutationTreeBridge::test_expand_and_record` shows the full call.

## Scope discipline

This demo is **not** a Nature-claim replica. ERA generated peer-reviewed
results across COVID hospitalization forecasting, scRNA-seq integration,
and zebrafish neuron prediction. Crucible's Phase 5.1 ships the
infrastructure (apply, sandbox, score, safety, tree bridge); domain
showcases live in `examples/flagship_param_golf/` (Crucible's own
parameter-golf result) and per-track HuggingFace dataset publishes.
