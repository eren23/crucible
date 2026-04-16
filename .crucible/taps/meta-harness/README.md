# crucible-meta-harness (local tap)

Meta-harness-style evolutionary optimization of task-specific harnesses,
delivered as a Crucible tap. Inspired by Stanford IRIS Lab's
[meta-harness](https://github.com/stanford-iris-lab/meta-harness).

## Packages

| Path | Type | Purpose |
|------|------|---------|
| `launchers/harness_runner/` | `launchers` | Remote entrypoint that loads a candidate `MemorySystem` implementation and runs the predict/learn loop. |
| `evaluations/harness_eval/` | `evaluations` | Multi-metric harness evaluation (accuracy, tokens, latency). Ships the `MemorySystem` ABC used by launcher + baselines. |
| `domain_specs/nlp_classification/` | `domain_specs` | Domain template for text-classification harness optimization. |
| `domain_specs/agent_scaffold/` | `domain_specs` | Domain template for agent scaffold optimization (terminal/tool-use tasks). |

## Install (from this local tap)

```bash
crucible tap add <path-to-this-tap>
crucible tap install harness_runner --type launchers
crucible tap install harness_eval --type evaluations
crucible tap install nlp_classification --type domain_specs
```

## Concepts

Harness optimization differs from model/architecture optimization: the base
model is held fixed and we evolve the *scaffolding* around it — memory
systems, retrieval strategies, prompt construction, tool use patterns.

Candidates are Python source files implementing the `MemorySystem`
interface. Crucible stores them under
`.crucible/search_trees/{tree}/candidates/{node_id}.py` and sets
`HARNESS_CANDIDATE_ID={node_id}` in the experiment config. The launcher
imports the candidate at runtime and runs the eval loop.

Metrics are tracked on the N-dimensional Pareto frontier built into
`SearchTree` (`HarnessOptimizer` + `tree_pareto` / `harness_frontier` MCP
tools). Evolution logs (per-iteration JSONL) live alongside `tree.yaml`.
