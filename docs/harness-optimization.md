---
layout: default
title: Harness Optimization
---

# Harness Optimization

Crucible can evolve task-specific **harnesses** — the scaffolding code around a
fixed base model (memory systems, retrieval strategies, prompt scaffolds,
agent control loops). Inspired by Stanford IRIS Lab's
[meta-harness](https://github.com/stanford-iris-lab/meta-harness), this is
orthogonal to architecture search: the model stays frozen, the *code
wrapping it* evolves.

The loop is **propose → validate → benchmark → Pareto frontier → repeat**:

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Propose    │ → │  Validate    │ → │  Benchmark   │ → │  Frontier    │
│  (LLM, 3-N   │   │ (syntax,     │   │ (fleet OR    │   │  (N-D Pareto │
│  candidates) │   │ interface,   │   │  local run,  │   │  + hyper-    │
│              │   │ constraints) │   │  multi-metric│   │  volume)     │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
        ↑                                                         │
        └─────────────────── evolution_log.jsonl ─────────────────┘
```

## Core Concepts

### Domain Spec

A YAML file that tells Crucible what you're optimizing. It defines:

- `interface` — the Python class + methods each candidate must implement.
- `metrics` — the N-dimensional Pareto axes (each `minimize` or `maximize`).
- `constraints` — parameter bounds enforced at validation time.
- `proposal_guidance` — free-form text injected into the LLM proposer prompt.
- `baselines` — reference implementations that seed the frontier.

Live at `.crucible/domain_specs/{name}/domain_spec.yaml` (or installed
globally from a tap under `~/.crucible-hub/plugins/domain_specs/`).

**Example** — NLP text classification (`.crucible/taps/meta-harness/domain_specs/nlp_classification/domain_spec.yaml`):

```yaml
name: nlp_classification
interface:
  class_name: MemorySystem
  required_methods:
    - name: predict
    - name: learn_from_batch
metrics:
  - name: accuracy
    direction: maximize
  - name: tokens_per_example
    direction: minimize
  - name: latency_ms
    direction: minimize
constraints:
  TEMPERATURE: {type: float, min: 0.0, max: 2.0}
proposal_guidance: |
  Favor diverse retrieval strategies over trivial prompt tweaks...
baselines:
  - name: no_memory
    path: baselines/no_memory.py
```

### Code-as-Candidate

Candidates are Python source files, not config dicts. The `HarnessOptimizer`
writes each candidate to `.crucible/search_trees/{tree}/candidates/{node_id}.py`
and sets `HARNESS_CANDIDATE_ID={node_id}` in the experiment config. The
launcher loads the file at dispatch time (parallel to how `MODEL_FAMILY`
points at architecture code on disk).

### N-Dimensional Pareto Frontier

`SearchTree` tracks any number of metrics simultaneously. A node is on the
frontier if no other completed node dominates it on every axis. The tree
maintains `frontier_node_ids` in metadata, refreshed on every result
record. Use `tree_pareto` / `harness_frontier` to query the current
frontier; `frontier_summary()` also returns `best_per_metric` and (for 2D)
the `hypervolume`.

Legacy single-metric trees still work — `primary_metric` +
`metric_direction` form a single-entry `metrics` list under the hood.

### Candidate Validation

Before any code goes near the fleet, `validate_candidate()` runs four
AST-based checks:

1. **Syntax** — candidate compiles.
2. **Interface** — the expected class exists with every required method.
3. **Constraints** — declared bounds (min/max/enum) are respected.
4. **Duplicate** — SHA-256 hash collision with previously seen candidates.

Failures are structured `ValidationResult` objects (never exceptions), so
the orchestrator decides whether to drop the candidate, re-prompt, or
escalate.

## MCP Tools

| Tool | Description |
|------|-------------|
| `harness_init` | Initialize a `HarnessOptimizer` for a domain+tree pair. |
| `harness_propose` | Generate N candidate implementations. |
| `harness_validate` | Validate candidates without dispatching. |
| `harness_iterate` | Run one full propose→validate→benchmark cycle. |
| `harness_frontier` | Current Pareto frontier snapshot. |
| `harness_evolution_log` | All past iteration records (JSONL). |
| `tree_pareto` | Pareto frontier for any search tree (not harness-specific). |

## Workflow

```
1. crucible tap add <meta-harness-tap-url>         # or use local tap
2. crucible tap install nlp_classification --type domain_specs
3. harness_init(domain_spec="nlp_classification", tree_name="nlp_run1")
4. harness_iterate()          # loop; each call appends evolution_log.jsonl
5. harness_frontier()         # inspect Pareto frontier
6. finding_promote(...)       # elevate winning candidates to the hub
```

## Meta-Harness Tap (in-repo at `.crucible/taps/meta-harness/`)

Ships four packages:

- **launchers/harness_runner** — remote entrypoint. Loads a candidate by
  `HARNESS_CANDIDATE_ID`, runs the predict/learn loop, emits metrics in
  the Crucible stdout format.
- **evaluations/harness_eval** — multi-metric evaluator + `MemorySystem`
  ABC used by launcher and baselines.
- **domain_specs/nlp_classification** — text classification domain template.
- **domain_specs/agent_scaffold** — agent scaffold optimization template.

## Persistence Layout

```
.crucible/search_trees/{tree_name}/
  tree.yaml                # SearchTreeMeta with metrics list, frontier_node_ids
  nodes.jsonl              # Append-only event ledger
  current_tree.yaml        # Snapshot of all nodes
  candidates/              # Code-as-candidate storage
    {node_id}.py
  evolution_log.jsonl      # Per-iteration summaries (proposed/validated/frontier)
```

## Design Tradeoffs

- **LLM-generated code writes to disk.** Validation is AST-based so
  malformed or interface-incomplete candidates never hit execution, but
  the runner does `exec` the code. Threat model assumes the agent and
  project are trusted; taps published to hub go through PR review.
- **2D hypervolume only** — higher-dimensional HV needs a dedicated
  library (pygmo, moarchiving). The frontier itself is N-D; only the HV
  scalar is restricted.
- **Dry-run mode** generates fixture candidates and synthesizes
  deterministic metric values so the whole pipeline is exercisable
  without LLM or GPU.
