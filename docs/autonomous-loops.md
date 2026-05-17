# Autonomous Loops

Phase 1 introduced three persisted-session drivers that wrap the [orchestrator contract](orchestrator-contract.md) into closed-loop workflows. The orchestrator drives the LLM round-trips; Crucible owns state, scheduling, budget, and judge enforcement.

| Driver | What it loops over | When to use |
|---|---|---|
| `autonomous_research_loop` | linear hypothesize → experiment → reflect | most projects: a flat batch of experiments per iteration |
| `tree_autonomous_loop` | branching search-tree expansion | when promising results should fork (UCB1, GRPO, agent-directed) |
| `harness_autonomous_loop` | meta-harness optimization | evolving task-specific harness code (memory systems, agent scaffolds) |

All three follow the same lifecycle: `start → submit/continue (N times) → status (any time) → cancel`. State persists as a yaml + jsonl event log under `.crucible/{type}_sessions/{session_id}.{yaml,jsonl}`. Sessions survive process restarts.

## Why drivers and not just `research_request_prompt`

You could roll your own loop with `research_request_prompt` + `research_submit` per the [orchestrator contract](orchestrator-contract.md). That's fine for one-shot work. The drivers exist because:

- **Iteration management.** The driver tracks which iteration / stage you're on and refuses to advance the state machine out-of-order.
- **State-snapshot tracking.** Each `request_prompt` records its snapshot; submits compare automatically without you plumbing the value through. Stale submits raise `StaleSubmitError` instead of silently overwriting.
- **Budget cap.** Phase 1.8 wall-clock × pod-rate cost tracking. When `budget_usd` is hit the session auto-cancels before the next prompt fires.
- **Judge separation.** Phase 1.9 hook calls `JudgePanel.assert_separated()` at `start`, so a mis-separated reward/eval judge fails before pod time is consumed.
- **Doom-loop detection.** Fingerprint windowing across submits — five identical prompts in a row aborts the session instead of burning iterations on a confused LLM.
- **Resumability.** Crash, restart, attach back to the session by its UUID. The yaml is the durable source of truth.

## `autonomous_research_loop`

Linear research loop. Each iteration: hypothesize → orchestrator runs experiments via pure fleet ops → reflect. Stage advance: `hypothesis → reflection → (next iter) hypothesis → ... → done`.

### Verbs

```python
autonomous_research_loop(action="start",
    iterations=3,             # how many hypothesis/reflection cycles
    tier="proxy",             # smoke | screen | proxy | medium | promotion
    focus_family="",          # bias hypothesis generation toward one architecture family
    budget_usd=50.0,          # opt-in wall-clock × pod-rate cap
    with_literature=False,    # pre-inject HF Papers context into hypothesis prompts
    literature_k=5,
)
# → {session_id, stage="hypothesis", iteration=0, system, user, schema, state_snapshot, ...}

autonomous_research_loop(action="submit",
    session_id="...",
    response={"hypotheses": [...]},   # validated against schema
    state_snapshot={"...": "..."},     # optional; falls back to session's tracked snapshot
)
# → {stage_applied, next_stage, iteration, iterations_completed, session_status, apply_result, next_prompt}

autonomous_research_loop(action="status", session_id="...")
# → full session yaml content

autonomous_research_loop(action="cancel", session_id="...", reason="...")
# → {session_status: "canceled", checkpoint_path, already_terminal}
```

### Full lifecycle example

```python
from crucible.core.config import load_config
from crucible.researcher import autonomous_session as autos

config = load_config("crucible.yaml")

# Iteration 0 hypothesis stage
started = autos.action_start(config, iterations=2, tier="proxy", budget_usd=25.0)
sid, snap = started["session_id"], started["state_snapshot"]

# Orchestrator calls its own LLM with (started["system"], started["user"], started["schema"])
hyp_response = call_my_llm(started)  # produces {"hypotheses": [...]}

after_hyp = autos.action_submit(config, session_id=sid, response=hyp_response, state_snapshot=snap)
# after_hyp["next_prompt"] is the reflection prompt for iteration 0.

# ... orchestrator now drives pure fleet ops:
#   design_batch_from_hypotheses(state_path=...)
#   design_enqueue_batch(...)
#   dispatch_experiments(...)
#   collect_results(...)
# ... once results land, reflection makes sense:

refl_prompt = after_hyp["next_prompt"]
refl_response = call_my_llm(refl_prompt)

after_refl = autos.action_submit(
    config, session_id=sid, response=refl_response,
    state_snapshot=refl_prompt["state_snapshot"],
)
# after_refl["iterations_completed"] == 1; after_refl["next_prompt"] is iteration 1's hypothesis.

# When all iterations complete, session_status flips to "done" and next_prompt is None.
```

### Stale-submit guard

```python
# Snapshot from start
snap_at_start = started["state_snapshot"]

# Meanwhile another agent added a finding, advancing the content_hash.
# Submit with stale snapshot:
autos.action_submit(config, session_id=sid, response=hyp_response, state_snapshot=snap_at_start)
# → StaleSubmitError: "submitted_snapshot=... current_snapshot=... Re-request the prompt with the latest state and retry."
```

This is the *correct* outcome: Crucible refuses to overwrite state that has moved out from under the orchestrator. The orchestrator should call `request_prompt` again, re-run its LLM with the new context, and resubmit.

### Budget cap

```python
autos.action_start(config, iterations=5, tier="proxy", budget_usd=10.0)
# → starts normally

# After enough wall-clock × $/hour fleet time has accumulated past $10:
autos.action_submit(...)  # → next build_prompt raises BudgetExceeded; session auto-cancels.
```

Budget tracking is best-effort wall-clock × fleet hourly rate. The session yaml records `spend_usd` so you can audit afterwards. Phase 1.8 added a `session_pod_filter` so HPO-driven pod costs from a parallel study don't inflate this session's spend.

### Literature pre-injection

`with_literature=True` pulls top-K HF Papers via `researcher.literature.search_papers(focus_family or project_name)` and injects abstracts into the hypothesis-stage `user` message. Network failures degrade silently — empty string in place of the literature block, never blocks the loop.

## `tree_autonomous_loop`

Branching search over an existing search tree. Each iteration: pick a node to expand → ask orchestrator for N children → enqueue, dispatch, collect, sync → repeat.

The wrinkle: Crucible doesn't dispatch fleet ops itself. After `submit` the orchestrator gets a response with `next_action`:

- `"submit"` — more nodes to expand, give the next prompt.
- `"external_dispatch"` — pending nodes exist, orchestrator must drive `tree_enqueue_pending → dispatch_experiments → collect_results → tree_sync_results` before continuing.
- `"done"` — session complete.

The `continue` verb re-probes for expandable nodes after the orchestrator's fleet ops complete.

### Verbs

```python
tree_autonomous_loop(action="start",
    tree_name="my-tree",     # must exist (call tree_create first)
    iterations=10,
    n_children=3,
    budget_usd=100.0,
)
# → {session_id, next_action="submit"|"external_dispatch"|"done", ...}

tree_autonomous_loop(action="submit", session_id, response, tree_snapshot?)
# → {applied, next_prompt}

tree_autonomous_loop(action="continue", session_id)
# → re-probe: returns next prompt, external_dispatch hint, or done.

tree_autonomous_loop(action="status", session_id)
tree_autonomous_loop(action="cancel", session_id, reason?)
```

### Lifecycle with external dispatch

```python
from crucible.researcher import tree_autonomous_session as tas
from crucible.mcp.tools import tree_create  # or via MCP

# Pre-condition: tree exists.
tree_create({"tree_name": "muon-vs-adamw", "root_config": {"OPTIMIZER": "adamw"}})

started = tas.action_start(config, tree_name="muon-vs-adamw", iterations=5)
sid = started["session_id"]

while True:
    if started["next_action"] == "done":
        break
    if started["next_action"] == "external_dispatch":
        # Orchestrator drives the fleet:
        run_pending_tree_nodes(tree_name="muon-vs-adamw")
        started = tas.action_continue(config, session_id=sid)
        continue
    # next_action == "submit": orchestrator calls its LLM with (system, user, schema)
    response = call_my_llm(started)
    applied = tas.action_submit(
        config, session_id=sid, response=response,
        tree_snapshot=started["tree_snapshot"],
    )
    started = applied["next_prompt"]
```

The pattern matches how `tree_auto_expand` works for one-shot expansion — the autonomous_loop just wraps it with iteration accounting + the external_dispatch protocol.

## `harness_autonomous_loop`

Meta-harness optimization — evolves harness code (e.g., agent scaffolds, memory systems) on a domain spec. Each iteration proposes N candidate Python harnesses, validates them, benchmarks them, and feeds results to the next proposal.

This driver is the most "agentic" of the three: the orchestrator's response is *Python source code* for a harness file, not a structured config dict. The schema is permissive — the file just needs to define the entry point declared in the domain spec.

### Verbs

```python
harness_autonomous_loop(action="start",
    domain_spec="path/to/nlp_classification.yaml",   # domain contract
    tree_name="harness-search",                       # accumulates candidates as tree nodes
    iterations=5,
    n_candidates=3,
    dry_run=False,                                    # skip benchmark? useful for code-quality probes
    budget_usd=50.0,
)
# → {session_id, system, user, schema, ...}

harness_autonomous_loop(action="submit", session_id, response=<python source string>)
# → {applied, scores, frontier, next_prompt}

harness_autonomous_loop(action="continue", session_id)   # re-build proposal prompt
harness_autonomous_loop(action="status", session_id)
harness_autonomous_loop(action="cancel", session_id, reason?)
```

The benchmark step calls into `HarnessOptimizer` (`researcher/harness_optimizer.py`) which evaluates candidates against the domain spec's tasks, computes N-dimensional Pareto frontiers, and writes them under `.crucible/search_trees/{tree_name}/candidates/{node_id}.py`.

See [harness-optimization.md](harness-optimization.md) for the underlying mechanics.

## Resumability

All three drivers persist enough state to resume after a crash. Two artifacts per session:

- `.crucible/{type}_sessions/{session_id}.yaml` — durable state (iteration, stage, status, budget, snapshot).
- `.crucible/{type}_sessions/{session_id}.jsonl` — append-only event log.

To resume:

```python
# Just reload by session_id. The driver picks up where it left off.
status = autos.action_status(config, session_id=sid)
if status["status"] == "running":
    # Re-build the current prompt (idempotent — same iteration/stage).
    from crucible.researcher.autonomous_session import AutonomousSession
    session = AutonomousSession(config, sid).load()
    state = autos._load_state(config)
    prompt = session.build_prompt(state)
```

The driver finds the same in-flight `.lock` file via fcntl advisory locking, so two processes can't drive the same session concurrently. The session-creation step holds a separate `.create.lock` under the sessions dir so two `action_start` calls for the same project return the same session_id rather than corrupting state.

## Judge-separation hook

When `config.judges` declares a `JudgePanel` (see [judge-separation.md](judge-separation.md)), all three drivers call `panel.assert_separated()` at `start`. A mis-separated config (reward judge and eval judge in the same family) raises `ConfigError` before any pod time is consumed:

```yaml
judges:
  reward_judge: {model: gemini-2.5-flash, family: gemini}
  eval_judge:   {model: claude-opus-4-7, family: claude}   # different family ✓
  enforce_separation: true
```

Opt out by setting `enforce_separation: false` (downgrades to warning) or omitting the `judges` block entirely.

## Doom-loop detection

Each driver maintains a sliding window of recent prompt fingerprints (system + user hash). When the window is fully identical (default 5 in a row) the next `build_prompt` raises `DoomLoopDetected` and marks the session errored. The signal here is "the orchestrator's LLM is stuck producing the same thing"; rather than burn iterations, the driver kills the session and surfaces a clear error.

The check is per-stage (a hypothesis prompt repeating five times trips; alternating hypothesis/reflection prompts do not), and *intentionally* spans iterations — a doom-loop is exactly "the same stage keeps coming back unchanged."

## When NOT to use the drivers

- **One-shot tool calls.** If you only want one hypothesis batch and one reflection pass, call `research_request_prompt` + `research_submit` directly. The driver overhead is for *loops*.
- **Pure-fleet workflows.** If hypotheses come from a notebook and you just want to run them, skip the driver and use `design_batch_from_hypotheses` + `design_enqueue_batch` directly.
- **Legacy autonomous mode.** `crucible research start` still works for the in-process Anthropic-keyed agent. Use it if you don't want to write an orchestrator and you're OK with `ANTHROPIC_API_KEY` inside Crucible.

## See also

- [orchestrator-contract.md](orchestrator-contract.md) — the underlying wire protocol.
- [judge-separation.md](judge-separation.md) — the LM-as-judge contract enforced at session start.
- [harness-optimization.md](harness-optimization.md) — what `harness_autonomous_loop` is looping over.
- `tests/integration/test_autonomous_loop_e2e.py` — runnable example of the full lifecycle.
- `examples/full_autonomous_discovery/` — end-to-end demo project.
