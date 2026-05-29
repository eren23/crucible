---
layout: default
title: Multi-Agent Hypothesis Tournament
---

# Multi-Agent Hypothesis Tournament

Crucible's tournament surface mirrors the multi-agent debate / Elo
pattern from DeepMind Co-Scientist (*Nature*, May 2026) — but stays
orchestrator-contract-only. Crucible persists the tournament state,
emits prompt envelopes, and updates Elo when results come back. The
LLM debate-judge runs in the *orchestrator's* process, not Crucible.

## When to use it

Use a tournament when you have **more candidate hypotheses than
compute** and need a cheap way to rank them before burning pod time.
The Co-Scientist paper shows Elo tournament ranking correlating well
with downstream experimental success — same finding applies here.

Skip the tournament when:
- You only have 1-3 hypotheses (just run them all).
- You can score each hypothesis cheaply (run them, use the real
  metric, skip the proxy).
- Your hypotheses are syntactically identical and only differ in
  small numeric knobs — Optuna (Tier 18) is the right tool.

## The contract

```
1. hypothesis_cluster(hypotheses)
   → keepers       # prune near-duplicates first
2. hypothesis_tournament_create(name, keepers)
3. loop N rounds:
     pair = hypothesis_tournament_pair(name)        # → {system, user, schema}
     llm  = orchestrator.run_llm(pair)              # debate judge
     hypothesis_tournament_submit(name, winner_id=llm.winner_id,
                                  loser_id=llm.loser_id,
                                  rationale=llm.rationale)
4. ranking = hypothesis_tournament_rank(name, top_k=K)
5. (optional) research_meta_review(track, tournament_name=name)
              → {system, user, schema}
              → orchestrator.run_llm(...) → note_add(...)
6. execute the top-K via design_enqueue_batch / dispatch_experiments.
```

`hypothesis_cluster` uses shingle-Jaccard by default (pure stdlib).
KMeans backend is opt-in via `backend='kmeans'` and requires
`scikit-learn`.

## Judge separation

Per `docs/judge-separation.md`, the tournament's **debate judge** is a
*third* judge — it must run in a model family different from both the
**reward judge** (used by harness_iterate / tree_expand_grpo) and the
**eval judge** (used to score the final results). Configure under
`judges:` in `crucible.yaml`:

```yaml
judges:
  reward_judge: {model: gemini-2.5-flash, family: gemini}
  eval_judge:   {model: claude-opus-4-7,  family: claude}
  audit_judge:  {model: qwen3-14b,        family: qwen}
  enforce_separation: true
```

The audit judge is the most natural choice for the tournament debate
role — already separated from reward + eval by contract.

## Storage layout

```
.crucible/tournaments/{name}/
  state.yaml      # current Elo + win/loss/draw per hypothesis
  events.jsonl    # append-only event log (create, pair, submit)
  .lock           # file lock for concurrent orchestrator instances
```

Two orchestrator processes can drive the same tournament — the file
lock serialises the mutating calls (pair / submit). Pairing in one
process while another submits is safe.

## Pairing policies

| Policy | When to use |
|---|---|
| `random` | Early rounds, when you want broad exploration. |
| `swiss` | Mid-tournament: pair by wins-minus-losses for tightest matches. |
| `elo_close` | Late rounds, when you want to break ties between close-rated hypotheses. (Default.) |

All three honor `no_repeat_within` (default 3) so the same pair
doesn't show up in consecutive rounds.

## Composing with the autonomous research loop

The tournament tools work alongside `autonomous_research_loop` — the
orchestrator owns the composition. A typical loop iteration:

1. `autonomous_research_loop(action='request_prompt', stage='hypothesis')`
   → orchestrator runs LLM → gets N hypotheses.
2. `hypothesis_cluster(hypotheses)` → drop near-duplicates → keepers.
3. `hypothesis_tournament_create(name=f"round_{i}", hypotheses=keepers)`.
4. M debate-judge rounds.
5. `hypothesis_tournament_rank(name, top_k=K)` → execution wave.
6. Run the wave via `design_enqueue_batch` → `dispatch_experiments` →
   `collect_results`.
7. `autonomous_research_loop(action='submit', stage='hypothesis', response=...)`
   so the loop's research state records what was selected and run.
8. Iterate.

The recipe at `docs/recipes/co-scientist-style-tournament.yaml`
captures this verbatim as a `recipe_get`-able playbook.

## Meta-review

`research_meta_review(track_name, tournament_name=..., top_k=...)`
returns a `{system, user, schema}` envelope that the orchestrator's
LLM completes to produce a research overview:

```json
{
  "synthesis": "1-3 paragraph summary grounded in the findings + Elo ranking",
  "top_directions": ["...", "..."],
  "open_questions": ["...", "..."],
  "suggested_next": ["...", "..."]
}
```

Save the response via `note_add` (and optionally cross-post via
`note_post_to_hf_discussions` for peer-agent visibility).

## Comparison to DeepMind Co-Scientist

| Co-Scientist agent | Crucible surface |
|---|---|
| Generation | Your orchestrator's hypothesis-stage LLM call (existing pattern) |
| Reflection | `autonomous_research_loop(stage='reflection')` |
| Ranking | `hypothesis_tournament_*` (Elo + pairing) |
| Evolution | `code_mutation_propose` / `code_mutation_apply` (Phase 5.1) |
| Proximity | `hypothesis_cluster` |
| Meta-review | `research_meta_review` |
| Supervisor | Your orchestrator's main loop |

Crucible deliberately doesn't bundle a Supervisor — that's the
orchestrator's job, and bundling it would conflict with the
no-LLM-keys-in-Crucible contract. The MCP tool surface is the
interface; the orchestrator's planner picks which tool to call when.
