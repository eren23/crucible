# Flagship Showcase — Parameter Golf Autonomous Discovery

> **Thrust C deliverable** (Phase 5.1 follow-on). A reproducible
> "Crucible discovered X on $50 of RunPod" recipe that ties together
> every surface shipped through Phase 5: autonomous loop,
> hypothesis tournament, code mutation, judge-separation contract,
> GIANTS-style synthesis, HF publish.

The point of this showcase is **not** to claim a new Nature paper. It
is to demonstrate, with a single command, that an open + reproducible
+ commodity-GPU + plugin-native stack can match what closed systems
present in headlines.

## The challenge

Parameter Golf: minimise validation BPB on the [Parameter Golf
benchmark][pg] under a 16 MB parameter budget and 240 GPU-min training
budget. The competition's original baseline is `train.py` from the
parameter-golf-dev repo at `e002a6d`.

[pg]: https://github.com/OpenAI/parameter-golf

## What this demo proves

| Crucible surface | How this demo exercises it |
|---|---|
| Phase 1 — autonomous loop | One MCP session drives 5 hypothesis ↔ reflection rounds |
| Phase 1 — judge separation | Reward/eval/audit judges in three distinct families |
| Phase 1 — literature pre-injection | First hypothesis is seeded with HF Papers context |
| Phase 2 — tool_router | Used between phases to decide the next call |
| Phase 3.4 — Optuna bridge | Optional HPO sub-loop over the top hypothesis |
| Phase 4.1 — paper draft | Final result is a `note_generate_paper_draft` markdown |
| Phase 4.2 — GIANTS synthesis | Round 3+ mines hub findings for cross-architecture moves |
| Phase 4.3 — peer sync | Optional cross-agent collab via HF Discussions |
| Phase 5.1 — code mutation | Round 4+ uses `code_mutation_propose` / `apply` |
| Phase 5.1 — tournament | Round 2+ uses `hypothesis_tournament_*` for ranking |
| Phase 5.1 — proximity dedupe | `hypothesis_cluster` filters before each tournament |
| Tier 14 — HF publish | Final leaderboard, findings, recipes pushed to HF |

## Budget

- Compute: ~$50 of RunPod spot (5 rounds × 8 variants × ~$1.25 each at smoke/screen tier).
- Wall clock: 4-6 hours end-to-end on a single 4090 spot pod.
- Orchestrator cost: ~$5-10 of LLM API depending on debate-judge model.

## Prerequisites

- RunPod API key in `.env` (`RUNPOD_API_KEY=...`)
- HuggingFace token in `.env` (`HF_TOKEN=...`) — for publish steps
- WandB key in `.env.runpod.local` (`WANDB_API_KEY=...`)
- An MCP-speaking orchestrator (Claude Code, Codex, etc.)
- Judges configured in `crucible.yaml`:
  ```yaml
  judges:
    reward_judge: {model: gemini-2.5-flash, family: gemini}
    eval_judge:   {model: claude-opus-4-7,  family: claude}
    audit_judge:  {model: qwen3-14b,        family: qwen}
    enforce_separation: true
  hf_collab:
    enabled: true
    leaderboard_repo: your-username/crucible-param-golf
    artifacts_repo:   your-username/crucible-param-golf-artifacts
  ```

## The 11-step playbook

> All commands run from the project root.

### Step 1 — Sanity check

```bash
crucible recipe get flagship-param-golf-discovery > /tmp/recipe.yaml
crucible mcp call code_mutation_list
crucible mcp call config_get_modalities
```

Verify the recipe loads, the Phase 5.1 code-mutation policies are
registered, and the training modality (LM / token) is built in.

### Step 2 — Seed literature

```bash
crucible mcp call research_literature_search --args \
  '{"query": "low parameter count language modelling efficient attention", "limit": 8}'
```

The HF Papers + arXiv hits feed the first hypothesis round.

### Step 3 — Open the autonomous loop

```bash
crucible mcp call autonomous_research_loop --args '{
  "action": "start",
  "iterations": 5,
  "tier": "screen",
  "budget_usd": 50,
  "with_literature": true,
  "literature_k": 8
}'
```

Returns `{session_id, stage: "hypothesis", system, user, schema,
state_snapshot}`. Orchestrator runs its LLM and continues at step 4.

### Step 4 — Round 1 hypothesis batch + dedupe

Orchestrator runs the hypothesis-stage LLM and produces 8-12 candidate
hypotheses. Then:

```bash
crucible mcp call hypothesis_cluster --args '{
  "hypotheses": <round_1_hypotheses>, "threshold": 0.7, "n": 3
}'
```

Use the returned `keepers` for the round-1 tournament.

### Step 5 — Round 1 tournament

```bash
crucible mcp call hypothesis_tournament_create --args '{
  "name": "round_1", "hypotheses": <keepers>
}'
# Loop 5 rounds:
crucible mcp call hypothesis_tournament_pair --args '{"name": "round_1", "policy": "elo_close"}'
# orchestrator runs debate-judge LLM (audit family — NOT same family as reward/eval)
crucible mcp call hypothesis_tournament_submit --args '{
  "name": "round_1", "winner_id": <id>, "loser_id": <id>, "rationale": <text>
}'
# After 5 rounds:
crucible mcp call hypothesis_tournament_rank --args '{"name": "round_1", "top_k": 4}'
```

### Step 6 — Provision + execute top-K

```bash
crucible mcp call provision_nodes --args '{"count": 4, "gpu_type": "RTX 4090"}'
crucible mcp call bootstrap_nodes
crucible mcp call design_enqueue_batch --args '{"hypotheses": <top_4>, "preset": "screen"}'
crucible mcp call dispatch_experiments
# wait, then:
crucible mcp call collect_results
crucible mcp call get_leaderboard
```

### Step 7 — Reflect + iterate

The reflection round uses the stateless orchestrator-contract pair
(`research_request_prompt` + `research_submit`). If you started the
loop via `autonomous_research_loop(action='start', ...)` and want the
session's budget guard / state snapshot to track each round, also
call `autonomous_research_loop(action='submit', session_id=..., response=...)`
after the stateless submit.

```bash
crucible mcp call research_request_prompt --args '{"stage": "reflection"}'
# orchestrator runs reflection LLM
crucible mcp call research_submit --args '{"stage": "reflection", "response": <json>}'

# Optional — keep the persisted session's budget guard in sync.
crucible mcp call autonomous_research_loop --args '{
  "action": "submit", "session_id": "<from start_loop>", "response": <json>
}'
```

This promotes the best result of round 1 to a finding via
`context_push_finding(auto_promote=True)`. Repeat steps 4-7 four more
times (rounds 2-5).

### Step 8 — Round 4+: code mutation

Once the loop has identified a promising architecture, switch from
config-tweaks to actual code mutations:

```bash
crucible mcp call code_mutation_propose --args '{
  "target_file": "src/crucible/models/architectures/baseline.py",
  "intent": "swap attention block for a windowed variant",
  "mutation_scope": ["src/crucible/models/"]
}'
# orchestrator runs LLM, gets {diff, hypothesis, rationale}
crucible mcp call code_mutation_apply --args '{
  "policy": "llm_diff",
  "target_file": "src/crucible/models/architectures/baseline.py",
  "mutation_scope": ["src/crucible/models/"],
  "llm_response": <json>,
  "scorer": {"cmd": ["python3", "train_gpt.py", "--preset", "smoke"], "score_pattern": "val_bpb:([0-9.]+)"}
}'
```

### Step 9 — Round 5: GIANTS synthesis

```bash
crucible mcp call design_synthesize_from_findings --args '{
  "policy": "memory_filter",
  "k": 3
}'
```

Generates new hypothesis prompts that combine two existing findings.
Feed each through the tournament + execute pipeline.

### Step 10 — Meta-review + paper draft

```bash
crucible mcp call research_meta_review --args '{
  "track_name": "param-golf-flagship",
  "tournament_name": "round_5",
  "top_k": 5
}'
# orchestrator runs LLM → save via note_add

crucible mcp call note_generate_paper_draft --args '{
  "action": "request_prompt",
  "track_name": "param-golf-flagship"
}'
# orchestrator runs LLM, then:
crucible mcp call note_generate_paper_draft --args '{
  "action": "submit",
  "track_name": "param-golf-flagship",
  "response": <json>
}'
```

### Step 11 — Publish

```bash
crucible mcp call hf_publish_leaderboard --args '{"top_n": 50, "challenge": "param-golf"}'
crucible mcp call hf_publish_findings --args '{}'
crucible mcp call hf_publish_recipes --args '{}'
```

## What's recorded

After step 11:

- `.crucible/notes/` — meta-review + paper draft markdown
- `.crucible/tournaments/round_{1..5}/` — full Elo + event history per round
- `.crucible/search_trees/` — code-mutation tree (if step 8 ran)
- `runs.jsonl` — every experiment with full provenance
- W&B project — every training run with `CRUCIBLE_VARIANT_NAME` tagged
- HF dataset `your-username/crucible-param-golf` — leaderboard + findings + recipes
- HF Discussions — peer-sync threads if you opted in

## Scope check

Per `docs/positioning.md`:
- ✅ Strengthens autonomous loop
- ✅ Strengthens reproducibility (everything versioned)
- ✅ Strengthens plugin architecture (uses every plugin family)
- ✅ Strengthens commodity-GPU fit ($50 spot budget)
- ❌ Drifts into excluded territory: no — this is parameter-golf,
  which is our origin and clearly on-thesis

## Cross-references

- Tournament docs: `docs/multi-agent-tournament.md`
- Tournament recipe: `docs/recipes/co-scientist-style-tournament.yaml`
- Judge-separation contract: `docs/judge-separation.md`
- Code mutation design: `docs/code-mutation-design.md`
- ERA-style mutation demo: `examples/code_mutation_era_replica/README.md`
- Original autonomous discovery demo (no GPU): `examples/full_autonomous_discovery/README.md`
