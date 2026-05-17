# Full Autonomous Discovery — End-to-End Demo

> **Phase 4.4 deliverable.** A self-contained project that exercises every
> Phase 1+2+3+4 surface in ~30 minutes on a laptop. No real GPU, no fleet,
> no LLM keys baked in.

The plan promised "a runnable, ~$5-of-spot-GPU project that goes from
`git clone` to a paper draft in 30 minutes." Since the loop has no LLM keys
inside Crucible, the orchestrator (you, via Claude Code / Codex / a local
LLM client) drives the LLM round-trips. The demo's training script is the
simulated trainer from `examples/basic/` — it produces realistic
`val_loss`/`val_bpb` lines without burning compute, so you can run the
**entire** loop locally and see the orchestrator-contract pattern
end-to-end before pointing it at a real GPU.

## What this demonstrates

| Phase | Surface | How it shows up here |
|-------|---------|----------------------|
| 1.1 | `autonomous_research_loop` | The whole loop is one MCP session |
| 1.2 | `state_snapshot` + content-hash stale-submit guard | Each submit references its prior snapshot |
| 1.8 | Cost budget guard | `budget_usd=5` passed to `action=start` |
| 2.1 | `tool_router` | Step 4 below — "what's next?" without guessing |
| 2.2 | Briefing `next_actions` | `get_research_briefing` shows the recommendation inline |
| 2.5 | `runs_search` | Step 8 — filter results across iterations |
| 3.1 | `research_arxiv_search` | Step 3 — seed the first hypothesis with literature |
| 3.4 | `hpo_create_study` (Optuna) | Optional: drive a parameter sweep |
| 4.1 | `note_generate_paper_draft` | Step 10 — final paper markdown |
| 4.2 | `design_synthesize_from_findings(policy=memory_filter)` | Step 7 — synthesis hypotheses |
| 4.3 | `research_peer_sync` | Optional: share top finding with peers on HF |

## Prerequisites

- Python 3.10+
- An MCP-speaking orchestrator (Claude Code, Codex, or any LLM client
  that can call MCP tools).
- ~30 minutes.

## Step-by-step playbook

> All commands are run from this directory (`examples/full_autonomous_discovery/`).
> Substitute `crucible` for whichever entry point you have installed
> (`PYTHONPATH=src .venv/bin/crucible` if running from the source tree).

### Step 1 — Setup (1 min)

```bash
cd examples/full_autonomous_discovery
crucible recipe list   # sanity check — should print "No recipes yet."
```

### Step 2 — Tool router sees an empty project (10s)

```bash
crucible mcp call tool_router
```

Expected: recommended_tool is `research_request_prompt` (Phase 2.1 router
branch: no completions, no pending hypotheses, no pods → kickstart with
a hypothesis).

### Step 3 — Seed the loop with arXiv literature (Phase 3.1, ~20s)

```bash
crucible mcp call research_arxiv_search --args '{"query": "weight tying language model", "limit": 5, "categories": ["cs.LG", "cs.CL"]}'
```

You'll get 5 recent papers as JSON. Hand them to your orchestrator —
they'll inform the next step's hypotheses.

### Step 4 — Open an autonomous loop (Phase 1.1, ~2 min)

```bash
crucible mcp call autonomous_research_loop --args '{"action": "start", "iterations": 3, "tier": "smoke", "budget_usd": 5, "with_literature": true, "literature_k": 5}'
```

Crucible returns `{session_id, iteration: 0, stage: "hypothesis", system,
user, schema, state_snapshot}`. The `user` block includes the literature
you pulled in Step 3 (because `with_literature=true` triggers Phase 3.1
inside the session driver).

### Step 5 — Orchestrator submits hypotheses

The orchestrator (your LLM client) calls its own LLM with `system` +
`user`, parses against `schema`, and submits the response:

```bash
crucible mcp call autonomous_research_loop --args '{
  "action": "submit",
  "session_id": "<SID>",
  "state_snapshot": <SNAP>,
  "response": {
    "hypotheses": [
      {"hypothesis": "tied embeddings reduce overfitting",
       "name": "tied_embed", "expected_impact": 0.05,
       "confidence": 0.7, "config": {"TIE_EMBEDDINGS": "true"},
       "rationale": "Inan et al 2017 shows it tightens train/val gap.",
       "family": "baseline"},
      {"hypothesis": "lower lr stabilizes the simulated trainer",
       "name": "lower_lr", "expected_impact": 0.04, "confidence": 0.6,
       "config": {"LR": "5e-4"}, "rationale": "smaller step size",
       "family": "baseline"}
    ]
  }
}'
```

### Step 6 — Run the experiments

```bash
# Each hypothesis becomes a config override; the simulated trainer runs.
crucible run experiment --preset smoke --set TIE_EMBEDDINGS=true --name tied_embed
crucible run experiment --preset smoke --set LR=5e-4 --name lower_lr
```

Each takes ~30 seconds. Leaderboard:

```bash
crucible analyze rank --top 5
```

### Step 7 — Reflection + memory-aware synthesis (Phase 4.2)

After 2-3 iterations have populated the findings ledger, switch to the
synthesis policy to pull the most promising cross-finding combos:

```bash
crucible mcp call design_synthesize_from_findings --args '{"scope": "track", "policy": "memory_filter", "k": 3}'
```

You'll get 3 orchestrator-shaped prompts ranked by confidence × recency
× cross-project diversity (Phase 4.2's scoring).

### Step 8 — Search the results (Phase 2.5)

```bash
crucible mcp call runs_search --args '{
  "where": "status == \"completed\" and result.val_loss < 5.0",
  "order_by": "result.val_loss",
  "direction": "asc",
  "limit": 10,
  "select": ["name", "result.val_loss", "model_bytes"]
}'
```

### Step 9 — Pull a research briefing (Phase 2.2)

```bash
crucible mcp call get_research_briefing
```

The response includes a `next_actions` field with the structured tool
recommendation, AND a `markdown_summary` with the rendered briefing
ending in "## Recommended Next Tool".

### Step 10 — Generate the paper draft (Phase 4.1)

```bash
# Step A: pull the prompt.
crucible mcp call note_generate_paper_draft --args '{"action": "request_prompt", "track_name": "default"}'
```

Hand the `system` + `user` to your orchestrator's LLM, get back the JSON
response, then:

```bash
# Step B: submit + render.
crucible mcp call note_generate_paper_draft --args '{
  "action": "submit",
  "track_name": "default",
  "response": { <the JSON from the orchestrator> }
}'
```

The returned `markdown` is your paper draft. Save it:

```bash
crucible mcp call note_generate_paper_draft --args '{"action": "submit", "track_name": "default", "response": <...>}' | \
  python3 -c "import sys, json; print(json.load(sys.stdin)['markdown'])" > paper-draft.md
```

### Step 11 (optional) — Share with peers (Phase 4.3)

If you've configured `hf_collab.leaderboard_repo` and have an `HF_TOKEN`:

```bash
crucible mcp call research_peer_sync --args '{"challenge_id": "weight-tying-2026"}'
```

Returns the URL of the shared HF Discussion thread + any peer agents'
top findings.

## What you have at the end

```
examples/full_autonomous_discovery/
├── crucible.yaml                  # the project spec
├── train.py                       # simulated trainer
├── experiments.jsonl              # 6-8 completed runs
├── .crucible/
│   ├── autonomous_sessions/       # 1 closed session (canceled / done)
│   ├── research_state.jsonl       # hypotheses + findings + beliefs
│   └── ...
└── paper-draft.md                 # rendered markdown paper
```

A new user can clone, run these 11 steps, and have a paper draft +
reproducibility bundle by lunch. The autonomous loop's closed-loop
nature is end-to-end visible.

## Going from this to a real GPU run

The exact same playbook works with a real fleet. Two changes:

1. Add a RunPod (or SSH) provider block to `crucible.yaml`:
   ```yaml
   provider:
     type: runpod
     gpu_types: ["NVIDIA GeForce RTX 4090"]
   ```
2. Replace `crucible run experiment` (steps 6) with the fleet flow:
   ```bash
   crucible fleet provision --count 1
   crucible fleet bootstrap
   crucible run enqueue --spec ./batch.json
   crucible run dispatch
   crucible run collect
   ```

The autonomous loop driving the experiment lifecycle doesn't change —
that's the point of the orchestrator-contract pattern. Local laptop or
fleet of A100s, same MCP surface.

## What's NOT in this demo

- Real code mutation (Phase 5+; design at `docs/code-mutation-design.md`)
- TUI cockpit interaction (run `crucible tui` separately; Phase 2.3)
- External_mcp servers (Phase 3.5; enable in `crucible.yaml` to wire
  Codex or Spider Chat into the loop)
- Real arXiv literature ingestion in the autonomous loop hook
  (`with_literature=true` is HF-Papers only today; arxiv as a side
  query is shown in Step 3)
