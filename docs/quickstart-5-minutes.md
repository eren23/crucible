---
layout: default
title: 5-Minute Quickstart
---

# 5-Minute Quickstart

From `git clone` to first leaderboard + next-action recommendation in five steps. **No RunPod account, no API keys, no cloud.** This runs entirely on your laptop using the built-in smoke preset.

The longer [Getting Started](getting-started) guide covers fleet provisioning, presets beyond smoke, hypotheses, and W&B integration. Come back to it once this 5-step path runs clean.

---

## Prerequisites

- Python 3.10 or newer
- ~500 MB free disk for the smoke-run artifacts

That's it. No GPU is required — the smoke preset is CPU-friendly.

---

## Step 1 — Install (60s)

```bash
git clone https://github.com/your-org/crucible.git
cd crucible
pip install -e ".[torch]"
```

The `[torch]` extra is enough for the smoke run. The full `[all]` extra (TUI, MCP, anthropic, wandb) ships with the optional features; you can add them later.

Verify the CLI registered:

```bash
crucible --help
```

---

## Step 2 — Copy the basic example (15s)

The shortest path is to start from the bundled example, which ships a minimal `crucible.yaml` and a dummy `train.py`:

```bash
cp -r examples/basic ../hello-crucible
cd ../hello-crucible
```

(If you already have your own `train.py` you'd rather use, `crucible init` instead — but you'll then need to point `training[0].script` in `crucible.yaml` at your file and adjust the policy/wandb defaults; see [Getting Started](getting-started).)

The example's `crucible.yaml` is laptop-friendly, but the production defaults that Crucible's policy enforcer expects need to be relaxed once for a local run. Append:

```bash
cat >> crucible.yaml <<'EOF'

execution_policy:
  require_remote: false
  required_provider: ""
  allow_local_dev: true

wandb:
  required: false
EOF
```

If you have a `WANDB_API_KEY` and want metrics tracked, skip the `wandb` block and set `wandb.project` to a project name instead — the smoke run will log there.

---

## Step 3 — Run a smoke experiment (10s)

```bash
crucible run experiment --preset smoke
```

The bundled `train.py` simulates a 500-step training run in ~6 seconds — Crucible's `OutputParser` reads its stdout for `step:X/Y train_loss:...` and `val_loss:.../val_bpb:...` patterns. When it finishes you'll see a `Status: completed` block with `val_loss`, `val_bpb`, `steps_completed`, and the serialized model byte count.

This run is now in your local results store. Confirm:

```bash
crucible analyze rank --top 5
```

You should see one row. That's your leaderboard.

---

## Step 4 — Ask Crucible what to do next (10s)

Crucible exposes a `tool_router` MCP tool that inspects your current state and recommends the next action. Run it through the CLI:

```bash
crucible mcp call tool_router
```

Output:

```json
{
  "recommended_tool": "get_leaderboard",
  "rationale": "New results are in. Inspect the leaderboard before forming the next hypothesis.",
  "alternatives": [
    {"tool": "research_request_prompt",
     "rationale": "Leaderboard has fresh entries. Request a reflection prompt (stage='reflection')."},
    {"tool": "context_push_finding",
     "rationale": "Record observations from the latest results as findings."}
  ],
  "state": {
    "nodes": {"total": 0, "ready": 0, ...},
    "queue": {"total": 0, ...},
    "completed_experiments": 1,
    "hypotheses_pending": 0,
    "active_session": null,
    "orphans_present": false
  }
}
```

If you'd run two experiments and one was still mid-flight, the recommendation would shift to `get_fleet_status` ("Experiments are running. Poll fleet status until they complete."). The router walks an 11-branch decision graph against the live state — fleet inventory, queue, leaderboard, research state, and any active autonomous session — and picks the right next step.

---

## Step 5 — Pull a research briefing (5s)

```bash
crucible mcp call get_research_briefing
```

The briefing returns project state, top leaderboard rows, hypotheses, findings, notes, and (as of Phase 2.2) a structured `next_actions` field carrying the same recommendation the router gave you. The markdown summary is the human-readable version:

```
# Research Briefing: hello-crucible

**Project:** hello-crucible (val_loss, minimize)
**Budget:** 10.0 / 10.0 hours remaining

## Recent Experiments (1)
- smoke (val_loss=5.234, status=completed)

...

## Recommended Next Tool
- **`get_leaderboard`** — New results are in. Inspect the leaderboard before forming the next hypothesis.
  - alt: `research_request_prompt` — Leaderboard has fresh entries. Request a reflection prompt (stage='reflection').
  - alt: `context_push_finding` — Record observations from the latest results as findings.
```

---

## Where to go next

You've now run Crucible's full read loop end-to-end. The write loop — generating new hypotheses, dispatching across a fleet, reflecting, synthesizing across findings — is the same shape, just with more steps. Three natural next directions:

1. **Run the full autonomous loop**: `crucible research run --iterations 3 --tier smoke --budget-usd 5`. Drives `hypothesize → batch → enqueue → dispatch → collect → reflect → repeat` via the orchestrator-contract protocol. Bring your own LLM through Claude Code, Codex, or a custom MCP client.
2. **Add a real GPU**: provision a RunPod node (`crucible fleet provision --count 1`) and re-run with `--preset proxy`. The fleet bootstrap → dispatch → collect chain is what production looks like.
3. **Import an autoresearch project**: if you already use Karpathy's `autoresearch` setup, run `crucible import autoresearch /path/to/project` to wrap it into a Crucible spec and inherit the fleet, judge-separation, and cross-project memory layers.

For the long version with every knob explained, see [Getting Started](getting-started). For the agent's view of the system, see [MCP Tools Reference](mcp-tools).
