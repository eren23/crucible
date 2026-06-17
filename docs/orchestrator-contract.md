# The Orchestrator Contract

Crucible is **infrastructure** — fleet, experiments, data, findings, search trees. Taste — hypothesis generation, reflection, planning — comes from outside. The orchestrator contract is the wire-protocol between the two.

This doc covers what the contract is, why it exists, the four call shapes you'll see, and how to drive it from any agent (Claude Code via MCP, a Python script, a custom CLI tool).

## TL;DR

```text
1. Crucible:    research_request_prompt(stage)            → {system, user, schema, state_snapshot}
2. Orchestrator: calls its own LLM with (system, user, schema), parses per schema
3. Crucible:    research_submit(stage, response, state_snapshot) → {applied, summary, ...}
4. Loop steps 1-3 for each stage. Pure fleet ops in between.
```

No `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `litellm` lives inside Crucible. The only creds Crucible touches are infrastructure creds (`RUNPOD_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`).

## Why this exists

Two reasons.

**Vendor neutrality.** Models change quarterly. Burning a hard dependency on one vendor's SDK into the infra layer means an LLM-vendor outage or a model deprecation can take the whole platform down. The orchestrator-contract path lets the same Crucible instance be driven by Claude, GPT-5, Gemini, Qwen, or a human at a terminal — the LLM client is the caller's problem, not ours.

**Audit + reproducibility.** Every prompt Crucible emits is captured in the session event log. Every response submitted is checked against a `state_snapshot` from the prompt time and rejected if state has moved. This means a session can be replayed: load the prompt, feed it to any model, compare the response. No magic round-trip you can't see.

## The four verbs

Stages produce prompts; orchestrators supply responses. The two non-stage verbs (`status`, `cancel`) are read-only and lifecycle-only.

### request_prompt(stage, ...)

Returns:

```python
{
    "system": "You are...",        # role/instructions
    "user":   "Current state...",  # state-conditioned task description
    "schema": {                     # JSON schema the response must satisfy
        "type": "object",
        "properties": {...},
        "required": [...]
    },
    "state_snapshot": {             # opaque object — pass back on submit
        "history_len": 12,
        "hypotheses_len": 4,
        "beliefs_len": 2,
        "findings_len": 1,
        "content_hash": "a5b5cd1b555684f9"
    }
}
```

Three valid `stage` values today:

| Stage | Purpose | Response shape |
|---|---|---|
| `hypothesis` | propose 3-5 experiments | `{hypotheses: [...]}` |
| `reflection` | update beliefs, promote/kill | `{beliefs, promote, kill, ...}` |
| `briefing` | read-only project summary | n/a (no submit) |

The schema is the source of truth — your LLM call must produce output that validates against it. Crucible's parsers are forgiving (string-coercion for numeric fields, name defaults if missing) but `config` on a hypothesis is required and an empty `hypotheses: []` is a no-op.

### submit(stage, response, state_snapshot)

Applies the parsed response to project state. Returns:

```python
{
    "applied": True,
    "summary": "Added 3 hypotheses",
    "..." : "stage-specific fields"
}
```

Or raises `StaleSubmitError` if `state_snapshot` doesn't match the state's current snapshot — see below.

### status

Pure read. Returns the session yaml content + paths to artifacts.

### cancel

Terminal transition. Returns `{session_status: "canceled", checkpoint_path: "...", already_terminal: bool}`.

## State snapshot — the stale-submit guard

Each `request_prompt` embeds a snapshot of the research state at prompt time:

```python
{
    "history_len": 12,        # count of experiment results
    "hypotheses_len": 4,      # count of hypotheses
    "beliefs_len": 2,         # count of beliefs
    "findings_len": 1,        # count of promoted findings
    "content_hash": "a5b5..." # SHA-256 of stable serialization (truncated)
}
```

When the orchestrator submits a response, it passes the snapshot back. Crucible recomputes the current snapshot and rejects the submit if any field differs:

- A length counter moves → another agent added experiments / hypotheses / beliefs in between.
- `content_hash` moves but lengths don't → a wholesale replacement happened (e.g., `update_beliefs` mutated existing entries in place).

Both modes raise `StaleSubmitError` with a message identifying which fields shifted. The orchestrator's correct response is to call `request_prompt` again with the now-current state, re-run its LLM, and retry the submit. No silent overwrite.

**Why both length and hash?** Length alone misses in-place mutations (`update_beliefs` keeps the same count but changes content). Hash alone is slower to debug because the message doesn't tell you whether the change was structural or content. Together they pinpoint the divergence.

## A complete round-trip

```python
from crucible.core.config import load_config
from crucible.researcher import orchestrator_api as oa
from crucible.researcher.state import ResearchState

config = load_config("crucible.yaml")
state = ResearchState(config.project_root / "research_state.jsonl")

# Step 1: ask Crucible for the hypothesis prompt.
prompt = oa.request_prompt(stage="hypothesis", config=config, state=state)

# Step 2: orchestrator calls its own LLM (any model, any client).
import anthropic
resp = anthropic.Anthropic().messages.create(
    model="claude-opus-4-7",
    system=prompt["system"],
    messages=[{"role": "user", "content": prompt["user"]}],
    max_tokens=4096,
)
parsed = json.loads(resp.content[0].text)

# Step 3: submit. Crucible validates state_snapshot + applies.
result = oa.submit_response(
    stage="hypothesis",
    response=parsed,
    config=config,
    state=state,
    iteration=0,
    state_snapshot=prompt["state_snapshot"],
)
# result["applied"] == True, result["summary"] == "Added 3 hypotheses"

# Now run the fleet (pure infra, no LLM):
#   design_batch_from_hypotheses → design_enqueue_batch
#   → dispatch_experiments → collect_results
# Then reflect:
prompt = oa.request_prompt(stage="reflection", config=config, state=state, iteration=0)
# ... same shape, same submit dance.
```

The same loop is wrapped by the persisted-session driver — see [autonomous-loops.md](autonomous-loops.md) — so you don't have to manage iteration counters and snapshots yourself.

## Driving from MCP

When Crucible is running as an MCP server (`crucible mcp serve`), an MCP-capable client (Claude Code, custom agent) sees:

- `research_request_prompt(stage, focus_family?, iteration?, literature_context?)`
- `research_submit(stage, response, state_snapshot?, iteration?)`

The schema is in each tool's MCP descriptor. The state_snapshot is plumbed through the tool args — the orchestrator must hand it back on submit if it wants the stale-submit guard active.

The legacy mode lives at `src/crucible/researcher/loop.py` and `src/crucible/researcher/llm_client.py`. It's preserved because the original Parameter Golf competition workflow used it; it doesn't share code with the orchestrator-contract path.

## What Crucible never does

- Call an LLM API directly (outside the legacy path).
- Hold an API key for an LLM provider.
- Pick a model — model selection is the orchestrator's responsibility.
- Validate that the orchestrator's LLM call was correct — Crucible only validates the parsed response shape.

## What you can build on top

Because the contract is wire-level (just JSON in / JSON out), you can:

- Replace the LLM with a deterministic mock for tests (see `tests/integration/test_autonomous_loop_e2e.py`).
- Drive the loop from a notebook with manual responses for a human-in-the-loop workflow.
- Plug in a multi-model ensemble: send the prompt to three models, pick the response by quorum, submit the winner.
- Replay an old session: re-emit the prompts from the event log, send to a new model, compare outputs.

The contract is the moat. Most autonomous-research platforms bake the LLM caller into the infra. Crucible doesn't, so the same platform runs against any LLM and any orchestrator strategy.

## See also

- [autonomous-loops.md](autonomous-loops.md) — the three persisted-session drivers (research, tree, harness) built on this contract.
- [judge-separation.md](judge-separation.md) — what to do when the orchestrator runs an LM-as-judge loop on top of the contract.
- [positioning.md](positioning.md) — how the contract relates to Crucible's overall niche.
