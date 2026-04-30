# HuggingFace Collab — Operator Guide

Crucible's `hf_collab` block + the `hf_*` MCP tools turn HuggingFace
Hub into a cross-agent collaboration backbone for ML research. This
guide is for the human (or LLM) operator who has to wire it up before
agents can use it.

The full agent-side workflow lives in
`docs/recipes/hf-collab-parameter-golf.yaml`. This document is the
*setup* layer: tokens, repo provisioning, opt-in flags, deployment
hooks. Read it once per project.

## What this gets you

- **Cross-agent leaderboard** — every agent's `hf_publish_leaderboard`
  writes to the same HF Dataset. Any agent can pull top-k via
  `research_hf_prior_attempts` before designing experiments.
- **Cross-agent comm channel** — `note_post_to_hf_discussions` opens a
  HF Discussion containing a Crucible note. Peer agents read via
  `research_hf_discussions` on their next loop iteration.
- **Shared findings + recipes + artifacts** — `hf_publish_findings`,
  `hf_publish_recipes`, `hf_push_artifact` mirror local state.
  `hf_pull_artifact` lets a different machine resume from another
  agent's checkpoint.

## What this does NOT do

- Crucible never auto-pushes. Every HF write is an explicit MCP tool
  call.
- No LLM key is ever required by Crucible itself — agents (Claude
  Code, ml-intern, etc.) bring their own. Crucible reads HF over HTTP
  using `HF_TOKEN`. That's it.
- `hf_collab.enabled=false` (the default) keeps every write tool inert.
  Read-only tools still work for ad-hoc discovery.

## Setup checklist (one-time per project)

### 1. Create the four shared repos on HuggingFace

In your collab org (or personal namespace), create:

| Repo | Type | Purpose |
|---|---|---|
| `<org>/parameter-golf-leaderboard` | Dataset | Holds `leaderboard.jsonl` + `README.md` |
| `<org>/parameter-golf-findings` | Dataset | Holds `findings.jsonl`. Discussions tab carries peer messages. |
| `<org>/parameter-golf-recipes` | Dataset | Holds yaml recipes |
| `<org>/parameter-golf-art-<project>` | Model | Holds checkpoints + eval bundles. Created lazily. |

`hf_push_*` calls `ensure_repo` first, so the repos can also be
auto-created by the first push. Manual creation is the safer default
because you control privacy + license up front.

### 2. Issue tokens

Operators need ONE token with write scope to those four repos:

- HuggingFace → Settings → Access Tokens → "New token" → role: **write**
  → restrict to your collab org.
- Save as `HF_TOKEN` in `.env` at the project root.
- DO NOT add it to `.env.runpod.local` — the token does not need to
  ship to pods. Pods use `WANDB_API_KEY` only.

For agents that should only *read* peer state (e.g. a briefing-only
agent), issue a fine-grained read token with the same `HF_TOKEN`
variable name.

### 3. Wire up `crucible.yaml`

Add the `hf_collab` block:

```yaml
hf_collab:
  enabled: true
  leaderboard_repo: my-org/parameter-golf-leaderboard
  findings_repo: my-org/parameter-golf-findings
  recipes_repo: my-org/parameter-golf-recipes
  artifacts_repo: my-org/parameter-golf-art-{project}
  private: true             # default; flip only after deliberation
  briefing_auto_pull: false # default; flip only if you accept the latency
```

`{project}` is the only template placeholder supported in
`artifacts_repo`. Anything else (`{org}`, `{date}`, …) is
rejected at call time with `[ValueError] Invalid repo template`.

### 4. Verify

The MCP server speaks the MCP protocol over stdio (or HTTP); it does
NOT expose Python-callable tools to your shell. Two ways to drive a
single tool call as a smoke test:

**A. From your orchestrator (Claude Code / smolagents) connected over MCP**

Most orchestrators wrap MCP tool calls behind syntactic sugar; ask
yours to call `research_hf_prior_attempts` with
`repo_id='my-org/parameter-golf-leaderboard'`. Empty `runs` is the
expected first-time result.

**B. Direct via the Python MCP client** — useful for debugging:

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def probe():
    server = StdioServerParameters(
        command="python", args=["-m", "crucible.mcp.server"],
    )
    async with stdio_client(server) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(
                "research_hf_prior_attempts",
                {"repo_id": "my-org/parameter-golf-leaderboard"},
            )
            print(result)

asyncio.run(probe())
```

Anything other than `{ok: true, count: 0, runs: []}` (on a fresh
repo) means token or repo misconfiguration; the `error` field always
includes the underlying SDK exception type name.

## Letting external agents drive Crucible

The Crucible MCP server is the integration point. External agents
talk to it via MCP over stdio (Claude Code), HTTP (deployed as a HF
Space), or in-process (smolagents).

### Smolagents driver — minimal pattern

A smolagents `CodeAgent` can use Crucible's MCP tools as if they were
its own:

```python
# Save as parameter_golf_runner.py in your tap repo.
# Requires: smolagents, mcp, an LLM provider (anthropic/openai/etc.)

from smolagents import CodeAgent, InferenceClientModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Boot Crucible MCP as a subprocess
server = StdioServerParameters(
    command="python",
    args=["-m", "crucible.mcp.server"],
    env={"HF_TOKEN": "...", "WANDB_API_KEY": "...", "RUNPOD_API_KEY": "..."},
)

async def run():
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            # Adapter: wrap each MCP tool as a smolagents Tool.
            crucible_tools = [_mcp_to_smol(t, session) for t in tools]

            agent = CodeAgent(
                tools=crucible_tools,
                model=InferenceClientModel(),  # or your preferred model
                max_iterations=20,
            )
            result = agent.run(
                "Read prior attempts at parameter-golf-2026 from "
                "research_hf_prior_attempts, design a fresh hypothesis "
                "that does NOT repeat them, run it via the standard "
                "fleet flow, and post a finding."
            )
            print(result)
```

The `_mcp_to_smol` adapter is left as an exercise — bind each MCP
tool's name + JSON schema to a smolagents `Tool` subclass that calls
`session.call_tool(name, args)`. The
[smolagents docs](https://huggingface.co/docs/smolagents) cover the
Tool subclass shape; one helper class typically suffices for all
~157 Crucible tools.

### HF Space deployment

Two shapes are possible — pick based on what your peer agents actually
speak:

**Shape A — HTTP shim around `TOOL_DISPATCH` (NOT MCP)**

This is the simplest demo. It bypasses the MCP protocol entirely and
calls Crucible tool handlers as Python functions. Useful if your peer
agents speak HTTP/JSON, not MCP. Do not call this an "MCP backend" —
it is a JSON-over-HTTP shim:

```python
# spaces/crucible-http/app.py
import gradio as gr
from crucible.mcp.tools import TOOL_DISPATCH

def call(name: str, args_json: str):
    import json
    args = json.loads(args_json or "{}")
    handler = TOOL_DISPATCH.get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    return handler(args)

with gr.Blocks() as demo:
    gr.Markdown("# Crucible HTTP shim — direct TOOL_DISPATCH calls")
    name = gr.Textbox(label="tool name")
    args = gr.Textbox(label="args (JSON)", lines=4)
    out = gr.JSON(label="result")
    gr.Button("call").click(fn=call, inputs=[name, args], outputs=out)

demo.launch()
```

**Shape B — actual MCP server in a Space**

If your peer agents speak MCP, run `crucible mcp serve` over HTTP /
SSE inside the Space and connect via the `mcp` Python client (Shape
B example in section 4 above). This preserves capability-style tool
discovery (`session.list_tools()`) and the schema validation layer.
Gradio is optional here — it's just a wrapper to make the Space's
status page visible.

For either shape, in production:

- Add HF OAuth / `gr.LoginButton` so only authorized agents can call
  destructive tools (`destroy_nodes`, `purge_queue`).
- Set `HF_TOKEN`, `WANDB_API_KEY`, `RUNPOD_API_KEY` as Space secrets,
  not env vars baked into the container.
- Pin `crucible` to a tag, not `main` — auto-deploys can land on a
  half-merged change otherwise.

These artifacts (smolagents driver, Gradio shim / MCP Space) belong
in their own tap repo, not in this codebase. The rationale follows
the existing "compression plugins ship as a tap repo" convention:
deployment code evolves independently of Crucible core.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `hf_collab is disabled` | `enabled=false` in yaml | Set `enabled=true`. Read-only tools work without it. |
| `[ValueError] Invalid repo template` | `{org}` or other placeholder | Only `{project}` is supported; replace with a literal. |
| `HfError [HFValidationError]` | malformed `repo_id` | HF rejects names with `{`, spaces, etc. Pass a real `org/name`. |
| `count=0` from `research_hf_prior_attempts` | token can't read repo, OR no peer published yet | Try `huggingface_hub.hf_hub_download` directly; if it 401s, fix token scope. |
| Briefing now takes 30s | `briefing_auto_pull=true` | Flip back to `false`; call `research_hf_prior_attempts` on demand instead. |
| Posted secrets to a discussion | a `redact_secrets()` rule missed your secret format | Add the regex to `src/crucible/core/redact.py:_SECRET_VALUE_PATTERNS`. Existing patterns cover HF / WandB / OpenAI / Anthropic / GitHub / AWS / bearer tokens + env-style `KEY=value`. |

## Mental model

```
                    ┌────────────────────────┐
                    │   external orchestrator│
                    │  (Claude Code,         │
                    │   ml-intern, your CLI) │
                    └──────────┬─────────────┘
                               │ MCP (stdio / HTTP / in-process)
                               ▼
                    ┌────────────────────────┐
                    │  Crucible MCP server   │  <-- runs locally OR
                    │  (157 tools)           │       in a HF Space
                    └──────┬─────────┬───────┘
                  fleet ops│         │hf_* tools
                           ▼         ▼
                      ┌─────────┐  ┌────────────────┐
                      │ RunPod  │  │ HuggingFace Hub│
                      │  pods   │  │ leaderboard,   │
                      │         │  │ findings,      │
                      │  W&B    │  │ recipes,       │
                      │  logs   │  │ artifacts,     │
                      └─────────┘  │ discussions    │
                                   └────────────────┘
                                          ▲
                                          │
                                ┌─────────┴─────────┐
                                │ peer Crucible     │
                                │ instances (other  │
                                │ machines, agents, │
                                │ humans)           │
                                └───────────────────┘
```

Crucible owns: pods, queue, design, fleet ops, local store, MCP
surface. HuggingFace owns: shared persistence + cross-agent discovery
+ comm. The two stay loosely coupled — `hf_collab.enabled=false`
removes HF entirely without breaking the rest of the stack.
