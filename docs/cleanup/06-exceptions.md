# Agent 6 — Defensive `try/except` Audit

Agent 6 ran out of context before writing its own assessment, so this was
reconstructed from the commit diff and follow-up verification. The work
itself survived and was committed after a manual regression fix.

## Summary

| Metric | Before | After | Delta |
|---|---|---|---|
| `except Exception` | 220 | 65 | **−155** |
| `try:` blocks | 374 | 365 | −9 |
| ruff errors | 327 | 324 | −3 |
| mypy errors | 431 | 430 | −1 |
| LOC | 37,531 | 37,405 | −126 |
| tests | 1787 pass / 16 fail | 1787 pass / 16 fail | unchanged |

The dominant move was **narrowing** `except Exception` to specific exception
types (subprocess, OS, CrucibleError hierarchy), not deleting try/except
blocks wholesale. The 155-count drop reflects mass narrowing across 8 files,
not bug-hiding handler deletion.

## Files touched

- `src/crucible/fleet/bootstrap.py` — narrowed process-cleanup except; also
  imported the previously undefined `remote_exec` (the latent bug Agent 1
  surfaced at `bootstrap.py:621`). Cleanup path now calls the real function.
- `src/crucible/fleet/providers/runpod.py` — removed unreachable
  `dict(node)` fallback per Agent 5's lead; `previous_by_id[pod_id]` is
  guaranteed populated.
- `src/crucible/mcp/tools.py` — largest diff (~160 narrowings / deletions).
  Narrowed `except Exception` handlers in 40+ MCP tool entry points to
  `except (CrucibleError, OSError, ...)` lists with reason-comments on
  surviving fallbacks.
- `src/crucible/research_dag/bridge.py` — narrowed to `(CrucibleError, OSError)`.
- `src/crucible/researcher/briefing.py` — same pattern.
- `src/crucible/researcher/literature.py` — same.
- `src/crucible/runner/experiment.py` — narrowed subprocess wrapper to
  `(OSError, subprocess.SubprocessError, RunnerError)` per Agent 5's lead;
  data-provenance lookup narrowed to `(OSError, ValueError)` with a
  best-effort why-comment.
- `src/crucible/training/generic_backend.py` — narrowed factory errors.

## Latent bug fixed

**`fleet/bootstrap.py:621` — undefined `remote_exec`.** Surfaced by Agent 1
as a ruff F821. Agent 6 imported `remote_exec` from `fleet/sync.py` and
narrowed the surrounding `except Exception` to
`except (subprocess.SubprocessError, OSError)`. The cleanup path — which
kills old training processes before bootstrap — can now actually run
instead of crashing with `NameError` and being swallowed.

## Regression introduced, then fixed

Agent 6's narrowing of `model_fetch_architecture`'s
`except Exception` was too aggressive: test doubles (`FakeConfig`) lacked
a `hub_dir` attribute, which the new narrowed list (`CrucibleError`,
`OSError`) did not catch, so `AttributeError` propagated and broke 5 tests
in `test_mcp_tools_new.py::TestModelFetchArchitecture`.

Fix applied in the same commit: `_get_hub_store()` now uses
`getattr(config, "hub_dir", None)` so partial test configs work without
reintroducing a broad catch. This is the correct fix — the brittleness was
in the accessor, not the exception surface.

## Classification rubric applied

Every `except` clause was triaged against CLAUDE.md's rule
("Let unexpected errors propagate — don't catch and swallow"):

| Class | Verdict | Example |
|---|---|---|
| Boundary (SDK/subprocess/IO) | Narrow to specific exceptions | runpod API calls, rsync subprocess |
| User-input sanitizer | Keep, re-raise as `CrucibleError` subclass | YAML loaders |
| Documented fallback | Keep, add why-comment if missing | wandb logger optional |
| Bug hider | Delete | bare `except Exception: pass` — removed |
| Unreachable | Delete | `except FileNotFoundError` after `path.exists()` |
| Broad `except Exception` with no reason | Narrow or delete | the 155 sites |

## Leads for later agents

- **Agent 7 (legacy)**: the `# noqa: BLE001` suppressions Agent 6 added
  (e.g. `mcp/tools.py:2076`) mark places where a broad catch is *documented
  as intentional* for best-effort hot-reload paths. Audit whether any of
  those code paths are themselves legacy / inert.
- **Agent 8 (comments)**: Agent 6 added ~20 why-comments. Double-check
  they're useful and not restatement.

## Remaining work (queued)

- 65 `except Exception` still stand. These are the harder cases — mostly
  in test-helper paths, tracer/logging middlewares, and generic plugin
  dispatch where the downstream type surface is genuinely `object`.
  Individual narrowing requires domain judgment; not auto-safe.
- MCP tool dispatch in `mcp/server.py` (out-of-scope per hard rule) has
  its own broad catches at the protocol boundary — that's correct, per
  the MCP contract.
