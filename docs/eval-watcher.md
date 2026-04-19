# Eval Watcher — Auto-Eval Daemon

The eval watcher is a background daemon that polls running fleet pods for new
checkpoints, SCPs them to the local machine, and runs a project-defined suite
of evaluation scripts on each one. Results are appended to a JSONL log that
survives Crucible restarts.

The motivation is the "ever-living" research workflow: once you've defined what
"good" means for a project (KNN, lift, planning recovery, whatever), the
daemon keeps the leaderboard fresh as new checkpoints land — no manual eval
runs needed.

## Quick start

1. Add an `eval_suite:` block to the project YAML.
2. Start the daemon: `mcp__crucible-fleet__eval_watch_start project_name=<name>`.
3. Inspect progress: `mcp__crucible-fleet__eval_watch_status`.
4. Stop when done: `mcp__crucible-fleet__eval_watch_stop`.

## Project YAML — `eval_suite` block

Add to any `.crucible/projects/<name>.yaml`:

```yaml
eval_suite:
  - script: evaluation/code_wm/full_val_knn.py
    args:
      - "--data"
      - "/path/to/data.h5"
      - "--device"
      - "cpu"
      - "--seed"
      - "42"
  - script: evaluation/code_wm/latent_planning.py
    args:
      - "--data"
      - "/path/to/data.h5"
      - "--n-samples"
      - "200"
```

Each script:
- MUST accept `--checkpoint <path>` and `--out <json_path>` (the daemon adds these).
- SHOULD write a JSON object to `--out` containing the metrics it computed.
- May accept any extra flags listed in `args:` (passed verbatim).

Script paths are resolved in order: absolute → project-local → any tap under
`~/.crucible-hub/taps/`.

## MCP tools

### `eval_watch_start`

Start the daemon for one project.

| arg | type | default | meaning |
|---|---|---|---|
| `project_name` | string | required | spec name (without `.yaml`) |
| `interval` | int | 300 | seconds between pod polls |
| `remote_pattern` | string | `/workspace/project/checkpoints/*.pt` | glob on each pod for checkpoints |
| `env` | object | {} | extra env vars passed to every eval script |

Returns: `{status: 'started'|'already_running', state, suite_size}`.

Idempotent — calling twice on the same daemon returns `already_running`.

### `eval_watch_status`

Returns daemon state + most recent N rows from the log.

| arg | type | default | meaning |
|---|---|---|---|
| `recent` | int | 10 | how many recent rows to include |

Returns: `{state: {running, project, last_poll_at, total_runs, ...}, recent: [...]}`.

### `eval_watch_stop`

Signal the daemon to halt and join the thread (15-second timeout).

Returns: `{status: 'stopped'|'not_running', state}`.

## How it works

Each poll:

1. Read `nodes.json` and filter nodes whose name starts with the project name
   (e.g., `wm_pred_fix-*` for project `wm_pred_fix`) AND whose `api_state`
   is `running` AND whose `ssh_host` is set.
2. For each matching node, `ssh ls <remote_pattern>` to list checkpoints.
3. For each remote checkpoint not yet pulled, `scp` it to
   `.crucible/eval_watch_ckpts/<node>_<step>.pt`.
4. Compute SHA-256 (16 hex chars) of the local checkpoint.
5. For each script in the suite, check whether `(sha, script_basename)` is
   already in the log. If yes, skip. If no, run the script.
6. Append a JSONL row to `.crucible/eval_watch.jsonl`.

Each row has the shape:

```json
{
  "label": "wm_pred_fix-06_code_wm_step3000",
  "script": "evaluation/code_wm/full_val_knn.py",
  "ckpt_sha": "ff61977762e264c3",
  "ok": true,
  "elapsed_s": 32.3,
  "result": { "knn_top5": 0.058, "eff_rank_online": 53.3, ... },
  "stdout_tail": "",
  "stderr_tail": "",
  "ran_at": "2026-04-19T13:30:00Z"
}
```

## Idempotency + persistence

- The same `(checkpoint SHA, script basename)` pair is never run twice.
- Re-starting the daemon picks up where it left off (state in
  `.crucible/eval_watch.state.json`, log in `.crucible/eval_watch.jsonl`).
- Pulled checkpoints are cached under `.crucible/eval_watch_ckpts/` and
  reused across polls.

## State + log layout

```
.crucible/
  eval_watch.state.json     # running flag, project, started_at, total_runs
  eval_watch.jsonl          # append-only log, one row per (ckpt, script) pair
  eval_watch_ckpts/         # pulled checkpoints (one per node × step)
    wm_pred_fix-02_code_wm_step1000.pt
    wm_pred_fix-02_code_wm_step2000.pt
    ...
```

## Limitations / future work

- **One daemon per Crucible process.** Starting the watcher on a second
  project while one is running returns `already_running`. To run multiple
  projects concurrently, restart Crucible between projects (state files
  scope cleanly per `.crucible/` directory, so per-project `cwd` works).
- **No GPU eval.** Evaluations run locally (CPU). For large checkpoints or
  expensive evals, a future version should support remote eval execution.
- **No auto-pause hook.** The original auto-pause TODO (stop pods when GPU
  util drops + no new checkpoints in N minutes) is not yet integrated. Same
  daemon, additional hook — to be added.
- **No W&B / Spider Chat sink.** Results land in JSONL only. A v1.1 will add
  optional auto-posting to W&B as `auto_eval/*` keys and to Spider Chat as
  notes tagged with the project name.
- **Polling, not pushing.** The daemon polls every `interval` seconds rather
  than receiving checkpoint-saved events. This is robust to network
  interruptions but adds latency; a v2 could combine with an inotify-style
  push.

## Common debugging

If the daemon shows `running: true` but `total_runs: 0`:

1. `ls nodes.json` — does it contain pods with the project's name prefix?
2. SSH manually to a pod and check `ls /workspace/project/checkpoints/*.pt`.
   If empty, training hasn't saved a checkpoint yet (default save interval is
   500-1000 steps).
3. Check `.crucible/eval_watch.jsonl` for `_poll_error` rows — they record
   any exception raised by the polling loop.

If a specific script always fails:

1. `eval_watch_status recent=20` — find the row with `ok: false`.
2. Read its `stderr_tail` and `stdout_tail` fields.
3. Run the script manually with the same args to reproduce.

## Example session

```text
> mcp__crucible-fleet__eval_watch_start project_name=wm_pred_fix interval=180
{
  "status": "started",
  "suite_size": 2,
  "state": { "running": true, "project": "wm_pred_fix", ... }
}

# ... wait for pods to checkpoint ...

> mcp__crucible-fleet__eval_watch_status recent=5
{
  "state": { "running": true, "total_runs": 8, "last_poll_at": "..." },
  "recent": [
    { "label": "wm_pred_fix-06_code_wm_step3000",
      "script": "evaluation/code_wm/full_val_knn.py",
      "ok": true, "elapsed_s": 32.1,
      "result": { "knn_top5": 0.477, ... } },
    ...
  ]
}

> mcp__crucible-fleet__eval_watch_stop
{ "status": "stopped", "state": { "running": false, ... } }
```
