# Authoring Crucible Community Plugins

Crucible's plugin system + community-tap workflow let you ship custom optimizers, callbacks, architectures, evaluators, and more without forking the platform. This doc is the single canonical path from "I have a custom optimizer in mind" to "it's installable via `crucible tap install my_optimizer`".

If you only need to use plugins someone else published, see [plugins.md](plugins.md). This doc is for authors.

## TL;DR

```bash
# 1. Scaffold a new tap (or use an existing one)
crucible tap init ~/my-tap --author you@example --license MIT

# 2. Edit the example plugin or add a new one
cd ~/my-tap
$EDITOR optimizers/example_optimizer/example_optimizer.py

# 3. Verify quality
crucible tap lint .          # repo-level checks (READMEs, no cruft, ...)
crucible tap validate .      # per-plugin manifest schema

# 4. Publish
git add . && git commit -m "Add my_optimizer"
git remote add origin https://github.com/<you>/<my-tap>
git push -u origin main

# 5. Anyone can install
crucible tap add https://github.com/<you>/<my-tap>
crucible tap install example_optimizer
```

## The plugin model

Crucible has 16 plugin families. Each one is a hook point — an interface Crucible's core calls into. You implement the interface in a small Python file (or sometimes a directory bundle), declare metadata in `plugin.yaml`, and the registry handles discovery + selection.

### The 16 plugin families

| Family | Interface | Builtin examples | Selected via |
|---|---|---|---|
| `optimizers` | `build(params, **kw) → torch.optim.Optimizer` | adam, adamw, muon, sgd | `OPTIMIZER=name` env var |
| `schedulers` | `build(optimizer, **kw) → LRScheduler` | cosine, constant, linear | `LR_SCHEDULE=name` |
| `callbacks` | training-loop hook class | grad_clip, nan_detector | `CALLBACKS=a,b,c` |
| `loggers` | log-event sink class | wandb, console, jsonl | `LOGGING_BACKEND=a,b` |
| `providers` | fleet provider class | runpod, ssh | `provider.type` in YAML |
| `architectures` | `CrucibleModel` subclass | baseline, looped, convloop, prefix_memory | `MODEL_FAMILY=name` |
| `data_adapters` | `DataAdapter` subclass | token, image_folder, synthetic_* | `data.adapter` in YAML |
| `data_sources` | dataset-fetch provider | huggingface, local_files, wandb_artifact | `data.source` |
| `objectives` | `TrainingObjective.compute(...)` | cross_entropy, mse, kl_divergence, diffusion, jepa | `OBJECTIVE=name` |
| `block_types` | composer block builder | attention_block, prefix_memory_block | YAML `block: type:` |
| `stack_patterns` | composer stack wiring | sequential, encoder_decoder_skip, looped | YAML `stack: pattern:` |
| `augmentations` | composer augmentation | smear_gate, bigram_hash, trigram_hash | YAML `augmentations:` |
| `activations` | activation function | relu_sq, gelu_sq, mish_sq | `ACTIVATION=name` |
| `launchers` | end-to-end training script (bundle) | (no builtins; community provides) | `LAUNCHER=name` |
| `evaluations` | eval-script bundle | (no builtins) | `eval_suite:` |
| `domain_specs` | `nlp_classification`, `agent_scaffold` | (tap-shipped only) | `harness_init(domain_spec=...)` |
| `evaluators` | benchmark-eval plugin | lm_eval_harness | `evaluators:` in YAML |
| `code_mutation` | code-mutation policy (Phase 5+ stub) | stub only | future |

### Plugin discovery and precedence

Plugins are resolved by name with **3-tier precedence**:

```
builtin (lowest)  →  ~/.crucible-hub/plugins/{type}/    (global, installed via taps)
                                                              ↓
                  →  .crucible/plugins/{type}/          (local, this project)  (highest)
```

A plugin called `lion` in your project's local `.crucible/plugins/optimizers/lion.py` overrides a hub-installed one with the same name. The hub-installed one in turn overrides a same-named builtin.

This is why your published tap plugins live under `~/.crucible-hub/plugins/{type}/` after `tap install`.

## Choosing a plugin shape: single-file vs bundle

Two physical layouts:

**Single-file plugins** (`.py`) — install as `~/.crucible-hub/plugins/{type}/{name}.py`. Recommended for optimizers, schedulers, callbacks, activations — anything self-contained.

**Bundle plugins** (directory) — install as `~/.crucible-hub/plugins/{type}/{name}/`. Required for `launchers`, `evaluations`, `domain_specs`. Also auto-selected for `architectures` that import sibling Python files.

You don't have to pick — the bundling heuristic (in `core/tap.py:install_package`) checks if the package directory has any `.py` files besides `{name}.py` and decides automatically.

## Authoring a plugin from scratch

### Step 1 — Scaffold a new tap

```bash
crucible tap init ~/my-tap --author you@example.com --license MIT
```

This drops the following structure:

```
~/my-tap/
├── README.md                                  one-paragraph what-it-is
├── LICENSE                                    MIT (or whatever you chose)
├── tap.yaml                                   top-level manifest
├── .gitignore                                 excludes data/, *.pt, etc.
├── .github/workflows/lint.yaml                runs `crucible tap lint .` on PR
└── optimizers/example_optimizer/
    ├── plugin.yaml                            well-formed manifest
    ├── example_optimizer.py                   minimal builder stub
    └── README.md                              describes the plugin
```

Verify the scaffold:

```bash
crucible tap lint ~/my-tap     # → 0 errors, 0 warnings
```

### Step 2 — Add your plugin

Pick a plugin type. For an optimizer:

```bash
mkdir -p ~/my-tap/optimizers/my_optimizer
cd ~/my-tap/optimizers/my_optimizer
```

Write the implementation in `my_optimizer.py`:

```python
"""my_optimizer — a custom Adam variant with bias correction off."""
from __future__ import annotations

from typing import Any

import torch


def build(params, *, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
          **kwargs: Any) -> torch.optim.Optimizer:
    """Return the optimizer instance.

    Crucible's registry calls this `build` function with `params` and a
    dict of env-var-derived kwargs. Argument names = env vars uppercased.
    """
    return torch.optim.Adam(params, lr=lr, betas=betas, eps=kwargs.get("eps", 1e-8))
```

Write the manifest in `plugin.yaml`:

```yaml
"name": "my_optimizer"
"type": "optimizers"
"version": "0.1.0"
"description": "Adam variant with optional bias correction"
"author": "you@example.com"
"tags": ["adam", "experimental"]
"crucible_compat": ">=0.2,<0.3"
"dependencies": ["torch>=2.0"]
"parameters":
  "LR": "Learning rate (default: 1e-3)"
  "BETA1": "Adam beta1 (default: 0.9)"
  "BETA2": "Adam beta2 (default: 0.999)"
```

Write a `README.md`:

```markdown
# my_optimizer

Adam variant with optional bias correction.

## Install

\`\`\`bash
crucible tap install my_optimizer
\`\`\`

## Configuration

| Env var | Default | What |
|---------|---------|------|
| `LR` | `1e-3` | learning rate |
| `BETA1` | `0.9` | Adam beta1 |
| `BETA2` | `0.999` | Adam beta2 |
```

### Step 3 — Verify

Two pre-flight commands:

```bash
crucible tap lint ~/my-tap           # repo-level: READMEs, naming, no cruft
crucible tap validate ~/my-tap       # per-plugin: manifest schema
```

Both must exit 0 before you publish. If they don't, the printed `fix:` hint tells you exactly what to change.

### Step 4 — Publish to your remote

```bash
cd ~/my-tap
git add .
git commit -m "Add optimizers/my_optimizer"
git remote add origin https://github.com/<you>/<my-tap>
git push -u origin main
```

That's it — your tap is live.

### Step 5 — Anyone installs it

```bash
crucible tap add https://github.com/<you>/<my-tap>
crucible tap search adam
crucible tap install my_optimizer
```

Then in a Crucible project:

```bash
OPTIMIZER=my_optimizer LR=3e-4 crucible run experiment
```

## The plugin.yaml schema

Every plugin folder has a `plugin.yaml` describing what it is. The schema is enforced by `crucible tap validate` and (Phase A.4) by `crucible tap publish` pre-flight.

### Required

| Field | Type | Notes |
|---|---|---|
| `name` | string | matches `[a-zA-Z_][a-zA-Z0-9_-]*`. Must equal the folder name. |
| `type` | string | one of the 16 plugin families |
| `version` | quoted string | semver (`M.m.p` optional `-suffix`) |
| `description` | string | single line, ≤ 500 chars (move detail to README.md) |

### Recommended (warnings only)

| Field | Type | Notes |
|---|---|---|
| `author` | string | maintainer handle or email |
| `tags` | list[string] | classification tags |
| `crucible_compat` | string | version range, e.g. `">=0.2,<0.3"`. Enforced at install time. |
| `dependencies` | list | Python deps or other plugins; surfaced as warnings if missing |

### Optional

| Field | Type | Notes |
|---|---|---|
| `config` | dict | default env vars the plugin sets |
| `parameters` | dict | runtime env var knobs documentation |
| `benchmarks` | list | performance metrics |
| `entry` | string | for launchers: the entry-point Python file |

### Example (well-formed)

```yaml
"name": "lion"
"type": "optimizers"
"version": "0.1.0"
"description": "Lion optimizer — sign-of-momentum update from Symbolic Discovery"
"author": "eren23"
"tags": ["lion", "momentum", "google-research"]
"crucible_compat": ">=0.2,<0.3"
"dependencies":
- "torch>=2.0"
"parameters":
  "LR": "Learning rate (default: 1e-4 — 3-10x lower than Adam)"
  "BETA1": "Momentum coefficient (default: 0.9)"
  "BETA2": "Momentum for the EMA (default: 0.99)"
"benchmarks":
  "test_loss": 2.421
```

## The tap.yaml schema (Phase A.3)

Top-level manifest at the tap-repo root. Optional today (filesystem walk still works without it), but `tap lint` warns if missing.

```yaml
"name": "my-tap"
"description": "Custom plugins for X"
"version": "0.1.0"
"author": "you@example.com"
"license": "MIT"
"crucible_compat": ">=0.2,<0.3"
"homepage": "https://github.com/<you>/<my-tap>"
"maintainer_contact": "<you@example.com>"
```

The future curated tap registry will rely on this — adding it now future-proofs your tap.

## The lint checks

`crucible tap lint <path>` runs 11 built-in checks. Each emits issues with a stable code (`L001`-`L011`), severity, and a paste-able fix hint.

| Code | Severity | What it catches |
|---|---|---|
| **L001** | error | `README.md` missing at tap repo root |
| **L002** | error | `LICENSE` / `LICENSE.txt` / `LICENSE.md` / `COPYING` missing |
| **L003** | warning | `tap.yaml` missing at repo root |
| **L004** | warning | per-plugin `README.md` missing |
| **L005** | error | cruft directories committed (`data/`, `checkpoints/`, `wandb/`, `_manuscript/`, `__pycache__/`, `.DS_Store`, etc.) |
| **L006** | error | files larger than 1 MB outside `.git/` |
| **L007** | error | plugin folder name ≠ manifest `name:` field |
| **L008** | warning | `description: >` or `description: \|` (multi-line block scalars) |
| **L009** | warning | `version:` is unquoted in YAML |
| **L010** | error | plugin `.py` file has a syntax error |
| **L011** | error/warning | per-plugin manifest schema violations (delegates to `validate_manifest`) |

CI configs can pin expectations by checking the per-issue codes in the JSON-ish output.

### Large-file policy

Files over 1 MB **don't belong in a tap**. The fix is to upload to HuggingFace and leave a pointer:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoints/best.pt",
    path_in_repo="best.pt",
    repo_id="myorg/my-checkpoints",
    repo_type="model",
)
```

Then in your plugin's `README.md`:

```markdown
## Pretrained weights

Available at https://huggingface.co/myorg/my-checkpoints/blob/main/best.pt
```

This keeps the tap cloneable in seconds and lets HF handle the storage.

## crucible_compat enforcement (Phase A.5)

When you declare `crucible_compat: ">=0.2,<0.3"` in `plugin.yaml`, `crucible tap install` checks the running Crucible version and **rejects the install** if it falls outside the range. Pass `--force` to override (with a warning).

```bash
crucible tap install my_optimizer                       # checked
crucible tap install my_optimizer --force               # override + warn
```

This is opt-in — plugins without `crucible_compat` pass through. But declaring one signals "I've tested against this range" to your users.

## Dependency declarations (Phase A.6)

`dependencies:` accepts two shapes:

```yaml
"dependencies":
- "torch>=2.0"                              # plain Python dep
- "name": "wm_base"                          # plugin dep
  "type": "architectures"
```

At install time, plain string deps are documented but not auto-checked (use your project's `pip` for that). Plugin deps trigger a **warning** if the named plugin isn't installed in the hub:

```bash
$ crucible tap install code_wm
Installed: code_wm (architectures) v0.1.0 from [community]
  -> ~/.crucible-hub/plugins/architectures/code_wm
  WARN: declared dependency not installed: architectures/wm_base
```

The user decides whether to install the dep first. No auto-install — that's Phase 5+.

## Publishing workflow — full reference

| Verb | What it does |
|---|---|
| `crucible tap init <path>` | scaffold new tap (README, LICENSE, tap.yaml, CI, example plugin) |
| `crucible tap lint <path>` | repo-level quality checks (11 built-ins) |
| `crucible tap validate <path>` | per-plugin manifest schema check |
| `crucible tap publish <name> --type T --tap T` | copy local plugin into tap repo, git add + commit; pre-flight validates |
| `crucible tap push <tap>` | git push the tap repo to its remote |
| `crucible tap submit-pr <tap>` | open a GitHub PR from a fork (uses `gh` CLI) |

End-to-end one-shot:

```bash
crucible tap init ~/my-tap
# ... edit ~/my-tap/optimizers/example_optimizer/example_optimizer.py ...
crucible tap lint ~/my-tap
git -C ~/my-tap remote add origin https://github.com/<you>/<my-tap>
git -C ~/my-tap push -u origin main
crucible tap add file://$HOME/my-tap         # add it to your own hub for testing
crucible tap install example_optimizer
```

## Common authoring mistakes (and what `tap lint` says)

| Mistake | Code | Fix |
|---|---|---|
| `version: 0.1.0` (unquoted) | L009 | use `version: "0.1.0"` |
| `description: >` multi-line block | L008 | collapse to one line, move detail to README |
| Folder `my-plugin/` but manifest `name: my_plugin` | L007 | match them — folder = name = installable identifier |
| Committed `data/` or `*.pt` files | L005, L006 | upload to HF, leave a pointer |
| No `README.md` in plugin folder | L004 | template-generate from `plugin.yaml.description` |
| No `LICENSE` at repo root | L002 | copy MIT/Apache-2.0/BSD-3-Clause boilerplate |
| Plugin file doesn't parse | L010 | fix the syntax error |
| Manifest missing `name` / `type` / `version` / `description` | L011 (error) | required fields |
| Manifest missing `author` / `tags` / `crucible_compat` / `dependencies` | L011 (warning) | recommended — declare them |

## What NOT to do

- **Don't reach into Crucible core.** Plugins must work via the published interface. If you need something the interface doesn't expose, propose a change to the registry.
- **Don't commit large binaries.** L006 rejects files over 1 MB. Use HuggingFace for weights, datasets, checkpoints.
- **Don't ship credentials.** `.env`, API tokens, anything matching `HF_TOKEN`, `WANDB_API_KEY`, `sk-*`, etc. The `redact_secrets()` Crucible helper exists for runtime defense but the publish path doesn't auto-redact — be careful what you commit.
- **Don't publish without `crucible tap lint`.** The CI workflow that `tap init` scaffolds runs it automatically — keep that workflow.
- **Don't squash someone else's plugin's name.** The 3-tier precedence means a same-named local plugin shadows a hub-installed one. If you fork someone's plugin, rename your fork.

## See also

- [plugins.md](plugins.md) — using plugins someone else published (architecture-plugin focus).
- [modality-guide.md](modality-guide.md) — building a plugin for a new training modality (diffusion, world models, etc.).
- [harness-optimization.md](harness-optimization.md) — domain-spec plugin authoring for meta-harness loops.
- [recipes/publish-first-plugin.yaml](recipes/publish-first-plugin.yaml) — the canonical step-by-step recipe.
- `crucible.core.plugin_schema` — Python source for the manifest validator.
- `crucible.core.tap_lint` — Python source for the lint engine + 11 built-in checks.
