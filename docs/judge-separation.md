# Judge-separation contract

When Crucible drives any LM-as-judge loop — harness candidate scoring,
GRPO tree expansion, recipe ranking — the judge model that **rewards**
during selection MUST differ from the judge model that **evaluates** for
the final ranking. Same model = identical reward hacks; same family =
correlated failure modes that the eval can't catch.

This contract is implemented as `JudgePanel.assert_separated()` in
`src/crucible/core/config.py`. It mirrors the recipe used by the
GIANTS paper (https://giants-insights.github.io/) — Gemini-2.5-Flash for
RL reward, Gemini-3-Pro for evaluation, plus Qwen3-14B and SciJudge-30B
as audit signals.

## Configure

Add a `judges:` block to `crucible.yaml`:

```yaml
judges:
  reward_judge:
    model: gemini-2.5-flash
    family: gemini
  eval_judge:
    model: claude-opus-4-7
    family: claude
  audit_judge:                # optional independent third judge
    model: qwen3-14b
    family: qwen
  enforce_separation: true    # default; set false to downgrade to warning
```

`family` groups models that share weights or training lineage. Use the
major-vendor or open-weights line: `claude`, `gemini`, `openai`, `qwen`,
`llama`, etc.

## When the contract fires

Tools that depend on LM-as-judge call `panel.assert_separated()` before
any LLM call. Currently:

- `harness_iterate` — fails with `ConfigError` before propose/validate.
- `tree_expand_grpo` — fails before any candidate scoring is consumed.

When `judges:` is absent or all model fields are blank, the panel is
*unconfigured* and enforcement is skipped. Opt-in only — existing
projects without judge configs see no behavior change.

## Failure modes

| Violation | Error |
|-----------|-------|
| `reward_judge.model == eval_judge.model` | `same model … reward-hacking will go undetected` |
| `reward_judge.family == eval_judge.family` (different model, same family) | `same family … separate the judge families` |
| `audit_judge.model` collides with reward or eval | `audit_judge model … collides with` |
| `audit_judge.family` collides with reward or eval | `audit_judge family … collides with` |

To downgrade any of these to a warning instead of a hard error, set
`enforce_separation: false` in the panel. Useful while migrating legacy
projects or doing one-off comparisons.

## Why

Reward-hacking is the dominant failure mode of LM-as-judge loops. Same
model → same blind spots. GIANTS measured a 35% relative gain on
insight-anticipation when they switched from a single judge to a
separated train-judge / eval-judge pair, and the win held even when the
trained policy was a 4B model competing against Gemini-3-Pro. The
mechanism: independent judges fail at independent things, so collisions
in one don't translate into reward signal in the other.

## Verification

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_judge_panel.py tests/test_judge_panel_wiring.py
```

For a project, validate by loading `crucible.yaml` and inspecting:

```python
from crucible.core.config import load_config
cfg = load_config()
cfg.judges.assert_separated()  # raises ConfigError on misconfig
```
