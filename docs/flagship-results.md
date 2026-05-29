---
layout: default
title: Flagship Validated Results
---

# Flagship Validated Results

A short page tracking outcomes from running the
`flagship-param-golf-discovery` recipe end-to-end. Each row is a
self-contained reproducibility bundle on HuggingFace — anyone can
clone the repo, run the recipe, and reach the same numbers.

This page is intentionally short. The goal is **proof that the loop
closes**, not a leaderboard race. Crucible's defensible niche is the
autonomous + reproducible + open + commodity-GPU intersection; this
page demonstrates the intersection works in practice.

## How to add a result here

When you run `flagship-param-golf-discovery` end-to-end:

1. Publish the artifacts via Tier 14
   (`hf_publish_leaderboard` / `findings` / `recipes`).
2. Open a PR to this file appending a row to the table below with:
   - Date
   - Compute spent (USD)
   - Wall-clock hours
   - Best val_bpb achieved
   - HuggingFace dataset / model repo link
   - One-line headline summarising the moves that mattered

PRs are intentionally low-ceremony — no peer review required. The
HuggingFace artifacts are the reproducibility evidence; the table
just indexes them.

## Results

| Date | Compute | Wall-clock | Best val_bpb | HF artifacts | Headline |
|---|---|---|---|---|---|
| _(seed run pending)_ | $50 | 5h | TBD | TBD | _placeholder — first end-to-end run after Phase 5.1 merge_ |

## What "result" means here

A "result" on this page is *not* a Nature-style claim. It's a
reproducibility bundle that demonstrates:

- A clean run of the `flagship-param-golf-discovery` recipe.
- The autonomous loop closed (`autonomous_research_loop` completed
  N iterations without orchestrator handholding beyond the LLM calls).
- The judge-separation contract held (no
  `JudgePanel.assert_separated()` rejection).
- The full provenance chain — from initial literature → hypothesis →
  tournament → mutation → execution → finding → paper draft —
  recovers from the artifacts alone.

If a recipe run hits a snag (judge mis-separation, scope violation,
exec failure mid-run), the right move is to fix the underlying issue
in the recipe / code and re-run, not to manually patch the result.
The bundle's value is the closed-loop reproducibility, not the raw
number.

## Cross-references

- The recipe: `.crucible/recipes/flagship-param-golf-discovery.yaml`
- The playbook: `examples/flagship_param_golf/README.md`
- Why this matters: `docs/positioning.md` (the "buyer" section)
- Roadmap context: `ROADMAP.md` (Phase 4 + 5 deliverables)
