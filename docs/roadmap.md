---
layout: default
title: Roadmap
---

# Roadmap

The canonical roadmap lives at [ROADMAP.md](../ROADMAP.md) in the project root. This page summarizes the current state.

## Current: v0.2.1-alpha

**What's working:** Full fleet loop (provision → bootstrap → dispatch → collect → destroy), 112 MCP tools, hub system, notes, research briefing, REST API, declarative architecture composition + Python plugins, modality-agnostic training (LM, diffusion, vision, world models), external project runner (run any codebase on fleet), 2 reference examples (DDPM, JEPA), W&B bridge, interactive TUI, unified plugin system (12 pluggable component types with 3-tier precedence), community taps (Homebrew-style plugin sharing), 1000 tests.

**What's next:**
- Pre-dispatch model family validation
- CI/CD pipeline and PyPI release
- Community tap index / web UI

## Planned Phases

### Phase 2: Integrate, Don't Reinvent
- SkyPilot provider for 20+ cloud support
- More provider plugins (Modal, Lambda, Vast.ai)
- Optuna/Ax integration for mathematical HPO
- Configurable output patterns

### Phase 3: Build Unique Value
- Hybrid search (LLM + Optuna)
- Research strategies as plugins
- Code-level search (AIDE-style)
- Auto-generated experiment reports

### What We Won't Build
- Experiment tracking UI (use W&B/MLflow)
- Kubernetes orchestration (use SkyPilot)
- Model serving / inference
- Dataset hosting (use HuggingFace)
- Optimization math (use Optuna/Ax)

See [ROADMAP.md](../ROADMAP.md) for the full plan with competitive analysis and contribution guide.
