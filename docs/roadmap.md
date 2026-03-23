---
layout: default
title: Roadmap
---

# Roadmap

The canonical roadmap lives at [ROADMAP.md](../ROADMAP.md) in the project root. This page summarizes the current state.

## Current: v0.2.0-alpha

**What's working:** Full fleet loop (provision → bootstrap → dispatch → collect → destroy), 64 MCP tools, hub system, notes, research briefing, REST API, declarative architecture composition + Python plugins, training module (extracted from train_gpt.py), W&B bridge, interactive TUI.

**What's next:**
- Test coverage push (fleet, architectures, components, runner)
- Provider plugin system (registry + auto-discovery, like architecture plugins)
- Pre-dispatch model family validation
- CI/CD pipeline and PyPI release

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
