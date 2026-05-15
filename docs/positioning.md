# Crucible Positioning

## The statement

**Crucible is the open research operating system for autonomous ML discovery on commodity GPUs — where hypothesis synthesis, fleet orchestration, and judge-separated loops compose into one closed loop.**

Short form: *for labs that can't afford DeepMind's compute but want Sakana's autonomy.*

## The four-way intersection

Crucible's defensible wedge is the rare overlap of four properties that no other player currently combines:

1. **Autonomous** — runs a closed `hypothesize → batch → execute → reflect → synthesize` loop, not a copilot.
2. **Reproducible** — fleet logs, configs, weights, recipes all version-controlled and shareable via taps + hub + HF.
3. **Open architecture** — plugin registry (15 extension points), MCP-first agent contract, no vendor lock-in on model, judge, optimizer, scheduler, provider, or evaluator.
4. **Commodity-GPU-native** — built around rented spot/on-demand GPUs (RunPod today, SSH anywhere, SkyPilot next), not frontier-lab infrastructure.

Take any one property away and a different incumbent wins. The intersection of all four is what Crucible owns.

## What Crucible IS

- A **fleet orchestrator** with transactional provisioning, orphan recovery, project-tagged isolation, and tiered experiment promotion.
- A **research orchestrator contract** — `research_request_prompt` / `research_submit` exchanges that let any LLM (Claude, GPT, Gemini, Llama via your runner) drive the loop. No LLM keys live in Crucible.
- A **plugin host** — optimizers, schedulers, callbacks, loggers, providers, architectures, data adapters, objectives, block types, stack patterns, augmentations, activations, data sources, evaluators, domain specs — all with 3-tier precedence (builtin < global < local).
- A **cross-project knowledge hub** — findings, tracks, recipes, and plugins shared via `~/.crucible-hub/` and Homebrew-style git taps.
- A **GIANTS-style synthesis engine** — pair-mines hub findings across projects/tracks and emits orchestrator-shaped prompts so your LLM can predict the experiment that synthesizes two parents. Provenance carried as `parent_finding_ids`.
- A **judge-separation contract** — reward judge and eval judge must be different models in different families, enforced at call time before pod time is consumed.
- A **harness optimizer** — meta-harness evolutionary loop over task-specific scaffolds (memory systems, agent harnesses) with N-D Pareto frontier tracking.
- A **publish/peer-research surface** — HF leaderboards, findings, recipes, model artifacts pushed outward; peer-agent prior attempts and discussions pulled inward.

## What Crucible explicitly is NOT

Scope discipline. These are real markets owned by real incumbents; Crucible does not compete here.

- **Not a wet-lab platform.** Periodic Labs, Profluent, Cradle, Inceptive, and FutureHouse Phoenix own biology/chemistry/materials with lab-integrated tools. Crucible is computational only.
- **Not a frontier-model competitor.** Gemini, Claude, GPT, DeepSeek-V3-class scale is not the goal. Crucible is model-agnostic — you supply the LLM via the orchestrator contract.
- **Not a paper pipeline (yet).** Sakana AI Scientist v2 has a peer-review breakthrough. Crucible's Phase 4 ships a paper-draft generator, but writing isn't the moat — the research loop is.
- **Not an enterprise dev-engineer agent.** Cognition Devin has SWE-1.6 and PR-merge enterprise sales. Crucible's buyer is a researcher, not an engineering ops team.
- **Not a Kubernetes / cluster manager.** SkyPilot / Modal / Anyscale own multi-cloud abstraction. Crucible integrates with them, doesn't replace them.
- **Not an HPO library.** Optuna and Ax own Bayesian / TPE / CMA-ES math. Crucible bridges Optuna, doesn't reinvent it.
- **Not a tracking dashboard.** W&B / MLflow own the visualization layer. Crucible logs to them.
- **Not a model registry.** HuggingFace owns model hosting. Crucible publishes to HF.

## The competitive landscape (May 2026)

### Closed-loop AI scientists (direct competition)

| Player | What they have | What we have they don't |
|---|---|---|
| **Sakana AI Scientist v2** | Peer-review-validated Nature 2026 paper, paper-generation pipeline | Fleet orchestration, plugin architecture, GIANTS synthesis (theirs is greedy-iterative) |
| **AlphaEvolve** (DeepMind) | Broke 56-year matrix-mult records; Gemini-powered | Judge-separation (they use monolithic Gemini); open architecture |
| **FunSearch** (DeepMind) | Validated on combinatorial algorithms; open-source | Fully autonomous (no required human backbone) |

### Co-scientist / copilot platforms

| Player | What they have | What we have they don't |
|---|---|---|
| **FutureHouse Platform** (Aviary + Crow/Phoenix/Falcon/Owl) | Open Aviary framework, biomedical domain integration, chemistry DBs, safety filters | Closed-loop autonomy (they're copilot-shaped) |
| **DeepMind AI Co-Scientist** | Genesis/DOE partnership, Gemini 2.0 multi-agent | Decoupled architecture, runs on your GPUs |
| **OpenAI Deep Research / Anthropic Managed Agents** | UX integration, web indexing | Closed-loop experiment orchestration on real compute |

### Orthogonal — strong, not direct competition

- **Periodic Labs / Profluent / Cradle / Inceptive** — wet-lab. Different problem.
- **MatterGen / GNoME** — materials prediction models, not orchestration.
- **SkyPilot / Modal / Anyscale Ray Tune** — pure infra; integrates with us.
- **W&B Sweeps / Optuna / Ax** — HPO math; bridges with us.
- **HuggingFace AutoTrain / Replicate** — finetune-as-a-service; different buyer.

### Where Crucible is hopelessly behind (don't chase)

- Wet-lab integration
- Peer-review breakthroughs (yet)
- Frontier model capability
- Enterprise sales motion

## The buyer

- An academic lab that wants Sakana's autonomy on a $50K/year compute budget.
- A startup that wants to run hypothesis-driven research without vendor lock-in.
- A serious independent researcher who needs reproducibility from fleet log to weight.
- A team running multi-agent research where two Crucible instances coordinate via shared HF Discussions.

The buyer is **not** an enterprise dev-tools customer, not a wet-lab researcher, not a hyperscale lab.

## Scope tests for new feature work

Before adding anything to Crucible, run it through these tests:

1. **Strengthens autonomous loop?** Hypothesis → execute → reflect → synthesize. If it helps close the loop, build.
2. **Strengthens reproducibility?** Logs, configs, recipes, provenance. If it makes results easier to reshare, build.
3. **Strengthens plugin architecture?** New extension points, taps, MCP tools. If it makes the platform more open, build.
4. **Strengthens commodity-GPU fit?** Rental spot/on-demand, multi-cloud, cost optimization. If it makes the platform cheaper to run, build.
5. **Drifts into excluded territory?** Wet-lab, frontier-model-scale, dev-agent, web UI, HPO math reinvention. If yes, drop or bridge to the incumbent.

## Reference

- Full audit and execution plan: `/Users/eren/.claude/plans/ai-native-discovery-engines-xuster-virtual-hare.md`
- Five-phase roadmap: [`../ROADMAP.md`](../ROADMAP.md)
- Judge-separation contract: [`./judge-separation.md`](./judge-separation.md)
- Orchestrator contract: see CLAUDE.md "Orchestrator contract" section
