# Autonomous Research Loop

Let Claude drive your ML experimentation: analyze results, generate hypotheses, design experiments, execute, reflect, promote or kill.

## Prerequisites

- `ANTHROPIC_API_KEY` in your `.env`
- A `program.md` file describing your research goals, model families, and hyperparameter space
- Working fleet setup (see [2-pod experiment](2pod_experiment.md)) OR local execution

## Setup

```bash
crucible init
# Edit crucible.yaml:
#   researcher.model: claude-sonnet-4-6-20250514
#   researcher.budget_hours: 10.0
#   researcher.program_file: program.md
```

## Dry Run (No LLM Calls)

Test the loop structure without spending API credits:

```bash
crucible research start --dry-run --tier smoke
```

This uses fixture hypotheses instead of calling Claude, and runs experiments locally with the smoke preset.

## Full Autonomous Loop

```bash
# 10 compute-hours budget, proxy tier experiments
crucible research start --budget-hours 10 --tier proxy

# Smaller budget, more iterations
crucible research start --budget-hours 5 --max-iterations 30 --tier smoke
```

## What Happens

Each iteration:
1. **Analyze** — Reads completed experiments, builds leaderboard and sensitivity analysis
2. **Hypothesize** — Claude generates 3-5 ranked experiment hypotheses based on results
3. **Design** — Converts hypotheses to experiment configs, includes baseline control
4. **Execute** — Dispatches to fleet (or runs locally if no fleet available)
5. **Collect** — Waits for experiments to complete, gathers results
6. **Reflect** — Claude compares predictions to outcomes, updates beliefs
7. **Promote/Kill** — Winners get promoted to next tier, dead ends are killed

## Check Status

```bash
crucible research status
```

Shows budget remaining, hypothesis count, experiment history, and current beliefs.

## The program.md File

This is your research brief for Claude. Include:

```markdown
# Research Program

## Objective
What metric are you optimizing? What constraints exist?

## Model Families
What architectures are available?

## Key Hyperparameters
What knobs can be tuned?

## Research Priorities
What should be explored first?

## Constraints
Budget, time limits, hardware limitations.
```

## Tips

- Start with `--dry-run` to verify the pipeline
- Use `--tier smoke` for fast iteration during development
- The researcher automatically includes a baseline control experiment
- Beliefs persist across restarts (stored in `research_state.jsonl`)
- Budget tracking prevents runaway spending
