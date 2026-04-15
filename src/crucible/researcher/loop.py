"""Autonomous multi-phase research loop.

Replaces the monolithic researcher with a structured explore/exploit cycle that uses
LLM-driven hypothesis generation, fleet-based experiment execution, and
automatic promotion/killing of experiment branches.
"""
from __future__ import annotations

import time
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.errors import CrucibleError
from crucible.researcher.batch_design import DEFAULT_TIER_COSTS, design_batch
from crucible.researcher.hypothesis import generate_hypotheses
from crucible.researcher.llm_client import AnthropicClient, LLMClient
from crucible.researcher.reflection import promote_or_kill, reflect_and_update
from crucible.researcher.state import ResearchState


class AutonomousResearcher:
    """Multi-phase autonomous research loop."""

    def __init__(
        self,
        config: ProjectConfig,
        budget_hours: float | None = None,
        max_iterations: int | None = None,
        tier: str = "proxy",
        backend: str = "torch",
        dry_run: bool = False,
        llm: LLMClient | None = None,
        baseline_config: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.tier = tier
        self.backend = backend
        self.dry_run = dry_run
        self.baseline_config = baseline_config

        budget = budget_hours or config.researcher.budget_hours
        self.max_iterations = max_iterations or config.researcher.max_iterations

        state_path = config.project_root / config.research_state_file
        self.state = ResearchState(state_path, budget_hours=budget)

        self.llm = llm or AnthropicClient(model=config.researcher.model)
        self._program_text: str | None = None
        self._iteration = 0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        print("=" * 60)
        print(f"Autonomous Researcher  --  {self.config.name}")
        print(f"Budget: {self.state.budget_remaining:.2f} compute-hours remaining")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Tier: {self.tier}  Backend: {self.backend}")
        print(f"Dry run: {self.dry_run}")
        print("=" * 60)

        while self.state.budget_remaining > 0 and self._iteration < self.max_iterations:
            self._iteration += 1
            print(f"\n{'=' * 60}")
            print(
                f"  Iteration {self._iteration}/{self.max_iterations}  "
                f"(budget left: {self.state.budget_remaining:.2f}h)"
            )
            print("=" * 60)

            try:
                analysis = self.analyze()

                # Literature awareness (best-effort, never blocks the loop)
                literature_context = ""
                try:
                    from crucible.researcher.literature import (
                        format_literature_context,
                        search_papers,
                        suggest_queries,
                    )

                    queries = suggest_queries(
                        self._get_program_text(),
                        self.state.beliefs,
                        self.state.get_findings() if hasattr(self.state, "get_findings") else [],
                    )
                    all_papers: list[dict[str, Any]] = []
                    seen: set[str] = set()
                    for q in queries[:3]:
                        for p in search_papers(q, limit=5):
                            if p["id"] not in seen:
                                seen.add(p["id"])
                                all_papers.append(p)
                    literature_context = format_literature_context(all_papers, max_papers=5)
                    if literature_context:
                        print(f"  Literature: found {len(all_papers)} relevant papers.")
                except Exception:
                    pass

                if self.dry_run:
                    hypotheses = _dry_run_hypotheses(self._iteration)
                    print(f"  [DRY RUN] Generated {len(hypotheses)} fixture hypotheses.")
                else:
                    hypotheses = generate_hypotheses(
                        analysis,
                        self._get_program_text(),
                        self.state,
                        self.llm,
                        self._iteration,
                        literature_context=literature_context,
                    )
                if not hypotheses:
                    print("LLM returned no hypotheses. Stopping.")
                    break
                batch = design_batch(
                    hypotheses,
                    self.state,
                    self.tier,
                    self.backend,
                    self._iteration,
                    baseline_config=self.baseline_config,
                )
                if not batch:
                    print("No executable experiments in batch. Stopping.")
                    break
                self.execute_batch(batch)
                self.collect_results()
                if self.dry_run:
                    print("  [DRY RUN] Skipping reflection (no LLM calls).")
                    promote_names, kill_names = [], []
                else:
                    promote_names, kill_names = reflect_and_update(
                        self.state, self.llm, metric_key=self.config.metrics.primary,
                    )
                promote_or_kill(self.state, promote_names, kill_names, self.tier)
                self.state.save()
            except KeyboardInterrupt:
                print("\nInterrupted. Saving state...")
                self.state.save()
                raise
            except CrucibleError as exc:
                print(f"ERROR in iteration {self._iteration}: {exc}")
                self.state.save()
            except Exception as exc:
                print(f"UNEXPECTED ERROR in iteration {self._iteration}: {exc}")
                self.state.save()
                time.sleep(5)

        print(f"\nResearch loop complete after {self._iteration} iterations.")
        self._print_final_leaderboard()
        self.state.save()

    # ------------------------------------------------------------------
    # Phase 1: Analyze
    # ------------------------------------------------------------------

    def analyze(self) -> str:
        """Structured result analysis returning a summary for LLM context."""
        from crucible.researcher.analysis import build_analysis

        return build_analysis(self.config, self.state)

    # ------------------------------------------------------------------
    # Phase 4: Execute Batch
    # ------------------------------------------------------------------

    def execute_batch(self, batch: list[dict[str, Any]]) -> None:
        """Submit experiments to the fleet for execution."""
        if self.dry_run:
            print("  [DRY RUN] Would submit batch:")
            import json

            for exp in batch:
                print(f"    {exp['name']}: {json.dumps(exp['config'], indent=2)}")
            return

        try:
            from crucible.fleet.manager import FleetManager

            fleet = FleetManager(self.config)
            fleet_experiments = [
                {
                    "name": exp["name"],
                    "config": exp["config"],
                    "tags": exp.get("tags", []),
                    "priority": exp.get("priority", 0),
                    "tier": exp.get("tier", self.tier),
                    "backend": exp.get("backend", self.backend),
                    "wave": exp.get("wave", f"auto_iter_{self._iteration}"),
                }
                for exp in batch
            ]
            added = fleet.enqueue(experiments=fleet_experiments, limit=0)
            print(f"  Enqueued {len(added)} experiments to fleet.")
            dispatched = fleet.dispatch(max_assignments=len(batch))
            running = [r for r in dispatched if r.get("lease_state") == "running"]
            print(f"  Dispatched {len(running)} experiments.")
        except ImportError as exc:
            print(f"  Fleet module not available ({exc}). Falling back to local execution.")
            self._execute_local(batch)
        except CrucibleError as exc:
            print(f"  Fleet operation failed ({exc}). Falling back to local execution.")
            self._execute_local(batch)

    def _execute_local(self, batch: list[dict[str, Any]]) -> None:
        """Fallback: run experiments locally one at a time."""
        try:
            from crucible.runner.experiment import run_experiment
        except ImportError:
            print("  Runner module not available. Cannot execute locally.")
            return

        tier_cost = DEFAULT_TIER_COSTS.get(self.tier, 0.5)
        for exp in batch:
            if self.state.budget_remaining < tier_cost:
                print(f"  Budget exhausted, skipping {exp['name']}.")
                break

            print(f"  Running locally: {exp['name']}")
            result = run_experiment(
                config=exp["config"],
                name=exp["name"],
                experiment_id=f"auto_{self._iteration:03d}_{exp['name'][:20]}",
                tags=exp.get("tags", ["autonomous"]),
                timeout_seconds=600,
                backend=exp.get("backend", self.backend),
                preset=exp.get("tier", self.tier),
                project_root=self.config.project_root,
                project_config=self.config,
            )
            self.state.record_result(
                experiment={"name": exp["name"], "config": exp["config"], "pod_hours": tier_cost},
                result=result,
            )
            if result.get("status") == "completed" and result.get("result"):
                metric_val = result["result"].get(self.config.metrics.primary)
                print(f"    Result: {self.config.metrics.primary}={metric_val}")
            else:
                print(f"    Result: {result.get('status', 'unknown')} - {result.get('error', '')[:100]}")

    # ------------------------------------------------------------------
    # Phase 5: Collect Results
    # ------------------------------------------------------------------

    def collect_results(self) -> None:
        """Wait for fleet experiments to complete and gather results."""
        if self.dry_run:
            print("  [DRY RUN] Skipping result collection.")
            return

        try:
            from crucible.fleet.manager import FleetManager

            fleet = FleetManager(self.config)
        except (ImportError, CrucibleError):
            return  # Results already collected by _execute_local

        wave_name = f"auto_iter_{self._iteration}"
        max_wait = 7200
        poll_interval = 60
        elapsed = 0

        print(f"  Waiting for wave '{wave_name}' to complete...")
        while elapsed < max_wait:
            fleet.collect()

            queue = fleet.queue_status()
            wave_items = [r for r in queue if r.get("wave") == wave_name]
            if not wave_items:
                print("  No items found for this wave in queue.")
                break

            terminal_states = {"completed", "finished"}
            terminal = [r for r in wave_items if r.get("lease_state") in terminal_states]
            print(f"  Progress: {len(terminal)}/{len(wave_items)} complete (elapsed: {elapsed}s)")

            if len(terminal) >= len(wave_items):
                print("  All experiments in wave complete.")
                break

            time.sleep(poll_interval)
            elapsed += poll_interval

        # Record results into state
        tier_cost = DEFAULT_TIER_COSTS.get(self.tier, 0.5)
        try:
            from crucible.analysis.results import merged_results

            all_results = merged_results(self.config)
        except ImportError:
            all_results = []

        for r in all_results:
            if wave_name in r.get("tags", []) or wave_name == r.get("wave"):
                if r.get("status") in {"completed", "failed", "timeout", "partial_recoverable"}:
                    already = any(
                        rec.get("experiment", {}).get("name") == r.get("name")
                        for rec in self.state.history
                    )
                    if not already:
                        self.state.record_result(
                            experiment={
                                "name": r.get("name", "unknown"),
                                "config": r.get("config", {}),
                                "pod_hours": tier_cost,
                            },
                            result=r,
                        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_program_text(self) -> str:
        if self._program_text is None:
            program_path = self.config.project_root / self.config.researcher.program_file
            if program_path.exists():
                self._program_text = program_path.read_text(encoding="utf-8")
            else:
                self._program_text = "(program.md not found)"
        return self._program_text

    def _print_final_leaderboard(self) -> None:
        try:
            from crucible.analysis.leaderboard import leaderboard
            from crucible.analysis.results import completed_results

            results = completed_results(self.config)
            top = leaderboard(results, top_n=10, cfg=self.config)
        except ImportError:
            top = []

        if not top:
            print("No completed experiments.")
            return
        primary = self.config.metrics.primary
        secondary = self.config.metrics.secondary or ""
        print("\nFinal leaderboard:")
        for i, r in enumerate(top, 1):
            res = r["result"]
            parts = [f"{primary}={res.get(primary, 'N/A')}"]
            if secondary:
                parts.append(f"{secondary}={res.get(secondary, 'N/A')}")
            print(f"  {i}. {r['name']}: {'  '.join(parts)}")
        if self.state.beliefs:
            print("\nFinal beliefs:")
            for b in self.state.beliefs:
                print(f"  - {b}")


# ---------------------------------------------------------------------------
# Dry-run fixture data
# ---------------------------------------------------------------------------


def _dry_run_hypotheses(iteration: int) -> list[dict[str, Any]]:
    """Return fixture hypotheses for dry-run mode (no LLM calls)."""
    return [
        {
            "hypothesis": "Dry-run fixture: test increased recurrence depth",
            "name": f"dryrun_recurrence_iter{iteration}",
            "expected_impact": 0.005,
            "confidence": 0.6,
            "config": {"MODEL_FAMILY": "looped", "RECURRENCE_STEPS": "12"},
            "rationale": "Fixture hypothesis for dry-run testing.",
            "family": "looped",
        },
        {
            "hypothesis": "Dry-run fixture: test wider hidden dim",
            "name": f"dryrun_wider_iter{iteration}",
            "expected_impact": 0.003,
            "confidence": 0.5,
            "config": {"MODEL_FAMILY": "baseline", "D_MODEL": "512"},
            "rationale": "Fixture hypothesis for dry-run testing.",
            "family": "baseline",
        },
    ]
