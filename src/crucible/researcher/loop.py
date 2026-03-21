"""Autonomous multi-phase research loop.

Replaces the monolithic researcher with a structured explore/exploit cycle that uses
LLM-driven hypothesis generation, fleet-based experiment execution, and
automatic promotion/killing of experiment branches.
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from crucible.core.config import ProjectConfig
from crucible.core.log import utc_now_iso
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

        state_path = config.project_root / "research_state.jsonl"
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
                hypotheses = generate_hypotheses(
                    analysis, self._get_program_text(), self.state, self.llm, self._iteration
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
                promote_names, kill_names = reflect_and_update(self.state, self.llm)
                promote_or_kill(self.state, promote_names, kill_names, self.tier)
                self.state.save()
            except KeyboardInterrupt:
                print("\nInterrupted. Saving state...")
                self.state.save()
                raise
            except Exception as exc:
                print(f"ERROR in iteration {self._iteration}: {exc}")
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
        try:
            from crucible.analysis.leaderboard import leaderboard, sensitivity_analysis
            from crucible.analysis.results import completed_results
        except ImportError:
            return "Analysis module not available. This is the first iteration."

        results = completed_results(self.config)
        if not results:
            return "No completed experiments yet. This is the first iteration."

        sections: list[str] = []

        # Overall leaderboard
        top = leaderboard(results, top_n=10)
        board_lines = ["## Leaderboard (top 10)"]
        for i, r in enumerate(top, 1):
            res = r["result"]
            board_lines.append(
                f"  {i}. {r['name']}: val_bpb={res.get('val_bpb', 'N/A')}  "
                f"val_loss={res.get('val_loss', 'N/A')}  "
                f"bytes={r.get('model_bytes', 'N/A')}"
            )
        sections.append("\n".join(board_lines))

        # Group by model family
        families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in results:
            family = r.get("config", {}).get("MODEL_FAMILY", "unknown")
            families[family].append(r)
        family_lines = ["## Results by Model Family"]
        for family, runs in sorted(families.items()):
            metrics = [r["result"].get("val_bpb") for r in runs if r.get("result", {}).get("val_bpb")]
            if metrics:
                best = min(metrics)
                worst = max(metrics)
                family_lines.append(
                    f"  {family}: {len(runs)} runs, best={best:.4f}, worst={worst:.4f}, spread={worst - best:.4f}"
                )
        sections.append("\n".join(family_lines))

        # Sensitivity analysis
        sens = sensitivity_analysis(results)
        if sens:
            sens_lines = ["## Sensitivity Analysis (top parameters by spread)"]
            ranked_sens = sorted(sens.items(), key=lambda kv: kv[1][-1][1] - kv[1][0][1], reverse=True)
            for key, pairs in ranked_sens[:10]:
                best_val, best_metric = pairs[0]
                worst_val, worst_metric = pairs[-1]
                spread = worst_metric - best_metric
                sens_lines.append(
                    f"  {key}: spread={spread:.4f}  best={best_val}({best_metric:.4f})  "
                    f"worst={worst_val}({worst_metric:.4f})"
                )
            sections.append("\n".join(sens_lines))

        # Detect diminishing returns
        recent = self.state.history[-5:]
        if len(recent) >= 3:
            recent_metrics = []
            for rec in recent:
                metric = rec.get("result", {}).get("val_bpb")
                if isinstance(metric, (int, float)):
                    recent_metrics.append(metric)
            if len(recent_metrics) >= 3 and top:
                recent_best = min(recent_metrics)
                overall_best = top[0]["result"].get("val_bpb", recent_best)
                if isinstance(overall_best, (int, float)) and abs(recent_best - overall_best) < 0.001:
                    sections.append(
                        "## Warning: Diminishing Returns\n"
                        "  Last 5 experiments have not improved on the overall best by >0.001.\n"
                        "  Consider exploring a different model family or hyperparameter axis."
                    )

        # Research state context
        sections.append(f"## Research State\n{self.state.get_history_summary()}")
        if self.state.beliefs:
            beliefs_str = "\n".join(f"  - {b}" for b in self.state.beliefs)
            sections.append(f"## Current Beliefs\n{beliefs_str}")

        return "\n\n".join(sections)

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
            added = fleet.enqueue(fleet_experiments, limit=0)
            print(f"  Enqueued {len(added)} experiments to fleet.")
            dispatched = fleet.dispatch(max_assignments=len(batch))
            running = [r for r in dispatched if r.get("lease_state") == "running"]
            print(f"  Dispatched {len(running)} experiments.")
        except (ImportError, Exception) as exc:
            print(f"  Fleet not available ({exc}). Falling back to local execution.")
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
                project_root=self.config.project_root,
            )
            self.state.record_result(
                experiment={"name": exp["name"], "config": exp["config"], "pod_hours": tier_cost},
                result=result,
            )
            if result.get("status") == "completed" and result.get("result"):
                metric = result["result"].get("val_bpb")
                print(f"    Result: primary_metric={metric}")
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
        except (ImportError, Exception):
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
            from crucible.analysis.results import load_all_results

            all_results = load_all_results(self.config)
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
            top = leaderboard(results, top_n=10)
        except ImportError:
            top = []

        if not top:
            print("No completed experiments.")
            return
        print("\nFinal leaderboard:")
        for i, r in enumerate(top, 1):
            res = r["result"]
            print(f"  {i}. {r['name']}: val_bpb={res.get('val_bpb', 'N/A')}  val_loss={res.get('val_loss', 'N/A')}")
        if self.state.beliefs:
            print("\nFinal beliefs:")
            for b in self.state.beliefs:
                print(f"  - {b}")
