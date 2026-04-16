"""Evolutionary optimization of task-specific harnesses.

``HarnessOptimizer`` wraps :class:`SearchTree` with meta-harness-style
propose → validate → benchmark → Pareto frontier → repeat flow:

  * Propose code candidates (Python source implementing a domain interface)
    via an LLM, informed by the current frontier and domain spec guidance.
  * Validate them (syntax, interface, constraints, duplicates).
  * Persist valid candidates as tree nodes (with ``HARNESS_CANDIDATE_ID``
    pointing at source on disk) and dispatch via the fleet; fall back to
    local execution when the fleet is unavailable.
  * Record multi-metric results and recompute the N-dimensional Pareto
    frontier on every ingest; append an iteration record to
    ``evolution_log.jsonl``.

The orchestrator is LLM-backend agnostic (any ``LLMClient`` Protocol impl)
and reuses Crucible's fleet + runner contracts unchanged. Harness-specific
remote behavior lives in the meta-harness tap (launcher + evaluator).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crucible.core.config import ProjectConfig
from crucible.core.errors import CrucibleError, HarnessOptimizerError
from crucible.researcher.candidate_validation import batch_validate
from crucible.researcher.domain_spec import DomainSpec, load_domain_spec
from crucible.researcher.evolution_log import (
    append_iteration,
    last_iteration,
)
from crucible.researcher.search_tree import SearchTree

if TYPE_CHECKING:
    from crucible.researcher.llm_client import LLMClient


_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n(?P<code>.*?)```", re.DOTALL
)
_JSON_BLOCK_RE = re.compile(r"```json\s*\n(?P<json>.*?)```", re.DOTALL)


class HarnessOptimizer:
    """Orchestrates the meta-harness evolutionary loop over a SearchTree."""

    def __init__(
        self,
        config: ProjectConfig,
        domain_spec: str | Path | DomainSpec,
        tree_name: str,
        *,
        tree_dir: Path | None = None,
        n_candidates: int = 3,
        llm: "LLMClient | None" = None,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.tree_name = tree_name
        self.n_candidates = n_candidates
        self.dry_run = dry_run

        if isinstance(domain_spec, DomainSpec):
            self.spec = domain_spec
        else:
            self.spec = load_domain_spec(domain_spec)

        if tree_dir is None:
            tree_dir = config.project_root / ".crucible" / "search_trees" / tree_name
        self.tree_dir = Path(tree_dir)

        if (self.tree_dir / "tree.yaml").exists():
            self.tree = SearchTree.load(self.tree_dir)
            # Refresh metrics from the spec so later changes stick.
            if self.spec.metrics:
                self.tree.meta["metrics"] = list(self.spec.metrics)
                self.tree.meta["primary_metric"] = self.spec.metrics[0]["name"]
                self.tree.meta["metric_direction"] = self.spec.metrics[0]["direction"]
                self.tree._save_meta()
        else:
            primary = self.spec.metrics[0]
            self.tree = SearchTree.create(
                tree_dir=self.tree_dir,
                name=tree_name,
                description=f"Harness optimization: {self.spec.name}",
                expansion_policy="agent_directed",
                pruning_policy="agent_directed",
                primary_metric=primary["name"],
                metric_direction=primary["direction"],
                metrics=list(self.spec.metrics),
                candidate_store_dir="candidates",
            )

        self._llm = llm
        self._iteration = last_iteration(self.tree_dir)

    # ------------------------------------------------------------------
    # LLM lazy load
    # ------------------------------------------------------------------

    @property
    def llm(self) -> "LLMClient":
        if self._llm is None:
            from crucible.researcher.llm_client import AnthropicClient

            self._llm = AnthropicClient(model=self.config.researcher.model)
        return self._llm

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    def propose_candidates(self, n: int | None = None) -> list[dict[str, Any]]:
        """Return a list of ``{name, hypothesis, code, rationale, config?}`` dicts."""
        n = n or self.n_candidates
        if self.dry_run:
            return self._dry_run_candidates(n)

        system, user = self._build_proposal_prompt(n)
        raw = self.llm.complete(system=system, user=user, max_tokens=8192)
        if not raw:
            raise HarnessOptimizerError("LLM returned no proposal text")
        return self._parse_candidates(raw)

    def _build_proposal_prompt(self, n: int) -> tuple[str, str]:
        """Assemble system + user prompt from spec + frontier + evolution log."""
        summary = self.tree.frontier_summary()
        log_tail = self._format_recent_log(max_records=3)
        interface = self.spec.interface
        methods = interface.get("required_methods") or []
        metrics_desc = "\n".join(
            f"  - {m['name']} ({m['direction']})" for m in self.spec.metrics
        )

        system = (
            "You are a research assistant proposing Python implementations of a "
            "harness class for an ML experiment. Each proposal must subclass or "
            "match the required interface and implement every listed method. "
            "Return proposals inside ```python ... ``` code fences, one block per "
            "candidate, and a trailing ```json``` block with a list of "
            "{name, hypothesis, rationale} metadata entries in the same order."
        )

        methods_block = "\n".join(
            f"  - {m['name']}({m.get('signature', '...')}) — "
            f"{m.get('description', '')}".rstrip(" —") for m in methods
        )
        baselines_block = "\n".join(
            f"  - {b['name']}: {b.get('description', '')}".rstrip(": ")
            for b in self.spec.baselines
        ) or "  (none)"

        user = (
            f"# Domain: {self.spec.name}\n\n"
            f"## Interface\n"
            f"Class: {interface.get('class_name', '<unspecified>')}\n"
            f"Required methods:\n{methods_block or '  (none)'}\n\n"
            f"## Metrics (Pareto-optimized)\n{metrics_desc}\n\n"
            f"## Baselines\n{baselines_block}\n\n"
            f"## Current Pareto frontier\n{json.dumps(summary, default=str)}\n\n"
            f"## Recent evolution history\n{log_tail or '(empty)'}\n\n"
            f"## Proposal guidance\n{self.spec.proposal_guidance or '(none)'}\n\n"
            f"Produce {n} diverse candidate implementations now."
        )
        return system, user

    def _format_recent_log(self, max_records: int = 3) -> str:
        from crucible.researcher.evolution_log import read_log

        records = read_log(self.tree_dir)[-max_records:]
        if not records:
            return ""
        lines = []
        for r in records:
            fs = r.get("frontier_summary") or {}
            lines.append(
                f"- iter {r.get('iteration')}: proposed={r.get('proposed')} "
                f"validated={r.get('validated')} frontier_size={fs.get('frontier_size')}"
            )
        return "\n".join(lines)

    def _parse_candidates(self, raw: str) -> list[dict[str, Any]]:
        """Extract code blocks + optional JSON metadata from an LLM response."""
        code_blocks = [m.group("code").strip() for m in _CODE_BLOCK_RE.finditer(raw)]
        # Drop the JSON block from code_blocks if the regex matched it.
        code_blocks = [c for c in code_blocks if not c.lstrip().startswith("[")]

        metadata: list[dict[str, Any]] = []
        json_match = _JSON_BLOCK_RE.search(raw)
        if json_match:
            try:
                parsed = json.loads(json_match.group("json"))
                if isinstance(parsed, list):
                    metadata = [m for m in parsed if isinstance(m, dict)]
            except json.JSONDecodeError:
                metadata = []

        iter_label = max(self._iteration, 1)
        candidates: list[dict[str, Any]] = []
        for i, code in enumerate(code_blocks):
            meta = metadata[i] if i < len(metadata) else {}
            candidates.append({
                "name": meta.get("name") or f"cand_{iter_label}_{i}",
                "hypothesis": meta.get("hypothesis", ""),
                "rationale": meta.get("rationale", ""),
                "code": code,
                "config": meta.get("config") or {},
            })
        return candidates

    def _dry_run_candidates(self, n: int) -> list[dict[str, Any]]:
        """Fixture candidates for exercising the pipeline without LLM calls."""
        class_name = self.spec.class_name or "HarnessCandidate"
        methods = self.spec.required_method_names or ["predict"]

        def _stub(name: str, body: str) -> str:
            method_bodies = "\n".join(
                f"    def {m}(self, *args, **kwargs):\n        {body}"
                for m in methods
            )
            base = f"({class_name})" if self.spec.class_name else ""
            return f"class {name}{base}:\n{method_bodies}\n"

        iter_label = max(self._iteration, 1)
        return [
            {
                "name": f"dryrun_cand_{iter_label}_{i}",
                "hypothesis": f"Dry-run fixture candidate {i}",
                "rationale": "Dry-run fixture; no LLM call.",
                "code": _stub(f"Dryrun{i}", "return None"),
                "config": {},
            }
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def validate_candidates(
        self, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Validate candidates against the domain spec; drop invalid ones.

        Returns only valid candidates, preserving order. The rejected set is
        available via the ``validation`` field on each input dict after the
        call returns.
        """
        seen = self._existing_candidate_hashes()
        annotated = batch_validate(candidates, self.spec, seen_hashes=seen)
        # Mutate input list in place so callers can inspect rejections.
        for i, entry in enumerate(annotated):
            candidates[i]["validation"] = entry["validation"]
            candidates[i]["valid"] = entry["valid"]
        return [c for c in annotated if c["valid"]]

    def _existing_candidate_hashes(self) -> set[str]:
        """Collect SHA-256 hashes of candidates already in the tree."""
        from crucible.researcher.candidate_validation import _hash_code

        cdir = self.tree._candidate_dir()
        if not cdir.exists():
            return set()
        hashes: set[str] = set()
        for path in cdir.glob("*.py"):
            try:
                hashes.add(_hash_code(path.read_text(encoding="utf-8")))
            except OSError:
                continue
        return hashes

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------

    def benchmark(self, candidates: list[dict[str, Any]]) -> list[str]:
        """Store code, add tree nodes, dispatch via fleet or local fallback.

        Returns the list of newly created node IDs (status ``pending`` until
        results arrive). Benchmarking is a fire-and-forget step here; call
        :meth:`collect_results` separately once runs complete.
        """
        node_ids: list[str] = []
        for cand in candidates:
            node_id = self.tree.add_root(
                name=cand["name"],
                config=dict(cand.get("config") or {}),
                hypothesis=cand.get("hypothesis", ""),
                rationale=cand.get("rationale", ""),
                tags=["harness-candidate", f"domain:{self.spec.name}"],
            )
            self.tree.store_candidate(node_id, cand["code"])
            # Wire the runner launcher: LAUNCH_SCRIPT tells run_remote.py to
            # delegate to the harness runner instead of the native loop.
            node = self.tree.get_node(node_id)
            if node is not None:
                launch_script = self.spec.evaluation.get("launch_script", "")
                if launch_script:
                    node["config"]["LAUNCH_SCRIPT"] = launch_script
                if self.spec.class_name:
                    node["config"].setdefault("HARNESS_CLASS_NAME", self.spec.class_name)
            node_ids.append(node_id)

        if self.dry_run:
            # Dry-run: immediately synthesize results from a deterministic heuristic.
            self._dry_run_complete(node_ids)
            return node_ids

        self._dispatch(node_ids)
        return node_ids

    def _dispatch(self, node_ids: list[str]) -> None:
        """Submit nodes via the fleet; fall back to local execution on error."""
        tier = self.spec.evaluation.get("tier") or "harness"
        backend = self.spec.evaluation.get("backend") or "harness"
        wave = f"harness_iter_{self._iteration}"

        experiments = []
        for nid in node_ids:
            node = self.tree.get_node(nid)
            if node is None:
                continue
            experiments.append({
                "name": node["experiment_name"],
                "config": dict(node["config"]),
                "tags": list(node.get("tags", [])),
                "tier": tier,
                "backend": backend,
                "wave": wave,
            })

        try:
            from crucible.fleet.manager import FleetManager

            fleet = FleetManager(self.config)
            fleet.enqueue(experiments=experiments, limit=0)
            fleet.dispatch(max_assignments=len(experiments))
        except (ImportError, CrucibleError):
            # Local fallback: run synchronously using the runner contract.
            self._execute_local(node_ids)

    def _execute_local(self, node_ids: list[str]) -> None:
        """Run candidates locally via the Crucible runner contract.

        Runner-side errors (``RunnerError``, ``CrucibleError``) are captured
        per-node so one bad candidate does not abort the batch. Unexpected
        errors propagate — they signal bugs in Crucible, not candidate code.
        """
        try:
            from crucible.runner.experiment import run_experiment
        except ImportError:
            return

        from crucible.core.errors import RunnerError

        for nid in node_ids:
            node = self.tree.get_node(nid)
            if node is None:
                continue
            try:
                result = run_experiment(
                    config=dict(node["config"]),
                    name=node["experiment_name"],
                    experiment_id=f"harness_{nid[:8]}",
                    tags=list(node.get("tags", [])),
                    timeout_seconds=int(self.spec.evaluation.get("timeout_seconds", 600)),
                    project_root=self.config.project_root,
                    project_config=self.config,
                )
            except (RunnerError, CrucibleError) as exc:
                # Expected runner failure: record a failed result and move on.
                result = {"status": "failed", "error": str(exc)}
            # Store whatever result keys we have; downstream frontier update
            # only cares about numeric metrics listed in the domain spec.
            metrics = result.get("result") or {}
            self.tree.record_result(nid, metrics)

    def _dry_run_complete(self, node_ids: list[str]) -> None:
        """Synthesize deterministic multi-metric results for dry-run flows."""
        for i, nid in enumerate(node_ids):
            fake = {}
            for j, m in enumerate(self.spec.metrics):
                # Stagger values so we generate a non-trivial frontier.
                base = 1.0 + 0.1 * (i % 3)
                fake[m["name"]] = base + 0.01 * (j + 1) * (1 if i % 2 == 0 else -1)
            self.tree.record_result(nid, fake)

    # ------------------------------------------------------------------
    # Iterate
    # ------------------------------------------------------------------

    def run_iteration(
        self, *, cost: dict[str, Any] | None = None, notes: str = ""
    ) -> dict[str, Any]:
        """Execute a full propose→validate→benchmark cycle and log it.

        Returns a summary dict containing counts, the frontier snapshot, and
        the evolution log record that was appended.
        """
        self._iteration += 1
        proposed = self.propose_candidates()
        valid = self.validate_candidates(list(proposed))
        node_ids = self.benchmark(valid) if valid else []
        frontier_summary = self.tree.frontier_summary()

        record = append_iteration(
            self.tree_dir,
            iteration=self._iteration,
            proposed=len(proposed),
            validated=len(valid),
            benchmarked=len(node_ids),
            frontier_summary=frontier_summary,
            cost=cost or {},
            notes=notes,
        )

        return {
            "iteration": self._iteration,
            "proposed": [c.get("name") for c in proposed],
            "validated": [c.get("name") for c in valid],
            "benchmarked_node_ids": node_ids,
            "frontier_summary": frontier_summary,
            "log_record": record,
        }

    def run(self, max_iterations: int = 5) -> list[dict[str, Any]]:
        """Run multiple iterations back-to-back. Returns per-iteration summaries."""
        out: list[dict[str, Any]] = []
        for _ in range(max_iterations):
            out.append(self.run_iteration())
        return out

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def frontier(self) -> dict[str, Any]:
        return self.tree.frontier_summary()

    def pareto_node_ids(self) -> list[str]:
        return self.tree.pareto_nodes()
