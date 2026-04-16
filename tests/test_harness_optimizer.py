"""Tests for HarnessOptimizer (dry-run covers the full pipeline without LLM)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible.core.config import load_config
from crucible.researcher.domain_spec import DomainSpec
from crucible.researcher.evolution_log import read_log
from crucible.researcher.harness_optimizer import HarnessOptimizer


def _write_project(root: Path) -> None:
    (root / "crucible.yaml").write_text(
        yaml.dump({
            "name": "harness_opt_test",
            "metrics": {"primary": "val_bpb", "direction": "minimize"},
        }),
        encoding="utf-8",
    )


@pytest.fixture
def spec() -> DomainSpec:
    return DomainSpec(
        name="demo",
        interface={
            "class_name": "Harness",
            "required_methods": [{"name": "predict"}],
        },
        metrics=[
            {"name": "accuracy", "direction": "maximize"},
            {"name": "tokens", "direction": "minimize"},
        ],
    )


def test_dry_run_single_iteration(tmp_path: Path, spec: DomainSpec) -> None:
    _write_project(tmp_path)
    cfg = load_config(tmp_path / "crucible.yaml")
    opt = HarnessOptimizer(cfg, domain_spec=spec, tree_name="t1", dry_run=True, n_candidates=3)
    summary = opt.run_iteration()

    assert summary["iteration"] == 1
    assert len(summary["proposed"]) == 3
    assert len(summary["validated"]) == 3
    assert len(summary["benchmarked_node_ids"]) == 3
    assert summary["frontier_summary"]["frontier_size"] >= 1
    # Evolution log has exactly one record
    records = read_log(opt.tree_dir)
    assert len(records) == 1


def test_dry_run_multiple_iterations(tmp_path: Path, spec: DomainSpec) -> None:
    _write_project(tmp_path)
    cfg = load_config(tmp_path / "crucible.yaml")
    opt = HarnessOptimizer(cfg, domain_spec=spec, tree_name="t2", dry_run=True, n_candidates=2)
    opt.run_iteration()
    opt.run_iteration()
    records = read_log(opt.tree_dir)
    assert [r["iteration"] for r in records] == [1, 2]
    # Cumulative nodes grew (2 per iter).
    assert opt.tree.meta["total_nodes"] == 4


def test_resumes_iteration_count(tmp_path: Path, spec: DomainSpec) -> None:
    _write_project(tmp_path)
    cfg = load_config(tmp_path / "crucible.yaml")
    opt1 = HarnessOptimizer(cfg, domain_spec=spec, tree_name="t3", dry_run=True)
    opt1.run_iteration()
    opt1.run_iteration()

    # Re-open; iteration counter should resume from log tail.
    opt2 = HarnessOptimizer(cfg, domain_spec=spec, tree_name="t3", dry_run=True)
    summary = opt2.run_iteration()
    assert summary["iteration"] == 3


def test_existing_tree_metrics_are_refreshed(tmp_path: Path, spec: DomainSpec) -> None:
    _write_project(tmp_path)
    cfg = load_config(tmp_path / "crucible.yaml")
    opt1 = HarnessOptimizer(cfg, domain_spec=spec, tree_name="t4", dry_run=True)
    opt1.run_iteration()

    # Tweak spec metrics and re-open — tree should reflect the new list.
    new_spec = DomainSpec(
        name="demo",
        interface=spec.interface,
        metrics=[
            {"name": "accuracy", "direction": "maximize"},
            {"name": "tokens", "direction": "minimize"},
            {"name": "latency_ms", "direction": "minimize"},
        ],
    )
    opt2 = HarnessOptimizer(cfg, domain_spec=new_spec, tree_name="t4", dry_run=True)
    assert opt2.tree.meta["metrics"] == list(new_spec.metrics)


def test_dry_run_parse_falls_back_without_json() -> None:
    # Direct test of the parser path: code blocks only, no JSON metadata.
    opt = object.__new__(HarnessOptimizer)  # bypass __init__
    opt._iteration = 0  # type: ignore[attr-defined]
    raw = """Some prose.
```python
class Harness:
    def predict(self, x): return 0
```
"""
    cands = HarnessOptimizer._parse_candidates(opt, raw)  # type: ignore[arg-type]
    assert len(cands) == 1
    assert "class Harness" in cands[0]["code"]
    assert cands[0]["name"]


def test_parse_candidates_with_json_metadata() -> None:
    opt = object.__new__(HarnessOptimizer)
    opt._iteration = 2  # type: ignore[attr-defined]
    raw = """
```python
class A:
    def predict(self, x): return 0
```

```python
class B:
    def predict(self, x): return 1
```

```json
[
  {"name": "variant_a", "hypothesis": "first", "rationale": "r1"},
  {"name": "variant_b", "hypothesis": "second", "rationale": "r2"}
]
```
"""
    cands = HarnessOptimizer._parse_candidates(opt, raw)  # type: ignore[arg-type]
    assert [c["name"] for c in cands] == ["variant_a", "variant_b"]
    assert cands[0]["hypothesis"] == "first"
    assert cands[1]["rationale"] == "r2"


def test_proposal_prompt_is_built_from_spec(tmp_path, spec: DomainSpec) -> None:
    _write_project(tmp_path)
    cfg = load_config(tmp_path / "crucible.yaml")
    opt = HarnessOptimizer(cfg, domain_spec=spec, tree_name="pp", dry_run=True)
    system, user = opt._build_proposal_prompt(3)
    assert "harness class" in system.lower()
    assert "# Domain: demo" in user
    assert "accuracy" in user and "tokens" in user
    assert "Required methods:" in user


def test_format_recent_log_with_entries(tmp_path, spec: DomainSpec) -> None:
    _write_project(tmp_path)
    cfg = load_config(tmp_path / "crucible.yaml")
    opt = HarnessOptimizer(cfg, domain_spec=spec, tree_name="fl", dry_run=True)
    opt.run_iteration()
    rendered = opt._format_recent_log(max_records=5)
    assert "iter 1" in rendered
    assert "frontier_size" in rendered


def test_local_execution_records_results_when_fleet_unavailable(
    tmp_path, spec: DomainSpec, monkeypatch
) -> None:
    """Cover the _execute_local path when fleet import fails."""
    _write_project(tmp_path)
    cfg = load_config(tmp_path / "crucible.yaml")
    opt = HarnessOptimizer(cfg, domain_spec=spec, tree_name="le", dry_run=False)

    # Force fleet path to raise ImportError
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "crucible.fleet.manager", None)

    # Stub the runner so we don't invoke real subprocesses.
    def fake_run_experiment(**_kw):
        return {"status": "completed", "result": {"accuracy": 0.7, "tokens": 42}}

    import crucible.runner.experiment as _exp
    monkeypatch.setattr(_exp, "run_experiment", fake_run_experiment)

    # Seed a proposal to skip the LLM.
    cands = opt._dry_run_candidates(1)
    valid = opt.validate_candidates(list(cands))
    node_ids = opt.benchmark(valid)
    assert node_ids
    # Because dry_run=False, benchmark dispatches; fleet is forced to fail,
    # so _execute_local runs via the stub above.
    node = opt.tree.get_node(node_ids[0])
    assert node is not None
    assert node["status"] == "completed"
    assert node["result_metrics"].get("accuracy") == 0.7


def test_fleet_dispatch_receives_tier_and_backend(
    tmp_path, spec: DomainSpec, monkeypatch
) -> None:
    """Regression guard: fleet queue contract requires tier + backend."""
    _write_project(tmp_path)
    cfg = load_config(tmp_path / "crucible.yaml")

    # Spec with explicit tier/backend in evaluation
    spec_with_tier = DomainSpec(
        name=spec.name,
        interface=spec.interface,
        metrics=list(spec.metrics),
        evaluation={"tier": "harness", "backend": "harness", "timeout_seconds": 60},
    )
    opt = HarnessOptimizer(
        cfg, domain_spec=spec_with_tier, tree_name="ft", dry_run=False
    )

    captured: dict = {}

    class FakeFleet:
        def __init__(self, _cfg):
            pass

        def enqueue(self, experiments, limit=0):
            captured["exps"] = list(experiments)
            return experiments

        def dispatch(self, max_assignments=0):
            return []

    import crucible.fleet.manager as _fm
    monkeypatch.setattr(_fm, "FleetManager", FakeFleet)

    cands = opt._dry_run_candidates(1)
    valid = opt.validate_candidates(list(cands))
    opt.benchmark(valid)

    assert captured["exps"], "fleet enqueue received empty experiments"
    exp = captured["exps"][0]
    assert exp["tier"] == "harness"
    assert exp["backend"] == "harness"
    assert exp["wave"].startswith("harness_iter_")
