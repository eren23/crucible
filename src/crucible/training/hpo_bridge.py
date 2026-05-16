"""Hyperparameter-optimization bridge (Phase 3.4).

Wraps Optuna (and, by extension, Ax via Optuna's BoTorchSampler) so
the autonomous research loop can drive a real Bayesian / TPE / CMA-ES
search through Crucible's fleet without reinventing the math.

Contract: tell-and-ask (Optuna's native API).

  bridge = HPOStudy(spec)            # define the param space + sampler
  trial = bridge.ask()               # → {trial_id, params: {LR: "1e-3", ...}}
  # Crucible enqueues + dispatches the trial as a normal experiment.
  # When results land, the orchestrator (or tree-expand policy) calls:
  bridge.tell(trial_id, score)       # feed back numerical metric
  # Repeat. bridge.best() returns the running best.

Persisted state under ``.crucible/hpo_studies/{name}.json`` so the
study can resume across process restarts. Optuna itself uses an
in-memory storage by default; we mirror trial outcomes to JSON so a
fresh Python process can rebuild the study via :meth:`HPOStudy.load`.

Optuna is an optional dependency. Construction without optuna
installed raises :class:`HPOImportError` with the actionable
``pip install optuna`` hint; never crashes inside a timer / dispatch
path.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.errors import CrucibleError


class HPOImportError(CrucibleError):
    """Optuna isn't installed. Raised at construction / load time only."""


class HPOConfigError(CrucibleError):
    """The HPO study spec is malformed (missing params, bad distribution)."""


class HPOStateError(CrucibleError):
    """Operation called in the wrong state (e.g., tell on unknown trial)."""


# Distributions we expose. Optuna has more; this is the MVP set.
_DISTRIBUTIONS = ("float", "int", "log_float", "categorical")


@dataclass
class TrialRecord:
    """One trial's persisted state. Mirrors Optuna's trial; persistable to JSON."""

    trial_id: int
    params: dict[str, Any]
    score: float | None = None
    status: str = "running"  # running | complete | failed | pruned
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None


class HPOStudy:
    """An ongoing HPO study backed by Optuna.

    Parameters
    ----------
    name:
        Study identifier; used as the persisted JSON filename.
    params:
        Map of env-var name → distribution spec. Each spec is a dict
        with ``type`` ∈ {float, int, log_float, categorical} and the
        relevant bounds:
          - float/int: ``low``, ``high``
          - log_float: ``low``, ``high`` (log-uniform sampling)
          - categorical: ``choices`` (list of str/int/float)
        Values are coerced to string when emitted as env-var
        overrides (every Crucible env-var override is a string).
    direction:
        ``"minimize"`` (default) or ``"maximize"``.
    sampler:
        Name of an Optuna sampler. ``"tpe"`` (default), ``"random"``,
        ``"cmaes"``, ``"botorch"``. ``botorch`` requires the
        ``optuna[botorch]`` extra; falls through to TPE if missing
        with a warning.
    seed:
        Random seed for the sampler. ``None`` = system random.
    storage_dir:
        Directory under which trial history is persisted. ``None``
        means in-memory only (no resume across process restarts).
    """

    def __init__(
        self,
        *,
        name: str,
        params: dict[str, dict[str, Any]],
        direction: str = "minimize",
        sampler: str = "tpe",
        seed: int | None = None,
        storage_dir: Path | None = None,
    ) -> None:
        self.name = name
        self.params = params
        self.direction = direction
        self.storage_dir = storage_dir
        self._trial_records: dict[int, TrialRecord] = {}

        self._validate_spec()
        self._study = self._build_optuna_study(sampler=sampler, seed=seed)

    # ------------------------------------------------------------------
    # Cross-process resume
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        *,
        name: str,
        storage_dir: Path,
        sampler: str = "tpe",
        seed: int | None = None,
    ) -> "HPOStudy":
        """Reconstruct an HPOStudy from its persisted JSON.

        Replays completed trials into the underlying Optuna study via
        :func:`optuna.trial.create_trial` + ``study.add_trial`` so the
        sampler's belief is current — not just the bookkeeping. Trials
        whose params don't match the persisted distributions are
        skipped (with a warning) rather than failing the whole load.

        Raises :class:`HPOConfigError` if the persisted file is missing
        or malformed.
        """
        import json

        path = storage_dir / f"{name}.json"
        if not path.exists():
            raise HPOConfigError(
                f"No persisted HPO study at {path}. "
                f"Call HPOStudy(...) directly to create."
            )

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise HPOConfigError(
                f"Persisted HPO study at {path} is malformed: {exc}"
            ) from exc

        study = cls(
            name=name,
            params=data["params"],
            direction=data.get("direction", "minimize"),
            sampler=sampler,
            seed=seed,
            storage_dir=storage_dir,
        )

        # Replay persisted trials through Optuna so the sampler updates
        # its posterior. The previous implementation only repopulated
        # _trial_records, leaving Optuna's study counter at 0 — which
        # broke subsequent tell() calls referencing old trial_ids.
        import optuna
        distributions = study._optuna_distributions()
        for t in data.get("trials", []):
            tid = t.get("trial_id")
            if not isinstance(tid, int):
                continue
            status = t.get("status", "complete")
            score = t.get("score")
            try:
                state = {
                    "complete": optuna.trial.TrialState.COMPLETE,
                    "failed": optuna.trial.TrialState.FAIL,
                    "pruned": optuna.trial.TrialState.PRUNED,
                }.get(status, optuna.trial.TrialState.FAIL)
                ft = optuna.trial.create_trial(
                    params=t.get("params", {}),
                    distributions=distributions,
                    value=float(score) if (status == "complete" and score is not None) else None,
                    state=state,
                )
                study._study.add_trial(ft)
            except Exception as exc:  # pragma: no cover — defensive
                from crucible.core.log import log_warn
                log_warn(
                    f"HPOStudy.load: skipping trial {tid} of study {name!r}: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            study._trial_records[tid] = TrialRecord(
                trial_id=tid,
                params=t.get("params", {}),
                score=score,
                status=status,
                created_at=t.get("created_at", 0.0),
                completed_at=t.get("completed_at"),
            )
        return study

    def _optuna_distributions(self):
        """Build the Optuna distribution objects matching ``self.params``."""
        import optuna.distributions as od

        out = {}
        for name, spec in self.params.items():
            ptype = spec["type"]
            if ptype == "float":
                out[name] = od.FloatDistribution(
                    low=float(spec["low"]), high=float(spec["high"]),
                )
            elif ptype == "log_float":
                out[name] = od.FloatDistribution(
                    low=float(spec["low"]), high=float(spec["high"]), log=True,
                )
            elif ptype == "int":
                out[name] = od.IntDistribution(
                    low=int(spec["low"]), high=int(spec["high"]),
                )
            elif ptype == "categorical":
                out[name] = od.CategoricalDistribution(choices=spec["choices"])
        return out

    # ------------------------------------------------------------------
    # ask + tell
    # ------------------------------------------------------------------

    def ask(self) -> dict[str, Any]:
        """Sample a new trial from the study.

        Returns ``{trial_id, params: {ENV_VAR: "stringified_value", ...}}``.
        The caller (orchestrator / tree-expand policy) translates
        ``params`` into a Crucible experiment config.
        """
        trial = self._study.ask()
        sampled = self._sample_from_trial(trial)
        record = TrialRecord(trial_id=trial.number, params=sampled)
        self._trial_records[trial.number] = record
        self._persist()
        return {
            "trial_id": trial.number,
            "params": {k: str(v) for k, v in sampled.items()},
        }

    def tell(
        self, trial_id: int, score: float, *, status: str = "complete"
    ) -> None:
        """Report a trial outcome back to the study.

        ``status`` is ``"complete"`` (success), ``"failed"`` (treat as
        worst), or ``"pruned"`` (Optuna's TrialState.PRUNED — count as
        early-stopped, sampler may update its belief).
        """
        record = self._trial_records.get(trial_id)
        if record is None:
            raise HPOStateError(
                f"Unknown trial_id {trial_id} for study {self.name!r}"
            )
        if record.status != "running":
            raise HPOStateError(
                f"Trial {trial_id} already finalized as {record.status!r}"
            )

        import optuna

        state_map = {
            "complete": optuna.trial.TrialState.COMPLETE,
            "failed": optuna.trial.TrialState.FAIL,
            "pruned": optuna.trial.TrialState.PRUNED,
        }
        state = state_map.get(status, optuna.trial.TrialState.FAIL)
        try:
            self._study.tell(trial_id, score if status == "complete" else None,
                             state=state)
        except Exception as exc:
            # Optuna may reject if the trial already settled in its
            # internal storage. Wrap as a typed error.
            raise HPOStateError(
                f"Optuna rejected tell for trial {trial_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        record.score = score if status == "complete" else None
        record.status = status
        record.completed_at = time.time()
        self._persist()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def best(self) -> dict[str, Any] | None:
        """Return the running best trial as a dict, or None if no trial completed.

        Shape: ``{trial_id, score, params}`` where ``params`` carries
        the actual sampled values (not stringified).
        """
        completed = [
            r for r in self._trial_records.values() if r.status == "complete"
        ]
        if not completed:
            return None
        better = min if self.direction == "minimize" else max
        best_record = better(completed, key=lambda r: r.score)
        return {
            "trial_id": best_record.trial_id,
            "score": best_record.score,
            "params": best_record.params,
        }

    def history(self) -> list[dict[str, Any]]:
        """Return all trial records as a list of dicts (for status / debugging)."""
        return [
            {
                "trial_id": r.trial_id,
                "params": r.params,
                "score": r.score,
                "status": r.status,
                "created_at": r.created_at,
                "completed_at": r.completed_at,
            }
            for r in self._trial_records.values()
        ]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_spec(self) -> None:
        if not isinstance(self.params, dict) or not self.params:
            raise HPOConfigError(
                "HPOStudy.params must be a non-empty dict of param → spec."
            )
        for name, spec in self.params.items():
            if not isinstance(spec, dict):
                raise HPOConfigError(
                    f"Param {name!r} spec must be a dict, got {type(spec).__name__}."
                )
            ptype = spec.get("type")
            if ptype not in _DISTRIBUTIONS:
                raise HPOConfigError(
                    f"Param {name!r} has unknown type {ptype!r}. "
                    f"Valid: {_DISTRIBUTIONS}."
                )
            if ptype in ("float", "int", "log_float"):
                if "low" not in spec or "high" not in spec:
                    raise HPOConfigError(
                        f"Param {name!r} ({ptype}) requires 'low' and 'high'."
                    )
                if spec["low"] >= spec["high"]:
                    raise HPOConfigError(
                        f"Param {name!r}: low ({spec['low']}) must be < high "
                        f"({spec['high']})."
                    )
            elif ptype == "categorical":
                if not isinstance(spec.get("choices"), list) or not spec["choices"]:
                    raise HPOConfigError(
                        f"Param {name!r} (categorical) requires non-empty "
                        "'choices' list."
                    )
        if self.direction not in ("minimize", "maximize"):
            raise HPOConfigError(
                f"direction must be 'minimize' or 'maximize', got {self.direction!r}"
            )

    def _build_optuna_study(self, *, sampler: str, seed: int | None):
        try:
            import optuna
        except ImportError as exc:
            raise HPOImportError(
                "Optuna isn't installed. Install with `pip install optuna` "
                "(or `pip install optuna[botorch]` for BoTorch sampling) to "
                "use the HPO bridge."
            ) from exc

        sampler_cls_map = {
            "tpe": optuna.samplers.TPESampler,
            "random": optuna.samplers.RandomSampler,
            "cmaes": optuna.samplers.CmaEsSampler,
        }
        sampler_cls = sampler_cls_map.get(sampler)
        if sampler == "botorch":
            try:
                from optuna.integration import BoTorchSampler
                sampler_inst = BoTorchSampler(seed=seed)
            except ImportError:
                from crucible.core.log import log_warn
                log_warn(
                    "optuna[botorch] not installed; falling back to TPE sampler."
                )
                sampler_inst = optuna.samplers.TPESampler(seed=seed)
        elif sampler_cls is not None:
            sampler_inst = sampler_cls(seed=seed)
        else:
            raise HPOConfigError(
                f"Unknown sampler {sampler!r}. Valid: tpe, random, cmaes, botorch."
            )

        return optuna.create_study(
            study_name=self.name,
            direction=self.direction,
            sampler=sampler_inst,
        )

    def _sample_from_trial(self, trial) -> dict[str, Any]:
        """Walk the spec, call trial.suggest_* per param type."""
        sampled: dict[str, Any] = {}
        for name, spec in self.params.items():
            ptype = spec["type"]
            if ptype == "float":
                sampled[name] = trial.suggest_float(
                    name, float(spec["low"]), float(spec["high"]),
                )
            elif ptype == "log_float":
                sampled[name] = trial.suggest_float(
                    name, float(spec["low"]), float(spec["high"]), log=True,
                )
            elif ptype == "int":
                sampled[name] = trial.suggest_int(
                    name, int(spec["low"]), int(spec["high"]),
                )
            elif ptype == "categorical":
                sampled[name] = trial.suggest_categorical(name, spec["choices"])
        return sampled

    def _persist(self) -> None:
        if self.storage_dir is None:
            return
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        path = self.storage_dir / f"{self.name}.json"
        path.write_text(
            json.dumps({
                "name": self.name,
                "direction": self.direction,
                "params": self.params,
                "trials": [
                    {
                        "trial_id": r.trial_id,
                        "params": r.params,
                        "score": r.score,
                        "status": r.status,
                        "created_at": r.created_at,
                        "completed_at": r.completed_at,
                    }
                    for r in sorted(
                        self._trial_records.values(), key=lambda x: x.trial_id
                    )
                ],
            }, indent=2),
            encoding="utf-8",
        )
