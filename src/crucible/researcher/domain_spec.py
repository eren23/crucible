"""Domain specification for harness optimization.

A ``DomainSpec`` is a YAML-backed contract that tells Crucible:
  - what interface each candidate must implement,
  - which baselines to start the frontier from,
  - which metrics to track (and whether each is minimized or maximized),
  - what parameter bounds / constraints to enforce at validation time,
  - freeform guidance the LLM proposer should read.

Domain specs live on disk at ``.crucible/domain_specs/{name}/domain_spec.yaml``
(or anywhere else if the loader is given an explicit path). They are
distributed across projects via the Crucible tap system (``domain_specs``
plugin type, installed under ``~/.crucible-hub/plugins/domain_specs/{name}/``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.errors import DomainSpecError
from crucible.core.io import read_yaml


@dataclass
class DomainSpec:
    """Structured domain specification.

    Attributes
    ----------
    name:
        Identifier; matches the directory name under ``domain_specs/``.
    interface:
        ``{class_name, required_methods: [{name, signature?, description?}]}``.
        Validators use ``required_methods`` to check candidates.
    baselines:
        List of ``{name, path, description?}`` entries pointing at reference
        implementations shipped with the domain (relative to the spec file).
    metrics:
        List of ``{name, direction, description?, unit?}``. ``direction`` is
        ``"minimize"`` or ``"maximize"``. Feeds the multi-metric Pareto
        frontier in :class:`crucible.researcher.search_tree.SearchTree`.
    constraints:
        ``{param_name: {min, max, type, enum?}}``. Used by the candidate
        validation pass before dispatch.
    proposal_guidance:
        Freeform text injected into the LLM proposer prompt.
    evaluation:
        ``{runner, timeout, datasets}`` hints for the launcher / evaluator.
    source_path:
        Path to the YAML file this spec was loaded from, for error messages.
    """

    name: str
    interface: dict[str, Any] = field(default_factory=dict)
    baselines: list[dict[str, Any]] = field(default_factory=list)
    metrics: list[dict[str, str]] = field(default_factory=list)
    constraints: dict[str, dict[str, Any]] = field(default_factory=dict)
    proposal_guidance: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    # --- derived -------------------------------------------------------

    @property
    def metric_names(self) -> list[str]:
        return [m["name"] for m in self.metrics]

    @property
    def required_method_names(self) -> list[str]:
        methods = self.interface.get("required_methods") or []
        return [m["name"] for m in methods if isinstance(m, dict) and "name" in m]

    @property
    def class_name(self) -> str:
        return self.interface.get("class_name") or ""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


_REQUIRED_TOP_LEVEL = ("name", "interface", "metrics")


# Any: ``metrics`` comes from ``yaml.safe_load``; validators inspect shape.
def _validate_metrics(metrics: Any, source: Path | None) -> list[dict[str, str]]:
    if not isinstance(metrics, list) or not metrics:
        raise DomainSpecError(
            f"metrics must be a non-empty list (source: {source})"
        )
    cleaned: list[dict[str, str]] = []
    for i, m in enumerate(metrics):
        if not isinstance(m, dict):
            raise DomainSpecError(f"metrics[{i}] must be a mapping")
        if "name" not in m or "direction" not in m:
            raise DomainSpecError(
                f"metrics[{i}] missing required keys name/direction (source: {source})"
            )
        if m["direction"] not in ("minimize", "maximize"):
            raise DomainSpecError(
                f"metrics[{i}].direction must be 'minimize' or 'maximize' "
                f"(got {m['direction']!r}, source: {source})"
            )
        cleaned.append({str(k): str(v) if k in ("name", "direction") else v for k, v in m.items()})
    return cleaned


# Any: ``interface`` comes from ``yaml.safe_load``; validators inspect shape.
def _validate_interface(interface: Any, source: Path | None) -> dict[str, Any]:
    if not isinstance(interface, dict):
        raise DomainSpecError(f"interface must be a mapping (source: {source})")
    if "class_name" not in interface:
        raise DomainSpecError(
            f"interface.class_name is required (source: {source})"
        )
    methods = interface.get("required_methods") or []
    if not isinstance(methods, list):
        raise DomainSpecError(
            f"interface.required_methods must be a list (source: {source})"
        )
    for i, m in enumerate(methods):
        if not isinstance(m, dict) or "name" not in m:
            raise DomainSpecError(
                f"interface.required_methods[{i}] must be a mapping with 'name' "
                f"(source: {source})"
            )
    return dict(interface)


# Any: ``constraints`` comes from ``yaml.safe_load``; validators inspect shape.
def _validate_constraints(
    constraints: Any, source: Path | None
) -> dict[str, dict[str, Any]]:
    if constraints is None:
        return {}
    if not isinstance(constraints, dict):
        raise DomainSpecError(f"constraints must be a mapping (source: {source})")
    cleaned: dict[str, dict[str, Any]] = {}
    for key, value in constraints.items():
        if not isinstance(value, dict):
            raise DomainSpecError(
                f"constraints[{key!r}] must be a mapping (source: {source})"
            )
        cleaned[str(key)] = dict(value)
    return cleaned


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_domain_spec(path: str | Path) -> DomainSpec:
    """Load a domain spec from a YAML file or a directory containing one.

    If *path* is a directory, the loader looks for ``domain_spec.yaml``
    inside it. Unknown fields are preserved in the returned object via
    plain attribute access on the underlying dict-valued fields.
    """
    p = Path(path)
    if p.is_dir():
        p = p / "domain_spec.yaml"
    if not p.exists():
        raise DomainSpecError(f"Domain spec file not found: {p}")

    data = read_yaml(p)
    if not isinstance(data, dict):
        raise DomainSpecError(f"Domain spec at {p} must be a YAML mapping")

    missing = [k for k in _REQUIRED_TOP_LEVEL if k not in data]
    if missing:
        raise DomainSpecError(
            f"Domain spec at {p} missing required fields: {missing}"
        )

    spec = DomainSpec(
        name=str(data["name"]),
        interface=_validate_interface(data["interface"], p),
        baselines=list(data.get("baselines") or []),
        metrics=_validate_metrics(data["metrics"], p),
        constraints=_validate_constraints(data.get("constraints"), p),
        proposal_guidance=str(data.get("proposal_guidance") or ""),
        evaluation=dict(data.get("evaluation") or {}),
        source_path=p,
    )
    return spec


def discover_domain_specs(search_dirs: list[Path]) -> dict[str, Path]:
    """Discover domain specs under the given directories.

    Returns a mapping of ``name -> path-to-yaml``. Later entries override
    earlier ones (matches the 3-tier plugin precedence: builtin < global <
    local). Directories without a ``domain_spec.yaml`` are skipped.
    """
    discovered: dict[str, Path] = {}
    for root in search_dirs:
        root = Path(root)
        if not root.exists() or not root.is_dir():
            continue
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            candidate = sub / "domain_spec.yaml"
            if candidate.exists():
                try:
                    data = read_yaml(candidate)
                    if isinstance(data, dict) and "name" in data:
                        discovered[str(data["name"])] = candidate
                except Exception:
                    # Silently skip malformed specs; they'll surface on load.
                    continue
    return discovered
