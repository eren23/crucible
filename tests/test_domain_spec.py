"""Tests for the domain spec loader."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from crucible.core.errors import DomainSpecError
from crucible.researcher.domain_spec import (
    DomainSpec,
    discover_domain_specs,
    load_domain_spec,
)


def _write_spec(dir_: Path, data: dict) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    target = dir_ / "domain_spec.yaml"
    target.write_text(yaml.dump(data), encoding="utf-8")
    return target


_MINIMAL = {
    "name": "demo",
    "interface": {
        "class_name": "Harness",
        "required_methods": [{"name": "predict"}],
    },
    "metrics": [{"name": "acc", "direction": "maximize"}],
}


def test_load_spec_from_directory(tmp_path: Path) -> None:
    _write_spec(tmp_path / "demo", _MINIMAL)
    spec = load_domain_spec(tmp_path / "demo")
    assert isinstance(spec, DomainSpec)
    assert spec.name == "demo"
    assert spec.class_name == "Harness"
    assert spec.required_method_names == ["predict"]
    assert spec.metric_names == ["acc"]


def test_load_spec_from_file(tmp_path: Path) -> None:
    path = _write_spec(tmp_path / "demo", _MINIMAL)
    spec = load_domain_spec(path)
    assert spec.name == "demo"


def test_missing_fields_raise(tmp_path: Path) -> None:
    bad = dict(_MINIMAL)
    bad.pop("metrics")
    _write_spec(tmp_path / "bad", bad)
    with pytest.raises(DomainSpecError):
        load_domain_spec(tmp_path / "bad")


def test_invalid_metric_direction(tmp_path: Path) -> None:
    bad = dict(_MINIMAL)
    bad["metrics"] = [{"name": "acc", "direction": "bogus"}]
    _write_spec(tmp_path / "bad2", bad)
    with pytest.raises(DomainSpecError):
        load_domain_spec(tmp_path / "bad2")


def test_interface_missing_class_name(tmp_path: Path) -> None:
    bad = dict(_MINIMAL)
    bad["interface"] = {"required_methods": [{"name": "predict"}]}
    _write_spec(tmp_path / "bad3", bad)
    with pytest.raises(DomainSpecError):
        load_domain_spec(tmp_path / "bad3")


def test_constraints_must_be_mapping(tmp_path: Path) -> None:
    bad = dict(_MINIMAL)
    bad["constraints"] = "not a dict"
    _write_spec(tmp_path / "bad4", bad)
    with pytest.raises(DomainSpecError):
        load_domain_spec(tmp_path / "bad4")


def test_discover_domain_specs(tmp_path: Path) -> None:
    _write_spec(tmp_path / "one", {**_MINIMAL, "name": "one"})
    _write_spec(tmp_path / "two", {**_MINIMAL, "name": "two"})
    (tmp_path / "ignored").mkdir()  # no domain_spec.yaml inside
    discovered = discover_domain_specs([tmp_path])
    assert set(discovered.keys()) == {"one", "two"}


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DomainSpecError):
        load_domain_spec(tmp_path / "nope")
