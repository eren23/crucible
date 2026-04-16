"""Tests for candidate validation (syntax, interface, constraints, duplicates)."""
from __future__ import annotations

import pytest

from crucible.researcher.candidate_validation import (
    batch_validate,
    validate_candidate,
)
from crucible.researcher.domain_spec import DomainSpec


@pytest.fixture
def spec() -> DomainSpec:
    return DomainSpec(
        name="demo",
        interface={
            "class_name": "Harness",
            "required_methods": [{"name": "predict"}, {"name": "learn_from_batch"}],
        },
        metrics=[{"name": "acc", "direction": "maximize"}],
        constraints={"TEMPERATURE": {"min": 0.0, "max": 2.0, "type": "float"}},
    )


_GOOD_DIRECT = """
class Harness:
    def predict(self, x): return 0
    def learn_from_batch(self, b): pass
"""

_GOOD_SUBCLASS = """
class My(Harness):
    def predict(self, x): return 0
    def learn_from_batch(self, b): pass
"""

_MISSING_METHOD = """
class My(Harness):
    def predict(self, x): return 0
"""

_NO_CLASS = """
def predict(x): return 0
"""

_BAD_SYNTAX = "def predict((:"


def test_valid_direct_class(spec: DomainSpec) -> None:
    result = validate_candidate(_GOOD_DIRECT, spec)
    assert result.valid
    assert not result.errors


def test_valid_subclass(spec: DomainSpec) -> None:
    result = validate_candidate(_GOOD_SUBCLASS, spec)
    assert result.valid


def test_missing_required_method(spec: DomainSpec) -> None:
    result = validate_candidate(_MISSING_METHOD, spec)
    assert not result.valid
    assert any("learn_from_batch" in e for e in result.errors)


def test_no_matching_class(spec: DomainSpec) -> None:
    result = validate_candidate(_NO_CLASS, spec)
    assert not result.valid
    assert any("Harness" in e for e in result.errors)


def test_syntax_error(spec: DomainSpec) -> None:
    result = validate_candidate(_BAD_SYNTAX, spec)
    assert not result.valid
    assert any("SyntaxError" in e for e in result.errors)


def test_constraint_bounds(spec: DomainSpec) -> None:
    ok = validate_candidate(_GOOD_DIRECT, spec, config={"TEMPERATURE": "1.2"})
    assert ok.valid
    bad = validate_candidate(_GOOD_DIRECT, spec, config={"TEMPERATURE": "3.5"})
    assert not bad.valid
    assert any("above max" in e for e in bad.errors)


def test_constraint_enum_rejection() -> None:
    s = DomainSpec(
        name="demo",
        interface={"class_name": "Harness", "required_methods": []},
        metrics=[{"name": "acc", "direction": "maximize"}],
        constraints={"MODE": {"enum": ["fast", "slow"]}},
    )
    ok = validate_candidate("class Harness: pass", s, config={"MODE": "fast"})
    assert ok.valid
    bad = validate_candidate("class Harness: pass", s, config={"MODE": "medium"})
    assert not bad.valid


def test_batch_validate_flags_duplicates(spec: DomainSpec) -> None:
    batch = [{"code": _GOOD_DIRECT}, {"code": _GOOD_DIRECT}]
    out = batch_validate(batch, spec)
    assert all(entry["valid"] for entry in out)
    warnings = out[1]["validation"]["warnings"]
    assert warnings and "Duplicate" in warnings[0]


def test_no_class_name_skips_interface_check() -> None:
    # When spec has no class_name, interface check is skipped; methods-only
    # candidates should still pass.
    s = DomainSpec(
        name="demo",
        interface={"class_name": "", "required_methods": []},
        metrics=[{"name": "acc", "direction": "maximize"}],
    )
    result = validate_candidate("def top_level(): pass", s)
    assert result.valid


def test_constraint_int_coercion() -> None:
    s = DomainSpec(
        name="demo",
        interface={"class_name": "H", "required_methods": []},
        metrics=[{"name": "a", "direction": "maximize"}],
        constraints={"N": {"min": 1, "max": 10, "type": "int"}},
    )
    good = validate_candidate("class H: pass", s, config={"N": "5"})
    assert good.valid
    bad_type = validate_candidate("class H: pass", s, config={"N": "not a number"})
    assert not bad_type.valid
    assert any("expected int" in e for e in bad_type.errors)
    below = validate_candidate("class H: pass", s, config={"N": "0"})
    assert not below.valid


def test_constraint_float_coercion_error() -> None:
    s = DomainSpec(
        name="demo",
        interface={"class_name": "H", "required_methods": []},
        metrics=[{"name": "a", "direction": "maximize"}],
        constraints={"F": {"min": 0.0, "max": 1.0, "type": "float"}},
    )
    bad = validate_candidate("class H: pass", s, config={"F": "bogus"})
    assert not bad.valid
    assert any("expected float" in e for e in bad.errors)


def test_qualified_base_class_reference() -> None:
    # Subclass written as `pkg.module.Harness` — attribute form.
    s = DomainSpec(
        name="demo",
        interface={
            "class_name": "Harness",
            "required_methods": [{"name": "predict"}],
        },
        metrics=[{"name": "a", "direction": "maximize"}],
    )
    code = """
class My(pkg.submod.Harness):
    def predict(self, x): return 0
"""
    result = validate_candidate(code, s)
    assert result.valid


def test_require_any_valid_raises_when_all_invalid(spec: DomainSpec) -> None:
    from crucible.core.errors import CandidateValidationError

    batch = [{"code": "def foo(): pass"}, {"code": "import x"}]
    with pytest.raises(CandidateValidationError):
        batch_validate(batch, spec, require_any_valid=True)


def test_require_any_valid_passes_with_one_valid(spec: DomainSpec) -> None:
    batch = [
        {"code": _MISSING_METHOD},
        {"code": _GOOD_DIRECT},
    ]
    out = batch_validate(batch, spec, require_any_valid=True)
    assert any(c["valid"] for c in out)
