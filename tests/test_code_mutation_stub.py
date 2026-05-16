"""Tests for the code_mutation interface stub — Phase 3.6.

The stub itself doesn't run code; the tests pin the contract so when
the real Phase 5+ implementation lands, the call sites don't shift.
"""
from __future__ import annotations

import pytest

from crucible.researcher.code_mutation import (
    CodeMutationError,
    CodeMutationNotImplemented,
    CodeMutationPolicy,
    MutationProposal,
    MutationResult,
    StubCodeMutationPolicy,
    get_code_mutation_policy,
    list_code_mutation_policies,
    register_code_mutation_policy,
)


class TestDataclassShapes:
    def test_proposal_required_fields(self):
        p = MutationProposal(
            name="add_dropout",
            target_file="src/model.py",
            diff="--- a/src/model.py\n+++ b/src/model.py\n@@ ...",
        )
        assert p.name == "add_dropout"
        assert p.target_file == "src/model.py"
        assert p.hypothesis == ""
        assert p.parent_node_id is None

    def test_result_default_failure_shape(self):
        r = MutationResult(proposal_name="x", success=False, error="not implemented")
        assert r.success is False
        assert r.score is None
        assert r.artifacts == {}


class TestRegistry:
    def test_stub_is_default_registered(self):
        assert "stub" in list_code_mutation_policies()

    def test_get_returns_stub_instance(self):
        policy = get_code_mutation_policy("stub")
        assert isinstance(policy, StubCodeMutationPolicy)

    def test_unknown_policy_raises_typed_error(self):
        with pytest.raises(CodeMutationError, match="No code_mutation policy"):
            get_code_mutation_policy("does_not_exist")

    def test_register_then_get(self):
        class _Test(CodeMutationPolicy):
            def validate(self, proposal):
                return []
            def apply(self, proposal):
                return MutationResult(proposal_name=proposal.name, success=True, score=0.5)
        try:
            register_code_mutation_policy("_test_policy", _Test)
            assert "_test_policy" in list_code_mutation_policies()
            policy = get_code_mutation_policy("_test_policy")
            assert isinstance(policy, _Test)
            result = policy.apply(MutationProposal(
                name="x", target_file="y", diff=""
            ))
            assert result.success is True
            assert result.score == 0.5
        finally:
            from crucible.researcher.code_mutation import _POLICY_REGISTRY
            _POLICY_REGISTRY.pop("_test_policy", None)


class TestStubBehavior:
    def test_validate_returns_not_implemented_hint(self):
        policy = StubCodeMutationPolicy()
        problems = policy.validate(MutationProposal(
            name="x", target_file="y", diff=""
        ))
        assert len(problems) == 1
        assert "not implemented" in problems[0].lower()
        assert "design" in problems[0].lower()

    def test_apply_raises_not_implemented(self):
        policy = StubCodeMutationPolicy()
        with pytest.raises(CodeMutationNotImplemented, match="Phase 5"):
            policy.apply(MutationProposal(
                name="x", target_file="y", diff=""
            ))

    def test_not_implemented_is_a_crucible_error(self):
        """The typed exception inherits from CrucibleError so existing
        catch-CrucibleError blocks in dispatchers handle it cleanly."""
        from crucible.core.errors import CrucibleError
        assert issubclass(CodeMutationNotImplemented, CodeMutationError)
        assert issubclass(CodeMutationError, CrucibleError)


class TestABCEnforcement:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            CodeMutationPolicy()  # type: ignore[abstract]

    def test_subclass_missing_apply_raises(self):
        class _Partial(CodeMutationPolicy):
            def validate(self, proposal):
                return []
            # apply not defined.
        with pytest.raises(TypeError):
            _Partial()  # type: ignore[abstract]
