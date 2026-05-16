"""Code-level mutation — interface stub (Phase 3.6).

Per the AI-native-discovery-engines plan, this is the *interface*
stub for code-mutation tree expansion. A real implementation
(AST/diff generation + safety filters + isolated-process execution +
scoring) is 4-6 weeks alone — the same surface Sakana AI Scientist
v2 spent the bulk of its engineering on — so the plan explicitly
defers the MVP to Phase 5+ and ships only this stub in Phase 3.

The stub gives downstream callers (tree expansion policies, the
autonomous loop, future Codex-MCP integration) a stable interface
to write against now, so when the real implementation lands the
plumbing on the Crucible side doesn't shift. The design doc that
this stub references lives at ``docs/code-mutation-design.md``.

What lives here:
- ``MutationProposal`` dataclass — orchestrator-supplied diff.
- ``MutationResult`` dataclass — execution outcome shape.
- ``CodeMutationPolicy`` ABC — the contract a real implementation
  must satisfy.
- ``StubCodeMutationPolicy`` — a no-op that raises a clear
  "not yet implemented" error when ``apply()`` is called. Lets
  callers register the policy + see it in tree-expand listings
  without crashing imports.

What does NOT live here yet (Phase 5+):
- Diff parsing / validation (the ``unidiff`` integration)
- Sandboxed subprocess execution
- AST-aware safety filters (e.g., reject diffs that touch
  ``security`` / ``credentials`` / ``__class__`` patterns)
- Score harvesting from the mutated run
- Rollback / version-store integration
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from crucible.core.errors import CrucibleError


class CodeMutationError(CrucibleError):
    """Code-mutation policy failed (parse, apply, exec, score)."""


class CodeMutationNotImplemented(CodeMutationError):
    """The stub policy was invoked. Wire a real policy or wait for Phase 5+."""


@dataclass
class MutationProposal:
    """One orchestrator-supplied mutation proposal.

    Attributes
    ----------
    name:
        Short identifier (becomes the experiment name).
    target_file:
        Path relative to project root that the diff applies to
        (e.g., ``"train.py"`` or ``"src/model.py"``).
    diff:
        Unified-diff text. The implementation parses with
        ``unidiff`` (or equivalent) in Phase 5+.
    hypothesis:
        Free-text what the mutation tests.
    rationale:
        Free-text why the orchestrator picked this mutation.
    parent_node_id:
        Optional ID of the tree node this mutation expands from.
    """

    name: str
    target_file: str
    diff: str
    hypothesis: str = ""
    rationale: str = ""
    parent_node_id: str | None = None


@dataclass
class MutationResult:
    """Outcome of applying + running a mutation."""

    proposal_name: str
    success: bool
    score: float | None = None
    error: str | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)


class CodeMutationPolicy(ABC):
    """Contract for code-mutation tree-expansion policies.

    Concrete implementations land in Phase 5+. The autonomous-loop
    and tree-expand subsystems treat any registered policy as a
    pluggable "given a mutation proposal, return an experiment-shaped
    result" surface — same shape as the existing tree expansion
    policies (UCB1, GRPO, agent-directed).
    """

    @abstractmethod
    def validate(self, proposal: MutationProposal) -> list[str]:
        """Pre-flight: list problems with the proposal, or [] if OK.

        Cheap check before consuming compute. Phase 5+ implementations
        will check the diff parses, the target file exists, and the
        AST-level safety filters don't reject.
        """
        ...

    @abstractmethod
    def apply(self, proposal: MutationProposal) -> MutationResult:
        """Apply the mutation + run + score. Returns a MutationResult.

        Implementations must NOT raise on training failures — return
        ``success=False`` with the error captured. Raising is reserved
        for programmer-error conditions (e.g., a wholly malformed
        proposal that the validator should have caught).
        """
        ...


class StubCodeMutationPolicy(CodeMutationPolicy):
    """Default policy that raises ``CodeMutationNotImplemented`` on apply.

    Registered so the tree-expand registry has *some* policy
    pluggable under the ``code_mutation`` name. The first time a
    real policy lands (Phase 5+) it shadows this stub.
    """

    def validate(self, proposal: MutationProposal) -> list[str]:
        return ["code_mutation policy not implemented yet — see docs/code-mutation-design.md"]

    def apply(self, proposal: MutationProposal) -> MutationResult:
        raise CodeMutationNotImplemented(
            "StubCodeMutationPolicy.apply: code-mutation execution is "
            "Phase 5+. See docs/code-mutation-design.md for the design "
            "doc and tracking issue. To wire a real implementation, "
            "subclass CodeMutationPolicy and register it via "
            "register_code_mutation_policy()."
        )


# ---------------------------------------------------------------------------
# Tiny registry (parallel to data_sources / evaluators)
# ---------------------------------------------------------------------------


_POLICY_REGISTRY: dict[str, type[CodeMutationPolicy]] = {}


def register_code_mutation_policy(
    name: str, policy_cls: type[CodeMutationPolicy]
) -> None:
    """Register a CodeMutationPolicy subclass under ``name``.

    Last writer wins, matching the project-wide plugin precedence
    (local > global > builtin). For Phase 3.6 only the builtin stub
    is registered; user policies plug in via this function.
    """
    _POLICY_REGISTRY[name] = policy_cls


def list_code_mutation_policies() -> list[str]:
    """Return registered policy names."""
    return sorted(_POLICY_REGISTRY.keys())


def get_code_mutation_policy(name: str = "stub") -> CodeMutationPolicy:
    """Look up + instantiate a policy by name (default: the stub)."""
    cls = _POLICY_REGISTRY.get(name)
    if cls is None:
        raise CodeMutationError(
            f"No code_mutation policy registered as {name!r}. "
            f"Registered: {list_code_mutation_policies()}"
        )
    return cls()


# Register the stub at import time so the registry is never empty.
register_code_mutation_policy("stub", StubCodeMutationPolicy)
