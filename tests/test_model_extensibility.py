"""Tests for model extensibility MCP tools and user_architectures auto-import."""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# We test the MCP tool functions directly, not via MCP protocol.
# Some tools need torch; skip if not installed.
torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_architectures_loaded():
    """Make sure the model registry is populated before each test.

    There is a circular import: ``models/__init__`` -> ``base`` ->
    ``components.__init__`` -> ``lora`` -> ``base`` (partially initialised).
    On the very first import the circular path raises ``ImportError``.
    Subsequent imports succeed because CPython caches the partially-loaded
    modules.  We tolerate the first failure so the test suite works in a
    clean interpreter.
    """
    try:
        import crucible.models.architectures  # noqa: F401
    except ImportError:
        # First import may fail due to circular import; retry once.
        import crucible.models.architectures  # noqa: F401


# ---------------------------------------------------------------------------
# Discovery tools
# ---------------------------------------------------------------------------


def test_model_list_families():
    from crucible.mcp.tools import model_list_families

    result = model_list_families({})
    assert "families" in result
    families = result["families"]
    for expected in ("baseline", "looped", "convloop", "prefix_memory"):
        assert expected in families, f"Missing family: {expected}"
    assert len(families) >= 4


def test_model_list_activations():
    from crucible.mcp.tools import model_list_activations

    result = model_list_activations({})
    assert "activations" in result
    activations = result["activations"]
    # There should be at least 9 built-in activations
    assert len(activations) >= 9
    for expected in ("relu_sq", "gelu_sq", "mish_sq", "x_absx"):
        assert expected in activations, f"Missing activation: {expected}"
    # Should be sorted
    assert activations == sorted(activations)


def test_model_list_components():
    from crucible.mcp.tools import model_list_components

    result = model_list_components({})
    assert "components" in result
    components = result["components"]
    assert "Block" in components
    assert "MLP" in components
    assert "RMSNorm" in components


def test_model_get_config_schema_baseline():
    from crucible.mcp.tools import model_get_config_schema

    result = model_get_config_schema({"family": "baseline"})
    assert "family" in result
    assert result["family"] == "baseline"
    assert "parameters" in result
    params = result["parameters"]
    assert "MODEL_DIM" in params
    assert params["MODEL_DIM"]["type"] == "int"
    assert params["MODEL_DIM"]["default"] == 512
    assert "ACTIVATION" in params


def test_model_get_config_schema_unknown():
    from crucible.mcp.tools import model_get_config_schema

    result = model_get_config_schema({"family": "nonexistent_family"})
    assert "error" in result
    assert "nonexistent_family" in result["error"]


# ---------------------------------------------------------------------------
# Validation tool
# ---------------------------------------------------------------------------


def test_model_validate_config_valid():
    from crucible.mcp.tools import model_validate_config

    result = model_validate_config({
        "config": {"MODEL_FAMILY": "baseline", "ACTIVATION": "relu_sq"}
    })
    assert result["valid"] is True
    assert result["errors"] == []


def test_model_validate_config_invalid_family():
    from crucible.mcp.tools import model_validate_config

    result = model_validate_config({
        "config": {"MODEL_FAMILY": "bogus_family", "ACTIVATION": "relu_sq"}
    })
    assert result["valid"] is False
    assert any("bogus_family" in e for e in result["errors"])


def test_model_validate_config_invalid_activation():
    from crucible.mcp.tools import model_validate_config

    result = model_validate_config({
        "config": {"MODEL_FAMILY": "baseline", "ACTIVATION": "bogus_act"}
    })
    assert result["valid"] is False
    assert any("bogus_act" in e for e in result["errors"])


def test_model_validate_config_both_invalid():
    from crucible.mcp.tools import model_validate_config

    result = model_validate_config({
        "config": {"MODEL_FAMILY": "nope", "ACTIVATION": "nada"}
    })
    assert result["valid"] is False
    assert len(result["errors"]) == 2


def test_model_validate_config_defaults():
    """With empty config, defaults (baseline + relu_sq) should be valid."""
    from crucible.mcp.tools import model_validate_config

    result = model_validate_config({"config": {}})
    assert result["valid"] is True


# ---------------------------------------------------------------------------
# Activation extensibility
# ---------------------------------------------------------------------------


def test_model_add_activation():
    from crucible.mcp.tools import model_add_activation
    from crucible.models.components.mlp import ACTIVATIONS

    result = model_add_activation({
        "name": "_test_swish",
        "code": "torch.sigmoid(x) * x",
    })
    assert result["status"] == "registered"
    assert "_test_swish" in result["activations"]
    assert "_test_swish" in ACTIVATIONS

    # Verify it actually works
    fn = ACTIVATIONS["_test_swish"]
    t = torch.randn(4, 8)
    out = fn(t)
    assert out.shape == t.shape

    # Cleanup
    del ACTIVATIONS["_test_swish"]


def test_model_add_activation_invalid():
    from crucible.mcp.tools import model_add_activation

    result = model_add_activation({
        "name": "_test_bad",
        "code": "this_is_not_valid(x)",
    })
    assert "error" in result
    assert "Invalid activation code" in result["error"]


# ---------------------------------------------------------------------------
# Template generation
# ---------------------------------------------------------------------------


def test_model_generate_template_valid_python():
    from crucible.mcp.tools import model_generate_template

    result = model_generate_template({"name": "test_arch"})
    assert "template" in result
    template = result["template"]

    # Template should be valid Python (parseable)
    tree = ast.parse(template)
    assert tree is not None

    # Should contain register_model call
    assert 'register_model("test_arch"' in template
    # Should contain a class definition
    assert "class TestArchGPT" in template


def test_model_generate_template_usage():
    from crucible.mcp.tools import model_generate_template

    result = model_generate_template({"name": "my_custom"})
    assert "usage" in result
    assert "model_add_architecture" in result["usage"]
    assert "my_custom" in result["usage"]


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------


def test_schema_registry():
    from crucible.models.registry import get_family_schema, register_schema

    # Register a test schema
    test_schema = {
        "PARAM_A": {"type": "int", "default": 42, "description": "Test param A"},
    }
    register_schema("_test_schema_family", test_schema)

    result = get_family_schema("_test_schema_family")
    assert result == test_schema

    # Unknown family returns generic stub
    result = get_family_schema("_nonexistent_schema_family")
    assert "note" in result

    # Cleanup
    from crucible.models.registry import _FAMILY_SCHEMAS
    del _FAMILY_SCHEMAS["_test_schema_family"]


# ---------------------------------------------------------------------------
# Registry source tracking & precedence
# ---------------------------------------------------------------------------


def test_registry_source_tracking():
    """Verify that register_model populates _REGISTRY_META with source info."""
    from crucible.models.registry import _REGISTRY, _REGISTRY_META, register_model

    # Builtins should have been registered by the autouse fixture
    assert "baseline" in _REGISTRY
    assert "baseline" in _REGISTRY_META
    assert _REGISTRY_META["baseline"]["source"] == "builtin"


def test_register_model_precedence(tmp_path):
    """Local overrides global, global overrides builtin."""
    from crucible.models.registry import (
        _REGISTRY,
        _REGISTRY_META,
        register_model,
    )

    # Register a test family as "global"
    _test_name = "_test_precedence_family"
    global_fn = lambda args: None  # noqa: E731
    local_fn = lambda args: None  # noqa: E731

    register_model(_test_name, global_fn, source="global")
    assert _REGISTRY_META[_test_name]["source"] == "global"
    assert _REGISTRY[_test_name] is global_fn

    # Local should override global
    register_model(_test_name, local_fn, source="local")
    assert _REGISTRY_META[_test_name]["source"] == "local"
    assert _REGISTRY[_test_name] is local_fn

    # Same-or-lower precedence should raise
    with pytest.raises(ValueError, match="already registered"):
        register_model(_test_name, global_fn, source="global")

    # Cleanup
    del _REGISTRY[_test_name]
    del _REGISTRY_META[_test_name]


def test_list_families_detailed():
    """list_families_detailed returns name+source dicts."""
    from crucible.models.registry import list_families_detailed

    detailed = list_families_detailed()
    assert isinstance(detailed, list)
    assert len(detailed) >= 4
    names = {d["name"] for d in detailed}
    assert "baseline" in names
    for entry in detailed:
        assert "name" in entry
        assert "source" in entry
        assert entry["source"] in {"builtin", "global", "local"}


def test_load_global_architectures(tmp_path):
    """load_global_architectures imports .py files and tags them as global."""
    from crucible.models.registry import (
        _REGISTRY,
        _REGISTRY_META,
        load_global_architectures,
    )

    # Write a minimal plugin to a temp dir
    plugin_code = '''
from crucible.models.registry import register_model

def _build_test_global(args):
    return None

register_model("_test_global_arch", _build_test_global)
'''
    (tmp_path / "test_global_arch.py").write_text(plugin_code)

    loaded = load_global_architectures(tmp_path)
    assert "test_global_arch" in loaded
    assert "_test_global_arch" in _REGISTRY
    assert _REGISTRY_META["_test_global_arch"]["source"] == "global"

    # Cleanup
    del _REGISTRY["_test_global_arch"]
    del _REGISTRY_META["_test_global_arch"]


def test_moe_baseline_registered():
    """The moe_baseline plugin should be auto-discovered and registered."""
    from crucible.models.registry import _REGISTRY

    assert "moe_baseline" in _REGISTRY, "moe_baseline plugin not registered"
