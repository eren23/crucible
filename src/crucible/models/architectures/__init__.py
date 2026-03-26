from __future__ import annotations

from pathlib import Path as _Path

import crucible.models.registry as _reg

# --- Builtins (source="builtin") ---
_reg._CURRENT_REGISTER_SOURCE = "builtin"
import crucible.models.architectures.baseline  # noqa: F401
import crucible.models.architectures.looped  # noqa: F401
import crucible.models.architectures.convloop  # noqa: F401
import crucible.models.architectures.prefix_memory  # noqa: F401

# --- Builtin YAML specs (source="builtin", opt-in via config) ---
# Only load builtin YAML specs when explicitly enabled — Python builtins
# take precedence by default.  YAML specs live in src/crucible/models/specs/.
try:
    from crucible.core.config import load_config as _load_cfg_builtin
    _cfg_builtin = _load_cfg_builtin()
    if getattr(_cfg_builtin, "compose_builtin_specs", False):
        _specs_dir = _Path(__file__).resolve().parent.parent / "specs"
        if _specs_dir.is_dir():
            _reg.load_global_architectures(_specs_dir, source="builtin")
except Exception:
    pass

# --- Global hub architectures (source="global") ---
# Loads both .py plugins and .yaml specs from the hub architectures directory.
_hub_loaded = False
try:
    from crucible.core.config import load_config as _load_cfg_global
    from crucible.core.hub import HubStore
    _cfg_global = _load_cfg_global()
    _hub_dir = HubStore.discover(config_hub_dir=getattr(_cfg_global, "hub_dir", ""))
    if _hub_dir is not None:
        for _source_dir in (
            _hub_dir / "architectures" / "plugins",
            _hub_dir / "architectures" / "specs",
        ):
            if _source_dir.is_dir():
                _reg.load_global_architectures(_source_dir, source="global")
                _hub_loaded = True
except Exception:
    pass

# --- Project-local architectures from .crucible/architectures/ (source="local") ---
# Loads both .py plugins and .yaml specs from the project architectures directory.
_reg._CURRENT_REGISTER_SOURCE = "local"
try:
    from crucible.core.config import load_config as _load_cfg
    _cfg = _load_cfg()
    _mirrored_global_arch_dir = _cfg.project_root / _cfg.store_dir / "architectures" / "_hub"
    if not _hub_loaded and _mirrored_global_arch_dir.is_dir():
        _reg.load_global_architectures(_mirrored_global_arch_dir, source="global")
    _local_arch_dir = _cfg.project_root / _cfg.store_dir / "architectures"
    if _local_arch_dir.is_dir():
        _reg.load_global_architectures(_local_arch_dir, source="local")
except Exception:
    pass

# Dev fallback: source-tree user_architectures/ (for examples/development only)
try:
    import crucible.models.user_architectures  # noqa: F401
except ImportError:
    pass
