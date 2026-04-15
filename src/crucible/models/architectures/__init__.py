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
except Exception as _exc_builtin:
    from crucible.core.log import log_warn as _lw1
    _lw1(f"Builtin YAML spec loading failed (non-fatal): {_exc_builtin}")

# --- Global hub architectures (source="global") ---
# Loads .py plugins and .yaml specs from hub architecture directories.
# The legacy path (``~/.crucible-hub/architectures/{plugins,specs}/``) is the
# one the mirror-fallback below is keyed off — only that path gates
# ``_hub_loaded``.  The tap-install path (``~/.crucible-hub/plugins/architectures/``,
# written by ``crucible tap install <name> --type architectures``) is loaded
# independently so it additively exposes tap plugins without bypassing the
# mirror loader that tests and fleet bootstrap depend on.
_hub_loaded = False
try:
    from crucible.core.config import load_config as _load_cfg_global
    from crucible.core.hub import HubStore
    _cfg_global = _load_cfg_global()
    _hub_dir = HubStore.discover(config_hub_dir=getattr(_cfg_global, "hub_dir", ""))
    if _hub_dir is not None:
        # Legacy hub paths — gate the mirror fallback on whether the loader
        # actually loaded something, not just on directory existence.  Empty
        # hub directories (created by ``crucible hub init``) would otherwise
        # trip _hub_loaded=True without contributing any architectures, and
        # then skip the mirror loader that fleet bootstrap and the integration
        # tests depend on.
        for _source_dir in (
            _hub_dir / "architectures" / "plugins",
            _hub_dir / "architectures" / "specs",
        ):
            if _source_dir.is_dir():
                _loaded = _reg.load_global_architectures(_source_dir, source="global")
                if _loaded:
                    _hub_loaded = True
        # Tap-install path — additive, does NOT affect _hub_loaded so the
        # mirror fallback below still runs when the legacy paths are empty.
        _tap_install_arch_dir = _hub_dir / "plugins" / "architectures"
        if _tap_install_arch_dir.is_dir():
            _reg.load_global_architectures(_tap_install_arch_dir, source="global")
except Exception as _exc_hub:
    from crucible.core.log import log_warn as _lw2
    _lw2(f"Global hub architecture loading failed (non-fatal): {_exc_hub}")

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
except Exception as _exc_local:
    from crucible.core.log import log_warn as _lw3
    _lw3(f"Local architecture loading failed (non-fatal): {_exc_local}")

# Dev fallback: source-tree user_architectures/ (for examples/development only)
try:
    import crucible.models.user_architectures  # noqa: F401
except ImportError:
    pass
