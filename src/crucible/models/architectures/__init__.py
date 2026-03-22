from __future__ import annotations

# Import all architecture modules so they self-register with the model registry.
import crucible.models.architectures.baseline  # noqa: F401
import crucible.models.architectures.looped  # noqa: F401
import crucible.models.architectures.convloop  # noqa: F401
import crucible.models.architectures.prefix_memory  # noqa: F401

# Auto-load user architectures (if directory exists)
try:
    import crucible.models.user_architectures  # noqa: F401
except ImportError:
    pass
