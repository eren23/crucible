"""World model example: JEPA on bouncing balls.

Importing this module registers the ``jepa_wm`` model family and
``bouncing_balls`` data adapter with Crucible's registries.
"""
from examples.world_model.model import register as _register_model
from examples.world_model.data_adapter import register as _register_adapter

_register_model()
_register_adapter()
