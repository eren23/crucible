"""User-defined architectures --- auto-imported on package load.

Place .py files here that call register_model() to register new
architecture families. They will be imported automatically.
"""
from __future__ import annotations

import importlib
import pkgutil

for _, name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f".{name}", __name__)
