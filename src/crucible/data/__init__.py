"""Data pipeline: manifest loading, HuggingFace downloads, and fleet sync."""

from crucible.data.download import DataManager
from crucible.data.manifest import load_manifest, resolve_shard_paths
from crucible.data.sync import probe_data_on_node, sync_data_to_node

__all__ = [
    "DataManager",
    "load_manifest",
    "resolve_shard_paths",
    "probe_data_on_node",
    "sync_data_to_node",
]
