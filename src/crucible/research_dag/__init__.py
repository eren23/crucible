"""Research DAG bridge between Crucible experiments and Spider Chat canvas.

Provides bidirectional sync: Crucible experiment nodes <-> Spider Chat canvas nodes.
Spider Chat is OPTIONAL — the bridge works in local-only mode without it.
"""
from crucible.research_dag.bridge import ResearchDAGBridge

__all__ = ["ResearchDAGBridge"]
