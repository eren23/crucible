"""Bridge between versioned experiment designs and the execution pipeline.

Converts ExperimentDesign dicts to ExperimentConfig dicts (for run_experiment)
and links execution results back to designs via the version store.
"""
from __future__ import annotations

from typing import Any

from crucible.core.types import ExperimentConfig, ExperimentDesign, VersionMeta


def design_to_experiment_config(
    design: ExperimentDesign, version_meta: VersionMeta
) -> ExperimentConfig:
    """Convert a versioned design to an executable ExperimentConfig.

    Tags include design:{name} and v:{version} for result linkage.
    """
    design_tags = list(design.get("tags", []))
    design_tags.append(f"design:{design['name']}")
    design_tags.append(f"v:{version_meta['version']}")

    return ExperimentConfig(
        name=design["name"],
        config=design.get("config", {}),
        tags=design_tags,
        tier=design.get("base_preset", "proxy"),
        backend=design.get("backend", "torch"),
    )


def link_result_to_design(
    store: Any,  # VersionStore — Any to avoid circular import
    design_name: str,
    run_id: str,
) -> VersionMeta | None:
    """Link an experiment result back to a design by updating linked_run_ids.

    Creates a new version of the design with the run_id added.
    Returns the new VersionMeta or None if the design doesn't exist.
    """
    current = store.get_current("experiment_design", design_name)
    if current is None:
        return None

    meta, content = current
    linked = list(content.get("linked_run_ids", []))
    if run_id not in linked:
        linked.append(run_id)
    content["linked_run_ids"] = linked

    return store.create(
        "experiment_design",
        design_name,
        content,
        summary=f"Linked run {run_id}",
        created_by=meta.get("created_by", "system"),
        tags=meta.get("tags", []),
    )
