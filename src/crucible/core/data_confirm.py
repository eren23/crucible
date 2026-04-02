"""Agent confirmation flow for data operations."""
from __future__ import annotations

from crucible.core.data_sources import DataStatus


def format_data_status_message(name: str, status_result) -> str:
    """Format a human-readable status message."""
    lines = [
        f"Data: {name}",
        f"Status: {status_result.status.value}",
    ]

    if status_result.last_prepared:
        lines.append(f"Last prepared: {status_result.last_prepared.strftime('%Y-%m-%d %H:%M')}")

    if status_result.shard_count:
        sc = status_result.shard_count
        train_count = sc.get("train", sc.get("total", "?"))
        val_count = sc.get("val", "?")
        lines.append(f"Shards: {train_count} train, {val_count} val")

    if status_result.issues:
        lines.append("Issues:")
        for issue in status_result.issues:
            lines.append(f"  - {issue}")

    lines.append("")
    lines.append("Choose:")
    lines.append("[1] Use existing — fast, may be stale")
    lines.append("[2] Reprocess — ensure freshness")
    lines.append("[3] Find new source — search alternatives")
    lines.append("[4] No data / skip — for testing only")

    return "\n".join(lines)


def should_reprocess(choice: str | int, status: DataStatus) -> tuple[bool | None, str]:
    """Parse user's confirmation choice.

    Args:
        choice: User's numeric choice (1-4).
        status: Reserved for future logic where status affects recommendation.

    Returns:
        Tuple of (should_reprocess: bool | None, action: str)
        should_reprocess is None when action is "find_new" or "skip"
    """
    if isinstance(choice, int):
        choice = str(choice)

    choice_map = {
        "1": (False, "using_existing"),
        "2": (True, "reprocessing"),
        "3": (None, "find_new"),
        "4": (False, "skip"),
    }

    result = choice_map.get(choice)
    if result is None:
        return False, f"Invalid choice: {choice}"
    return result
