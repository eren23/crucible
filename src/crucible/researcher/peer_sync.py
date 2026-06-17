"""Peer coordination via HuggingFace Discussions (Phase 4.3).

Lets two or more Crucible agents racing on the same problem ("challenge_id")
share their current top finding through a shared HF Discussion thread.
Each agent:
1. Computes its top finding (best leaderboard row → finding payload).
2. Pulls peers' previously-posted top findings from the discussion thread.
3. Posts its own top finding as a new comment.

The orchestrator typically calls this once per loop iteration — after
collecting results, before forming the next hypothesis — so the next
hypothesis can reference what peers have found.

Built on the existing Tier-15 :mod:`hf_discussions` infrastructure.
All HF I/O is best-effort: a network outage degrades to "no peer
signal this iteration" rather than blocking the loop.

Discussion title convention: ``"crucible-peer-sync:<challenge_id>"``.
First agent to call ``research_peer_sync`` for a challenge_id opens
the thread; subsequent calls find + reuse it.
"""
from __future__ import annotations

from typing import Any

from crucible.core.errors import HfError
from crucible.core.log import log_warn
from crucible.core.redact import redact_secrets
from crucible.researcher.hf_discussions import list_discussions, post_discussion

_DISCUSSION_TITLE_PREFIX = "crucible-peer-sync:"


def title_for_challenge(challenge_id: str) -> str:
    return f"{_DISCUSSION_TITLE_PREFIX}{challenge_id}"


def find_existing_thread(
    repo_id: str,
    *,
    challenge_id: str,
    repo_type: str = "dataset",
    token: str | None = None,
) -> dict[str, Any] | None:
    """Look up an existing peer-sync discussion on the named repo.

    Returns the discussion record or None if no open thread exists.
    """
    title = title_for_challenge(challenge_id)
    threads = list_discussions(
        repo_id, repo_type=repo_type, status="open", limit=100, token=token,
    )
    for d in threads:
        if d.get("title") == title:
            return d
    return None


def render_finding_post(
    *,
    agent_id: str,
    challenge_id: str,
    top_finding: dict[str, Any],
    leaderboard_row: dict[str, Any] | None = None,
    iso_now: str,
) -> str:
    """Render a peer-share body in the convention the next agent expects.

    Format (markdown body of the discussion comment):

        ## crucible-peer-finding (v1)
        agent_id: <agent_id>
        challenge_id: <challenge_id>
        ts: <iso_now>
        leaderboard_metric: <primary_metric>=<value>

        ### Top finding

        <finding markdown>
    """
    lines: list[str] = []
    lines.append("## crucible-peer-finding (v1)")
    lines.append(f"agent_id: {agent_id}")
    lines.append(f"challenge_id: {challenge_id}")
    lines.append(f"ts: {iso_now}")
    if leaderboard_row:
        metric = (
            leaderboard_row.get("primary_metric")
            or leaderboard_row.get("metric")
            or "score"
        )
        value = leaderboard_row.get("primary_value") or leaderboard_row.get(metric)
        if value is not None:
            lines.append(f"leaderboard_metric: {metric}={value}")
        name = leaderboard_row.get("name") or ""
        if name:
            lines.append(f"leaderboard_run: {name}")
    lines.append("")
    lines.append("### Top finding")
    lines.append("")
    title = (top_finding.get("title") or "").strip()
    if title:
        lines.append(f"**{title}**")
        lines.append("")
    body = (top_finding.get("body") or "").strip()
    if body:
        lines.append(body)
    lines.append("")
    confidence_raw = top_finding.get("confidence")
    category = top_finding.get("category", "observation")
    # Coerce confidence to float — LLMs sometimes return numeric values
    # as strings (e.g., "0.85"), which would crash the :.2f format spec.
    # Fall back to the raw string if coercion fails so the value still
    # shows up in the post.
    if confidence_raw is not None:
        try:
            conf_val = float(confidence_raw)
            lines.append(f"_category={category}, confidence={conf_val:.2f}_")
        except (TypeError, ValueError):
            lines.append(f"_category={category}, confidence={confidence_raw}_")
    # Apply secret redaction so a finding accidentally containing an env
    # dump or stack trace doesn't leak credentials to a public repo.
    return redact_secrets("\n".join(lines))


def sync_peer_finding(
    *,
    repo_id: str,
    challenge_id: str,
    agent_id: str,
    top_finding: dict[str, Any],
    leaderboard_row: dict[str, Any] | None = None,
    iso_now: str,
    repo_type: str = "dataset",
    token: str | None = None,
) -> dict[str, Any]:
    """Post our top finding to the shared peer-sync thread + pull peers'.

    Returns ``{thread_num, thread_url, posted_url, peer_count, peers}``
    where ``peers`` is the list of previously-posted findings from
    other agents (anyone whose author/agent_id differs from ours).

    Best-effort. On failure to post, returns the read-side result so
    the orchestrator can still see peer findings.
    """
    existing = find_existing_thread(
        repo_id, challenge_id=challenge_id, repo_type=repo_type, token=token,
    )

    body = render_finding_post(
        agent_id=agent_id,
        challenge_id=challenge_id,
        top_finding=top_finding,
        leaderboard_row=leaderboard_row,
        iso_now=iso_now,
    )

    posted_url = ""
    thread_num = 0
    thread_url = ""

    try:
        if existing is None:
            opened = post_discussion(
                repo_id,
                title=title_for_challenge(challenge_id),
                description=body,
                repo_type=repo_type,
                token=token,
            )
            thread_num = opened["num"]
            thread_url = opened["url"]
            posted_url = opened["url"]
        else:
            thread_num = existing["num"]
            thread_url = existing["url"]
            # Post a follow-up comment to the existing thread.
            posted_url = _post_comment_or_new_discussion(
                repo_id=repo_id,
                num=thread_num,
                body=body,
                repo_type=repo_type,
                token=token,
            )
    except HfError as exc:
        log_warn(f"peer_sync: write side failed for {repo_id!r}: {exc}")

    peers = _fetch_peer_findings(
        repo_id=repo_id,
        challenge_id=challenge_id,
        thread_num=thread_num,
        my_agent_id=agent_id,
        repo_type=repo_type,
        token=token,
    )

    return {
        "challenge_id": challenge_id,
        "agent_id": agent_id,
        "thread_num": thread_num,
        "thread_url": thread_url,
        "posted_url": posted_url,
        "peer_count": len(peers),
        "peers": peers,
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _post_comment_or_new_discussion(
    *,
    repo_id: str,
    num: int,
    body: str,
    repo_type: str,
    token: str | None,
) -> str:
    """Append a comment to an existing discussion thread.

    If the installed ``huggingface_hub`` version lacks
    ``HfApi.comment_discussion``, raises :class:`HfError` rather than
    silently opening a new top-level discussion — the previous
    fallback violated the peer-read contract (peers reading the
    original thread never saw the orphan reply).
    """
    try:
        from huggingface_hub import HfApi  # type: ignore

        from crucible.core.hf_writer import resolve_token

        api = HfApi(token=resolve_token(token))
        comment = getattr(api, "comment_discussion", None)
        if not callable(comment):
            raise HfError(
                "huggingface_hub.HfApi.comment_discussion is unavailable "
                "in this SDK version; cannot append to discussion thread "
                f"#{num}. Upgrade huggingface_hub to >=0.13 or post the "
                "finding manually."
            )
        d = comment(
            repo_id=repo_id, repo_type=repo_type,
            discussion_num=num, comment=body,
        )
        url = getattr(d, "url", None)
        return str(url or "")
    except HfError:
        raise
    except Exception as exc:
        raise HfError(f"comment_discussion failed: {exc}") from exc


def _fetch_peer_findings(
    *,
    repo_id: str,
    challenge_id: str,
    thread_num: int,
    my_agent_id: str,
    repo_type: str,
    token: str | None,
) -> list[dict[str, Any]]:
    """Read the named thread and parse out peer agents' top-finding posts.

    Returns the list of {agent_id, ts, body, leaderboard_metric, url}
    dicts, one per peer comment. Comments without the v1 header are
    skipped silently (treat as off-protocol noise).
    """
    if thread_num <= 0:
        return []
    try:
        from huggingface_hub import HfApi  # type: ignore

        from crucible.core.hf_writer import resolve_token

        api = HfApi(token=resolve_token(token))
        get = getattr(api, "get_discussion_details", None)
        if get is None:
            return []
        details = get(repo_id=repo_id, repo_type=repo_type, discussion_num=thread_num)
    except Exception as exc:
        log_warn(f"peer_sync: read details failed for thread {thread_num}: {exc}")
        return []

    events = getattr(details, "events", None) or []
    peers: list[dict[str, Any]] = []
    for ev in events:
        # Different SDK versions name the event types differently; we
        # only care about comments with a content/text payload.
        content = (
            getattr(ev, "content", None)
            or getattr(ev, "text", None)
            or getattr(ev, "comment", None)
            or ""
        )
        if not isinstance(content, str) or "crucible-peer-finding" not in content:
            continue
        peer = _parse_peer_post(content)
        if not peer or peer.get("agent_id") == my_agent_id:
            continue
        peers.append(peer)
    return peers


def _parse_peer_post(body: str) -> dict[str, Any] | None:
    """Pull the v1 header fields out of a discussion comment body.

    Read-side redaction (H.1.2): peer agents post via
    :func:`render_finding_post`, which redacts before write. But a peer
    running an older Crucible (no redaction) or a different system
    entirely could post an unredacted env dump. Apply
    :func:`redact_secrets` to the body we return so our MCP response
    doesn't forward another agent's leaked credentials verbatim.
    """
    if "crucible-peer-finding" not in body:
        return None
    out: dict[str, Any] = {"body": redact_secrets(body)}
    for line in body.splitlines():
        if line.startswith("agent_id:"):
            out["agent_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("challenge_id:"):
            out["challenge_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("ts:"):
            out["ts"] = line.split(":", 1)[1].strip()
        elif line.startswith("leaderboard_metric:"):
            out["leaderboard_metric"] = line.split(":", 1)[1].strip()
        elif line.startswith("leaderboard_run:"):
            out["leaderboard_run"] = line.split(":", 1)[1].strip()
    if "agent_id" not in out:
        return None
    return out
