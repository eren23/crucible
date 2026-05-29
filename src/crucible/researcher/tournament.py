"""Hypothesis tournament — Elo + pairing + persistence (Thrust B1).

Implements the orchestrator-contract surface for DeepMind Co-Scientist's
self-play / debate / Elo tournament pattern, without any LLM keys in
Crucible. The orchestrator runs the LLM-as-debate-judge step; this
module just supplies the pairing prompt and persists the outcome.

Storage layout under ``.crucible/tournaments/{name}/``:
    state.yaml     — current ratings + metadata snapshot
    events.jsonl   — append-only event log (pair, submit, dedupe)

Pairing policies:
    random        — uniform random pair (with-replacement-safe via
                    not-recently-judged history)
    swiss         — pair each hypothesis with the one closest in
                    score-by-wins; greedy fill, no rematch within
                    ``swiss_no_repeat_within`` rounds
    elo_close     — pair the two hypotheses with the closest Elo
                    that haven't been paired recently
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.errors import CrucibleError
from crucible.core.file_lock import file_lock
from crucible.core.io import append_jsonl, atomic_write_yaml, read_jsonl, read_yaml
from crucible.core.log import utc_now_iso


class TournamentError(CrucibleError):
    """Tournament create / pair / submit failure."""


_DEFAULT_K = 32.0
_DEFAULT_INITIAL_RATING = 1500.0
_VALID_POLICIES = frozenset({"random", "swiss", "elo_close"})


@dataclass
class Hypothesis:
    """Tournament-side hypothesis record.

    Lightweight wrapper around an arbitrary dict — the tournament
    doesn't care what's inside the ``payload``, only that each entry
    has a unique ``id``. The orchestrator's debate-judge prompt
    serialises the payload verbatim.
    """

    id: str
    payload: dict[str, Any]
    rating: float = _DEFAULT_INITIAL_RATING
    wins: int = 0
    losses: int = 0
    draws: int = 0


@dataclass
class PairingResult:
    """Output of ``pair`` — the matchup the orchestrator should judge."""

    pair_id: str
    left: Hypothesis
    right: Hypothesis
    policy: str
    round_number: int


@dataclass
class TournamentRank:
    """One entry in a ``rank()`` response."""

    id: str
    rating: float
    wins: int
    losses: int
    draws: int
    payload: dict[str, Any]


def _expected_score(a: float, b: float) -> float:
    """Standard Elo expected-score formula."""
    return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))


def _update_elo(
    rating_a: float, rating_b: float, score_a: float, k: float = _DEFAULT_K
) -> tuple[float, float]:
    """Returns (new_a, new_b) after one Elo update.

    ``score_a`` is 1.0 for A wins, 0.0 for B wins, 0.5 for draw.
    """
    expected_a = _expected_score(rating_a, rating_b)
    new_a = rating_a + k * (score_a - expected_a)
    new_b = rating_b + k * ((1.0 - score_a) - (1.0 - expected_a))
    return new_a, new_b


class Tournament:
    """Persistent hypothesis tournament.

    Concurrency: every mutating call (``create``, ``pair``, ``submit``)
    takes an exclusive file lock on the tournament directory. Two
    orchestrator instances driving the same tournament can interleave
    cleanly; the slower one will simply see an updated state and may
    re-judge a pair the faster one already submitted (which the Elo
    math handles — repeated identical pairings just shrink the rating
    delta as expectations converge).
    """

    def __init__(self, tournament_dir: Path) -> None:
        self.dir = Path(tournament_dir)
        self.state_path = self.dir / "state.yaml"
        self.events_path = self.dir / "events.jsonl"

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        tournament_dir: Path,
        name: str,
        hypotheses: list[dict[str, Any]],
        *,
        description: str = "",
        initial_rating: float = _DEFAULT_INITIAL_RATING,
        k_factor: float = _DEFAULT_K,
        seed: int | None = None,
    ) -> "Tournament":
        """Create a new tournament with the given hypotheses.

        Each hypothesis dict must have an ``id`` (string). If absent, one
        is generated. The rest of the dict is preserved verbatim as the
        ``payload``.

        Concurrency: holds the tournament directory's file lock for the
        full exists-check → mkdir → write sequence so two processes
        racing on the same directory cannot both pass the existence
        check and stomp on each other's state.yaml (M4 fix).
        """
        tournament = cls(tournament_dir)
        # mkdir first so the .lock file has a home. parents=True +
        # exist_ok=True is safe — the lock-protected existence check
        # below is the actual race guard.
        tournament.dir.mkdir(parents=True, exist_ok=True)
        with file_lock(tournament.dir / ".lock"):
            if tournament.state_path.exists():
                raise TournamentError(
                    f"Tournament already exists at {tournament_dir}; "
                    "use Tournament(dir) to open it"
                )
            now = utc_now_iso()
            entries: list[dict[str, Any]] = []
            seen_ids: set[str] = set()
            for raw in hypotheses:
                hid = str(raw.get("id") or f"h_{uuid.uuid4().hex[:8]}")
                if hid in seen_ids:
                    raise TournamentError(f"duplicate hypothesis id: {hid!r}")
                seen_ids.add(hid)
                entries.append({
                    "id": hid,
                    "payload": {k: v for k, v in raw.items() if k != "id"},
                    "rating": initial_rating,
                    "wins": 0,
                    "losses": 0,
                    "draws": 0,
                })
            state = {
                "name": name,
                "description": description,
                "initial_rating": initial_rating,
                "k_factor": k_factor,
                "seed": seed,
                "created_at": now,
                "updated_at": now,
                "round_number": 0,
                "hypotheses": entries,
                "recent_pairings": [],
                # M5 fix: open pairings tracked here so submit() can
                # validate it's settling a real pair_id, not a fabricated
                # one. Map pair_id → {left, right, round}. Pruned when
                # a submit settles them or when the window rolls.
                "open_pairings": {},
            }
            atomic_write_yaml(tournament.state_path, state)
            append_jsonl(tournament.events_path, {
                "event": "create",
                "timestamp": now,
                "name": name,
                "hypothesis_count": len(entries),
            })
        return tournament

    # ------------------------------------------------------------------
    # State load/save
    # ------------------------------------------------------------------

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            raise TournamentError(f"Tournament not found at {self.dir}")
        return read_yaml(self.state_path)

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = utc_now_iso()
        atomic_write_yaml(self.state_path, state)

    def state(self) -> dict[str, Any]:
        """Return a snapshot of the current tournament state (read-only)."""
        return self._load_state()

    def hypotheses(self) -> list[Hypothesis]:
        state = self._load_state()
        return [_to_hypothesis(entry) for entry in state["hypotheses"]]

    # ------------------------------------------------------------------
    # Pairing
    # ------------------------------------------------------------------

    def pair(
        self,
        *,
        policy: str = "elo_close",
        no_repeat_within: int = 3,
        rng_seed: int | None = None,
    ) -> PairingResult:
        """Return the next matchup the orchestrator should judge.

        ``no_repeat_within`` is the number of recent pairings that
        cannot reappear in this round (sliding window over the
        ``recent_pairings`` list in state).

        The returned ``pair_id`` is recorded in ``state.open_pairings``
        and must be echoed back via ``submit(pair_id=...)`` — stale or
        fabricated submits are rejected (M5 fix).
        """
        if policy not in _VALID_POLICIES:
            raise TournamentError(
                f"unknown policy {policy!r}; expected one of {sorted(_VALID_POLICIES)}"
            )
        with file_lock(self.dir / ".lock"):
            state = self._load_state()
            entries = state["hypotheses"]
            if len(entries) < 2:
                raise TournamentError(
                    f"need ≥2 hypotheses to pair; have {len(entries)}"
                )
            recent_pairs = {
                tuple(sorted(p)) for p in state["recent_pairings"][-no_repeat_within * len(entries):]
            }
            rng = random.Random(rng_seed) if rng_seed is not None else random
            left_idx, right_idx = _pick_pair(entries, policy, recent_pairs, rng)
            pair_id = f"p_{uuid.uuid4().hex[:8]}"
            state["round_number"] = int(state["round_number"]) + 1
            state["recent_pairings"].append([entries[left_idx]["id"], entries[right_idx]["id"]])
            # Bound the history to a reasonable window so it doesn't grow forever.
            state["recent_pairings"] = state["recent_pairings"][-1000:]
            # M5: register this pair_id so submit can validate it.
            # Backward-compat: older state.yaml may lack open_pairings;
            # tolerate via setdefault so opening a pre-fix tournament
            # doesn't crash.
            open_pairs = state.setdefault("open_pairings", {})
            open_pairs[pair_id] = {
                "left": entries[left_idx]["id"],
                "right": entries[right_idx]["id"],
                "round": state["round_number"],
            }
            # Bound open pairings so a buggy orchestrator that never
            # submits doesn't grow state.yaml without bound. Keep the
            # most recent 200 issued pair_ids.
            if len(open_pairs) > 200:
                drop = sorted(open_pairs)[: len(open_pairs) - 200]
                for k in drop:
                    open_pairs.pop(k, None)
            self._save_state(state)
            append_jsonl(self.events_path, {
                "event": "pair",
                "pair_id": pair_id,
                "policy": policy,
                "round": state["round_number"],
                "left": entries[left_idx]["id"],
                "right": entries[right_idx]["id"],
                "timestamp": utc_now_iso(),
            })
            return PairingResult(
                pair_id=pair_id,
                left=_to_hypothesis(entries[left_idx]),
                right=_to_hypothesis(entries[right_idx]),
                policy=policy,
                round_number=state["round_number"],
            )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        *,
        winner_id: str,
        loser_id: str,
        rationale: str = "",
        draw: bool = False,
        pair_id: str | None = None,
    ) -> dict[str, Any]:
        """Record a debate-judge outcome and update Elo.

        ``draw=True`` ignores ``winner_id`` / ``loser_id`` ordering and
        applies a 0.5 / 0.5 split.

        ``pair_id`` (M5 fix) — when provided, must match a previously
        issued pair_id from ``pair()`` AND name the same two
        hypotheses. Stale or fabricated pair_ids are rejected before
        Elo is touched. ``pair_id=None`` is accepted for backward
        compatibility but should be considered deprecated.
        """
        if winner_id == loser_id:
            raise TournamentError(f"winner_id == loser_id == {winner_id!r}")
        with file_lock(self.dir / ".lock"):
            state = self._load_state()
            by_id = {e["id"]: e for e in state["hypotheses"]}
            if winner_id not in by_id:
                raise TournamentError(f"unknown winner_id: {winner_id!r}")
            if loser_id not in by_id:
                raise TournamentError(f"unknown loser_id: {loser_id!r}")
            # M5: pair_id integrity gate.
            open_pairs = state.setdefault("open_pairings", {})
            if pair_id is not None:
                if pair_id not in open_pairs:
                    raise TournamentError(
                        f"pair_id {pair_id!r} not open — already settled, expired, "
                        "or never issued"
                    )
                expected = open_pairs[pair_id]
                expected_set = {expected["left"], expected["right"]}
                actual_set = {winner_id, loser_id}
                if expected_set != actual_set:
                    raise TournamentError(
                        f"pair_id {pair_id!r} was issued for "
                        f"{sorted(expected_set)}, submit got {sorted(actual_set)}"
                    )
                open_pairs.pop(pair_id, None)
            a = by_id[winner_id]
            b = by_id[loser_id]
            score_a = 0.5 if draw else 1.0
            new_a, new_b = _update_elo(
                a["rating"], b["rating"], score_a, k=state["k_factor"]
            )
            delta_a = new_a - a["rating"]
            delta_b = new_b - b["rating"]
            a["rating"] = new_a
            b["rating"] = new_b
            if draw:
                a["draws"] += 1
                b["draws"] += 1
            else:
                a["wins"] += 1
                b["losses"] += 1
            self._save_state(state)
            event = {
                "event": "submit",
                "pair_id": pair_id,
                "winner_id": winner_id,
                "loser_id": loser_id,
                "draw": draw,
                "rationale": rationale,
                "rating_delta": {
                    winner_id: round(delta_a, 4),
                    loser_id: round(delta_b, 4),
                },
                "timestamp": utc_now_iso(),
            }
            append_jsonl(self.events_path, event)
            return {
                "winner_id": winner_id,
                "loser_id": loser_id,
                "draw": draw,
                "winner_new_rating": round(new_a, 4),
                "loser_new_rating": round(new_b, 4),
                "rationale": rationale,
            }

    # ------------------------------------------------------------------
    # Ranking
    # ------------------------------------------------------------------

    def rank(self, top_k: int | None = None) -> list[TournamentRank]:
        """Return hypotheses sorted by Elo (descending). ``top_k=None`` returns all."""
        state = self._load_state()
        entries = sorted(state["hypotheses"], key=lambda e: -e["rating"])
        if top_k is not None:
            entries = entries[: max(0, int(top_k))]
        return [
            TournamentRank(
                id=e["id"],
                rating=e["rating"],
                wins=e["wins"],
                losses=e["losses"],
                draws=e["draws"],
                payload=dict(e["payload"]),
            )
            for e in entries
        ]

    # ------------------------------------------------------------------
    # Event log access
    # ------------------------------------------------------------------

    def events(self) -> list[dict[str, Any]]:
        return read_jsonl(self.events_path)


# ---------------------------------------------------------------------------
# Pairing implementations
# ---------------------------------------------------------------------------


def _pick_pair(
    entries: list[dict[str, Any]],
    policy: str,
    recent_pairs: set[tuple[str, str]],
    rng: random.Random,
) -> tuple[int, int]:
    n = len(entries)
    candidates = [(i, j) for i in range(n) for j in range(i + 1, n)]
    fresh = [
        (i, j)
        for i, j in candidates
        if tuple(sorted((entries[i]["id"], entries[j]["id"]))) not in recent_pairs
    ]
    pool = fresh or candidates  # fall back if everything recent
    if policy == "random":
        return rng.choice(pool)
    if policy == "swiss":
        # Swiss: pair by closeness in wins-minus-losses score.
        score = [e["wins"] - e["losses"] for e in entries]
        pool.sort(key=lambda ij: abs(score[ij[0]] - score[ij[1]]))
        return pool[0]
    if policy == "elo_close":
        rating = [e["rating"] for e in entries]
        pool.sort(key=lambda ij: abs(rating[ij[0]] - rating[ij[1]]))
        return pool[0]
    raise TournamentError(f"unhandled policy: {policy!r}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Orchestrator-contract prompt envelope
# ---------------------------------------------------------------------------


JUDGE_SYSTEM_PROMPT = (
    "You are a research-design debate judge. Two hypotheses are "
    "presented; pick the one more likely to improve the primary "
    "metric, and explain why in 1-3 sentences. Output strict JSON "
    "matching the supplied schema — no markdown fences, no prose "
    "preamble.\n\n"
    "Rules:\n"
    "- Be specific. Vague preferences (\"more interesting\") are "
    "not useful.\n"
    "- If both are equally promising, return \"draw\": true.\n"
    "- The hypothesis you pick must be one of the two supplied IDs.\n"
    "- Honor the project's judge-separation contract: this debate "
    "judge runs in a separate family from the reward and eval judges.\n"
)


def judge_request_prompt(pair: PairingResult, *, context: str = "") -> dict[str, Any]:
    """Build the orchestrator-contract envelope for one debate-judge round.

    Returns ``{system, user, schema, pair_id}``. The orchestrator runs
    its own LLM with this envelope, then submits via
    :meth:`Tournament.submit` (with ``winner_id`` / ``loser_id`` taken
    from the LLM's JSON response).
    """
    user = (
        f"# Debate round {pair.round_number}\n"
        f"pair_id: {pair.pair_id}\n\n"
        f"## Hypothesis A — id: {pair.left.id} (rating {pair.left.rating:.1f})\n"
        f"```yaml\n{_yamlify(pair.left.payload)}\n```\n\n"
        f"## Hypothesis B — id: {pair.right.id} (rating {pair.right.rating:.1f})\n"
        f"```yaml\n{_yamlify(pair.right.payload)}\n```\n"
    )
    if context:
        user += f"\n## Context\n{context}\n"

    schema = {
        "type": "object",
        "required": ["winner_id", "loser_id", "rationale"],
        "properties": {
            "winner_id": {
                "type": "string",
                "enum": [pair.left.id, pair.right.id],
            },
            "loser_id": {
                "type": "string",
                "enum": [pair.left.id, pair.right.id],
            },
            "rationale": {"type": "string"},
            "draw": {"type": "boolean"},
        },
    }
    return {
        "system": JUDGE_SYSTEM_PROMPT,
        "user": user,
        "schema": schema,
        "pair_id": pair.pair_id,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_hypothesis(entry: dict[str, Any]) -> Hypothesis:
    return Hypothesis(
        id=entry["id"],
        payload=dict(entry["payload"]),
        rating=float(entry["rating"]),
        wins=int(entry["wins"]),
        losses=int(entry["losses"]),
        draws=int(entry["draws"]),
    )


def _yamlify(payload: dict[str, Any]) -> str:
    """Render a payload dict as compact YAML for the debate prompt."""
    import yaml as _yaml
    return _yaml.safe_dump(payload, sort_keys=True, default_flow_style=False).strip()
