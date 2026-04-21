"""Bidirectional sync between Crucible experiments and Spider Chat canvas.

ResearchDAGBridge is the core integration point. It:
  - Pushes Crucible experiment nodes to Spider Chat as information nodes
  - Pulls manually-created Spider Chat nodes back as Crucible hypotheses
  - Maintains mapping state in .crucible/research_dag/

Spider Chat is OPTIONAL. The bridge works in local-only mode when Spider Chat
is unreachable — DAG state is tracked locally, and canvas sync is attempted
best-effort. All operations degrade gracefully.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
from pathlib import Path
from typing import Any

from crucible.core.errors import ResearchDAGError
from crucible.research_dag.dag_state import DAGState
from crucible.research_dag.node_format import (
    crucible_node_type,
    format_experiment_content,
    format_finding_content,
)

log = logging.getLogger(__name__)

_CANVAS_TTL = 60  # seconds — re-probe canvas connectivity after this interval

# Spider Chat HTTP failures — any of these means the canvas is unreachable or
# returned an unexpected payload. All operations degrade gracefully.
_HTTP_FAILURES = (
    ResearchDAGError,
    urllib.error.URLError,
    TimeoutError,
    json.JSONDecodeError,
    OSError,
)


class ResearchDAGBridge:
    """Bidirectional bridge between Crucible experiments and Spider Chat canvas.

    Works in two modes:
      - **Connected**: Spider Chat is reachable — full sync with canvas visualization.
      - **Local-only**: Spider Chat is down or unconfigured — DAG state tracked locally.

    All push/pull operations handle Spider Chat unavailability gracefully.
    """

    def __init__(
        self,
        project_dir: Path,
        spiderchat_url: str = "",
        spiderchat_token: str = "",
    ) -> None:
        self.project_dir = Path(project_dir)
        self.spiderchat_url = spiderchat_url.rstrip("/") if spiderchat_url else ""
        self.spiderchat_token = spiderchat_token
        self.state = DAGState(self.project_dir / ".crucible" / "research_dag")
        self._canvas_available: bool | None = None  # None = not checked yet
        self._canvas_checked_at: float = 0.0

    @property
    def canvas_connected(self) -> bool:
        """Whether Spider Chat canvas is reachable. Re-probes after TTL expiry."""
        if not self.spiderchat_url:
            return False
        now = time.monotonic()
        if self._canvas_available is None or (now - self._canvas_checked_at) > _CANVAS_TTL:
            self._canvas_available = self._check_canvas()
            self._canvas_checked_at = now
        return self._canvas_available

    def _check_canvas(self) -> bool:
        """Probe Spider Chat backend. Returns True if reachable."""
        if not self.spiderchat_url:
            return False
        try:
            self._http_request("GET", "/api/health")
            return True
        except _HTTP_FAILURES:
            try:
                self._http_request("GET", "/api/flows?limit=1")
                return True
            except _HTTP_FAILURES:
                log.info("Spider Chat not reachable at %s — local-only mode", self.spiderchat_url)
                return False

    def _invalidate_canvas_cache(self) -> None:
        """Force re-probe on next canvas_connected check."""
        self._canvas_available = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init(self, flow_id: str = "", project_name: str = "") -> dict[str, Any]:
        """Initialize the bridge for a project.

        Args:
            flow_id: Spider Chat flow ID to sync with. Empty = local-only mode.
            project_name: Human-readable project name.

        Returns:
            Summary of the initialized state.
        """
        self.state.init(
            flow_id=flow_id,
            project_name=project_name,
            spiderchat_url=self.spiderchat_url,
        )
        self.state.load()
        mode = "connected" if (flow_id and self.canvas_connected) else "local-only"
        log.info("Research DAG bridge initialized: flow_id=%s project=%s mode=%s", flow_id, project_name, mode)
        result = self.state.summary()
        result["mode"] = mode
        return result

    def load(self) -> None:
        """Load existing bridge state from disk."""
        self.state.load()

    # ------------------------------------------------------------------
    # Crucible → Spider Chat (best-effort)
    # ------------------------------------------------------------------

    def push_experiment_node(
        self,
        experiment: dict[str, Any],
        flow_id: str = "",
        parent_canvas_ids: list[str] | None = None,
        primary_metric: str = "val_bpb",
        best_metric: float | None = None,
    ) -> str:
        """Push a Crucible experiment to Spider Chat as an information node.

        Returns canvas node ID, or local placeholder if Spider Chat unavailable.
        """
        crucible_id = experiment.get("node_id") or experiment.get("experiment_name", "")
        if not crucible_id:
            raise ResearchDAGError("Experiment must have node_id or experiment_name")

        result = experiment.get("result")
        content = format_experiment_content(
            experiment, result=result, best_metric=best_metric, primary_metric=primary_metric
        )

        status = experiment.get("status", "pending")
        node_type = crucible_node_type(experiment)
        metric_val = experiment.get("result_metric")

        existing_canvas_id = self.state.get_canvas_node_id(crucible_id)
        fid = flow_id or self.state.flow_id

        if existing_canvas_id:
            # Update existing — best-effort canvas push
            if fid and self.canvas_connected and not existing_canvas_id.startswith("local-"):
                self._update_canvas_node(fid, existing_canvas_id, content, status, metric_val, node_type)
            self.state.update_mapping(crucible_id, status, result_metric=metric_val)
            return existing_canvas_id

        canvas_node_id = ""
        if fid and self.canvas_connected:
            canvas_node_id = self._try_create_canvas_node(
                flow_id=fid,
                content=content,
                name=experiment.get("experiment_name", crucible_id),
                crucible_id=crucible_id,
                crucible_status=status,
                crucible_metric=metric_val,
                crucible_type=node_type,
                parent_canvas_ids=parent_canvas_ids,
            )

        if not canvas_node_id:
            canvas_node_id = f"local-{crucible_id}"

        self.state.add_mapping(
            crucible_node_id=crucible_id,
            canvas_node_id=canvas_node_id,
            node_type=node_type,
            status=status,
            canvas_flow_id=fid,
            result_metric=float(metric_val) if metric_val is not None else None,
        )
        return canvas_node_id

    def push_result_update(
        self,
        crucible_node_id: str,
        result: dict[str, Any],
        primary_metric: str = "val_bpb",
        best_metric: float | None = None,
    ) -> None:
        """Update an existing canvas node with experiment results (best-effort)."""
        canvas_id = self.state.get_canvas_node_id(crucible_node_id)
        if not canvas_id:
            log.debug("No canvas mapping for crucible node %s, skipping", crucible_node_id)
            return

        metric_val = result.get(primary_metric)

        # Rebuild content with results so canvas node shows actual data, not "Pending"
        content = format_experiment_content(
            {"node_id": crucible_node_id, "result": result, "status": "completed"},
            result=result,
            best_metric=best_metric,
            primary_metric=primary_metric,
        )

        if self.state.flow_id and self.canvas_connected and not canvas_id.startswith("local-"):
            self._update_canvas_node(
                flow_id=self.state.flow_id,
                node_id=canvas_id,
                content=content,
                status="completed",
                metric=float(metric_val) if metric_val is not None else None,
                node_type="experiment",
            )

        self.state.update_status(crucible_node_id, "completed")

    def push_finding(
        self,
        finding: dict[str, Any],
        source_canvas_ids: list[str] | None = None,
        flow_id: str = "",
    ) -> str:
        """Push a research finding (best-effort). Returns canvas node ID or local placeholder."""
        fid = flow_id or self.state.flow_id
        content = format_finding_content(finding)
        finding_id = finding.get("finding_id", finding.get("title", "finding"))

        canvas_node_id = ""
        if fid and self.canvas_connected:
            canvas_node_id = self._try_create_canvas_node(
                flow_id=fid,
                content=content,
                name=finding.get("title", "Research Finding"),
                crucible_id=f"finding-{finding_id}",
                crucible_status="completed",
                crucible_metric=finding.get("confidence"),
                crucible_type="finding",
                parent_canvas_ids=source_canvas_ids,
            )

        if not canvas_node_id:
            canvas_node_id = f"local-finding-{finding_id}"

        self.state.add_mapping(
            crucible_node_id=f"finding-{finding_id}",
            canvas_node_id=canvas_node_id,
            node_type="finding",
            status="completed",
            canvas_flow_id=fid,
        )

        return canvas_node_id

    # ------------------------------------------------------------------
    # Spider Chat → Crucible (requires connection)
    # ------------------------------------------------------------------

    def pull_manual_nodes(self, flow_id: str = "") -> list[dict[str, Any]]:
        """Detect Spider Chat canvas nodes not yet mapped to Crucible.

        Returns empty list if Spider Chat is unavailable.
        """
        fid = flow_id or self.state.flow_id
        if not fid or not self.canvas_connected:
            return []

        all_nodes = self._get_flow_nodes(fid)
        synced_canvas_ids = self.state.get_synced_canvas_ids()

        manual_hypotheses: list[dict[str, Any]] = []
        for node in all_nodes:
            node_id = node.get("id", "")
            if node_id in synced_canvas_ids:
                continue
            # Reverse-lookup dedup: skip if already tracked under a different crucible ID
            if self.state.get_crucible_node_id(node_id):
                continue

            node_data = node.get("data", {})
            node_type = node.get("type", "")
            if node_type != "information":
                continue

            content = node_data.get("content", "")
            name = node_data.get("name", node_id)

            hypothesis = {
                "name": name,
                "hypothesis": content[:500] if content else f"Manual node: {name}",
                "config": {},
                "rationale": "Manually created in Spider Chat canvas",
                "generation_method": "manual_canvas",
                "canvas_node_id": node_id,
                "tags": node_data.get("tags", []),
            }
            manual_hypotheses.append(hypothesis)

            self.state.add_mapping(
                crucible_node_id=f"manual-{node_id}",
                canvas_node_id=node_id,
                node_type="manual",
                status="pending",
                canvas_flow_id=fid,
            )

        return manual_hypotheses

    def pull_connections(self, flow_id: str = "") -> list[dict[str, Any]]:
        """Detect edges in Spider Chat that represent experiment dependencies.

        Returns empty list if Spider Chat is unavailable.
        """
        fid = flow_id or self.state.flow_id
        if not fid or not self.canvas_connected:
            return []

        edges = self._get_flow_edges(fid)
        connections: list[dict[str, Any]] = []

        for edge in edges:
            source_canvas = edge.get("source", "")
            target_canvas = edge.get("target", "")
            source_crucible = self.state.get_crucible_node_id(source_canvas)
            target_crucible = self.state.get_crucible_node_id(target_canvas)

            if source_crucible and target_crucible:
                connections.append({
                    "source_crucible_id": source_crucible,
                    "target_crucible_id": target_crucible,
                    "source_canvas_id": source_canvas,
                    "target_canvas_id": target_canvas,
                })

        return connections

    # ------------------------------------------------------------------
    # Full sync
    # ------------------------------------------------------------------

    def sync(
        self,
        tree_nodes: list[dict[str, Any]],
        flow_id: str = "",
        primary_metric: str = "val_bpb",
        best_metric: float | None = None,
    ) -> dict[str, Any]:
        """Full bidirectional sync.

        Works in both connected and local-only mode.
        In local-only mode: still tracks DAG state, skips canvas operations.
        """
        fid = flow_id or self.state.flow_id
        pushed = 0
        updated = 0
        pulled = 0
        skipped_canvas = 0

        for node in tree_nodes:
            crucible_id = node.get("node_id", "")
            existing = self.state.get_canvas_node_id(crucible_id)

            if existing:
                mapping = self.state.get_mapping(crucible_id)
                # Detect status OR metric value changes
                status_changed = mapping and mapping.get("status") != node.get("status")
                stored_metric = mapping.get("result_metric") if mapping else None
                result_changed = (
                    node.get("result_metric") is not None
                    and node.get("result_metric") != stored_metric
                )
                if status_changed or result_changed:
                    self.push_experiment_node(
                        node, flow_id=fid,
                        primary_metric=primary_metric,
                        best_metric=best_metric,
                    )
                    updated += 1
            else:
                parent_canvas_ids = []
                parent_id = node.get("parent_node_id")
                if parent_id:
                    parent_canvas = self.state.get_canvas_node_id(parent_id)
                    if parent_canvas:
                        parent_canvas_ids.append(parent_canvas)

                canvas_id = self.push_experiment_node(
                    node, flow_id=fid,
                    parent_canvas_ids=parent_canvas_ids,
                    primary_metric=primary_metric,
                    best_metric=best_metric,
                )
                pushed += 1
                if canvas_id.startswith("local-"):
                    skipped_canvas += 1

        manual = self.pull_manual_nodes(fid)
        pulled = len(manual)

        mode = "connected" if self.canvas_connected else "local-only"

        return {
            "mode": mode,
            "pushed": pushed,
            "updated": updated,
            "pulled": pulled,
            "skipped_canvas": skipped_canvas,
            "manual_hypotheses": manual,
            "total_mappings": len(self.state.get_all_mappings()),
        }

    def status(self) -> dict[str, Any]:
        """Return current bridge status."""
        result = self.state.summary()
        result["mode"] = "connected" if self.canvas_connected else "local-only"
        result["spiderchat_url"] = self.spiderchat_url or "(not configured)"
        return result

    # ------------------------------------------------------------------
    # Spider Chat HTTP API (private, best-effort)
    # ------------------------------------------------------------------

    def _http_request(self, method: str, path: str, body: dict | None = None) -> dict[str, Any]:
        """Make HTTP request to Spider Chat backend. Raises on failure."""
        import urllib.parse
        import urllib.request
        import urllib.error

        # Validate URL is well-formed to prevent injection
        parsed = urllib.parse.urlparse(self.spiderchat_url)
        if parsed.scheme not in ("http", "https"):
            raise ResearchDAGError(f"Invalid Spider Chat URL scheme: {parsed.scheme}")

        url = f"{self.spiderchat_url}{path}"
        data = json.dumps(body).encode("utf-8") if body else None

        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.spiderchat_token:
            req.add_header("Authorization", f"Bearer {self.spiderchat_token}")

        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _safe_path_id(raw_id: str) -> str:
        """URL-encode an ID for safe use in URL paths."""
        import urllib.parse
        return urllib.parse.quote(raw_id, safe="")

    def _try_create_canvas_node(
        self,
        flow_id: str,
        content: str,
        name: str,
        crucible_id: str,
        crucible_status: str,
        crucible_metric: float | None,
        crucible_type: str,
        parent_canvas_ids: list[str] | None = None,
    ) -> str:
        """Create an information node in Spider Chat canvas. Returns '' on failure."""
        node_data: dict[str, Any] = {
            "type": "information",
            "content": content,
            "name": name,
            "crucibleId": crucible_id,
            "crucibleStatus": crucible_status,
            "crucibleType": crucible_type,
            "useForGeneration": True,
        }
        if crucible_metric is not None:
            node_data["crucibleMetric"] = crucible_metric

        body: dict[str, Any] = {
            "type": "information",
            "data": node_data,
        }
        if parent_canvas_ids:
            body["parentIds"] = parent_canvas_ids

        try:
            resp = self._http_request("POST", f"/api/flows/{self._safe_path_id(flow_id)}/nodes", body)
        except _HTTP_FAILURES:
            log.warning("Spider Chat unreachable — node %s tracked locally only", crucible_id)
            self._invalidate_canvas_cache()
            return ""

        node_id = resp.get("id") or resp.get("nodeId") or resp.get("node", {}).get("id", "")
        if not node_id:
            log.warning("Spider Chat did not return node ID for %s", crucible_id)
            return ""

        if parent_canvas_ids:
            for parent_id in parent_canvas_ids:
                try:
                    self._http_request("POST", f"/api/flows/{self._safe_path_id(flow_id)}/edges", {
                        "source": parent_id,
                        "target": node_id,
                    })
                except _HTTP_FAILURES:
                    log.debug("Failed to create edge %s -> %s", parent_id, node_id)

        log.info("Created canvas node %s for crucible %s", node_id, crucible_id)
        return node_id

    def _update_canvas_node(
        self,
        flow_id: str,
        node_id: str,
        content: str | None,
        status: str,
        metric: float | None,
        node_type: str,
    ) -> None:
        """Update an existing Spider Chat canvas node (best-effort, no raise)."""
        update_data: dict[str, Any] = {
            "crucibleStatus": status,
            "crucibleType": node_type,
        }
        if content is not None:
            update_data["content"] = content
        if metric is not None:
            update_data["crucibleMetric"] = metric

        try:
            self._http_request("PATCH", f"/api/flows/{self._safe_path_id(flow_id)}/nodes/{self._safe_path_id(node_id)}", {
                "data": update_data,
            })
        except _HTTP_FAILURES:
            log.debug("Failed to update canvas node %s", node_id)
            self._invalidate_canvas_cache()

    def _get_flow_nodes(self, flow_id: str) -> list[dict[str, Any]]:
        """Get all nodes from a Spider Chat flow. Returns [] on failure."""
        try:
            resp = self._http_request("GET", f"/api/flows/{self._safe_path_id(flow_id)}")
            return resp.get("nodes", [])
        except _HTTP_FAILURES:
            log.debug("Failed to get flow nodes for %s", flow_id)
            self._invalidate_canvas_cache()
            return []

    def _get_flow_edges(self, flow_id: str) -> list[dict[str, Any]]:
        """Get all edges from a Spider Chat flow. Returns [] on failure."""
        try:
            resp = self._http_request("GET", f"/api/flows/{self._safe_path_id(flow_id)}")
            return resp.get("edges", [])
        except _HTTP_FAILURES:
            log.debug("Failed to get flow edges for %s", flow_id)
            self._invalidate_canvas_cache()
            return []
