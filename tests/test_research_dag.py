"""Tests for the research_dag bridge module."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from crucible.core.errors import ResearchDAGError
from crucible.research_dag.bridge import ResearchDAGBridge, _CANVAS_TTL
from crucible.research_dag.dag_state import DAGState
from crucible.research_dag.node_format import (
    crucible_node_type,
    format_experiment_content,
    format_finding_content,
    format_review_content,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experiment(
    node_id: str = "exp-001",
    name: str = "test-exp",
    status: str = "pending",
    result: dict | None = None,
    result_metric: float | None = None,
    config: dict | None = None,
    hypothesis: str = "Test hypothesis",
    generation_method: str = "manual",
) -> dict:
    exp = {
        "node_id": node_id,
        "experiment_name": name,
        "hypothesis": hypothesis,
        "rationale": "Test rationale",
        "config": config or {"MODEL_FAMILY": "baseline", "D_MODEL": "128"},
        "status": status,
        "generation_method": generation_method,
    }
    if result is not None:
        exp["result"] = result
    if result_metric is not None:
        exp["result_metric"] = result_metric
    return exp


def _make_bridge(tmp_path: Path, url: str = "") -> ResearchDAGBridge:
    bridge = ResearchDAGBridge(project_dir=tmp_path, spiderchat_url=url)
    bridge.init(project_name="test-project")
    return bridge


# ===========================================================================
# DAGState tests
# ===========================================================================


class TestDAGState:
    def test_init_creates_directory(self, tmp_path):
        state_dir = tmp_path / "dag_state"
        state = DAGState(state_dir)
        state.init(flow_id="flow-123", project_name="test")

        assert state_dir.exists()
        assert (state_dir / "config.jsonl").exists()

    def test_add_mapping_persists_across_reload(self, tmp_path):
        state_dir = tmp_path / "dag_state"
        state = DAGState(state_dir)
        state.init(flow_id="flow-123")
        state.load()

        state.add_mapping("c1", "canvas-1", node_type="experiment", status="pending")

        # New instance, reload from disk
        state2 = DAGState(state_dir)
        state2.load()

        assert state2.get_canvas_node_id("c1") == "canvas-1"

    def test_update_status_persists(self, tmp_path):
        state_dir = tmp_path / "dag_state"
        state = DAGState(state_dir)
        state.init(flow_id="flow-123")
        state.load()

        state.add_mapping("c1", "canvas-1", status="pending")
        state.update_status("c1", "completed")

        state2 = DAGState(state_dir)
        state2.load()

        mapping = state2.get_mapping("c1")
        assert mapping is not None
        assert mapping["status"] == "completed"

    def test_get_canvas_node_id(self, tmp_path):
        state = DAGState(tmp_path / "ds")
        state.init(flow_id="f1")
        state.load()
        state.add_mapping("c1", "canvas-1")

        assert state.get_canvas_node_id("c1") == "canvas-1"
        assert state.get_canvas_node_id("c2") is None

    def test_get_crucible_node_id_reverse(self, tmp_path):
        state = DAGState(tmp_path / "ds")
        state.init(flow_id="f1")
        state.load()
        state.add_mapping("c1", "canvas-1")
        state.add_mapping("c2", "canvas-2")

        assert state.get_crucible_node_id("canvas-1") == "c1"
        assert state.get_crucible_node_id("canvas-2") == "c2"
        assert state.get_crucible_node_id("canvas-3") is None

    def test_get_synced_sets(self, tmp_path):
        state = DAGState(tmp_path / "ds")
        state.init(flow_id="f1")
        state.load()
        state.add_mapping("c1", "canvas-1")
        state.add_mapping("c2", "canvas-2")

        assert state.get_synced_crucible_ids() == {"c1", "c2"}
        assert state.get_synced_canvas_ids() == {"canvas-1", "canvas-2"}

    def test_summary(self, tmp_path):
        state = DAGState(tmp_path / "ds")
        state.init(flow_id="f1", project_name="my-proj")
        state.load()
        state.add_mapping("c1", "cv1", node_type="experiment", status="completed")
        state.add_mapping("c2", "cv2", node_type="hypothesis", status="pending")

        s = state.summary()
        assert s["flow_id"] == "f1"
        assert s["project_name"] == "my-proj"
        assert s["total_mappings"] == 2
        assert s["status_breakdown"]["completed"] == 1
        assert s["status_breakdown"]["pending"] == 1
        assert s["type_breakdown"]["experiment"] == 1
        assert s["type_breakdown"]["hypothesis"] == 1

    def test_empty_load(self, tmp_path):
        state = DAGState(tmp_path / "nonexistent")
        state.load()

        assert state.flow_id == ""
        assert state.get_all_mappings() == []

    def test_multiple_inits_keeps_latest(self, tmp_path):
        state_dir = tmp_path / "ds"
        state = DAGState(state_dir)
        state.init(flow_id="old-flow")
        state.init(flow_id="new-flow")
        state.load()

        assert state.flow_id == "new-flow"

    def test_get_mapping(self, tmp_path):
        state = DAGState(tmp_path / "ds")
        state.init(flow_id="f1")
        state.load()
        state.add_mapping("c1", "cv1", node_type="experiment", status="pending")

        m = state.get_mapping("c1")
        assert m is not None
        assert m["canvas_node_id"] == "cv1"
        assert m["status"] == "pending"
        assert state.get_mapping("nonexistent") is None

    def test_atomic_save(self, tmp_path):
        """Verify _save_mappings produces a valid file after update_status."""
        state = DAGState(tmp_path / "ds")
        state.init(flow_id="f1")
        state.load()
        state.add_mapping("c1", "cv1", status="pending")
        state.add_mapping("c2", "cv2", status="pending")

        state.update_status("c1", "completed")

        # Verify file is valid by reloading
        state2 = DAGState(tmp_path / "ds")
        state2.load()
        assert state2.get_mapping("c1")["status"] == "completed"
        assert state2.get_mapping("c2")["status"] == "pending"


# ===========================================================================
# Bridge tests (local-only, no HTTP)
# ===========================================================================


class TestBridgeLocalOnly:
    def test_local_only_no_url(self, tmp_path):
        bridge = _make_bridge(tmp_path, url="")
        assert bridge.canvas_connected is False

    def test_local_only_push_returns_local_id(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        exp = _make_experiment()
        canvas_id = bridge.push_experiment_node(exp)
        assert canvas_id.startswith("local-")

    def test_local_only_push_creates_mapping(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        exp = _make_experiment(node_id="x1")
        bridge.push_experiment_node(exp)

        assert bridge.state.get_canvas_node_id("x1") is not None

    def test_local_only_push_update_existing(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        exp = _make_experiment(node_id="x1", status="pending")
        cid1 = bridge.push_experiment_node(exp)

        # Push again with updated status
        exp2 = _make_experiment(node_id="x1", status="completed")
        cid2 = bridge.push_experiment_node(exp2)

        assert cid1 == cid2  # Same canvas ID returned
        assert bridge.state.get_mapping("x1")["status"] == "completed"

    def test_local_only_result_update_no_crash(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        # Update for a node that doesn't exist — should not raise
        bridge.push_result_update("nonexistent", {"val_bpb": 1.0})

    def test_local_only_finding(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        finding = {"title": "Wide models help", "category": "observation", "confidence": 0.8}
        cid = bridge.push_finding(finding)

        assert cid.startswith("local-finding-")
        assert bridge.state.get_canvas_node_id("finding-Wide models help") is not None

    def test_local_only_pull_returns_empty(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        result = bridge.pull_manual_nodes()
        assert result == []

    def test_local_only_sync(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        nodes = [
            _make_experiment(node_id="n1", name="exp-1"),
            _make_experiment(node_id="n2", name="exp-2", status="completed",
                             result={"val_bpb": 1.1}, result_metric=1.1),
        ]
        result = bridge.sync(tree_nodes=nodes)

        assert result["mode"] == "local-only"
        assert result["pushed"] == 2
        assert result["skipped_canvas"] == 2
        assert result["pulled"] == 0
        assert result["total_mappings"] == 2

    def test_sync_detects_status_change(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        node = _make_experiment(node_id="n1", status="pending")
        bridge.sync(tree_nodes=[node])

        # Now sync again with status changed
        node_updated = _make_experiment(node_id="n1", status="completed")
        result = bridge.sync(tree_nodes=[node_updated])

        assert result["updated"] == 1
        assert result["pushed"] == 0

    def test_sync_idempotent_no_spurious_updates(self, tmp_path):
        """Third sync with same data should produce updated=0."""
        bridge = _make_bridge(tmp_path)
        node = _make_experiment(node_id="n1", status="completed",
                                result={"val_bpb": 1.0}, result_metric=1.0)

        # First sync — pushes
        r1 = bridge.sync(tree_nodes=[node])
        assert r1["pushed"] == 1

        # Second sync — same data, should NOT update
        r2 = bridge.sync(tree_nodes=[node])
        assert r2["updated"] == 0
        assert r2["pushed"] == 0

        # Third sync — still same, still no update
        r3 = bridge.sync(tree_nodes=[node])
        assert r3["updated"] == 0

    def test_status_reports_mode(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        s = bridge.status()
        assert s["mode"] == "local-only"
        assert s["spiderchat_url"] == "(not configured)"

    def test_push_experiment_no_id_raises(self, tmp_path):
        bridge = _make_bridge(tmp_path)
        with pytest.raises(ResearchDAGError, match="must have node_id"):
            bridge.push_experiment_node({"config": {}, "status": "pending"})


# ===========================================================================
# Node format tests
# ===========================================================================


class TestNodeFormat:
    def test_format_experiment_pending(self):
        exp = _make_experiment()
        content = format_experiment_content(exp)
        assert "Pending" in content
        assert "Test hypothesis" in content

    def test_format_experiment_with_result(self):
        exp = _make_experiment()
        result = {"val_bpb": 1.234, "val_loss": 2.5}
        content = format_experiment_content(exp, result=result)
        assert "1.2340" in content
        assert "2.5000" in content

    def test_format_experiment_best_marker(self):
        exp = _make_experiment()
        result = {"val_bpb": 1.0}
        content = format_experiment_content(exp, result=result, best_metric=1.0, primary_metric="val_bpb")
        assert "\u2605" in content  # star character

    def test_format_experiment_not_best(self):
        exp = _make_experiment()
        result = {"val_bpb": 2.0}
        content = format_experiment_content(exp, result=result, best_metric=1.0, primary_metric="val_bpb")
        assert "\u2605" not in content

    def test_format_finding(self):
        finding = {
            "title": "Wide models help",
            "body": "Empirically verified.",
            "category": "observation",
            "confidence": 0.85,
            "source_experiments": ["exp-1", "exp-2"],
        }
        content = format_finding_content(finding)
        assert "Wide models help" in content
        assert "observation" in content
        assert "85%" in content
        assert "exp-1" in content

    def test_format_review(self):
        review = {
            "recommendation": "accept",
            "hypothesis_consistency": 0.9,
            "reasoning": "Results match.",
            "suspicious_patterns": ["Metric plateau"],
        }
        content = format_review_content(review)
        assert "Accepted" in content
        assert "Results match" in content
        assert "Metric plateau" in content

    def test_crucible_node_type_manual(self):
        assert crucible_node_type({"generation_method": "manual"}) == "manual"

    def test_crucible_node_type_experiment(self):
        assert crucible_node_type({"generation_method": "llm", "result": {"val_bpb": 1.0}}) == "experiment"

    def test_crucible_node_type_hypothesis(self):
        assert crucible_node_type({"generation_method": "llm"}) == "hypothesis"

    def test_crucible_node_type_manual_canvas(self):
        assert crucible_node_type({"generation_method": "manual_canvas"}) == "manual"

    def test_format_experiment_config_table(self):
        exp = _make_experiment(config={"LR": "3e-4", "BATCH_SIZE": "64"})
        content = format_experiment_content(exp)
        assert "| BATCH_SIZE | `64` |" in content
        assert "| LR | `3e-4` |" in content

    def test_format_finding_no_body(self):
        finding = {"title": "Minimal", "category": "belief", "confidence": 0.5}
        content = format_finding_content(finding)
        assert "Minimal" in content
        assert "50%" in content

    def test_format_experiment_generation_method(self):
        exp = _make_experiment(generation_method="llm_auto_expand")
        content = format_experiment_content(exp)
        assert "llm_auto_expand" in content


# ===========================================================================
# Connected-mode tests (mocked HTTP)
# ===========================================================================


def _mock_bridge(tmp_path: Path) -> ResearchDAGBridge:
    """Create a bridge with mocked canvas connectivity."""
    bridge = ResearchDAGBridge(project_dir=tmp_path, spiderchat_url="http://mock:3000")
    bridge._canvas_available = True
    bridge._canvas_checked_at = time.monotonic()
    bridge.state.init(flow_id="flow-abc", project_name="mock-project")
    bridge.state.load()
    return bridge


class TestBridgeConnected:
    def test_push_creates_canvas_node(self, tmp_path):
        bridge = _mock_bridge(tmp_path)
        exp = _make_experiment(node_id="c1", name="baseline")

        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {"id": "canvas-node-1"}
            cid = bridge.push_experiment_node(exp)

        assert cid == "canvas-node-1"
        assert bridge.state.get_canvas_node_id("c1") == "canvas-node-1"
        # Verify POST was called to create node
        calls = mock_http.call_args_list
        assert any(c[0][0] == "POST" and "/nodes" in c[0][1] for c in calls)

    def test_push_with_parents_creates_edges(self, tmp_path):
        bridge = _mock_bridge(tmp_path)
        exp = _make_experiment(node_id="c1")

        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {"id": "canvas-child"}
            bridge.push_experiment_node(exp, parent_canvas_ids=["canvas-parent"])

        # Verify edge creation POST
        edge_calls = [c for c in mock_http.call_args_list
                      if c[0][0] == "POST" and "/edges" in c[0][1]]
        assert len(edge_calls) == 1
        assert edge_calls[0][0][2]["source"] == "canvas-parent"
        assert edge_calls[0][0][2]["target"] == "canvas-child"

    def test_push_update_existing_node(self, tmp_path):
        bridge = _mock_bridge(tmp_path)

        # First push
        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {"id": "cv1"}
            bridge.push_experiment_node(_make_experiment(node_id="c1", status="pending"))

        # Second push — same crucible ID, different status
        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {}
            cid = bridge.push_experiment_node(_make_experiment(node_id="c1", status="completed"))

        assert cid == "cv1"  # Same canvas ID returned
        # Verify PATCH was called (update, not create)
        assert any(c[0][0] == "PATCH" for c in mock_http.call_args_list)

    def test_push_result_update_connected(self, tmp_path):
        bridge = _mock_bridge(tmp_path)

        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {"id": "cv1"}
            bridge.push_experiment_node(_make_experiment(node_id="c1"))

        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {}
            bridge.push_result_update("c1", {"val_bpb": 0.95})

        assert bridge.state.get_mapping("c1")["status"] == "completed"
        assert any(c[0][0] == "PATCH" for c in mock_http.call_args_list)

    def test_push_finding_connected(self, tmp_path):
        bridge = _mock_bridge(tmp_path)
        finding = {"title": "Discovery", "confidence": 0.8, "category": "observation"}

        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {"id": "cv-finding"}
            cid = bridge.push_finding(finding)

        assert cid == "cv-finding"
        assert bridge.state.get_canvas_node_id("finding-Discovery") == "cv-finding"

    def test_push_falls_back_on_http_error(self, tmp_path):
        bridge = _mock_bridge(tmp_path)
        exp = _make_experiment(node_id="c1")

        with patch.object(bridge, "_http_request", side_effect=ConnectionError("down")):
            cid = bridge.push_experiment_node(exp)

        assert cid.startswith("local-")
        assert bridge._canvas_available is None  # Cache invalidated

    def test_push_falls_back_on_no_id_returned(self, tmp_path):
        bridge = _mock_bridge(tmp_path)
        exp = _make_experiment(node_id="c1")

        with patch.object(bridge, "_http_request", return_value={}):
            cid = bridge.push_experiment_node(exp)

        assert cid.startswith("local-")

    def test_pull_manual_nodes(self, tmp_path):
        bridge = _mock_bridge(tmp_path)

        mock_nodes = [
            {"id": "n1", "type": "information", "data": {"name": "My Idea", "content": "Try X", "tags": ["arch"]}},
            {"id": "n2", "type": "chat", "data": {"prompt": "chat node"}},  # Should be skipped
            {"id": "n3", "type": "information", "data": {"name": "Another", "content": "Try Y"}},
        ]

        with patch.object(bridge, "_http_request", return_value={"nodes": mock_nodes, "edges": []}):
            hypotheses = bridge.pull_manual_nodes()

        assert len(hypotheses) == 2
        assert hypotheses[0]["name"] == "My Idea"
        assert hypotheses[0]["hypothesis"] == "Try X"
        assert hypotheses[0]["tags"] == ["arch"]
        assert hypotheses[1]["name"] == "Another"

    def test_pull_manual_skips_already_synced(self, tmp_path):
        bridge = _mock_bridge(tmp_path)

        # Pre-register a mapping
        bridge.state.add_mapping("c1", "n1", node_type="experiment")

        mock_nodes = [
            {"id": "n1", "type": "information", "data": {"name": "Already synced", "content": "Old"}},
            {"id": "n2", "type": "information", "data": {"name": "New", "content": "Fresh"}},
        ]

        with patch.object(bridge, "_http_request", return_value={"nodes": mock_nodes, "edges": []}):
            hypotheses = bridge.pull_manual_nodes()

        assert len(hypotheses) == 1
        assert hypotheses[0]["name"] == "New"

    def test_pull_connections(self, tmp_path):
        bridge = _mock_bridge(tmp_path)
        bridge.state.add_mapping("c1", "cv1")
        bridge.state.add_mapping("c2", "cv2")

        mock_edges = [
            {"source": "cv1", "target": "cv2"},
            {"source": "cv2", "target": "cv-unknown"},  # No mapping — skipped
        ]

        with patch.object(bridge, "_http_request", return_value={"nodes": [], "edges": mock_edges}):
            connections = bridge.pull_connections()

        assert len(connections) == 1
        assert connections[0]["source_crucible_id"] == "c1"
        assert connections[0]["target_crucible_id"] == "c2"

    def test_sync_connected_mode(self, tmp_path):
        bridge = _mock_bridge(tmp_path)

        node_counter = [0]

        def mock_http(method, path, body=None):
            if method == "POST" and "/nodes" in path:
                node_counter[0] += 1
                return {"id": f"cv-{node_counter[0]}"}
            if method == "POST" and "/edges" in path:
                return {}
            if method == "GET":
                return {"nodes": [], "edges": []}
            return {}

        nodes = [
            _make_experiment(node_id="n1", name="exp-1"),
            _make_experiment(node_id="n2", name="exp-2", status="completed"),
        ]

        with patch.object(bridge, "_http_request", side_effect=mock_http):
            result = bridge.sync(tree_nodes=nodes)

        assert result["mode"] == "connected"
        assert result["pushed"] == 2
        assert result["skipped_canvas"] == 0
        assert result["total_mappings"] == 2

    def test_sync_parent_edge_resolution(self, tmp_path):
        """Verify sync resolves parent_node_id to parent canvas IDs for edge creation."""
        bridge = _mock_bridge(tmp_path)

        created_edges = []

        def mock_http(method, path, body=None):
            if method == "POST" and "/nodes" in path:
                return {"id": f"cv-{body['data']['crucibleId']}"}
            if method == "POST" and "/edges" in path:
                created_edges.append(body)
                return {}
            if method == "GET":
                return {"nodes": [], "edges": []}
            return {}

        nodes = [
            _make_experiment(node_id="root"),
            {**_make_experiment(node_id="child"), "parent_node_id": "root"},
        ]

        with patch.object(bridge, "_http_request", side_effect=mock_http):
            bridge.sync(tree_nodes=nodes)

        # The child should have an edge from root's canvas ID
        assert any(e["source"] == "cv-root" and e["target"] == "cv-child" for e in created_edges)

    def test_update_canvas_node_invalidates_on_failure(self, tmp_path):
        bridge = _mock_bridge(tmp_path)
        assert bridge._canvas_available is True

        with patch.object(bridge, "_http_request", side_effect=ConnectionError("down")):
            bridge._update_canvas_node("f1", "n1", "content", "completed", 1.0, "experiment")

        assert bridge._canvas_available is None  # Invalidated

    def test_get_flow_nodes_returns_empty_on_failure(self, tmp_path):
        bridge = _mock_bridge(tmp_path)

        with patch.object(bridge, "_http_request", side_effect=ConnectionError("down")):
            nodes = bridge._get_flow_nodes("f1")

        assert nodes == []
        assert bridge._canvas_available is None

    def test_get_flow_edges_returns_empty_on_failure(self, tmp_path):
        bridge = _mock_bridge(tmp_path)

        with patch.object(bridge, "_http_request", side_effect=ConnectionError("down")):
            edges = bridge._get_flow_edges("f1")

        assert edges == []

    def test_check_canvas_health_endpoint(self, tmp_path):
        bridge = ResearchDAGBridge(project_dir=tmp_path, spiderchat_url="http://mock:3000")

        with patch.object(bridge, "_http_request", return_value={}) as mock_http:
            result = bridge._check_canvas()

        assert result is True
        mock_http.assert_called_once_with("GET", "/api/health")

    def test_check_canvas_fallback_to_flows(self, tmp_path):
        bridge = ResearchDAGBridge(project_dir=tmp_path, spiderchat_url="http://mock:3000")

        call_count = [0]

        def mock_http(method, path, body=None):
            call_count[0] += 1
            if "/health" in path:
                raise ConnectionError("no health endpoint")
            return {}  # /flows works

        with patch.object(bridge, "_http_request", side_effect=mock_http):
            result = bridge._check_canvas()

        assert result is True
        assert call_count[0] == 2  # health failed, flows succeeded

    def test_check_canvas_both_fail(self, tmp_path):
        bridge = ResearchDAGBridge(project_dir=tmp_path, spiderchat_url="http://mock:3000")

        with patch.object(bridge, "_http_request", side_effect=ConnectionError("down")):
            result = bridge._check_canvas()

        assert result is False

    def test_http_request_sets_auth_header(self, tmp_path):
        bridge = ResearchDAGBridge(
            project_dir=tmp_path, spiderchat_url="http://mock:3000", spiderchat_token="secret-token"
        )

        with patch("urllib.request.urlopen") as mock_urlopen:
            from io import BytesIO
            mock_resp = BytesIO(b'{"ok": true}')
            mock_resp.status = 200
            mock_resp.__enter__ = lambda s: s
            mock_resp.__exit__ = lambda s, *a: None
            mock_urlopen.return_value = mock_resp

            bridge._http_request("GET", "/api/test")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer secret-token"
        assert req.get_header("Content-type") == "application/json"

    def test_status_connected_mode(self, tmp_path):
        bridge = _mock_bridge(tmp_path)
        s = bridge.status()
        assert s["mode"] == "connected"
        assert s["spiderchat_url"] == "http://mock:3000"

    def test_push_result_update_sends_content(self, tmp_path):
        """Bug fix: push_result_update must send result content, not None."""
        bridge = _mock_bridge(tmp_path)

        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {"id": "cv1"}
            bridge.push_experiment_node(_make_experiment(node_id="c1"))

        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {}
            bridge.push_result_update("c1", {"val_bpb": 0.95, "val_loss": 1.5})

        # Verify PATCH was called with content (not None)
        patch_calls = [c for c in mock_http.call_args_list if c[0][0] == "PATCH"]
        assert len(patch_calls) == 1
        data = patch_calls[0][0][2]["data"]
        assert data.get("content") is not None
        assert "0.9500" in data["content"]

    def test_sync_detects_result_change_same_status(self, tmp_path):
        """Bug fix: sync must detect result/metric changes even when status unchanged."""
        bridge = _mock_bridge(tmp_path)

        # First sync — completed with no result yet
        node_v1 = _make_experiment(node_id="n1", status="completed")

        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {"id": "cv1"}
            bridge.sync(tree_nodes=[node_v1])

        # Second sync — same status but now has result
        node_v2 = {**_make_experiment(node_id="n1", status="completed"),
                    "result": {"val_bpb": 1.0}, "result_metric": 1.0}

        with patch.object(bridge, "_http_request") as mock_http:
            mock_http.return_value = {}
            result = bridge.sync(tree_nodes=[node_v2])

        assert result["updated"] == 1

    def test_safe_path_id_encodes_special_chars(self, tmp_path):
        bridge = _mock_bridge(tmp_path)
        assert bridge._safe_path_id("abc123") == "abc123"
        assert bridge._safe_path_id("a/b") == "a%2Fb"
        assert bridge._safe_path_id("a?b=c") == "a%3Fb%3Dc"
        assert bridge._safe_path_id("a#b") == "a%23b"

    def test_http_request_rejects_non_http_url(self, tmp_path):
        bridge = ResearchDAGBridge(project_dir=tmp_path, spiderchat_url="ftp://evil.com")
        with pytest.raises(ResearchDAGError, match="Invalid.*scheme"):
            bridge._http_request("GET", "/api/test")


# ===========================================================================
# Canvas connectivity tests (mocked)
# ===========================================================================


class TestCanvasConnectivity:
    def test_canvas_ttl_recheck(self, tmp_path):
        bridge = ResearchDAGBridge(project_dir=tmp_path, spiderchat_url="http://fake:3000")
        bridge.init(project_name="test")

        check_count = 0

        def mock_check():
            nonlocal check_count
            check_count += 1
            return False

        with patch.object(bridge, "_check_canvas", side_effect=mock_check):
            # First access
            _ = bridge.canvas_connected
            assert check_count == 1

            # Immediate second access — cached, no re-probe
            _ = bridge.canvas_connected
            assert check_count == 1

            # Simulate TTL expiry
            bridge._canvas_checked_at = time.monotonic() - _CANVAS_TTL - 1
            _ = bridge.canvas_connected
            assert check_count == 2

    def test_canvas_recovery_after_ttl(self, tmp_path):
        bridge = ResearchDAGBridge(project_dir=tmp_path, spiderchat_url="http://fake:3000")
        bridge.init(project_name="test")

        call_results = [False, True]  # First check fails, second succeeds

        def mock_check():
            return call_results.pop(0) if call_results else True

        with patch.object(bridge, "_check_canvas", side_effect=mock_check):
            assert bridge.canvas_connected is False

            # Expire TTL
            bridge._canvas_checked_at = time.monotonic() - _CANVAS_TTL - 1
            assert bridge.canvas_connected is True

    def test_invalidate_cache_forces_recheck(self, tmp_path):
        bridge = ResearchDAGBridge(project_dir=tmp_path, spiderchat_url="http://fake:3000")
        bridge.init(project_name="test")

        check_count = 0

        def mock_check():
            nonlocal check_count
            check_count += 1
            return False

        with patch.object(bridge, "_check_canvas", side_effect=mock_check):
            _ = bridge.canvas_connected
            assert check_count == 1

            bridge._invalidate_canvas_cache()
            _ = bridge.canvas_connected
            assert check_count == 2
