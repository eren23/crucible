"""Tests for RunPod GraphQL functions in crucible.fleet.providers.runpod.

Covers all 10 GraphQL-related functions:
- runpod_graphql (core transport)
- runpod_stop_pod, runpod_start_pod, runpod_start_spot_pod
- runpod_list_network_volumes, runpod_create_network_volume, runpod_delete_network_volume
- runpod_list_gpu_types
- runpod_list_templates, runpod_create_template

All network I/O is mocked via urllib.request.urlopen (transport-level tests)
or via runpod_graphql itself (higher-level tests).
"""

from __future__ import annotations

import io
import json
from typing import Any
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError

import pytest

from crucible.core.errors import FleetError
from crucible.fleet.providers.runpod import (
    runpod_create_network_volume,
    runpod_create_template,
    runpod_delete_network_volume,
    runpod_graphql,
    runpod_list_gpu_types,
    runpod_list_network_volumes,
    runpod_list_templates,
    runpod_start_pod,
    runpod_start_spot_pod,
    runpod_stop_pod,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_urlopen_response(data: dict[str, Any]) -> MagicMock:
    """Build a mock that behaves like the context manager returned by urlopen."""
    body = json.dumps(data).encode("utf-8")
    resp = MagicMock()
    resp.read.return_value = body
    resp.__enter__ = lambda self: self
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_http_error(code: int = 500, body: str = "Internal Server Error") -> HTTPError:
    """Build an HTTPError with a readable body."""
    return HTTPError(
        url="https://api.runpod.io/graphql",
        code=code,
        msg="Server Error",
        hdrs={},  # type: ignore[arg-type]
        fp=io.BytesIO(body.encode("utf-8")),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure RUNPOD_API_KEY is set for every test."""
    monkeypatch.setenv("RUNPOD_API_KEY", "test-key-abc123")


# =========================================================================
# 1. runpod_graphql — core transport
# =========================================================================


class TestRunpodGraphql:
    """Tests for the low-level runpod_graphql() transport function."""

    # --- happy path ---

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_happy_path_returns_data_dict(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"myself": {"id": "user-1"}}}
        )
        result = runpod_graphql("query { myself { id } }")
        assert result == {"myself": {"id": "user-1"}}

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_variables_included_in_payload(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response({"data": {"ok": True}})
        runpod_graphql(
            "mutation m($id: String!) { m(id: $id) }",
            variables={"id": "pod-1"},
        )
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["variables"] == {"id": "pod-1"}
        assert "mutation" in sent["query"]

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_no_variables_omits_key_from_payload(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response({"data": {"ok": True}})
        runpod_graphql("query { ok }")
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert "variables" not in sent

    # --- authorization header ---

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_authorization_header_contains_api_key(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response({"data": {"ok": True}})
        runpod_graphql("query { ok }")
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer test-key-abc123"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_content_type_is_json(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response({"data": {"ok": True}})
        runpod_graphql("query { ok }")
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Content-type") == "application/json"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_uses_post_method(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response({"data": {"ok": True}})
        runpod_graphql("query { ok }")
        req = mock_urlopen.call_args[0][0]
        assert req.get_method() == "POST"

    # --- timeout ---

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_timeout_parameter_passed_through(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response({"data": {"ok": True}})
        runpod_graphql("query { ok }", timeout=42)
        call_args = mock_urlopen.call_args
        # urlopen(req, timeout=N) — timeout is a keyword arg
        if len(call_args[0]) > 1:
            assert call_args[0][1] == 42
        else:
            assert call_args[1].get("timeout") == 42

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_default_timeout_is_60(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response({"data": {"ok": True}})
        runpod_graphql("query { ok }")
        call_args = mock_urlopen.call_args
        if len(call_args[0]) > 1:
            assert call_args[0][1] == 60
        else:
            assert call_args[1].get("timeout") == 60

    # --- GraphQL error handling ---

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_errors_raises_fleet_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "Pod not found"}]}
        )
        with pytest.raises(FleetError, match="Pod not found"):
            runpod_graphql("query { fail }")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_errors_uses_first_error_message(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "first"}, {"message": "second"}]}
        )
        with pytest.raises(FleetError, match="first"):
            runpod_graphql("query { fail }")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_error_without_message_key(self, mock_urlopen: MagicMock) -> None:
        """When the error dict has no 'message' key, it stringifies the dict."""
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"code": "FORBIDDEN"}]}
        )
        with pytest.raises(FleetError, match="GraphQL error"):
            runpod_graphql("query { fail }")

    # --- data is None ---

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_data_none_raises_fleet_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response({"data": None})
        with pytest.raises(FleetError, match="returned no data"):
            runpod_graphql("query { fail }")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_missing_data_key_raises_fleet_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response({"something": "else"})
        with pytest.raises(FleetError, match="returned no data"):
            runpod_graphql("query { fail }")

    # --- HTTP / network errors ---

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_error_raises_fleet_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(401, "Unauthorized")
        with pytest.raises(FleetError, match="401.*Unauthorized"):
            runpod_graphql("query { fail }")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_500_raises_fleet_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(500, "server exploded")
        with pytest.raises(FleetError, match="500"):
            runpod_graphql("query { fail }")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_url_error_raises_fleet_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = URLError("DNS resolution failed")
        with pytest.raises(FleetError, match="DNS resolution failed"):
            runpod_graphql("query { fail }")

    # --- missing API key ---

    def test_missing_api_key_raises_fleet_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("RUNPOD_API_KEY")
        with pytest.raises(FleetError, match="RUNPOD_API_KEY"):
            runpod_graphql("query { ok }")


# =========================================================================
# 2. runpod_stop_pod
# =========================================================================


class TestRunpodStopPod:
    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_happy_path(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {
                "data": {
                    "podStop": {
                        "id": "pod-abc",
                        "desiredStatus": "EXITED",
                        "lastStatusChange": "2026-04-01T00:00:00Z",
                    }
                }
            }
        )
        result = runpod_stop_pod("pod-abc")
        assert result["id"] == "pod-abc"
        assert result["desiredStatus"] == "EXITED"
        assert "lastStatusChange" in result

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_sends_correct_pod_id_variable(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"podStop": {"id": "pod-xyz"}}}
        )
        runpod_stop_pod("pod-xyz")
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["variables"]["podId"] == "pod-xyz"
        assert "podStop" in sent["query"]

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_returns_empty_dict_when_mutation_returns_null(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"podStop": None}}
        )
        result = runpod_stop_pod("pod-abc")
        assert result == {}

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "Pod not found"}]}
        )
        with pytest.raises(FleetError, match="Pod not found"):
            runpod_stop_pod("nonexistent")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(500, "Internal Server Error")
        with pytest.raises(FleetError, match="500"):
            runpod_stop_pod("pod-abc")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_uses_120s_timeout(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"podStop": {"id": "p"}}}
        )
        runpod_stop_pod("p")
        call_args = mock_urlopen.call_args
        if len(call_args[0]) > 1:
            assert call_args[0][1] == 120
        else:
            assert call_args[1].get("timeout") == 120


# =========================================================================
# 3. runpod_start_pod
# =========================================================================


class TestRunpodStartPod:
    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_happy_path(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {
                "data": {
                    "podResume": {
                        "id": "pod-abc",
                        "costPerHr": 0.44,
                        "desiredStatus": "RUNNING",
                        "lastStatusChange": "2026-04-01T00:00:00Z",
                    }
                }
            }
        )
        result = runpod_start_pod("pod-abc")
        assert result["id"] == "pod-abc"
        assert result["costPerHr"] == 0.44
        assert result["desiredStatus"] == "RUNNING"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_sends_correct_pod_id_variable(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"podResume": {"id": "pod-123"}}}
        )
        runpod_start_pod("pod-123")
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["variables"]["podId"] == "pod-123"
        assert "podResume" in sent["query"]

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_returns_empty_dict_when_mutation_returns_null(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"podResume": None}}
        )
        result = runpod_start_pod("pod-abc")
        assert result == {}

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "Insufficient funds"}]}
        )
        with pytest.raises(FleetError, match="Insufficient funds"):
            runpod_start_pod("pod-abc")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(403, "Forbidden")
        with pytest.raises(FleetError, match="403"):
            runpod_start_pod("pod-abc")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_uses_120s_timeout(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"podResume": {"id": "p"}}}
        )
        runpod_start_pod("p")
        call_args = mock_urlopen.call_args
        if len(call_args[0]) > 1:
            assert call_args[0][1] == 120
        else:
            assert call_args[1].get("timeout") == 120


# =========================================================================
# 4. runpod_start_spot_pod
# =========================================================================


class TestRunpodStartSpotPod:
    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_happy_path(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {
                "data": {
                    "podBidResume": {
                        "id": "pod-spot",
                        "costPerHr": 0.22,
                        "desiredStatus": "RUNNING",
                        "lastStatusChange": "2026-04-01T00:00:00Z",
                    }
                }
            }
        )
        result = runpod_start_spot_pod("pod-spot", bid_per_gpu=0.25, gpu_count=2)
        assert result["id"] == "pod-spot"
        assert result["costPerHr"] == 0.22

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_sends_correct_variables(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"podBidResume": {"id": "pod-spot"}}}
        )
        runpod_start_spot_pod("pod-spot", bid_per_gpu=0.30, gpu_count=4)
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["variables"]["podId"] == "pod-spot"
        assert sent["variables"]["bidPerGpu"] == 0.30
        assert sent["variables"]["gpuCount"] == 4

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_default_gpu_count_is_one(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"podBidResume": {"id": "pod-spot"}}}
        )
        runpod_start_spot_pod("pod-spot", bid_per_gpu=0.25)
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["variables"]["gpuCount"] == 1

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_returns_empty_dict_when_mutation_returns_null(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"podBidResume": None}}
        )
        result = runpod_start_spot_pod("pod-spot", 0.25)
        assert result == {}

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "Bid too low"}]}
        )
        with pytest.raises(FleetError, match="Bid too low"):
            runpod_start_spot_pod("pod-spot", 0.01)

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(502, "Bad Gateway")
        with pytest.raises(FleetError, match="502"):
            runpod_start_spot_pod("pod-spot", 0.25)

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_uses_120s_timeout(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"podBidResume": {"id": "p"}}}
        )
        runpod_start_spot_pod("p", 0.25)
        call_args = mock_urlopen.call_args
        if len(call_args[0]) > 1:
            assert call_args[0][1] == 120
        else:
            assert call_args[1].get("timeout") == 120


# =========================================================================
# 5. runpod_list_network_volumes
# =========================================================================


class TestRunpodListNetworkVolumes:
    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_happy_path(self, mock_urlopen: MagicMock) -> None:
        volumes = [
            {"id": "vol-1", "dataCenterId": "US-TX-3", "name": "data", "size": 100},
            {"id": "vol-2", "dataCenterId": "EU-SE-1", "name": "models", "size": 50},
        ]
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"myself": {"networkVolumes": volumes}}}
        )
        result = runpod_list_network_volumes()
        assert len(result) == 2
        assert result[0]["id"] == "vol-1"
        assert result[1]["name"] == "models"
        assert result[1]["size"] == 50

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_empty_list(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"myself": {"networkVolumes": []}}}
        )
        assert runpod_list_network_volumes() == []

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_null_network_volumes_returns_empty_list(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"myself": {"networkVolumes": None}}}
        )
        assert runpod_list_network_volumes() == []

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_null_myself_returns_empty_list(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"myself": None}}
        )
        assert runpod_list_network_volumes() == []

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_missing_myself_key_returns_empty_list(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {}}
        )
        assert runpod_list_network_volumes() == []

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "Authentication failed"}]}
        )
        with pytest.raises(FleetError, match="Authentication failed"):
            runpod_list_network_volumes()

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(503, "Service Unavailable")
        with pytest.raises(FleetError, match="503"):
            runpod_list_network_volumes()


# =========================================================================
# 6. runpod_create_network_volume
# =========================================================================


class TestRunpodCreateNetworkVolume:
    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_happy_path(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {
                "data": {
                    "createNetworkVolume": {
                        "id": "vol-new",
                        "name": "training-data",
                        "size": 200,
                        "dataCenterId": "US-TX-3",
                    }
                }
            }
        )
        result = runpod_create_network_volume("training-data", 200, "US-TX-3")
        assert result["id"] == "vol-new"
        assert result["name"] == "training-data"
        assert result["size"] == 200
        assert result["dataCenterId"] == "US-TX-3"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_sends_correct_input_variables(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"createNetworkVolume": {"id": "vol-new"}}}
        )
        runpod_create_network_volume("my-vol", 150, "EU-SE-1")
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        inp = sent["variables"]["input"]
        assert inp["name"] == "my-vol"
        assert inp["size"] == 150
        assert inp["dataCenterId"] == "EU-SE-1"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_returns_empty_dict_when_mutation_returns_null(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"createNetworkVolume": None}}
        )
        result = runpod_create_network_volume("vol", 10, "US-TX-3")
        assert result == {}

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "Datacenter not found"}]}
        )
        with pytest.raises(FleetError, match="Datacenter not found"):
            runpod_create_network_volume("vol", 10, "INVALID")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(400, "Bad Request")
        with pytest.raises(FleetError, match="400"):
            runpod_create_network_volume("vol", 10, "US-TX-3")


# =========================================================================
# 7. runpod_delete_network_volume
# =========================================================================


class TestRunpodDeleteNetworkVolume:
    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_happy_path_true_does_not_raise(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"deleteNetworkVolume": True}}
        )
        # Should complete without error
        runpod_delete_network_volume("vol-123")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_sends_correct_volume_id(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"deleteNetworkVolume": True}}
        )
        runpod_delete_network_volume("vol-xyz")
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["variables"]["id"] == "vol-xyz"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_false_raises_fleet_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"deleteNetworkVolume": False}}
        )
        with pytest.raises(FleetError, match="Failed to delete network volume vol-123"):
            runpod_delete_network_volume("vol-123")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_null_raises_fleet_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"deleteNetworkVolume": None}}
        )
        with pytest.raises(FleetError, match="Failed to delete network volume vol-abc"):
            runpod_delete_network_volume("vol-abc")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_missing_key_raises_fleet_error(self, mock_urlopen: MagicMock) -> None:
        """When the data dict doesn't contain the deleteNetworkVolume key at all."""
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"somethingElse": True}}
        )
        with pytest.raises(FleetError, match="Failed to delete"):
            runpod_delete_network_volume("vol-gone")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_error_message_includes_volume_id(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"deleteNetworkVolume": False}}
        )
        with pytest.raises(FleetError, match="vol-special"):
            runpod_delete_network_volume("vol-special")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "Volume in use"}]}
        )
        with pytest.raises(FleetError, match="Volume in use"):
            runpod_delete_network_volume("vol-123")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(404, "Not Found")
        with pytest.raises(FleetError, match="404"):
            runpod_delete_network_volume("vol-123")


# =========================================================================
# 8. runpod_list_gpu_types
# =========================================================================


class TestRunpodListGpuTypes:
    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_happy_path_flattens_lowest_price(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {
                "data": {
                    "gpuTypes": [
                        {
                            "lowestPrice": {
                                "gpuName": "RTX 4090",
                                "gpuTypeId": "NVIDIA GeForce RTX 4090",
                                "minimumBidPrice": 0.25,
                                "uninterruptablePrice": 0.44,
                                "minMemory": 24,
                                "minVcpu": 8,
                            }
                        },
                        {
                            "lowestPrice": {
                                "gpuName": "RTX 3090",
                                "gpuTypeId": "NVIDIA GeForce RTX 3090",
                                "minimumBidPrice": 0.16,
                                "uninterruptablePrice": 0.22,
                                "minMemory": 24,
                                "minVcpu": 4,
                            }
                        },
                    ]
                }
            }
        )
        result = runpod_list_gpu_types()
        assert len(result) == 2
        # Flattened: inner dicts directly, not wrapped
        assert result[0]["gpuName"] == "RTX 4090"
        assert result[1]["gpuName"] == "RTX 3090"
        assert result[1]["minimumBidPrice"] == 0.16
        assert "lowestPrice" not in result[0]

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_filters_out_entries_with_null_gpu_name(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {
                "data": {
                    "gpuTypes": [
                        {
                            "lowestPrice": {
                                "gpuName": "RTX 4090",
                                "gpuTypeId": "RTX4090",
                                "minimumBidPrice": 0.25,
                                "uninterruptablePrice": 0.44,
                                "minMemory": 24,
                                "minVcpu": 8,
                            }
                        },
                        {
                            "lowestPrice": {
                                "gpuName": None,
                                "gpuTypeId": "unknown",
                                "minimumBidPrice": None,
                                "uninterruptablePrice": None,
                                "minMemory": None,
                                "minVcpu": None,
                            }
                        },
                    ]
                }
            }
        )
        result = runpod_list_gpu_types()
        assert len(result) == 1
        assert result[0]["gpuName"] == "RTX 4090"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_filters_out_entries_with_missing_gpu_name_key(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {
                "data": {
                    "gpuTypes": [
                        {"lowestPrice": {"gpuTypeId": "no-name"}},
                        {
                            "lowestPrice": {
                                "gpuName": "A100",
                                "gpuTypeId": "A100",
                                "minimumBidPrice": 1.0,
                                "uninterruptablePrice": 2.0,
                                "minMemory": 80,
                                "minVcpu": 16,
                            }
                        },
                    ]
                }
            }
        )
        result = runpod_list_gpu_types()
        assert len(result) == 1
        assert result[0]["gpuName"] == "A100"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_filters_out_null_lowest_price(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {
                "data": {
                    "gpuTypes": [
                        {"lowestPrice": None},
                        {
                            "lowestPrice": {
                                "gpuName": "H100",
                                "gpuTypeId": "H100",
                                "minimumBidPrice": 2.0,
                                "uninterruptablePrice": 3.5,
                                "minMemory": 80,
                                "minVcpu": 16,
                            }
                        },
                    ]
                }
            }
        )
        result = runpod_list_gpu_types()
        assert len(result) == 1
        assert result[0]["gpuName"] == "H100"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_sends_gpu_count_variable(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"gpuTypes": []}}
        )
        runpod_list_gpu_types(gpu_count=4)
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["variables"]["input"]["gpuCount"] == 4

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_default_gpu_count_is_one(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"gpuTypes": []}}
        )
        runpod_list_gpu_types()
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["variables"]["input"]["gpuCount"] == 1

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_sends_secure_cloud_true(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"gpuTypes": []}}
        )
        runpod_list_gpu_types(secure_cloud=True)
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["variables"]["input"]["secureCloud"] is True

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_sends_secure_cloud_false(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"gpuTypes": []}}
        )
        runpod_list_gpu_types(secure_cloud=False)
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert sent["variables"]["input"]["secureCloud"] is False

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_omits_secure_cloud_when_none(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"gpuTypes": []}}
        )
        runpod_list_gpu_types(secure_cloud=None)
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert "secureCloud" not in sent["variables"]["input"]

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_empty_gpu_types_list(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"gpuTypes": []}}
        )
        assert runpod_list_gpu_types() == []

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_null_gpu_types(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"gpuTypes": None}}
        )
        assert runpod_list_gpu_types() == []

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "Rate limit exceeded"}]}
        )
        with pytest.raises(FleetError, match="Rate limit exceeded"):
            runpod_list_gpu_types()

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(429, "Too Many Requests")
        with pytest.raises(FleetError, match="429"):
            runpod_list_gpu_types()


# =========================================================================
# 9. runpod_list_templates
# =========================================================================


class TestRunpodListTemplates:
    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_happy_path(self, mock_urlopen: MagicMock) -> None:
        templates = [
            {
                "id": "tmpl-1",
                "name": "PyTorch 2.0",
                "imageName": "runpod/pytorch:2.0",
                "dockerArgs": "",
                "containerDiskInGb": 20,
                "volumeInGb": 40,
                "volumeMountPath": "/workspace",
                "ports": "22/tcp,8888/http",
                "isServerless": False,
            },
            {
                "id": "tmpl-2",
                "name": "CUDA Base",
                "imageName": "nvidia/cuda:12.0",
                "dockerArgs": "",
                "containerDiskInGb": 10,
                "volumeInGb": 20,
                "volumeMountPath": "/workspace",
                "ports": "22/tcp",
                "isServerless": False,
            },
        ]
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"myself": {"podTemplates": templates}}}
        )
        result = runpod_list_templates()
        assert len(result) == 2
        assert result[0]["id"] == "tmpl-1"
        assert result[0]["name"] == "PyTorch 2.0"
        assert result[1]["imageName"] == "nvidia/cuda:12.0"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_empty_list(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"myself": {"podTemplates": []}}}
        )
        assert runpod_list_templates() == []

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_null_templates_returns_empty_list(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"myself": {"podTemplates": None}}}
        )
        assert runpod_list_templates() == []

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_null_myself_returns_empty_list(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"myself": None}}
        )
        assert runpod_list_templates() == []

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_missing_myself_key_returns_empty_list(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {}}
        )
        assert runpod_list_templates() == []

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "Unauthorized"}]}
        )
        with pytest.raises(FleetError, match="Unauthorized"):
            runpod_list_templates()

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(500, "Server Error")
        with pytest.raises(FleetError, match="500"):
            runpod_list_templates()


# =========================================================================
# 10. runpod_create_template
# =========================================================================


class TestRunpodCreateTemplate:
    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_happy_path(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"saveTemplate": {"id": "tmpl-new", "name": "My Template"}}}
        )
        result = runpod_create_template("My Template", "runpod/pytorch:2.0")
        assert result["id"] == "tmpl-new"
        assert result["name"] == "My Template"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_sends_required_fields(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"saveTemplate": {"id": "tmpl-x", "name": "test"}}}
        )
        runpod_create_template("test", "nvidia/cuda:12.0")
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        inp = sent["variables"]["input"]
        assert inp["name"] == "test"
        assert inp["imageName"] == "nvidia/cuda:12.0"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_default_parameters(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"saveTemplate": {"id": "tmpl-x", "name": "test"}}}
        )
        runpod_create_template("test", "image:latest")
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        inp = sent["variables"]["input"]
        assert inp["containerDiskInGb"] == 20
        assert inp["volumeInGb"] == 40
        assert inp["volumeMountPath"] == "/workspace"
        assert inp["ports"] == "22/tcp,8888/http"
        assert inp["dockerArgs"] == ""
        assert inp["isServerless"] is False
        # No env key when env=None
        assert "env" not in inp

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_custom_parameters(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"saveTemplate": {"id": "tmpl-c", "name": "custom"}}}
        )
        runpod_create_template(
            "custom",
            "image:latest",
            container_disk_gb=50,
            volume_gb=100,
            volume_mount_path="/data",
            ports="22/tcp",
            docker_args="--shm-size=16g",
            is_serverless=True,
        )
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        inp = sent["variables"]["input"]
        assert inp["containerDiskInGb"] == 50
        assert inp["volumeInGb"] == 100
        assert inp["volumeMountPath"] == "/data"
        assert inp["ports"] == "22/tcp"
        assert inp["dockerArgs"] == "--shm-size=16g"
        assert inp["isServerless"] is True

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_env_vars_sent_as_key_value_list(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"saveTemplate": {"id": "tmpl-e", "name": "env-test"}}}
        )
        runpod_create_template(
            "env-test",
            "image:latest",
            env={"WANDB_API_KEY": "abc123", "HF_TOKEN": "hf_xyz"},
        )
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        inp = sent["variables"]["input"]
        env_list = inp["env"]
        assert isinstance(env_list, list)
        assert len(env_list) == 2
        env_dict = {e["key"]: e["value"] for e in env_list}
        assert env_dict["WANDB_API_KEY"] == "abc123"
        assert env_dict["HF_TOKEN"] == "hf_xyz"

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_no_env_omits_env_key(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"saveTemplate": {"id": "tmpl-x", "name": "test"}}}
        )
        runpod_create_template("test", "image:latest")
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert "env" not in sent["variables"]["input"]

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_empty_env_dict_omits_env_key(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"saveTemplate": {"id": "tmpl-x", "name": "test"}}}
        )
        runpod_create_template("test", "image:latest", env={})
        req = mock_urlopen.call_args[0][0]
        sent = json.loads(req.data.decode("utf-8"))
        assert "env" not in sent["variables"]["input"]

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_returns_empty_dict_when_mutation_returns_null(
        self, mock_urlopen: MagicMock
    ) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"data": {"saveTemplate": None}}
        )
        result = runpod_create_template("test", "image:latest")
        assert result == {}

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_graphql_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_urlopen_response(
            {"errors": [{"message": "Template name already exists"}]}
        )
        with pytest.raises(FleetError, match="Template name already exists"):
            runpod_create_template("dup", "image:latest")

    @patch("crucible.fleet.providers.runpod.urlrequest.urlopen")
    def test_http_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = _make_http_error(500, "Internal Server Error")
        with pytest.raises(FleetError, match="500"):
            runpod_create_template("test", "image:latest")
