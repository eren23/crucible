"""RunPod REST API provider: provision, destroy, refresh, wait, inventory_record_from_api.

Uses stdlib ``urllib`` only -- no ``requests`` dependency required.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from crucible.core.errors import FleetError
from crucible.core.log import log_info, log_success, log_warn, utc_now_iso
from crucible.fleet.inventory import BAD_API_STATES
from crucible.fleet.provider import FleetProvider
from crucible.fleet.sync import remote_exec, ssh_ok

RUNPOD_REST_BASE = os.environ.get("RUNPOD_REST_BASE", "https://rest.runpod.io/v1")
DEFAULT_IMAGE_NAME = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
DEFAULT_GPU_TYPE_IDS = ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090"]
DEFAULT_OVERNIGHT_GPU_TYPE_IDS = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090"]
DEFAULT_PORTS = ["22/tcp", "8888/http"]
DEFAULT_PUBLIC_KEY_PATH = os.environ.get(
    "RUNPOD_SSH_PUBLIC_KEY_PATH", "~/.ssh/id_ed25519_runpod.pub",
)


# ---------------------------------------------------------------------------
# API helpers (module-level, usable without a provider instance)
# ---------------------------------------------------------------------------

def runpod_api_key() -> str:
    """Read the RunPod API key from the environment."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise FleetError("RUNPOD_API_KEY is required for automatic pod provisioning.")
    return api_key


def read_public_key(path_text: str) -> str:
    """Read and validate an SSH public key file."""
    path = Path(path_text).expanduser()
    if not path.exists():
        raise FleetError(f"RunPod SSH public key not found: {path}")
    value = path.read_text(encoding="utf-8").strip()
    if not value.startswith("ssh-"):
        raise FleetError(f"Unexpected SSH public key format in {path}")
    return value


def runpod_request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
) -> Any:
    """Perform a single RunPod REST API call."""
    api_key = runpod_api_key()
    url = RUNPOD_REST_BASE.rstrip("/") + path
    if query:
        encoded = urlparse.urlencode(
            {k: v for k, v in query.items() if v is not None}, doseq=True,
        )
        if encoded:
            url += "?" + encoded
    body = None
    headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urlrequest.Request(url, data=body, headers=headers, method=method)
    try:
        with urlrequest.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            if not raw:
                return None
            return json.loads(raw)
    except urlerror.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise FleetError(
            f"RunPod API {method} {path} failed: {exc.code} {body_text}",
        ) from exc
    except urlerror.URLError as exc:
        raise FleetError(f"RunPod API {method} {path} failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Low-level API operations
# ---------------------------------------------------------------------------

def runpod_list_api_pods(*, name: str | None = None) -> list[dict[str, Any]]:
    """List pods from the RunPod REST API."""
    payload = runpod_request(
        "GET", "/pods",
        query={"name": name, "computeType": "GPU", "includeMachine": "true"},
    )
    if not isinstance(payload, list):
        raise FleetError("Unexpected RunPod /pods response: expected a list.")
    return payload


def runpod_get_api_pod(pod_id: str) -> dict[str, Any]:
    """Get a single pod's details."""
    payload = runpod_request("GET", f"/pods/{pod_id}")
    if not isinstance(payload, dict):
        raise FleetError(f"Unexpected RunPod /pods/{pod_id} response: expected a dict.")
    return payload


def runpod_delete_api_pod(pod_id: str) -> None:
    """Delete a pod by ID."""
    runpod_request("DELETE", f"/pods/{pod_id}")


def destroy_stale_named_pods(
    *,
    prefixes: list[str],
    keep_ids: set[str] | None = None,
) -> int:
    """Destroy RunPod pods whose names match *prefixes*, excluding *keep_ids*."""
    keep_ids = keep_ids or set()
    destroyed = 0
    for pod in runpod_list_api_pods():
        name = str(pod.get("name") or "")
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        pod_id = str(pod.get("id") or "")
        if pod_id in keep_ids:
            continue
        runpod_delete_api_pod(pod_id)
        destroyed += 1
    return destroyed


def default_cloud_types(*, interruptible: bool) -> list[str]:
    """Return the cloud-type fallback order based on interruptibility."""
    return ["COMMUNITY", "SECURE"] if interruptible else ["SECURE", "COMMUNITY"]


# ---------------------------------------------------------------------------
# Pod payload / port parsing
# ---------------------------------------------------------------------------

def parse_port_mapping(raw: dict[str, Any], internal_port: str) -> int | None:
    """Extract a mapped external port number from the API response."""
    mappings = raw.get("portMappings") or {}
    if isinstance(mappings, dict):
        value = mappings.get(str(internal_port))
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def build_pod_payload(
    *,
    name: str,
    gpu_type_ids: list[str],
    image_name: str,
    cloud_type: str,
    interruptible: bool,
    container_disk_gb: int,
    volume_gb: int,
    volume_mount_path: str,
    public_key: str,
    ports: list[str],
) -> dict[str, Any]:
    """Construct the JSON body for a POST /pods request."""
    return {
        "allowedCudaVersions": ["12.8"],
        "cloudType": cloud_type,
        "computeType": "GPU",
        "containerDiskInGb": container_disk_gb,
        "env": {
            "JUPYTER_PASSWORD": uuid.uuid4().hex[:16],
            "PUBLIC_KEY": public_key,
            "SSH_PUBLIC_KEY": public_key,
        },
        "gpuCount": 1,
        "gpuTypeIds": gpu_type_ids,
        "gpuTypePriority": "availability",
        "imageName": image_name,
        "interruptible": interruptible,
        "minRAMPerGPU": 8,
        "minVCPUPerGPU": 2,
        "name": name,
        "ports": ports,
        "supportPublicIp": True,
        "volumeInGb": volume_gb,
        "volumeMountPath": volume_mount_path,
    }


def create_api_pod(
    *,
    name: str,
    gpu_type_ids: list[str],
    image_name: str,
    cloud_type: str,
    interruptible: bool,
    container_disk_gb: int,
    volume_gb: int,
    volume_mount_path: str,
    public_key: str,
    ports: list[str],
) -> dict[str, Any]:
    """Create a RunPod pod and return the raw API response."""
    payload = build_pod_payload(
        name=name, gpu_type_ids=gpu_type_ids, image_name=image_name,
        cloud_type=cloud_type, interruptible=interruptible,
        container_disk_gb=container_disk_gb, volume_gb=volume_gb,
        volume_mount_path=volume_mount_path, public_key=public_key,
        ports=ports,
    )
    created = runpod_request("POST", "/pods", payload=payload)
    if not isinstance(created, dict):
        raise FleetError("Unexpected RunPod pod creation response: expected a dict.")
    return created


# ---------------------------------------------------------------------------
# Node record translation
# ---------------------------------------------------------------------------

def inventory_record_from_api(
    raw: dict[str, Any],
    *,
    previous: dict[str, Any] | None = None,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a RunPod API response into a fleet node record dict.

    Parameters
    ----------
    defaults : dict, optional
        Provider defaults from ``crucible.yaml`` ``provider.defaults``.
        Used as fallbacks for ``workspace_path``, ``python_bin``,
        ``env_source`` when no previous record exists.
    """
    previous = previous or {}
    defaults = defaults or {}
    gpu = raw.get("gpu") or {}
    api_state = str(raw.get("desiredStatus") or raw.get("status") or "").lower()
    local_state = str(previous.get("state") or "").lower()
    state = local_state or api_state or "new"
    if local_state in {"", "creating", "new"} and api_state:
        state = api_state
    return {
        "name": raw.get("name") or previous.get("name") or raw["id"],
        "node_id": raw["id"],
        "pod_id": raw["id"],  # backward compat
        "gpu": gpu.get("displayName") or previous.get("gpu") or "-",
        "interruptible": bool(
            raw.get("interruptible", previous.get("interruptible", True)),
        ),
        "cost_per_hr": float(
            raw.get("adjustedCostPerHr")
            or raw.get("costPerHr")
            or previous.get("cost_per_hr")
            or 0,
        ),
        "ssh_host": raw.get("publicIp") or previous.get("ssh_host") or "",
        "ssh_port": parse_port_mapping(raw, "22") or previous.get("ssh_port") or 22,
        "user": previous.get("user", "root"),
        "ssh_key": previous.get("ssh_key", defaults.get("ssh_key", "~/.ssh/id_ed25519_runpod")),
        "workspace_path": previous.get("workspace_path", defaults.get("workspace_path", "/workspace/project")),
        "python_bin": previous.get("python_bin", defaults.get("python_bin", "python3")),
        "env_source": previous.get("env_source", defaults.get("env_source", ".env.local")),
        "state": state,
        "api_state": api_state,
        "env_ready": bool(previous.get("env_ready", False)),
        "dataset_ready": bool(previous.get("dataset_ready", False)),
        "git_sha": previous.get("git_sha"),
        "last_seen_at": previous.get("last_seen_at"),
        "replacement": bool(previous.get("replacement", False)),
        "provider": "runpod",
    }


# ---------------------------------------------------------------------------
# RunPodProvider class
# ---------------------------------------------------------------------------

class RunPodProvider(FleetProvider):
    """Full RunPod REST API fleet provider.

    Implements provisioning, destruction, refresh (per-pod API query),
    and two-phase wait (API endpoint ready, then SSH ready).
    """

    provider_name = "runpod"

    def __init__(
        self,
        *,
        image_name: str = DEFAULT_IMAGE_NAME,
        gpu_type_ids: list[str] | None = None,
        cloud_types: list[str] | None = None,
        interruptible: bool = True,
        container_disk_gb: int = 20,
        volume_gb: int = 40,
        volume_mount_path: str = "/workspace",
        public_key_path: str = DEFAULT_PUBLIC_KEY_PATH,
        ports: list[str] | None = None,
        ssh_key: str = "~/.ssh/id_ed25519_runpod",
        defaults: dict[str, Any] | None = None,
    ) -> None:
        self.image_name = image_name
        self.gpu_type_ids = gpu_type_ids or list(DEFAULT_GPU_TYPE_IDS)
        self.cloud_types = cloud_types or default_cloud_types(interruptible=interruptible)
        self.interruptible = interruptible
        self.container_disk_gb = container_disk_gb
        self.volume_gb = volume_gb
        self.volume_mount_path = volume_mount_path
        self.public_key_path = public_key_path
        self.ports = ports or list(DEFAULT_PORTS)
        self.ssh_key = ssh_key
        self.defaults = defaults or {}

    # -- FleetProvider interface ------------------------------------------

    def provision(
        self,
        *,
        count: int,
        name_prefix: str,
        start_index: int = 1,
        replacement: bool = False,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        public_key = read_public_key(self.public_key_path)
        # Apply per-call overrides (e.g. from project spec pod config)
        eff_container_disk = kwargs.pop("container_disk_gb", self.container_disk_gb)
        eff_volume_gb = kwargs.pop("volume_gb", self.volume_gb)
        eff_gpu_type_ids = kwargs.pop("gpu_type_ids", None)
        if isinstance(eff_gpu_type_ids, str):
            eff_gpu_type_ids = [eff_gpu_type_ids]
        eff_image = kwargs.pop("image_name", self.image_name)

        created: list[dict[str, Any]] = []
        for index in range(start_index, start_index + count):
            ordinal = index - start_index + 1
            name = f"{name_prefix}-{index:02d}"
            last_error: str | None = None
            log_info(f"Creating pod {ordinal}/{count}: {name} (container={eff_container_disk}GB, volume={eff_volume_gb}GB)")
            for cloud_type in self.cloud_types:
                try:
                    raw = create_api_pod(
                        name=name,
                        gpu_type_ids=eff_gpu_type_ids or self.gpu_type_ids,
                        image_name=eff_image,
                        cloud_type=cloud_type,
                        interruptible=self.interruptible,
                        container_disk_gb=eff_container_disk,
                        volume_gb=eff_volume_gb,
                        volume_mount_path=self.volume_mount_path,
                        public_key=public_key,
                        ports=self.ports,
                    )
                    node = inventory_record_from_api(raw, defaults=self.defaults)
                    node["state"] = "creating"
                    node["replacement"] = replacement
                    node["ssh_key"] = self.ssh_key
                    created.append(node)
                    log_success(f"Created {name} ({cloud_type})")
                    break
                except FleetError as exc:
                    last_error = str(exc)
                    log_warn(
                        f"{name} create attempt failed on {cloud_type}: {last_error}",
                    )
            else:
                raise FleetError(last_error or f"Failed to create pod {name}")
        return created

    def destroy(
        self,
        nodes: list[dict[str, Any]],
        *,
        selected_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        remaining: list[dict[str, Any]] = []
        for node in nodes:
            if selected_names and node["name"] not in selected_names:
                remaining.append(node)
                continue
            pod_id = node.get("pod_id") or node.get("node_id")
            if pod_id:
                try:
                    runpod_delete_api_pod(pod_id)
                except FleetError:
                    pass  # Best-effort: pod may already be gone
        return remaining

    def refresh(self, nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        previous_by_id: dict[str, dict[str, Any]] = {}
        for n in nodes:
            nid = n.get("pod_id") or n.get("node_id")
            if nid:
                previous_by_id[nid] = n
        refreshed: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for node in nodes:
            pod_id = node.get("pod_id") or node.get("node_id")
            if not pod_id:
                refreshed.append(node)
                continue
            seen_ids.add(pod_id)
            try:
                api = runpod_get_api_pod(pod_id)
                refreshed.append(
                    inventory_record_from_api(api, previous=previous_by_id.get(pod_id), defaults=self.defaults),
                )
            except FleetError:
                failed = dict(previous_by_id.get(pod_id, node))
                failed["api_state"] = "lost"
                failed["state"] = "lost"
                refreshed.append(failed)

        # Reconcile: add any pods from the API that aren't in inventory (orphan recovery)
        try:
            all_pods = self.list_all_pods()
            for pod in all_pods:
                pod_id = str(pod.get("id") or "")
                if pod_id and pod_id not in seen_ids:
                    log_info(f"Reconciled orphan pod: {pod.get('name', '?')} ({pod_id})")
                    refreshed.append(
                        inventory_record_from_api(pod, previous=None, defaults=self.defaults),
                    )
        except Exception:
            pass  # Best-effort reconciliation

        return refreshed

    def wait_ready(
        self,
        nodes: list[dict[str, Any]],
        *,
        timeout_seconds: int = 900,
        poll_seconds: int = 15,
        stalled_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        """Wait for API endpoints to be reachable, then for SSH connectivity."""
        stalled_seconds = stalled_seconds or min(timeout_seconds, 120)
        nodes = self._wait_for_api_ready(
            nodes,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            stalled_seconds=stalled_seconds,
        )
        nodes = self._wait_for_ssh_ready(
            nodes,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            stalled_seconds=stalled_seconds,
        )
        return nodes

    # -- API-level operations (beyond inventory) --------------------------

    def list_all_pods(self) -> list[dict[str, Any]]:
        """List all pods from the RunPod API, regardless of local inventory."""
        return runpod_list_api_pods()

    def destroy_all_pods(self) -> list[str]:
        """Destroy ALL pods from the RunPod account. Returns list of destroyed pod IDs."""
        pods = self.list_all_pods()
        destroyed: list[str] = []
        for pod in pods:
            pod_id = str(pod.get("id") or "")
            if not pod_id:
                continue
            try:
                runpod_delete_api_pod(pod_id)
                destroyed.append(pod_id)
            except FleetError:
                pass  # Best-effort: pod may already be gone
        return destroyed

    def destroy_pods_by_id(self, pod_ids: list[str]) -> list[str]:
        """Destroy specific pods by ID. Returns list of destroyed pod IDs."""
        destroyed: list[str] = []
        for pod_id in pod_ids:
            try:
                runpod_delete_api_pod(pod_id)
                destroyed.append(pod_id)
            except FleetError:
                pass  # Best-effort: pod may already be gone
        return destroyed

    # -- Internal helpers -------------------------------------------------

    def _wait_for_api_ready(
        self,
        nodes: list[dict[str, Any]],
        *,
        timeout_seconds: int,
        poll_seconds: int,
        stalled_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        deadline = time.time() + timeout_seconds
        current = nodes
        last_ready = -1
        last_progress = time.time()
        while time.time() < deadline:
            current = self.refresh(current)
            pending = [
                n for n in current if not n.get("ssh_host") or not n.get("ssh_port")
            ]
            ready = len(current) - len(pending)
            if ready != last_ready:
                log_info(f"API ready {ready}/{len(current)}")
                last_ready = ready
                last_progress = time.time()
            if not pending:
                return current
            if (
                stalled_seconds is not None
                and time.time() - last_progress >= stalled_seconds
            ):
                names = ", ".join(n["name"] for n in pending)
                raise TimeoutError(
                    f"API readiness stalled for {stalled_seconds}s: {names}",
                )
            time.sleep(poll_seconds)
        names = ", ".join(
            n["name"] for n in current
            if not n.get("ssh_host") or not n.get("ssh_port")
        )
        raise TimeoutError(f"Timed out waiting for RunPod endpoints: {names}")

    def _wait_for_ssh_ready(
        self,
        nodes: list[dict[str, Any]],
        *,
        timeout_seconds: int,
        poll_seconds: int,
        stalled_seconds: int | None = None,
    ) -> list[dict[str, Any]]:
        deadline = time.time() + timeout_seconds
        pending = {n["name"]: n for n in nodes}
        total = len(nodes)
        last_ready = -1
        last_progress = time.time()
        while time.time() < deadline and pending:
            next_pending: dict[str, Any] = {}
            for node in pending.values():
                if ssh_ok(node):
                    node["last_seen_at"] = utc_now_iso()
                    node["state"] = "new"
                    continue
                next_pending[node["name"]] = node
            pending = next_pending
            ready = total - len(pending)
            if ready != last_ready:
                log_info(f"SSH ready {ready}/{total}")
                last_ready = ready
                last_progress = time.time()
            if (
                stalled_seconds is not None
                and pending
                and time.time() - last_progress >= stalled_seconds
            ):
                names = ", ".join(n["name"] for n in pending.values())
                raise TimeoutError(
                    f"SSH readiness stalled for {stalled_seconds}s: {names}",
                )
            if pending:
                time.sleep(poll_seconds)
        if pending:
            names = ", ".join(n["name"] for n in pending.values())
            raise TimeoutError(f"Timed out waiting for SSH readiness: {names}")
        return nodes
