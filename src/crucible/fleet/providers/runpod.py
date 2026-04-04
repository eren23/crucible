"""RunPod fleet provider: REST + GraphQL APIs.

Uses stdlib ``urllib`` only -- no ``requests`` dependency required.
REST API (``rest.runpod.io/v1``) handles pod CRUD.
GraphQL API (``api.runpod.io/graphql``) handles stop/start, network volumes,
GPU availability, and templates.
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
RUNPOD_GRAPHQL_URL = os.environ.get("RUNPOD_GRAPHQL_URL", "https://api.runpod.io/graphql")
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


def runpod_graphql(
    query: str,
    variables: dict[str, Any] | None = None,
    *,
    timeout: int = 60,
) -> dict[str, Any]:
    """Perform a RunPod GraphQL query/mutation.

    Returns the ``data`` dict from the response.  Raises ``FleetError``
    on HTTP errors or GraphQL-level errors.
    """
    api_key = runpod_api_key()
    payload: dict[str, Any] = {"query": query}
    if variables:
        payload["variables"] = variables
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    req = urlrequest.Request(RUNPOD_GRAPHQL_URL, data=body, headers=headers, method="POST")
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            result = json.loads(raw)
    except urlerror.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise FleetError(f"RunPod GraphQL failed: {exc.code} {body_text}") from exc
    except urlerror.URLError as exc:
        raise FleetError(f"RunPod GraphQL failed: {exc}") from exc

    errors = result.get("errors")
    if errors:
        msg = errors[0].get("message", str(errors[0]))
        raise FleetError(f"RunPod GraphQL error: {msg}")
    data = result.get("data")
    if data is None:
        raise FleetError(f"RunPod GraphQL returned no data: {raw[:200]}")
    return data


# ---------------------------------------------------------------------------
# Low-level API operations (REST)
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
# GraphQL operations: pod lifecycle (stop / start)
# ---------------------------------------------------------------------------

def runpod_stop_pod(pod_id: str) -> dict[str, Any]:
    """Stop a running pod (preserves disk).  Returns the mutation result."""
    data = runpod_graphql(
        """
        mutation stopPod($podId: String!) {
            podStop(input: {podId: $podId}) {
                id
                desiredStatus
                lastStatusChange
            }
        }
        """,
        variables={"podId": pod_id},
        timeout=120,
    )
    return data.get("podStop") or {}


def runpod_start_pod(pod_id: str) -> dict[str, Any]:
    """Resume a stopped on-demand pod.  Returns the mutation result."""
    data = runpod_graphql(
        """
        mutation podResume($podId: String!) {
            podResume(input: {podId: $podId}) {
                id
                costPerHr
                desiredStatus
                lastStatusChange
            }
        }
        """,
        variables={"podId": pod_id},
        timeout=120,
    )
    return data.get("podResume") or {}


def runpod_start_spot_pod(
    pod_id: str,
    bid_per_gpu: float,
    gpu_count: int = 1,
) -> dict[str, Any]:
    """Resume a stopped spot/interruptible pod with a bid price."""
    data = runpod_graphql(
        """
        mutation podBidResume($podId: String!, $bidPerGpu: Float!, $gpuCount: Int!) {
            podBidResume(input: {podId: $podId, bidPerGpu: $bidPerGpu, gpuCount: $gpuCount}) {
                id
                costPerHr
                desiredStatus
                lastStatusChange
            }
        }
        """,
        variables={"podId": pod_id, "bidPerGpu": bid_per_gpu, "gpuCount": gpu_count},
        timeout=120,
    )
    return data.get("podBidResume") or {}


# ---------------------------------------------------------------------------
# GraphQL operations: network volumes
# ---------------------------------------------------------------------------

def runpod_list_network_volumes() -> list[dict[str, Any]]:
    """List all network volumes in the account."""
    data = runpod_graphql(
        """
        query getNetworkVolumes {
            myself {
                networkVolumes {
                    id
                    dataCenterId
                    name
                    size
                }
            }
        }
        """,
    )
    myself = data.get("myself") or {}
    return myself.get("networkVolumes") or []


def runpod_create_network_volume(
    name: str,
    size_gb: int,
    datacenter_id: str,
) -> dict[str, Any]:
    """Create a persistent network volume.  Returns ``{id, name, size, dataCenterId}``."""
    data = runpod_graphql(
        """
        mutation createNetworkVolume($input: CreateNetworkVolumeInput!) {
            createNetworkVolume(input: $input) {
                id
                name
                size
                dataCenterId
            }
        }
        """,
        variables={
            "input": {
                "name": name,
                "size": size_gb,
                "dataCenterId": datacenter_id,
            },
        },
    )
    return data.get("createNetworkVolume") or {}


def runpod_delete_network_volume(volume_id: str) -> None:
    """Delete a network volume by ID."""
    data = runpod_graphql(
        """
        mutation deleteNetworkVolume($id: String!) {
            deleteNetworkVolume(input: {id: $id})
        }
        """,
        variables={"id": volume_id},
    )
    if not data.get("deleteNetworkVolume"):
        raise FleetError(f"Failed to delete network volume {volume_id}")


# ---------------------------------------------------------------------------
# GraphQL operations: GPU availability & pricing
# ---------------------------------------------------------------------------

def runpod_list_gpu_types(
    gpu_count: int = 1,
    secure_cloud: bool | None = None,
) -> list[dict[str, Any]]:
    """List available GPU types with lowest spot & on-demand prices."""
    gql_input: dict[str, Any] = {"gpuCount": gpu_count}
    if secure_cloud is not None:
        gql_input["secureCloud"] = secure_cloud
    data = runpod_graphql(
        """
        query LowestPrice($input: GpuLowestPriceInput!) {
            gpuTypes {
                lowestPrice(input: $input) {
                    gpuName
                    gpuTypeId
                    minimumBidPrice
                    uninterruptablePrice
                    minMemory
                    minVcpu
                }
            }
        }
        """,
        variables={"input": gql_input},
    )
    gpu_types = data.get("gpuTypes") or []
    # Flatten: each element is {"lowestPrice": {...}} — extract the inner dict
    result: list[dict[str, Any]] = []
    for gt in gpu_types:
        price = gt.get("lowestPrice")
        if price and price.get("gpuName"):
            result.append(price)
    return result


# ---------------------------------------------------------------------------
# GraphQL operations: templates
# ---------------------------------------------------------------------------

def runpod_list_templates() -> list[dict[str, Any]]:
    """List the user's own templates."""
    data = runpod_graphql(
        """
        query getTemplates {
            myself {
                podTemplates {
                    id
                    name
                    imageName
                    dockerArgs
                    containerDiskInGb
                    volumeInGb
                    volumeMountPath
                    ports
                    isServerless
                }
            }
        }
        """,
    )
    myself = data.get("myself") or {}
    return myself.get("podTemplates") or []


def runpod_create_template(
    name: str,
    image: str,
    *,
    container_disk_gb: int = 20,
    volume_gb: int = 40,
    volume_mount_path: str = "/workspace",
    ports: str = "22/tcp,8888/http",
    docker_args: str = "",
    env: dict[str, str] | None = None,
    is_serverless: bool = False,
) -> dict[str, Any]:
    """Create a pod template.  Returns ``{id, name}``."""
    gql_input: dict[str, Any] = {
        "name": name,
        "imageName": image,
        "containerDiskInGb": container_disk_gb,
        "volumeInGb": volume_gb,
        "volumeMountPath": volume_mount_path,
        "ports": ports,
        "dockerArgs": docker_args,
        "isServerless": is_serverless,
    }
    if env:
        gql_input["env"] = [{"key": k, "value": v} for k, v in env.items()]
    data = runpod_graphql(
        """
        mutation saveTemplate($input: SaveTemplateInput!) {
            saveTemplate(input: $input) {
                id
                name
            }
        }
        """,
        variables={"input": gql_input},
    )
    return data.get("saveTemplate") or {}


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
    gpu_count: int = 1,
    network_volume_id: str | None = None,
    template_id: str | None = None,
) -> dict[str, Any]:
    """Construct the JSON body for a POST /pods request."""
    if container_disk_gb < 0 or volume_gb < 0:
        raise FleetError("RunPod disk sizes must be non-negative.")
    if not network_volume_id and volume_gb > 0 and volume_gb < container_disk_gb:
        raise FleetError(
            "RunPod volume size must be greater than or equal to container disk size.",
        )
    payload: dict[str, Any] = {
        "allowedCudaVersions": ["12.8"],
        "cloudType": cloud_type,
        "computeType": "GPU",
        "containerDiskInGb": container_disk_gb,
        "env": {
            "JUPYTER_PASSWORD": uuid.uuid4().hex[:16],
            # SSH key is set via top-level "publicKey" field (not env vars).
            # Env-var approach (PUBLIC_KEY / SSH_PUBLIC_KEY) is unreliable on
            # community pods — depends on image entrypoint parsing them.
        },
        "gpuCount": gpu_count,
        "gpuTypeIds": gpu_type_ids,
        "gpuTypePriority": "availability",
        "imageName": image_name,
        "interruptible": interruptible,
        "minRAMPerGPU": 8,
        "minVCPUPerGPU": 2,
        "name": name,
        "ports": ports,
        # RunPod API-level SSH key injection — reliable across all pod types.
        "publicKey": public_key,
        "supportPublicIp": True,
        "volumeInGb": volume_gb,
        "volumeMountPath": volume_mount_path,
    }
    if network_volume_id:
        payload["networkVolumeId"] = network_volume_id
    if template_id:
        payload["templateId"] = template_id
    return payload


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
    gpu_count: int = 1,
    network_volume_id: str | None = None,
    template_id: str | None = None,
) -> dict[str, Any]:
    """Create a RunPod pod and return the raw API response."""
    payload = build_pod_payload(
        name=name, gpu_type_ids=gpu_type_ids, image_name=image_name,
        cloud_type=cloud_type, interruptible=interruptible,
        container_disk_gb=container_disk_gb, volume_gb=volume_gb,
        volume_mount_path=volume_mount_path, public_key=public_key,
        ports=ports, gpu_count=gpu_count,
        network_volume_id=network_volume_id, template_id=template_id,
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
    # Intentional stop/start lifecycle transitions
    if api_state in {"stopped", "exited"} and local_state not in {"stopped", "exited", "starting"}:
        state = "stopped"
    elif local_state == "stopped" and api_state in {"running", "ready", ""}:
        state = "starting"
    elif local_state == "starting" and api_state in {"running", "ready"}:
        # Pod resumed — transition to "new" (SSH wait will promote to "ready")
        state = "new"
    return {
        "name": raw.get("name") or previous.get("name") or raw["id"],
        "node_id": raw["id"],
        "pod_id": raw["id"],  # backward compat
        "gpu": gpu.get("displayName") or previous.get("gpu") or "-",
        "gpu_count": raw.get("gpuCount") or previous.get("gpu_count") or 1,
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
        "network_volume_id": raw.get("networkVolumeId") or previous.get("network_volume_id") or "",
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
        gpu_count: int = 1,
        network_volume_id: str = "",
        template_id: str = "",
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
        self.gpu_count = gpu_count
        self.network_volume_id = network_volume_id
        self.template_id = template_id

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
        eff_interruptible = bool(kwargs.pop("interruptible", self.interruptible))
        eff_cloud_types = kwargs.pop("cloud_types", None)
        eff_gpu_count = kwargs.pop("gpu_count", self.gpu_count)
        eff_network_volume_id = kwargs.pop("network_volume_id", self.network_volume_id) or None
        eff_template_id = kwargs.pop("template_id", self.template_id) or None
        if eff_cloud_types is None:
            eff_cloud_types = (
                list(self.cloud_types)
                if eff_interruptible == self.interruptible
                else default_cloud_types(interruptible=eff_interruptible)
            )
        else:
            eff_cloud_types = list(eff_cloud_types)
        if not eff_interruptible:
            eff_cloud_types = [ct for ct in eff_cloud_types if str(ct).upper() != "COMMUNITY"] or ["SECURE"]

        created: list[dict[str, Any]] = []
        for index in range(start_index, start_index + count):
            ordinal = index - start_index + 1
            name = f"{name_prefix}-{index:02d}"
            last_error: str | None = None
            log_info(f"Creating pod {ordinal}/{count}: {name} (container={eff_container_disk}GB, volume={eff_volume_gb}GB, gpus={eff_gpu_count})")
            for cloud_type in eff_cloud_types:
                try:
                    raw = create_api_pod(
                        name=name,
                        gpu_type_ids=eff_gpu_type_ids or self.gpu_type_ids,
                        image_name=eff_image,
                        cloud_type=cloud_type,
                        interruptible=eff_interruptible,
                        container_disk_gb=eff_container_disk,
                        volume_gb=eff_volume_gb,
                        volume_mount_path=self.volume_mount_path,
                        public_key=public_key,
                        ports=self.ports,
                        gpu_count=eff_gpu_count,
                        network_volume_id=eff_network_volume_id,
                        template_id=eff_template_id,
                    )
                    node = inventory_record_from_api(raw, defaults=self.defaults)
                    node["state"] = "creating"
                    node["replacement"] = replacement
                    node["ssh_key"] = self.ssh_key
                    if eff_network_volume_id:
                        node["network_volume_id"] = eff_network_volume_id
                    created.append(node)
                    log_success(f"Created {name} ({cloud_type}, {eff_gpu_count} GPU(s))")
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

    # -- Stop / Start lifecycle -------------------------------------------

    def stop(
        self,
        nodes: list[dict[str, Any]],
        *,
        selected_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Stop running pods.  Preserves disk and bootstrap state."""
        updated: list[dict[str, Any]] = []
        for node in nodes:
            if selected_names and node["name"] not in selected_names:
                updated.append(node)
                continue
            pod_id = node.get("pod_id") or node.get("node_id")
            if not pod_id:
                updated.append(node)
                continue
            try:
                runpod_stop_pod(pod_id)
                node = dict(node)
                node["state"] = "stopped"
                node["api_state"] = "stopped"
                log_info(f"Stopped {node['name']} ({pod_id})")
            except FleetError as exc:
                log_warn(f"Failed to stop {node.get('name', '?')}: {exc}")
            updated.append(node)
        return updated

    def start(
        self,
        nodes: list[dict[str, Any]],
        *,
        selected_names: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Start stopped pods and wait for SSH readiness."""
        started: list[dict[str, Any]] = []
        for node in nodes:
            if selected_names and node["name"] not in selected_names:
                started.append(node)
                continue
            pod_id = node.get("pod_id") or node.get("node_id")
            if not pod_id:
                started.append(node)
                continue
            try:
                if node.get("interruptible"):
                    cost = node.get("cost_per_hr", 0.0)
                    bid = max(cost, 0.10)  # floor: never bid below $0.10/gpu
                    gpu_count = node.get("gpu_count", 1)
                    runpod_start_spot_pod(pod_id, bid_per_gpu=bid, gpu_count=gpu_count)
                else:
                    runpod_start_pod(pod_id)
                node = dict(node)
                node["state"] = "starting"
                node["api_state"] = "starting"
                log_info(f"Started {node['name']} ({pod_id})")
            except FleetError as exc:
                log_warn(f"Failed to start {node.get('name', '?')}: {exc}")
            started.append(node)
        # Wait for SSH on the started nodes, preserving original order
        starting_names = {n["name"] for n in started if n.get("state") == "starting"}
        if starting_names:
            waited = self.wait_ready(
                [n for n in started if n["name"] in starting_names],
                timeout_seconds=600,
            )
            wait_map = {n["name"]: n for n in waited}
            started = [wait_map.get(n["name"], n) for n in started]
        return started

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
