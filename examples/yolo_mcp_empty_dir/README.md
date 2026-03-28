# YOLO MCP Demo -- Complete Setup Guide

Run YOLO object detection training on a cloud GPU through Crucible's MCP tools. No code modifications needed.

## What this demo does

1. Provisions an RTX 4090 GPU on RunPod
2. Clones the Ultralytics YOLO repo and installs it
3. Trains YOLOv8n on the COCO8 dataset
4. Collects mAP metrics back to your local machine

Total cost: ~$0.10-0.50 depending on run length. Training time: 2-15 minutes.

## Prerequisites

- **RunPod account** with API key ([runpod.io](https://www.runpod.io/console/user/settings))
- **Weights & Biases account** with API key ([wandb.ai/authorize](https://wandb.ai/authorize))
- **SSH key** at `~/.ssh/id_ed25519_runpod` (added to your RunPod account)
- **Python 3.9+** with Crucible installed

### Generate SSH key (if you don't have one)

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_runpod -N ""
```

Then add the public key (`~/.ssh/id_ed25519_runpod.pub`) to RunPod:
Settings -> SSH Public Keys -> Add SSH Key

### Install Crucible

```bash
pip install crucible-ml[mcp]
```

## Setup

### 1. Create your working directory

```bash
mkdir yolo-demo && cd yolo-demo
crucible init
```

### 2. Copy the demo config files

```bash
# Copy the project spec
cp /path/to/crucible/examples/yolo_mcp_empty_dir/.crucible/projects/yolo-demo.yaml \
   .crucible/projects/yolo-demo.yaml

# Copy the crucible config (or edit the generated one)
cp /path/to/crucible/examples/yolo_mcp_empty_dir/crucible.yaml ./crucible.yaml
```

Or if you cloned the repo, just work directly from the example directory:
```bash
cd examples/yolo_mcp_empty_dir
```

### 3. Set up your secrets

```bash
cp .env.example .env
# Edit .env with your actual API keys
```

### 4. Connect to MCP

**Option A: Claude Code (recommended)**

Create `.mcp.json` in your project directory:
```json
{
  "mcpServers": {
    "crucible": {
      "command": "crucible",
      "args": ["mcp", "serve"]
    }
  }
}
```

Then start Claude Code from the project directory.

**Option B: Claude Desktop**

Add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "crucible": {
      "command": "/path/to/your/venv/bin/crucible",
      "args": ["mcp", "serve"],
      "cwd": "/path/to/your/yolo-demo"
    }
  }
}
```

## Running the Demo

Use these MCP tool calls in order:

### Step 1: Verify your setup
```json
{"tool": "list_projects"}
```
Should return `yolo-demo` in the list.

### Step 2: Provision a GPU
```json
{"tool": "provision_project", "arguments": {"project_name": "yolo-demo", "count": 1}}
```
Creates one RunPod instance with an RTX 4090.

### Step 3: Wait for SSH, then refresh
```json
{"tool": "fleet_refresh"}
```
Repeat every 30 seconds until the node shows `ssh_host` populated.

### Step 4: Bootstrap the environment
```json
{"tool": "bootstrap_project", "arguments": {"project_name": "yolo-demo"}}
```
This clones Ultralytics, creates a venv, and installs dependencies. Takes 2-5 minutes.

### Step 5: Launch training
```json
{
  "tool": "run_project",
  "arguments": {
    "project_name": "yolo-demo",
    "overrides": {
      "MODEL": "yolov8n.pt",
      "DATA": "coco8.yaml",
      "EPOCHS": "10",
      "RUN_NAME": "yolo-baseline"
    }
  }
}
```

### Step 6: Monitor (optional)
```json
{"tool": "get_fleet_status", "arguments": {"include_metrics": true}}
```

### Step 7: Collect results
```json
{"tool": "collect_project_results", "arguments": {"run_id": "<run_id from step 5>"}}
```
Returns parsed metrics: precision, recall, mAP50, mAP50-95.

### Step 8: Clean up
```json
{"tool": "destroy_nodes"}
```

## What Ultralytics parameters can you change?

Pass any of these as overrides in `run_project`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL` | `yolov8n.pt` | YOLO model checkpoint (yolov8n/s/m/l/x, yolo11n/s/m/l/x) |
| `DATA` | `coco8.yaml` | Dataset config (coco8, coco128, coco, or custom) |
| `EPOCHS` | `10` | Training epochs |
| `IMGSZ` | `640` | Image size |
| `BATCH` | `16` | Batch size |
| `RUN_NAME` | `mcp-demo` | Name for this run |

## How it works

Crucible's project runner:
1. Provisions a RunPod pod with the GPU and container image specified in the project spec
2. Clones the Ultralytics repo into the pod
3. Creates a Python venv and installs Ultralytics + wandb
4. Runs `yolo detect train` with your parameters
5. After training, parses `results.csv` and emits `metric:*` lines that Crucible collects
6. Results are rsynced back to your local machine

The project spec (`.crucible/projects/yolo-demo.yaml`) declares all of this declaratively -- no code to write.

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `RUNPOD_API_KEY not set` | Missing .env | Create `.env` with your API key |
| `SSH key not found` | Missing SSH key | Run `ssh-keygen` command above |
| `provision failed` | No GPU quota | Check RunPod dashboard for available GPUs |
| `bootstrap timed out` | Slow network | Retry `bootstrap_project` |
| `No metrics collected` | Training didn't finish | Check logs with `get_run_logs` |
