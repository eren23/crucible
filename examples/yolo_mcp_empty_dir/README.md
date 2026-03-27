# YOLO MCP Empty-Dir Demo

This example shows the smallest workable setup for running an external YOLO fine-tuning flow through Crucible's MCP tools without modifying any source code.

## What "empty dir" means here

Strict empty-dir MCP-only bootstrapping is not supported today. Crucible can run a predeclared external project over MCP, but it does not yet have MCP tools for:

- creating `crucible.yaml`
- creating `.crucible/projects/<name>.yaml`

So the practical boundary is:

1. create a new directory
2. copy these two config files into it
3. do everything else through MCP

## Files

- `crucible.yaml` configures RunPod and leaderboard ranking for parsed detection metrics
- `.crucible/projects/yolo-demo.yaml` declares the external project that MCP will provision, bootstrap, and run

## MCP flow

Start the MCP server from the example directory:

```bash
cd examples/yolo_mcp_empty_dir
crucible mcp serve
```

Then run this sequence from your MCP client:

1. `config_get_project()`
2. `list_projects()`
3. `provision_project(project_name="yolo-demo", count=1)`
4. `fleet_refresh()` until the node has SSH info
5. `bootstrap_project(project_name="yolo-demo")`
6. `run_project(project_name="yolo-demo", overrides={...})`
7. `get_fleet_status(include_metrics=true)` while training
8. `collect_project_results(run_id=...)`
9. `get_leaderboard(top_n=3)`

## Suggested runs

Run these sequentially, collecting each before launching the next:

- smoke: `MODEL=yolov8n.pt`, `DATA=coco8.yaml`, `EPOCHS=3`, `RUN_NAME=yolo-smoke`
- baseline: `MODEL=yolov8n.pt`, `DATA=coco8.yaml`, `EPOCHS=10`, `RUN_NAME=yolo-baseline`
- full: `MODEL=yolov8n.pt`, `DATA=coco8.yaml`, `EPOCHS=30`, `RUN_NAME=yolo-full`

## Important caveats

- This template emits `metric:*` lines from Ultralytics `results.csv` after training so `collect_project_results()` can parse structured metrics without custom code.
- Keep this demo single-threaded: launch one run, wait, collect it, then launch the next.
- `wandb` is still installed so you can inspect runs separately if you want, but Crucible ranking for this demo uses stdout metrics.

## Environment

This template assumes these variables are available to the MCP server process:

- `RUNPOD_API_KEY`
- `WANDB_API_KEY`
- optionally `WANDB_ENTITY`
