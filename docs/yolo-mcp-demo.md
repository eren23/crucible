---
layout: default
title: YOLO MCP Demo
---

# YOLO MCP Demo

This walkthrough shows the smallest practical way to fine-tune a YOLO model through Crucible's MCP tools from a fresh directory.

## The honest boundary

This is not strict zero-file startup.

Today, Crucible can use MCP to provision GPUs, bootstrap an external project, launch training, monitor it, and collect results. It does **not** yet expose MCP tools to create:

- `crucible.yaml`
- `.crucible/projects/<name>.yaml`

So the current "empty dir" story is:

1. create a new directory
2. drop in two small config files
3. use MCP for everything else

## Demo Shape

The example template lives in `examples/yolo_mcp_empty_dir/`.

It pins a small YOLO checkpoint and declares one external project:

- model: `yolov8n.pt`
- dataset: `coco8.yaml`
- provider: RunPod
- GPU: RTX 4090
- primary metric: `map50_95_b`

## MCP Sequence

Use these calls in order:

1. `config_get_project()`
2. `list_projects()`
3. `provision_project(project_name="yolo-demo", count=1)`
4. `fleet_refresh()` until the pod exposes SSH details
5. `bootstrap_project(project_name="yolo-demo")`
6. `run_project(project_name="yolo-demo", overrides={...})`
7. `get_fleet_status(include_metrics=true)` while the run is live
8. `collect_project_results(run_id=...)`
9. `get_leaderboard(top_n=3)` after all runs have been collected

## Recommended Run Set

Run these one at a time:

- smoke: 3 epochs
- baseline: 10 epochs
- full: 30 epochs

Use `run_project()` overrides like:

```json
{
  "project_name": "yolo-demo",
  "overrides": {
    "MODEL": "yolov8n.pt",
    "DATA": "coco8.yaml",
    "EPOCHS": "10",
    "RUN_NAME": "yolo-baseline"
  }
}
```

## Why It Uses Stdout Metrics

Ultralytics writes `results.csv` for each run. This template appends a short post-train command that reads that file and prints `metric:*` lines, which Crucible already knows how to parse.

That keeps the demo simple and makes result collection deterministic.

## Why It Stays Sequential

Keep the flow single-threaded:

1. launch one run
2. wait for it to finish
3. collect it
4. launch the next run

That gives you a clean one-run-at-a-time story for the post and avoids overlapping remote state while the external project is still proving out.

## Blogpost Angle

If you want to turn this into a short post, the clean framing is:

- hook: "I opened a fresh directory and fine-tuned YOLO over MCP"
- truth-in-advertising: "fresh directory" still needs two config files today
- body: MCP provisions the GPU, bootstraps Ultralytics, launches runs, and collects parsed mAP
- close: the orchestration loop is real; the remaining gap is workspace/spec creation over MCP

## References

- [Ultralytics Documentation](https://docs.ultralytics.com/)
- [YOLO Performance Metrics Guide](https://github.com/ultralytics/ultralytics/blob/main/docs/en/guides/yolo-performance-metrics.md)
