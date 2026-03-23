#!/usr/bin/env python3
"""Integration test: Run 3 SOTA experiments on 2 RunPod pods via Crucible.

Run from the parameter-golf repo directory:
    cd /Users/eren/Documents/ai/parameter-golf
    PYTHONPATH=/Users/eren/Documents/AI/parameter-golf_dev/src python3 \
        /Users/eren/Documents/AI/parameter-golf_dev/scripts/test_2pod.py

This will:
1. Load crucible.yaml
2. Provision 2 RunPod pods (RTX 4090/3090)
3. Wait for SSH connectivity
4. Bootstrap (sync code, install deps, download data)
5. Enqueue first 3 SOTA experiments from next_batch_merged.json
6. Dispatch (2 parallel, 1 queued)
7. Monitor until all complete (~1.5 hours)
8. Print leaderboard
9. Ask before destroying pods
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure we can import crucible
try:
    from crucible.core.config import load_config
    from crucible.core.env import load_env_files
    from crucible.core.log import log_info, log_step, log_success, log_warn, log_error
except ImportError:
    print("ERROR: Cannot import crucible. Set PYTHONPATH to crucible's src/ directory.")
    print("  PYTHONPATH=/Users/eren/Documents/AI/parameter-golf_dev/src python3 scripts/test_2pod.py")
    sys.exit(1)

# Configuration
SPEC_FILE = "specs/next_batch_merged.json"
NUM_PODS = 2
NUM_EXPERIMENTS = 3  # First 3 from the spec
NAME_PREFIX = "crucible-test"
MONITOR_INTERVAL = 60  # seconds between status checks


def main() -> None:
    # ── Load config ──
    config = load_config()
    load_env_files(config.project_root)

    print("=" * 60)
    print("  Crucible 2-Pod Integration Test")
    print(f"  Project: {config.name}")
    print(f"  Provider: {config.provider.type}")
    print(f"  Runner: {config.runner_script}")
    print("=" * 60)

    # ── Load spec and pick first N experiments ──
    spec_path = config.project_root / SPEC_FILE
    if not spec_path.exists():
        log_error(f"Spec file not found: {spec_path}")
        sys.exit(1)

    all_experiments = json.loads(spec_path.read_text(encoding="utf-8"))
    experiments = all_experiments[:NUM_EXPERIMENTS]
    print(f"\nSelected {len(experiments)} experiments from {len(all_experiments)} total:")
    for exp in experiments:
        print(f"  - {exp['name']} (tier: {exp['tier']})")

    # ── Pre-flight checks ──
    ssh_key = Path(config.provider.ssh_key).expanduser()
    if not ssh_key.exists():
        log_error(f"SSH key not found: {ssh_key}")
        sys.exit(1)
    log_success(f"SSH key: {ssh_key}")

    import os
    if not os.environ.get("RUNPOD_API_KEY"):
        log_error("RUNPOD_API_KEY not set in environment")
        sys.exit(1)
    log_success("RUNPOD_API_KEY is set")

    runner_path = config.project_root / config.runner_script
    if not runner_path.exists():
        log_error(f"Runner script not found: {runner_path}")
        sys.exit(1)
    log_success(f"Runner script: {runner_path}")

    # ── Confirm ──
    print(f"\nAbout to provision {NUM_PODS} RunPod pods and run {NUM_EXPERIMENTS} experiments.")
    print(f"Estimated cost: ~${NUM_PODS * 0.5 * 1.5:.2f} (2x GPU for ~1.5h)")
    reply = input("Continue? [y/N] ").strip().lower()
    if reply not in ("y", "yes"):
        print("Aborted.")
        sys.exit(0)

    # ── Import fleet manager ──
    from crucible.fleet.manager import FleetManager

    fm = FleetManager(config)

    # ── PHASE 1: Provision ──
    log_step("PHASE 1: Provisioning pods")
    try:
        nodes = fm.provision_and_wait(count=NUM_PODS, name_prefix=NAME_PREFIX)
        log_success(f"Provisioned {len(nodes)} pods")
        for n in nodes:
            print(f"  {n['name']}: {n.get('gpu', '?')} @ {n.get('ssh_host', '?')}:{n.get('ssh_port', 22)}")
    except Exception as exc:
        log_error(f"Provisioning failed: {exc}")
        sys.exit(1)

    # ── PHASE 2: Bootstrap ──
    log_step("PHASE 2: Bootstrapping pods")
    try:
        nodes = fm.bootstrap(nodes, train_shards=1)
        ready = [n for n in nodes if n.get("state") == "ready"]
        log_success(f"Bootstrapped {len(ready)}/{len(nodes)} pods")
    except Exception as exc:
        log_error(f"Bootstrap failed: {exc}")
        _offer_destroy(fm)
        sys.exit(1)

    # ── PHASE 3: Enqueue ──
    log_step("PHASE 3: Enqueuing experiments")
    added = fm.enqueue(experiments=experiments)
    log_success(f"Enqueued {len(added)} experiments")
    for item in added:
        print(f"  {item['experiment_name']} -> {item['run_id']}")

    # ── PHASE 4: Dispatch ──
    log_step("PHASE 4: Dispatching")
    queue = fm.dispatch(nodes, max_assignments=NUM_PODS)
    running = [r for r in queue if r.get("lease_state") == "running"]
    queued = [r for r in queue if r.get("lease_state") == "queued"]
    log_success(f"Running: {len(running)}, Queued: {len(queued)}")

    # ── PHASE 5: Monitor ──
    log_step("PHASE 5: Monitoring")
    try:
        _monitor_loop(fm, experiments, nodes)
    except KeyboardInterrupt:
        print("\nInterrupted!")

    # ── PHASE 6: Results ──
    log_step("PHASE 6: Final results")
    fm.collect(nodes)
    _print_results(fm)

    # ── PHASE 7: Cleanup ──
    _offer_destroy(fm)


def _monitor_loop(
    fm: FleetManager,
    experiments: list[dict],
    nodes: list[dict],
) -> None:
    """Poll for completion until all experiments finish."""
    experiment_names = {exp["name"] for exp in experiments}
    start_time = time.time()

    while True:
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60):02d}s"

        # Collect results
        fm.collect(nodes)

        # Check queue status
        queue = fm.queue_status()
        our_items = [r for r in queue if r.get("experiment_name") in experiment_names]

        running = [r for r in our_items if r.get("lease_state") == "running"]
        completed = [r for r in our_items if r.get("lease_state") in ("completed", "finished")]
        queued_items = [r for r in our_items if r.get("lease_state") == "queued"]

        print(
            f"  [{elapsed_str}] "
            f"Running: {len(running)}, "
            f"Completed: {len(completed)}/{len(experiments)}, "
            f"Queued: {len(queued_items)}"
        )

        # Check if all done
        if len(completed) >= len(experiments):
            log_success(f"All {len(experiments)} experiments complete!")
            break

        # If there are queued items and idle pods, dispatch more
        if queued_items:
            from crucible.fleet.inventory import load_nodes_snapshot
            current_nodes = load_nodes_snapshot(fm.nodes_file)
            fm.dispatch(current_nodes, max_assignments=NUM_PODS)

        time.sleep(MONITOR_INTERVAL)


def _print_results(fm: FleetManager) -> None:
    """Print a simple leaderboard from fleet results."""
    from crucible.core.io import read_jsonl

    results = read_jsonl(fm.fleet_results_file)
    completed = [
        r for r in results
        if r.get("status") == "completed" and r.get("result")
    ]

    if not completed:
        log_warn("No completed results found yet.")
        return

    completed.sort(key=lambda r: r["result"].get("val_bpb", 999))

    print(f"\n{'Rank':>4}  {'Name':<45}  {'val_bpb':>8}  {'val_loss':>8}  {'bytes':>10}")
    print("-" * 85)
    for i, r in enumerate(completed, 1):
        res = r["result"]
        bpb = res.get("val_bpb", "N/A")
        loss = res.get("val_loss", "N/A")
        bpb_str = f"{bpb:.4f}" if isinstance(bpb, (int, float)) else str(bpb)
        loss_str = f"{loss:.4f}" if isinstance(loss, (int, float)) else str(loss)
        print(f"{i:>4}  {r.get('name', '?'):<45}  {bpb_str:>8}  {loss_str:>8}  {r.get('model_bytes', 'N/A'):>10}")


def _offer_destroy(fm: FleetManager) -> None:
    """Ask user whether to destroy pods."""
    print()
    reply = input("Destroy pods? [y/N] ").strip().lower()
    if reply in ("y", "yes"):
        try:
            fm.destroy()
            log_success("Pods destroyed.")
        except Exception as exc:
            log_error(f"Destroy failed: {exc}")
    else:
        print("Pods kept alive. Run 'crucible fleet destroy' when done.")


if __name__ == "__main__":
    main()
