#!/usr/bin/env python3
"""Le-WM inference and evaluation script.

Loads a trained Le-WM checkpoint, runs CEM planning in PushT,
records videos, and reports success rate.

Usage:
    # With pretrained weights:
    python eval_lewm.py --checkpoint pusht/lewm --episodes 10

    # With our trained checkpoint:
    python eval_lewm.py --checkpoint lewm --episodes 5 --video

    # Just record a video:
    python eval_lewm.py --checkpoint lewm --video --no-eval

Requires: stable-worldmodel[env], torch
Checkpoint: _object.ckpt in $STABLEWM_HOME (default ~/.stable_worldmodel/)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Le-WM inference")
    parser.add_argument("--checkpoint", default="lewm",
                        help="Checkpoint name under $STABLEWM_HOME")
    parser.add_argument("--env", default="swm/PushT-v1")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-steps", type=int, default=300)
    parser.add_argument("--video-dir", default="./videos")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--cem-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--cem-topk", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    import torch
    import stable_worldmodel as swm

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load model
    print(f"Loading: {args.checkpoint}")
    cost_model = swm.policy.AutoCostModel(args.checkpoint)
    cost_model = cost_model.to(device)
    cost_model.requires_grad_(False)
    params = sum(p.numel() for p in cost_model.parameters())
    print(f"Params: {params:,} on {device}")

    # Environment
    env = swm.World(args.env, num_envs=1, image_shape=(224, 224))

    # CEM planning
    plan_config = swm.PlanConfig(
        horizon=args.horizon,
        receding_horizon=args.horizon,
        action_block=args.action_block,
    )
    solver = swm.solver.CEMSolver(
        model=cost_model, batch_size=1,
        num_samples=args.cem_samples,
        n_steps=args.cem_steps,
        topk=args.cem_topk,
        device=device, seed=args.seed,
    )
    policy = swm.policy.WorldModelPolicy(solver=solver, config=plan_config)
    env.set_policy(policy)
    print("CEM policy ready")

    # Video
    if args.video:
        vdir = Path(args.video_dir)
        vdir.mkdir(parents=True, exist_ok=True)
        print(f"Recording video ({args.video_steps} steps)...")
        t0 = time.time()
        env.record_video(video_path=vdir, max_steps=args.video_steps, seed=args.seed)
        print(f"Saved in {time.time()-t0:.1f}s: {[v.name for v in vdir.glob('*.mp4')]}")

    # Evaluate
    results = {}
    if not args.no_eval:
        print(f"Running {args.episodes} episodes...")
        t0 = time.time()
        results = env.evaluate(episodes=args.episodes, seed=args.seed)
        print(f"\nResults ({time.time()-t0:.1f}s):")
        for k, v in results.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    env.close()

    if args.output_json:
        out = {"checkpoint": args.checkpoint, "results": {
            k: v for k, v in results.items() if isinstance(v, (int, float, str, bool))
        }, "params": params}
        Path(args.output_json).write_text(json.dumps(out, indent=2))
        print(f"Saved: {args.output_json}")

    print("Done!")


if __name__ == "__main__":
    main()
