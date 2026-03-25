"""Bouncing ball data adapter for world model training.

Self-contained: generates synthetic video sequences of colored squares
bouncing inside a frame.  No external data dependency.

Each batch contains:
    - ``frames``: [B, T, 3, H, W] — video frames with bouncing objects
    - ``actions``: [B, T-1, 2] — velocity perturbations applied between frames
"""
from __future__ import annotations

from typing import Any

from crucible.training.data_adapters import DataAdapter, register_data_adapter


class BouncingBallAdapter(DataAdapter):
    """Generates bouncing-ball video for world model training.

    Physics: colored squares move with constant velocity inside the frame,
    bouncing elastically off walls.  Actions are velocity perturbations
    applied to the first object between frames.
    """

    def __init__(
        self,
        num_objects: int = 3,
        object_size: int = 4,
        action_dim: int = 2,
        action_scale: float = 2.0,
        velocity_scale: float = 3.0,
    ):
        self.num_objects = num_objects
        self.object_size = object_size
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.velocity_scale = velocity_scale

    def next_batch(
        self,
        batch_size: int = 8,
        num_frames: int = 4,
        image_size: int = 32,
        device: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import torch

        frames = torch.zeros(batch_size, num_frames, 3, image_size, image_size)
        actions = torch.zeros(batch_size, num_frames - 1, self.action_dim)
        max_pos = image_size - self.object_size

        for b in range(batch_size):
            # Random initial state
            positions = torch.rand(self.num_objects, 2) * max_pos
            velocities = (torch.rand(self.num_objects, 2) - 0.5) * self.velocity_scale
            colors = torch.rand(self.num_objects, 3) * 0.8 + 0.2  # avoid too dark

            for t in range(num_frames):
                # Render
                for obj in range(self.num_objects):
                    px = int(positions[obj, 0].clamp(0, max_pos).item())
                    py = int(positions[obj, 1].clamp(0, max_pos).item())
                    for c in range(3):
                        frames[b, t, c, py:py + self.object_size, px:px + self.object_size] = colors[obj, c]

                if t < num_frames - 1:
                    # Action: perturb first object's velocity
                    action = (torch.rand(self.action_dim) - 0.5) * self.action_scale
                    actions[b, t] = action
                    velocities[0, :self.action_dim] += action

                    # Physics step
                    positions = positions + velocities

                    # Wall bouncing
                    for obj in range(self.num_objects):
                        for dim in range(2):
                            if positions[obj, dim] < 0:
                                positions[obj, dim] = -positions[obj, dim]
                                velocities[obj, dim] = abs(velocities[obj, dim])
                            elif positions[obj, dim] > max_pos:
                                overshoot = positions[obj, dim] - max_pos
                                positions[obj, dim] = max_pos - overshoot
                                velocities[obj, dim] = -abs(velocities[obj, dim])

        # Normalize to [-1, 1]
        frames = frames * 2.0 - 1.0

        if device is not None:
            frames = frames.to(device)
            actions = actions.to(device)

        return {"frames": frames, "actions": actions}

    @classmethod
    def modality(cls) -> str:
        return "world_model"


def register() -> None:
    """Register the bouncing ball adapter with Crucible."""
    register_data_adapter("bouncing_balls", BouncingBallAdapter)
