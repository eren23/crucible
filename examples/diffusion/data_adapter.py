"""MNIST image data adapter for diffusion training.

Downloads MNIST via torchvision and serves normalized 28x28 grayscale
images as ``{"images": [B, 1, 28, 28]}`` tensors in [-1, 1].
"""
from __future__ import annotations

from typing import Any

from crucible.training.data_adapters import DataAdapter, register_data_adapter


class MNISTAdapter(DataAdapter):
    """Loads MNIST images for diffusion training."""

    def __init__(self, root: str = "./data", train: bool = True):
        from torchvision import datasets, transforms

        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # -> [-1, 1]
        ])
        self.dataset = datasets.MNIST(root, train=train, download=True, transform=tfm)
        self._idx = 0

    def next_batch(
        self,
        batch_size: int = 32,
        device: Any = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        import torch

        images = []
        for _ in range(batch_size):
            img, _ = self.dataset[self._idx % len(self.dataset)]
            images.append(img)
            self._idx += 1

        batch = torch.stack(images)
        if device is not None:
            batch = batch.to(device)
        return {"images": batch}

    @classmethod
    def modality(cls) -> str:
        return "diffusion"


def register() -> None:
    """Register the MNIST adapter with Crucible."""
    register_data_adapter("mnist_images", MNISTAdapter)
