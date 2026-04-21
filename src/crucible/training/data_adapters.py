"""Data adapters -- uniform batch interface for all modalities.

Each adapter wraps a modality-specific data source and returns batches as
``dict[str, Tensor]`` so the generic training backend can be data-agnostic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


class DataAdapter:
    """Base class for data adapters."""

    def next_batch(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Return the next training batch as a dict of tensors."""
        raise NotImplementedError

    @classmethod
    def modality(cls) -> str:
        """Modality tag matching the model it feeds."""
        return "generic"


class TokenDataAdapter(DataAdapter):
    """Wraps the existing :class:`DistributedTokenLoader`.

    Translates ``(x, y)`` tuples into ``{'input_ids': x, 'target_ids': y}``.
    """

    def __init__(self, loader: object):
        # loader is a DistributedTokenLoader; typed as object to avoid
        # circular import with data loading module.
        self.loader = loader

    def next_batch(
        self,
        global_tokens: int = 0,
        seq_len: int = 0,
        grad_accum_steps: int = 1,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        x, y = self.loader.next_batch(global_tokens, seq_len, grad_accum_steps)
        return {"input_ids": x, "target_ids": y}

    @classmethod
    def modality(cls) -> str:
        return "lm"


# ---------------------------------------------------------------------------
# Vision adapters
# ---------------------------------------------------------------------------

class ImageFolderAdapter(DataAdapter):
    """Loads images from torchvision datasets or a directory.

    Supports shortcut names like ``"mnist"`` and ``"cifar10"`` which
    auto-download via torchvision.  For custom directories, pass a
    ``root`` path.
    """

    def __init__(
        self,
        dataset_name: str = "cifar10",
        root: str = "./data",
        image_size: int = 32,
        channels: int = 3,
        train: bool = True,
    ):
        import torch
        from torchvision import datasets, transforms

        tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.Grayscale(num_output_channels=channels) if channels == 1 else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * channels, [0.5] * channels),
        ])

        name_lower = dataset_name.lower()
        if name_lower == "mnist":
            self.dataset = datasets.MNIST(root, train=train, download=True, transform=tfm)
        elif name_lower in ("cifar10", "cifar-10"):
            self.dataset = datasets.CIFAR10(root, train=train, download=True, transform=tfm)
        elif name_lower in ("cifar100", "cifar-100"):
            self.dataset = datasets.CIFAR100(root, train=train, download=True, transform=tfm)
        elif name_lower in ("fashionmnist", "fashion-mnist", "fashion_mnist"):
            self.dataset = datasets.FashionMNIST(root, train=train, download=True, transform=tfm)
        else:
            self.dataset = datasets.ImageFolder(root, transform=tfm)

        self._loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=1, shuffle=True, drop_last=True,
        )
        self._iter = iter(self._loader)

    def next_batch(
        self,
        batch_size: int = 8,
        device: torch.device | str | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        import torch

        images_list = []
        labels_list = []
        if len(self._loader.dataset) == 0:
            from crucible.core.errors import DataError
            raise DataError("Dataset is empty — cannot produce a batch")

        for _ in range(batch_size):
            try:
                img, label = next(self._iter)
            except StopIteration:
                self._iter = iter(self._loader)
                img, label = next(self._iter)
            images_list.append(img.squeeze(0))
            labels_list.append(label)

        images = torch.stack(images_list)
        if isinstance(labels_list[0], torch.Tensor):
            labels = torch.stack(labels_list)
        else:
            labels = torch.tensor(labels_list)

        if device is not None:
            images = images.to(device)
            labels = labels.to(device)

        return {"images": images, "labels": labels}

    @classmethod
    def modality(cls) -> str:
        return "vision"


class SyntheticImageAdapter(DataAdapter):
    """Generates random images for pipeline testing.  No data dependency."""

    def __init__(self, channels: int = 3, num_classes: int = 10):
        self.channels = channels
        self.num_classes = num_classes

    def next_batch(
        self,
        batch_size: int = 8,
        image_size: int = 32,
        device: torch.device | str | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        import torch

        images = torch.randn(batch_size, self.channels, image_size, image_size)
        labels = torch.randint(0, self.num_classes, (batch_size,))
        if device is not None:
            images = images.to(device)
            labels = labels.to(device)
        return {"images": images, "labels": labels}

    @classmethod
    def modality(cls) -> str:
        return "vision"


# ---------------------------------------------------------------------------
# Video / sequence adapters
# ---------------------------------------------------------------------------

class SyntheticVideoAdapter(DataAdapter):
    """Generates bouncing-ball video sequences for world model testing.

    Self-contained physics sim: colored squares bounce inside the frame.
    Actions represent velocity perturbations applied to one square.
    """

    def __init__(
        self,
        num_objects: int = 2,
        object_size: int = 4,
        action_dim: int = 2,
    ):
        self.num_objects = num_objects
        self.object_size = object_size
        self.action_dim = action_dim

    def next_batch(
        self,
        batch_size: int = 4,
        num_frames: int = 4,
        image_size: int = 32,
        device: torch.device | str | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        import torch

        frames = torch.zeros(batch_size, num_frames, 3, image_size, image_size)
        actions = torch.zeros(batch_size, num_frames - 1, self.action_dim)

        for b in range(batch_size):
            # Initialize object positions and velocities
            positions = torch.rand(self.num_objects, 2) * (image_size - self.object_size)
            velocities = (torch.rand(self.num_objects, 2) - 0.5) * 4.0
            colors = torch.rand(self.num_objects, 3)

            for t in range(num_frames):
                # Render frame
                for obj_idx in range(self.num_objects):
                    px = int(positions[obj_idx, 0].clamp(0, image_size - self.object_size))
                    py = int(positions[obj_idx, 1].clamp(0, image_size - self.object_size))
                    for c in range(3):
                        frames[b, t, c, py:py + self.object_size, px:px + self.object_size] = colors[obj_idx, c]

                # Apply physics + actions
                if t < num_frames - 1:
                    # Generate action (velocity perturbation for first object)
                    action = (torch.rand(self.action_dim) - 0.5) * 2.0
                    actions[b, t, :] = action
                    velocities[0, :self.action_dim] += action

                    # Step physics
                    positions += velocities

                    # Bounce off walls
                    for obj_idx in range(self.num_objects):
                        for dim in range(2):
                            if positions[obj_idx, dim] < 0:
                                positions[obj_idx, dim] = -positions[obj_idx, dim]
                                velocities[obj_idx, dim] = abs(velocities[obj_idx, dim])
                            elif positions[obj_idx, dim] > image_size - self.object_size:
                                overshoot = positions[obj_idx, dim] - (image_size - self.object_size)
                                positions[obj_idx, dim] = (image_size - self.object_size) - overshoot
                                velocities[obj_idx, dim] = -abs(velocities[obj_idx, dim])

        if device is not None:
            frames = frames.to(device)
            actions = actions.to(device)

        return {"frames": frames, "actions": actions}

    @classmethod
    def modality(cls) -> str:
        return "world_model"


# ---------------------------------------------------------------------------
# Registry (PluginRegistry-backed)
# ---------------------------------------------------------------------------

from crucible.core.plugin_registry import PluginRegistry

_ADAPTER_REGISTRY = PluginRegistry[type["DataAdapter"]]("data_adapter")
DATA_ADAPTER_REGISTRY: dict[str, type["DataAdapter"]] = _ADAPTER_REGISTRY._registry  # convenience alias


def register_data_adapter(name: str, cls: type["DataAdapter"], *, source: str = "builtin") -> None:
    """Register a data adapter class under *name*.

    Supports 3-tier precedence (builtin < global < local) via *source*.
    """
    _ADAPTER_REGISTRY.register(name, cls, source=source)


def build_data_adapter(name: str, **kwargs: Any) -> DataAdapter:
    """Instantiate a registered data adapter by name."""
    cls = _ADAPTER_REGISTRY.get(name)
    if cls is None:
        raise KeyError(
            f"Unknown data adapter '{name}'. Available: {sorted(_ADAPTER_REGISTRY.list_plugins())}"
        )
    return cls(**kwargs)


def list_data_adapters() -> list[str]:
    """Return sorted list of registered data adapter names."""
    return _ADAPTER_REGISTRY.list_plugins()


def list_data_adapters_detailed() -> list[dict[str, str]]:
    """Return data adapters with source metadata."""
    return _ADAPTER_REGISTRY.list_plugins_detailed()


# Register built-ins
register_data_adapter("token", TokenDataAdapter)
register_data_adapter("image_folder", ImageFolderAdapter)
register_data_adapter("synthetic_images", SyntheticImageAdapter)
register_data_adapter("synthetic_video", SyntheticVideoAdapter)
