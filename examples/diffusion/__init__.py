"""Diffusion model example: DDPM on MNIST.

Importing this module registers the ``ddpm_unet`` model family and
``mnist_images`` data adapter with Crucible's registries.
"""
from examples.diffusion.model import register as _register_model
from examples.diffusion.data_adapter import register as _register_adapter

_register_model()
_register_adapter()
