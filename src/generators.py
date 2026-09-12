"""
Generators Facade: Exposes both GAN (CTGAN) and Diffusion (TabDDPM) synthesizers.
Modular implementations are separated into:
  - src.generator_gan: Conditional Tabular GAN synthesis
  - src.generator_diffusion: Tabular Denoising Diffusion synthesis
"""

from src.generator_gan import generate_ctgan_synthetic_data, clip_to_bounds, TARGET_COL
from src.generator_diffusion import generate_tabddpm_synthetic_data

__all__ = [
    "generate_ctgan_synthetic_data",
    "generate_tabddpm_synthetic_data",
    "clip_to_bounds",
    "TARGET_COL"
]
