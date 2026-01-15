"""
FontDiffusion training package.

Provides training infrastructure with support for:
- Phase 1: Diffusion model training
- Phase 2: Style contrastive refinement
- FSTDiff: Font style transformation with MSSE and FST modules
"""

from .trainer import FontDiffuserTrainer
from .config import TrainingConfig

__all__ = [
    "FontDiffuserTrainer",
    "TrainingConfig",
]
