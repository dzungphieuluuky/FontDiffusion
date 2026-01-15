"""
FontDiffusion inference package.

Provides optimized and distributed sampling pipelines for font generation.
Includes single-image, batch, and multi-GPU inference support.
"""

from .sample_optimized import (
    load_fontdiffuser_pipeline,
    get_content_transform,
    get_style_transform,
)
from .sample_batch import (
    FontManager,
    QualityEvaluator,
    GenerationTracker,
    parse_args,
    create_args_namespace,
    load_characters,
    load_style_images,
    save_checkpoint,
    log_to_wandb,
)
from .sample_distributed import main as run_distributed_sampling
__all__ = [
    # sample_optimized
    "load_fontdiffuser_pipeline",
    "sampling",
    "get_content_transform",
    "get_style_transform",
    # sample_batch
    "FontManager",
    "QualityEvaluator",
    "GenerationTracker",
    "parse_args",
    "create_args_namespace",
    "load_characters",
    "load_style_images",
    "save_checkpoint",
    "log_to_wandb",
    "run_distributed_sampling",
]