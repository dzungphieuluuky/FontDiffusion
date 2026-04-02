"""
Visualization script to display content images before and after skeleton-distance transform.

Shows side-by-side comparisons of:
1. Original content image (RGB)
2. Skeleton-distance transformed image

Usage:
    python tools/visualize_skeleton_transform.py \
        --data_root my_dataset \
        --phase train \
        --num_samples 10 \
        --save_dir results/skeleton_viz \
        --use_skeleton_content \
        --skeleton_method medial_axis \
        --skeleton_distance_method hybrid
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from src.configs.fontdiffuser import get_parser
from src.modules.skeleton_distance_transform import SkeletonDistanceTransform
from src.dataset.font_dataset_fst import FontDataset


logger = logging.getLogger(__name__)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert tensor to numpy array for visualization.

    Args:
        tensor: Tensor of shape (C, H, W) or (H, W)

    Returns:
        Numpy array in range [0, 1]
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach().numpy()

    # Denormalize if needed (values in [-1, 1] range from normalization)
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2

    # Clip to [0, 1]
    tensor = np.clip(tensor, 0, 1)

    return tensor


def visualize_single_sample(
    original_image: torch.Tensor,
    transformed_image: torch.Tensor,
    skeleton_transform: SkeletonDistanceTransform,
    sample_name: str = "sample",
    save_path: Optional[str] = None,
) -> None:
    """
    Create side-by-side visualization of original and transformed content image.

    Args:
        original_image: (3, H, W) original RGB image tensor
        transformed_image: (3, H, W) skeleton-distance transformed image tensor
        skeleton_transform: SkeletonDistanceTransform instance
        sample_name: Name of the sample for title
        save_path: Optional path to save figure
    """
    # Handle batch dimension if present
    if original_image.dim() == 4:
        original_image = original_image[0]
    if transformed_image.dim() == 4:
        transformed_image = transformed_image[0]

    # Convert to numpy for visualization
    orig_np = tensor_to_numpy(original_image)
    trans_np = tensor_to_numpy(transformed_image)

    # Extract first channel for visualization
    if orig_np.shape[0] == 3:
        orig_vis = orig_np[0]  # (H, W)
    else:
        orig_vis = orig_np[0]

    if trans_np.shape[0] == 3:
        # All 3 channels should be identical in skeleton-distance output
        trans_vis = trans_np[0]  # (H, W)
    else:
        trans_vis = trans_np[0]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Original
    im0 = axes[0].imshow(orig_vis, cmap="gray")
    axes[0].set_title(
        f"{sample_name}\nOriginal RGB Content Image", fontsize=12, fontweight="bold"
    )
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Transformed
    im1 = axes[1].imshow(trans_vis, cmap="gray")
    axes[1].set_title(
        f"{sample_name}\nSkeleton-Distance Transform\n"
        f"(method={skeleton_transform.method}, distance={skeleton_transform.distance_method})",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_detailed_pipeline(
    original_image: torch.Tensor,
    transformed_image: torch.Tensor,
    skeleton_transform: SkeletonDistanceTransform,
    sample_name: str = "sample",
    save_path: Optional[str] = None,
) -> None:
    """
    Create detailed 3-panel visualization showing transformation process.

    Panels:
    1. Original RGB content image
    2. Skeleton-distance transformed image
    3. Overlay showing topology preservation

    Args:
        original_image: (3, H, W) original RGB image tensor
        transformed_image: (3, H, W) skeleton-distance transformed image tensor
        skeleton_transform: SkeletonDistanceTransform instance
        sample_name: Name for title
        save_path: Optional path to save figure
    """
    # Handle batch dimension if present
    if original_image.dim() == 4:
        original_image = original_image[0]
    if transformed_image.dim() == 4:
        transformed_image = transformed_image[0]

    orig_np = tensor_to_numpy(original_image)
    trans_np = tensor_to_numpy(transformed_image)

    # Extract first channel for visualization
    if orig_np.shape[0] == 3:
        orig_vis = orig_np[0]
    else:
        orig_vis = orig_np[0]

    if trans_np.shape[0] == 3:
        trans_vis = trans_np[0]
    else:
        trans_vis = trans_np[0]

    # Create detailed figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Original
    im0 = axes[0].imshow(orig_vis, cmap="gray")
    axes[0].set_title(
        "Original RGB Content\n(with stroke thickness)", fontsize=11, fontweight="bold"
    )
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Skeleton-Distance (fused output)
    im1 = axes[1].imshow(trans_vis, cmap="hot")
    axes[1].set_title(
        "Skeleton-Distance Fused\n(topology preserved)", fontsize=11, fontweight="bold"
    )
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Overlay: skeleton on original to show topology preservation
    orig_rgb = np.stack([orig_vis, orig_vis, orig_vis], axis=0)
    orig_rgb = np.transpose(orig_rgb, (1, 2, 0))  # (H, W, 3)

    axes[2].imshow(orig_rgb, cmap="gray", alpha=0.4, label="Original")
    axes[2].imshow(trans_vis, cmap="hot", alpha=0.8, label="Transform")
    axes[2].set_title(
        "Overlay\n(skeleton guides topology)", fontsize=11, fontweight="bold"
    )
    axes[2].axis("off")

    fig.suptitle(
        f"{sample_name} | "
        f"Method: {skeleton_transform.method} | "
        f"Distance: {skeleton_transform.distance_method}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"  ✓ Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_dataset_samples(
    dataset: FontDataset,
    skeleton_transform: SkeletonDistanceTransform,
    num_samples: int = 10,
    save_dir: Optional[str] = None,
    detailed: bool = False,
) -> None:
    """
    Visualize multiple dataset samples with skeleton transform applied.

    Args:
        dataset: FontDataset instance
        skeleton_transform: SkeletonDistanceTransform instance
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
        detailed: Whether to show detailed pipeline (3 panels)
    """
    logger.info(f"\nVisualizing {num_samples} dataset samples...")

    num_to_process = min(num_samples, len(dataset))

    for idx in range(num_to_process):
        # Get sample from dataset
        sample = dataset[idx]

        # Extract content image (already transformed by dataset)
        content_image = sample["content_image"]  # (3, H, W) with skeleton applied

        # Extract original content image if available
        if "content_image_original" in sample:
            original_image = sample["content_image_original"]
        else:
            logger.warning(f"  ⚠ No original image stored for sample {idx}")
            continue

        # Extract sample name from path
        target_image_path = sample.get("target_image_path", f"sample_{idx}")
        sample_name = Path(target_image_path).stem

        # Create sample name (parse style+content format if available)
        if "+" in sample_name:
            parts = sample_name.split("+")
            if len(parts) == 2:
                style, content = parts
                sample_name = f"{style}+{content}"

        # Visualize
        if detailed:
            viz_fn = visualize_detailed_pipeline
        else:
            viz_fn = visualize_single_sample

        viz_path = None
        if save_dir:
            viz_path = f"{save_dir}/{idx:03d}_{sample_name}.png"

        viz_fn(
            original_image,
            content_image,
            skeleton_transform,
            sample_name=sample_name,
            save_path=viz_path,
        )

        if (idx + 1) % 5 == 0:
            logger.info(f"  Processed {idx + 1}/{num_to_process} samples")

    logger.info(f"✓ Visualization complete! ({num_to_process} samples)")
    if save_dir:
        logger.info(f"  Results saved to: {save_dir}")


def main():
    """Main entry point for visualization script."""
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # Get centralized parser from configs
    parser = get_parser()

    # Add visualization-specific arguments (not in main parser)
    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset phase to visualize",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to visualize",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed 3-panel pipeline visualization",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Display plots interactively instead of saving",
    )

    args = parser.parse_args()

    # Validate required arguments
    if not args.data_root:
        parser.error("--data_root is required")
    if not args.use_skeleton_content:
        parser.error("--use_skeleton_content flag is required for this script")

    logger.info("=" * 80)
    logger.info("Content Image Skeleton-Distance Transform Visualization")
    logger.info("=" * 80)
    logger.info(f"Dataset root: {args.data_root}")
    logger.info(f"Phase: {args.phase}")
    logger.info(f"\nSkeleton Transform Configuration:")
    logger.info(f"  Method: {args.skeleton_method}")
    logger.info(f"  Distance Method: {args.skeleton_distance_method}")
    logger.info(f"  Max Distance: {args.skeleton_max_distance}")
    logger.info(f"  Sigma: {args.skeleton_sigma}")
    logger.info(f"  Fusion Method: {args.skeleton_fusion_method}")
    logger.info(f"\nVisualization:")
    logger.info(f"  Num Samples: {args.num_samples}")
    logger.info(
        f"  Style: {'Detailed (3-panel)' if args.detailed else 'Simple (2-panel)'}"
    )
    logger.info(
        f"  Output: {'Display' if args.display else f'Save to {args.save_dir}'}"
    )
    logger.info("=" * 80)

    # Build transforms matching trainer_fst.py setup
    content_image_size = (
        args.content_image_size
        if isinstance(args.content_image_size, tuple)
        else (args.content_image_size, args.content_image_size)
    )
    style_image_size = (
        args.style_image_size
        if isinstance(args.style_image_size, tuple)
        else (args.style_image_size, args.style_image_size)
    )
    target_size = (args.resolution, args.resolution)

    content_transforms = transforms.Compose(
        [
            transforms.Resize(
                content_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    style_transforms = transforms.Compose(
        [
            transforms.Resize(
                style_image_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    target_transforms = transforms.Compose(
        [
            transforms.Resize(
                target_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    logger.info(f"\n✓ Transforms prepared")
    logger.info(f"  Content size: {content_image_size}")
    logger.info(f"  Style size: {style_image_size}")
    logger.info(f"  Target size: {target_size}")

    # Build skeleton config from args
    skeleton_config = {
        "method": args.skeleton_method,
        "distance_method": args.skeleton_distance_method,
        "max_distance": args.skeleton_max_distance,
        "sigma": args.skeleton_sigma,
        "output_mode": args.skeleton_output_mode,
        "normalize": True,
        "fusion_method": args.skeleton_fusion_method,
    }
    logger.info(f"\n✓ Skeleton config: {skeleton_config}")

    # Create dataset with skeleton transform enabled (matching trainer_fst.py)
    logger.info(f"\nLoading dataset from {args.data_root}...")
    dataset = FontDataset(
        args=args,
        phase=args.phase,
        transforms=[content_transforms, style_transforms, target_transforms],
        scr=args.phase_2,
        use_fst=args.use_fst,
        style_source_same_prob=args.style_source_same_prob,
        num_consistency_pairs=args.num_consistency_pairs,
        num_identity_pairs=args.num_identity_pairs,
        identity_pair_mode=args.identity_pair_mode,
        use_skeleton_transform=args.use_skeleton_content,
        skeleton_config=skeleton_config,
    )
    logger.info(f"✓ Loaded dataset with {len(dataset)} samples")
    logger.info("✓ Skeleton transform will be applied during data loading")

    # Create skeleton transform instance for reference
    skeleton_transform = SkeletonDistanceTransform(**skeleton_config)
    logger.info(f"\n✓ Created skeleton transform: {skeleton_transform}")

    # Visualize samples
    save_dir_to_use = args.save_dir if not args.display else None
    visualize_dataset_samples(
        dataset=dataset,
        skeleton_transform=skeleton_transform,
        num_samples=args.num_samples,
        save_dir=save_dir_to_use,
        detailed=args.detailed,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Visualization Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
"python -m src.tools.visualize_skeleton_transform --detailed --display --data_root data --use_skeleton_content --skeleton_max_distance 12  --skeleton_sigma 1.5"
