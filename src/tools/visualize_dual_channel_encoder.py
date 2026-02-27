"""
Visualization tool for DualChannelContentEncoder transformations.

Displays the original content image and its 3-channel RGB output after
skeleton-distance fusion and channel expansion.

Usage:
    python tools/visualize_dual_channel_encoder.py \
        --test_image_path path/to/image.png \
        --ckpt_dir ckpt/ \
        --fusion_method concat \
        --save_output results/
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import logging

from src.builders.build import build_content_encoder
from src.configs.fontdiffuser import get_parser
from src.modules.skeleton_distance_transform import DualChannelContentEncoder

logger = logging.getLogger(__name__)


def load_image(image_path: str, resolution: int = 96) -> torch.Tensor:
    """
    Load and preprocess image for model input.
    
    Args:
        image_path: Path to image file
        resolution: Target resolution (default 96x96)
    
    Returns:
        torch.Tensor: (1, 1, H, W) normalized image tensor
    """
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    
    # Resize to target resolution
    img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
    
    # Convert to numpy
    img_np = np.array(img, dtype=np.float32) / 255.0
    
    # Convert to tensor and add batch/channel dimensions
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    return img_tensor


def visualize_transformation(
    original: np.ndarray,
    transformed_rgb: np.ndarray,
    fusion_method: str,
    save_path: str = None,
) -> None:
    """
    Visualize original image and transformed RGB output side-by-side.
    
    Args:
        original: (H, W) original grayscale image
        transformed_rgb: (H, W, 3) transformed RGB image
        fusion_method: Fusion method used
        save_path: Optional path to save visualization
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original Content Image\n(Grayscale, 1 channel)')
    axes[0].axis('off')
    
    # Transformed RGB
    axes[1].imshow(transformed_rgb)
    axes[1].set_title(
        f'Transformed Output\n(RGB, 3 channels, fusion={fusion_method})'
    )
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_detailed_channels(
    original: np.ndarray,
    skeleton: np.ndarray,
    distance: np.ndarray,
    fused: np.ndarray,
    expanded_rgb: np.ndarray,
    fusion_method: str,
    save_path: str = None,
) -> None:
    """
    Detailed visualization of the entire transformation pipeline.
    
    Args:
        original: (H, W) original grayscale
        skeleton: (H, W) skeleton channel
        distance: (H, W) distance channel
        fused: (H, W) fused skeleton-distance
        expanded_rgb: (H, W, 3) final RGB output
        fusion_method: Fusion method used
        save_path: Optional path to save
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Input and intermediate
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original Image\n(1 channel)')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(skeleton, cmap='gray')
    axes[0, 1].set_title('Skeleton Channel\n(extracted topology)')
    axes[0, 1].axis('off')
    
    im = axes[0, 2].imshow(distance, cmap='hot')
    axes[0, 2].set_title('Distance Channel\n(influence map)')
    axes[0, 2].axis('off')
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    # Row 2: Fusion and output
    axes[1, 0].imshow(fused, cmap='gray')
    axes[1, 0].set_title(f'Fused Output\n({fusion_method}, 1 channel)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(expanded_rgb)
    axes[1, 1].set_title('Expanded to RGB\n(3 channels, replicated)')
    axes[1, 1].axis('off')
    
    # Statistics
    axes[1, 2].axis('off')
    stats_text = (
        f"Transformation Pipeline\n"
        f"{'=' * 30}\n\n"
        f"Fusion Method: {fusion_method}\n\n"
        f"Input Shape: {original.shape}\n"
        f"Skeleton Shape: {skeleton.shape}\n"
        f"Distance Shape: {distance.shape}\n"
        f"Fused Shape: {fused.shape}\n"
        f"Output Shape: {expanded_rgb.shape}\n\n"
        f"Original Range: [{original.min():.3f}, {original.max():.3f}]\n"
        f"Skeleton Range: [{skeleton.min():.3f}, {skeleton.max():.3f}]\n"
        f"Distance Range: [{distance.min():.3f}, {distance.max():.3f}]\n"
        f"Fused Range: [{fused.min():.3f}, {fused.max():.3f}]\n"
    )
    axes[1, 2].text(
        0.1, 0.5, stats_text,
        fontsize=10, family='monospace',
        verticalalignment='center'
    )
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"✓ Detailed visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    """Main entry point for visualization."""
    # Use centralized parser
    parser = get_parser()
    parser.add_argument(
        '--test_image_path',
        type=str,
        required=True,
        help='Path to content image to visualize',
    )
    parser.add_argument(
        '--fusion_method',
        type=str,
        choices=['concat', 'add', 'weighted'],
        default='concat',
        help='Channel fusion method for DualChannelContentEncoder',
    )
    parser.add_argument(
        '--save_output',
        type=str,
        default='results/visualizations/',
        help='Directory to save output visualizations',
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed channel-by-channel visualization',
    )
    
    args = parser.parse_args()
    # convert style image size and content image size to tuples
    if isinstance(args.style_image_size, int):
        args.style_image_size = (args.style_image_size, args.style_image_size)
    if isinstance(args.content_image_size, int):
        args.content_image_size = (args.content_image_size, args.content_image_size)
        
    # Validate input
    content_path = Path(args.test_image_path)
    if not content_path.exists():
        logger.error(f"✗ Content image not found: {content_path}")
        return 1
    
    logger.info(f"Loading content image: {content_path}")
    
    # Load image
    img_tensor = load_image(str(content_path), resolution=96)
    original_np = img_tensor[0, 0].numpy()
    
    # Load content encoder
    logger.info("Building content encoder...")
    content_encoder = build_content_encoder(args)
    
    # Wrap with DualChannelContentEncoder
    logger.info(f"Wrapping encoder with DualChannelContentEncoder (fusion={args.fusion_method})")
    dual_encoder = DualChannelContentEncoder(
        original_encoder=content_encoder,
        fusion_method=args.fusion_method,
        learnable_weights=False,
    )
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dual_encoder = dual_encoder.to(device)
    img_tensor = img_tensor.to(device)
    
    logger.info(f"Input shape: {img_tensor.shape}")
    
    # Forward pass
    logger.info("Processing image through DualChannelContentEncoder...")
    with torch.no_grad():
        output, residual_features = dual_encoder(img_tensor)
    
    logger.info(f"Output shape: {output.shape}")
    
    # Convert output to numpy
    output_np = output[0].cpu().permute(1, 2, 0).numpy()
    output_np = np.clip(output_np, 0, 1)  # Ensure [0, 1] range
    
    # Create save directory
    save_dir = Path(args.save_output)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original and output images
    logger.info("Saving individual images...")
    
    # Save original
    original_pil = Image.fromarray((original_np * 255).astype(np.uint8))
    original_path = save_dir / f"{content_path.stem}_original.png"
    original_pil.save(original_path)
    logger.info(f"✓ Original image saved: {original_path}")
    
    # Save output
    output_pil = Image.fromarray((output_np * 255).astype(np.uint8))
    output_path = save_dir / f"{content_path.stem}_transformed.png"
    output_pil.save(output_path)
    logger.info(f"✓ Transformed image saved: {output_path}")
    
    # Create visualizations
    logger.info("Creating visualization...")
    
    if args.detailed:
        # For detailed view, we need to extract skeleton and distance from intermediate steps
        # This requires modifying the forward pass, so we'll show the simplified version
        logger.warning(
            "Note: Detailed channel visualization requires access to intermediate "
            "skeleton-distance outputs. Currently showing combined visualization."
        )
    
    # Simple side-by-side comparison
    viz_path = save_dir / f"{content_path.stem}_comparison.png"
    visualize_transformation(
        original_np,
        output_np,
        args.fusion_method,
        save_path=str(viz_path),
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("Transformation Summary")
    logger.info("=" * 80)
    logger.info(f"Input image: {content_path}")
    logger.info(f"Input shape: {img_tensor.shape} (B, C, H, W)")
    logger.info(f"Output shape: {output.shape} (B, C, H, W)")
    logger.info(f"Fusion method: {args.fusion_method}")
    logger.info(f"Device: {device}")
    logger.info(f"\nOutput files:")
    logger.info(f"  - Original: {original_path}")
    logger.info(f"  - Transformed: {output_path}")
    logger.info(f"  - Comparison: {viz_path}")
    logger.info("=" * 80 + "\n")
    
    return 0


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    exit(main())