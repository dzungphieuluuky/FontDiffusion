"""
Hybrid Skeleton-Distance Transform for Content Preprocessing

This module transforms content images into skeleton-distance maps that preserve
topology (where to draw) while removing style information (how thick to draw).

Key Benefits:
- Prevents style leakage from content images (e.g., NomNaTong tapering)
- Provides topological guidance without dictating stroke thickness
- Forces the model to rely on style encoder for thickness/decorations

Pipeline:
1. Topological Thinning (Skeletonization) - Extract 1-pixel medial axis
2. Distance Field Generation - Create smooth influence map
3. Normalization and Clipping - Create bounded "tube" of influence
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize, medial_axis
from skimage.filters import gaussian
from typing import Tuple, Optional, Literal
import cv2


class SkeletonDistanceTransform(nn.Module):
    """
    Transforms binary character images into skeleton-distance representations.
    
    This removes stroke thickness information while preserving topology,
    preventing the model from copying source image style characteristics.
    """
    
    def __init__(
        self,
        method: Literal["skeletonize", "medial_axis", "zhang_suen"] = "medial_axis",
        distance_method: Literal["edt", "gaussian", "hybrid"] = "hybrid",
        max_distance: float = 10.0,
        sigma: float = 3.0,
        output_mode: Literal["skeleton_only", "distance_only", "dual_channel"] = "dual_channel",
        normalize: bool = True,
    ):
        """
        Args:
            method: Skeletonization algorithm
                - "skeletonize": scikit-image morphological thinning
                - "medial_axis": Distance-based medial axis
                - "zhang_suen": Zhang-Suen thinning algorithm
            distance_method: How to create distance field
                - "edt": Euclidean Distance Transform
                - "gaussian": Gaussian blur of skeleton
                - "hybrid": EDT with Gaussian smoothing
            max_distance: Maximum influence radius (pixels)
            sigma: Gaussian sigma for smoothing
            output_mode: What to output
                - "skeleton_only": Binary skeleton (1 channel)
                - "distance_only": Distance map (1 channel)
                - "dual_channel": Both (2 channels)
            normalize: Whether to normalize output to [0, 1]
        """
        super().__init__()
        
        self.method = method
        self.distance_method = distance_method
        self.max_distance = max_distance
        self.sigma = sigma
        self.output_mode = output_mode
        self.normalize = normalize
        
        # Register as buffer (non-trainable but moves with model)
        self.register_buffer('initialized', torch.tensor(True))
    
    def skeletonize_image(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Extract 1-pixel skeleton from binary image.
        
        Args:
            binary_image: (H, W) binary numpy array
            
        Returns:
            skeleton: (H, W) binary skeleton
        """
        if self.method == "skeletonize":
            # scikit-image morphological thinning
            skeleton = skeletonize(binary_image > 0.5)
        
        elif self.method == "medial_axis":
            # Distance-based medial axis (more robust for fonts)
            skeleton, distance = medial_axis(
                binary_image > 0.5,
                return_distance=True
            )
        
        elif self.method == "zhang_suen":
            # Zhang-Suen algorithm (OpenCV)
            binary_uint8 = (binary_image > 0.5).astype(np.uint8) * 255
            skeleton = cv2.ximgproc.thinning(
                binary_uint8,
                thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
            skeleton = skeleton > 0
        
        else:
            raise ValueError(f"Unknown skeletonization method: {self.method}")
        
        return skeleton.astype(np.float32)
    
    def create_distance_field(
        self,
        skeleton: np.ndarray,
        original_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Create smooth distance field from skeleton.
        
        Args:
            skeleton: (H, W) binary skeleton
            original_shape: Original image shape
            
        Returns:
            distance_field: (H, W) smooth distance map
        """
        if self.distance_method == "edt":
            # Euclidean Distance Transform
            # Compute distance from each pixel to nearest skeleton pixel
            distance = distance_transform_edt(skeleton == 0)
            
            # Invert so skeleton has highest values
            distance = self.max_distance - distance
            distance = np.clip(distance, 0, self.max_distance)
            
            # Normalize to [0, 1]
            if self.normalize:
                distance = distance / self.max_distance
        
        elif self.distance_method == "gaussian":
            # Gaussian blur of skeleton
            distance = gaussian(skeleton, sigma=self.sigma)
            
            if self.normalize:
                # Normalize to [0, 1]
                distance = distance / (distance.max() + 1e-8)
        
        elif self.distance_method == "hybrid":
            # EDT followed by Gaussian smoothing
            distance = distance_transform_edt(skeleton == 0)
            distance = self.max_distance - distance
            distance = np.clip(distance, 0, self.max_distance)
            
            # Smooth with Gaussian
            distance = gaussian(distance, sigma=self.sigma)
            
            if self.normalize:
                distance = distance / self.max_distance
        
        else:
            raise ValueError(f"Unknown distance method: {self.distance_method}")
        
        return distance.astype(np.float32)
    
    def process_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single image through the skeleton-distance pipeline.
        
        Args:
            image: (H, W) or (H, W, C) numpy array
            
        Returns:
            output: Processed image (channels depend on output_mode)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Ensure binary
        binary = (image > 0.5).astype(np.float32)
        
        # Handle empty images
        if binary.sum() == 0:
            if self.output_mode == "dual_channel":
                return np.zeros((2, *binary.shape), dtype=np.float32)
            else:
                return np.zeros_like(binary)
        
        # Step 1: Skeletonization
        skeleton = self.skeletonize_image(binary)
        
        # Step 2: Distance field
        distance = self.create_distance_field(skeleton, binary.shape)
        
        # Step 3: Output based on mode
        if self.output_mode == "skeleton_only":
            return skeleton
        elif self.output_mode == "distance_only":
            return distance
        else:  # dual_channel
            # Stack skeleton and distance as 2 channels
            return np.stack([skeleton, distance], axis=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform batch of images.
        
        Args:
            x: (B, C, H, W) - Input images (usually grayscale, C=1)
            
        Returns:
            output: (B, C_out, H, W) where C_out depends on output_mode
                - skeleton_only: C_out = 1
                - distance_only: C_out = 1
                - dual_channel: C_out = 2
        """
        device = x.device
        batch_size = x.shape[0]
        
        # Process each image in batch
        outputs = []
        
        for i in range(batch_size):
            # Convert to numpy
            img = x[i].cpu().numpy()
            
            # If input is multi-channel, use first channel
            if img.shape[0] > 1:
                img = img[0]
            else:
                img = img.squeeze(0)
            
            # Process
            processed = self.process_single_image(img)
            
            # Ensure correct shape
            if len(processed.shape) == 2:
                processed = processed[np.newaxis, ...]  # Add channel dim
            
            outputs.append(processed)
        
        # Stack batch
        output = np.stack(outputs, axis=0)
        
        # Convert back to tensor
        output = torch.from_numpy(output).to(device)
        
        return output


class AdaptiveSkeletonDistanceTransform(SkeletonDistanceTransform):
    """
    Adaptive version that adjusts parameters based on stroke thickness.
    
    For thin strokes: smaller max_distance
    For thick strokes: larger max_distance
    """
    
    def __init__(
        self,
        min_distance: float = 5.0,
        max_distance: float = 15.0,
        **kwargs
    ):
        """
        Args:
            min_distance: Minimum influence radius
            max_distance: Maximum influence radius
            **kwargs: Other parameters for base class
        """
        super().__init__(max_distance=max_distance, **kwargs)
        self.min_distance = min_distance
    
    def estimate_stroke_thickness(self, binary_image: np.ndarray) -> float:
        """
        Estimate average stroke thickness using distance transform.
        
        Args:
            binary_image: (H, W) binary image
            
        Returns:
            Average stroke thickness in pixels
        """
        if binary_image.sum() == 0:
            return self.min_distance
        
        # Distance transform from background to foreground
        dist = distance_transform_edt(binary_image > 0.5)
        
        # Average of non-zero distances (represents stroke radius)
        stroke_radius = dist[dist > 0].mean()
        
        # Diameter = 2 * radius
        stroke_thickness = 2 * stroke_radius
        
        return float(stroke_thickness)
    
    def process_single_image(self, image: np.ndarray) -> np.ndarray:
        """
        Process with adaptive max_distance based on stroke thickness.
        
        Args:
            image: (H, W) numpy array
            
        Returns:
            Processed image
        """
        # Convert to binary
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        binary = (image > 0.5).astype(np.float32)
        
        # Estimate stroke thickness
        thickness = self.estimate_stroke_thickness(binary)
        
        # Adapt max_distance (use half the stroke thickness + margin)
        adaptive_max_dist = np.clip(
            thickness * 0.5 + 3.0,
            self.min_distance,
            self.max_distance
        )
        
        # Temporarily update max_distance
        original_max_dist = self.max_distance
        self.max_distance = adaptive_max_dist
        
        # Process
        result = super().process_single_image(image)
        
        # Restore original
        self.max_distance = original_max_dist
        
        return result


class DualChannelContentEncoder(nn.Module):
    """
    Modified content encoder that accepts BOTH 1-channel and 2-channel input.
    
    - If input is 1-channel (normal mode): Expand to 3 channels (RGB replication)
    - If input is 2-channel (skeleton mode): Fuse channels, then expand to 3 channels
    
    This ensures ContentEncoder always receives 3-channel input as expected.
    """
    
    def __init__(
        self,
        original_encoder: nn.Module,
        fusion_method: Literal["concat", "add", "weighted"] = "concat",
        learnable_weights: bool = False,
    ):
        """
        Args:
            original_encoder: Original ContentEncoder module (expects 3 channels)
            fusion_method: How to combine skeleton and distance channels
                - "concat": Concatenate and use 1x1 conv to reduce to 1 channel
                - "add": Simple addition
                - "weighted": Weighted sum (optionally learnable)
            learnable_weights: Whether fusion weights are learnable
        """
        super().__init__()
        
        self.original_encoder = original_encoder
        self.fusion_method = fusion_method
        self.fusion_conv: nn.Module = nn.Linear(1, 1)  # type: ignore
        
        if fusion_method == "concat":
            # 1x1 conv to reduce 2 channels to 1
            self.fusion_conv = nn.Conv2d(2, 1, kernel_size=1, bias=False)
            nn.init.xavier_uniform_(self.fusion_conv.weight)
        
        elif fusion_method == "weighted":
            if learnable_weights:
                # Learnable weights (initialized to equal weighting)
                self.skeleton_weight = nn.Parameter(torch.tensor(0.5))
                self.distance_weight = nn.Parameter(torch.tensor(0.5))
            else:
                # Fixed weights (skeleton less important than distance field)
                self.register_buffer('skeleton_weight', torch.tensor(0.3))
                self.register_buffer('distance_weight', torch.tensor(0.7))
    
    def _expand_to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand single-channel tensor to 3-channel RGB by replication.
        
        Args:
            x: (B, 1, H, W) single-channel tensor
            
        Returns:
            (B, 3, H, W) RGB tensor
        """
        if x.shape[1] == 1:
            return x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            return x
        else:
            raise ValueError(
                f"Expected 1 or 3 channels, got {x.shape[1]} channels"
            )
    
    def forward(
        self, 
        content_input: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            content_input: (B, C, H, W) where C can be 1 or 2
                - C=1: Normal content image (grayscale)
                - C=2: Skeleton-distance transformed [skeleton, distance]
            
        Returns:
            Same as original encoder output: (h, residual_features)
                - h: Final feature map
                - residual_features: List of intermediate features
        """
        num_channels = content_input.shape[1]
        
        # Handle 1-channel input (normal mode - no skeleton transform)
        if num_channels == 1:
            # Expand grayscale to RGB by replication
            rgb_input = self._expand_to_rgb(content_input)  # (B, 3, H, W)
            return self.original_encoder(rgb_input)
        
        # Handle 2-channel input (skeleton-distance mode)
        elif num_channels == 2:
            # Fuse channels based on method
            if self.fusion_method == "concat":
                # 1x1 conv fusion
                fused = self.fusion_conv(content_input)  # (B, 1, H, W)
            
            elif self.fusion_method == "add":
                # Simple addition
                fused = content_input.sum(dim=1, keepdim=True)  # (B, 1, H, W)
            
            elif self.fusion_method == "weighted":
                # Weighted sum
                skeleton = content_input[:, 0:1, :, :]
                distance = content_input[:, 1:2, :, :]
                
                fused = (
                    self.skeleton_weight * skeleton +
                    self.distance_weight * distance
                )  # (B, 1, H, W)
            
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")
            
            # Expand fused (1-channel) to RGB
            rgb_input = self._expand_to_rgb(fused)  # (B, 3, H, W)
            
            # Pass through original encoder
            return self.original_encoder(rgb_input)
        
        else:
            raise ValueError(
                f"DualChannelContentEncoder expects 1 or 2 channels, got {num_channels}. "
                f"Input shape: {content_input.shape}"
            )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"DualChannelContentEncoder(\n"
            f"  fusion_method={self.fusion_method},\n"
            f"  accepts: 1-channel (normal) or 2-channel (skeleton-distance),\n"
            f"  expands to 3-channel RGB before passing to original_encoder,\n"
            f"  original_encoder={self.original_encoder.__class__.__name__}\n"
            f")"
        )
# ============================================================================
# Utility Functions
# ============================================================================

def visualize_skeleton_distance_transform(
    original_image: np.ndarray,
    skeleton: np.ndarray,
    distance: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Visualize the transformation pipeline.
    
    Args:
        original_image: (H, W) original binary image
        skeleton: (H, W) skeleton
        distance: (H, W) distance field
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Image\n(with stroke thickness)')
    axes[0].axis('off')
    
    # Skeleton
    axes[1].imshow(skeleton, cmap='gray')
    axes[1].set_title('Skeleton\n(1-pixel medial axis)')
    axes[1].axis('off')
    
    # Distance field
    im = axes[2].imshow(distance, cmap='hot')
    axes[2].set_title('Distance Field\n(smooth influence map)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    # Overlay
    axes[3].imshow(original_image, cmap='gray', alpha=0.3)
    axes[3].imshow(skeleton, cmap='Reds', alpha=0.7)
    axes[3].set_title('Overlay\n(skeleton on original)')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def batch_visualize_transforms(
    images: torch.Tensor,
    transform: SkeletonDistanceTransform,
    num_samples: int = 4,
    save_dir: Optional[str] = None,
):
    """
    Visualize transformations for a batch of images.
    
    Args:
        images: (B, C, H, W) batch of images
        transform: SkeletonDistanceTransform instance
        num_samples: Number of samples to visualize
        save_dir: Directory to save visualizations
    """
    from pathlib import Path
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Transform batch
    transformed = transform(images)
    
    for i in range(min(num_samples, images.shape[0])):
        # Get original
        orig = images[i, 0].cpu().numpy()
        
        # Get transformed
        if transform.output_mode == "dual_channel":
            skel = transformed[i, 0].cpu().numpy()
            dist = transformed[i, 1].cpu().numpy()
        else:
            skel = None
            dist = transformed[i, 0].cpu().numpy()
        
        # Visualize
        if skel is not None:
            save_path = f"{save_dir}/sample_{i}.png" if save_dir else None
            visualize_skeleton_distance_transform(orig, skel, dist, save_path)


# ============================================================================
# Example Usage
# ============================================================================

def example_usage():
    """Example showing how to use skeleton-distance transform."""
    
    # Create transform
    transform = SkeletonDistanceTransform(
        method="medial_axis",
        distance_method="hybrid",
        max_distance=10.0,
        sigma=3.0,
        output_mode="dual_channel",
    )
    
    # Create dummy image (simulating a character)
    import torch
    
    # Simulate a thick stroke character
    dummy_image = torch.zeros(1, 1, 96, 96)
    dummy_image[0, 0, 30:70, 40:45] = 1.0  # Vertical stroke
    dummy_image[0, 0, 45:50, 30:70] = 1.0  # Horizontal stroke
    
    # Transform
    transformed = transform(dummy_image)
    
    print(f"Input shape: {dummy_image.shape}")
    print(f"Output shape: {transformed.shape}")
    print(f"Output channels: {transformed.shape[1]} (skeleton + distance)")
    
    # Visualize
    orig = dummy_image[0, 0].cpu().numpy()
    skel = transformed[0, 0].cpu().numpy()
    dist = transformed[0, 1].cpu().numpy()
    
    visualize_skeleton_distance_transform(orig, skel, dist)


if __name__ == "__main__":
    example_usage()