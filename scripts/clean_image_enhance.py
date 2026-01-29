"""
Enhanced Image Cleaning Script for FontDiffuser Style References

Improvements over original:
1. Better handling of colored/gradient backgrounds (common in internet images)
2. Multiple binarization methods with automatic selection
3. Better noise removal for low-quality images
4. Stroke thickness normalization
5. Optional visual verification mode
6. Batch processing with progress tracking
7. Error handling and logging

Usage:
    python clean_style_images.py input_folder --output cleaned --size 256
    python clean_style_images.py input_folder --method auto --verify
"""

import cv2
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Tuple, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class CharacterImageCleaner:
    """Enhanced character image cleaning for font style references."""
    
    def __init__(
        self,
        target_size: int = 256,
        padding: int = 30,
        method: str = 'auto',
        denoise_strength: int = 3,
        min_stroke_thickness: int = 2,
    ):
        """
        Initialize cleaner.
        
        Args:
            target_size: Output image size (square)
            padding: Padding around character in pixels
            method: Binarization method ('otsu', 'adaptive', 'auto')
            denoise_strength: Denoising kernel size (odd number)
            min_stroke_thickness: Minimum stroke thickness in pixels
        """
        self.target_size = target_size
        self.padding = padding
        self.method = method
        self.denoise_strength = denoise_strength if denoise_strength % 2 == 1 else denoise_strength + 1
        self.min_stroke_thickness = min_stroke_thickness
    
    def detect_background_color(self, gray: np.ndarray) -> str:
        """
        Detect if background is light or dark.
        
        Args:
            gray: Grayscale image
            
        Returns:
            'light' or 'dark'
        """
        # Sample corners to detect background
        h, w = gray.shape
        border_size = min(h, w) // 10
        
        corners = [
            gray[:border_size, :border_size],  # Top-left
            gray[:border_size, -border_size:],  # Top-right
            gray[-border_size:, :border_size],  # Bottom-left
            gray[-border_size:, -border_size:],  # Bottom-right
        ]
        
        avg_corner_brightness = np.mean([corner.mean() for corner in corners])
        
        return 'light' if avg_corner_brightness > 127 else 'dark'
    
    def remove_colored_background(self, img: np.ndarray) -> np.ndarray:
        """
        Remove colored backgrounds (common in internet images).
        
        Args:
            img: BGR image
            
        Returns:
            Grayscale image with background removed
        """
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Calculate saturation mask (colored regions)
        saturation = hsv[:, :, 1]
        
        # If image has high saturation variance, it likely has colored background
        if saturation.std() > 30:
            # Create mask for low-saturation areas (likely text)
            # This works for both colored backgrounds and colored text
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Use color information to enhance separation
            # Characters are often more uniform in color than backgrounds
            b, g, r = cv2.split(img)
            
            # Calculate color variance
            color_std = np.std([b, g, r], axis=0)
            
            # Areas with low color variance are likely text
            text_mask = color_std < np.percentile(color_std, 30)
            
            # Enhance text regions
            enhanced = gray.copy()
            enhanced[text_mask] = np.clip(enhanced[text_mask] * 1.2, 0, 255).astype(np.uint8)
            
            return enhanced
        else:
            # Simple grayscale conversion
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    def binarize_otsu(self, gray: np.ndarray) -> np.ndarray:
        """Binarize using Otsu's method."""
        # Apply slight blur before Otsu
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return binary
    
    def binarize_adaptive(self, gray: np.ndarray, block_size: int = 25, c: int = 10) -> np.ndarray:
        """Binarize using adaptive thresholding."""
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, c
        )
        return binary
    
    def binarize_sauvola(self, gray: np.ndarray, window_size: int = 25, k: float = 0.2) -> np.ndarray:
        """
        Binarize using Sauvola's method (better for degraded documents).
        
        This is excellent for scanned historical documents with uneven illumination.
        """
        # Ensure window size is odd
        if window_size % 2 == 0:
            window_size += 1
        
        # Calculate local mean and standard deviation
        mean = cv2.boxFilter(gray.astype(np.float32), -1, (window_size, window_size))
        sqmean = cv2.boxFilter(gray.astype(np.float32)**2, -1, (window_size, window_size))
        std = np.sqrt(sqmean - mean**2)
        
        # Sauvola threshold
        R = std.max() if std.max() > 0 else 128
        threshold = mean * (1 + k * ((std / R) - 1))
        
        binary = (gray > threshold).astype(np.uint8) * 255
        binary = cv2.bitwise_not(binary)
        
        return binary
    
    def auto_select_method(self, gray: np.ndarray) -> np.ndarray:
        """
        Automatically select best binarization method.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Binary image
        """
        # Try different methods
        otsu_binary = self.binarize_otsu(gray)
        adaptive_binary = self.binarize_adaptive(gray)
        
        # Evaluate which gives better result
        # Metric: foreground ratio should be reasonable (5-60% for typical characters)
        otsu_fg_ratio = np.sum(otsu_binary > 0) / otsu_binary.size
        adaptive_fg_ratio = np.sum(adaptive_binary > 0) / adaptive_binary.size
        
        # Check if image has uneven illumination
        h, w = gray.shape
        quadrants = [
            gray[:h//2, :w//2],
            gray[:h//2, w//2:],
            gray[h//2:, :w//2],
            gray[h//2:, w//2:]
        ]
        illumination_variance = np.std([q.mean() for q in quadrants])
        
        # If high illumination variance, use adaptive or Sauvola
        if illumination_variance > 20:
            sauvola_binary = self.binarize_sauvola(gray)
            sauvola_fg_ratio = np.sum(sauvola_binary > 0) / sauvola_binary.size
            
            # Choose based on foreground ratio
            if 0.05 < sauvola_fg_ratio < 0.6:
                return sauvola_binary
            elif 0.05 < adaptive_fg_ratio < 0.6:
                return adaptive_binary
            else:
                return otsu_binary
        
        # Otherwise use Otsu if ratio is reasonable, else adaptive
        if 0.05 < otsu_fg_ratio < 0.6:
            return otsu_binary
        else:
            return adaptive_binary
    
    def remove_noise(self, binary: np.ndarray, min_component_size: int = 20) -> np.ndarray:
        """
        Remove noise using connected component analysis.
        
        Args:
            binary: Binary image
            min_component_size: Minimum component size to keep
            
        Returns:
            Cleaned binary image
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Keep only large components (likely the character)
        cleaned = np.zeros_like(binary)
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_component_size:
                cleaned[labels == i] = 255
        
        # If we removed everything, return original
        if np.sum(cleaned) == 0:
            return binary
        
        return cleaned
    
    def normalize_stroke_thickness(self, binary: np.ndarray) -> np.ndarray:
        """
        Normalize stroke thickness for consistent style.
        
        Args:
            binary: Binary image
            
        Returns:
            Image with normalized strokes
        """
        # Calculate average stroke thickness using distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        avg_thickness = np.mean(dist_transform[dist_transform > 0]) * 2
        
        # If strokes are too thin, dilate slightly
        if avg_thickness < self.min_stroke_thickness:
            kernel_size = int(self.min_stroke_thickness - avg_thickness) + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            binary = cv2.dilate(binary, kernel, iterations=1)
        
        # If strokes are too thick, erode slightly
        elif avg_thickness > self.min_stroke_thickness * 3:
            kernel_size = 2
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            binary = cv2.erode(binary, kernel, iterations=1)
        
        return binary
    
    def center_character(self, binary: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Center character on canvas with proper padding.
        
        Args:
            binary: Binary image (white text on black background)
            
        Returns:
            (centered_image, success)
        """
        # Find bounding box
        coords = cv2.findNonZero(binary)
        if coords is None:
            return np.zeros((self.target_size, self.target_size), dtype=np.uint8), False
        
        x, y, w, h = cv2.boundingRect(coords)
        character_crop = binary[y:y+h, x:x+w]
        
        # Calculate resizing to fit canvas while maintaining aspect ratio
        max_glyph_dim = self.target_size - (self.padding * 2)
        
        if w == 0 or h == 0:
            return np.zeros((self.target_size, self.target_size), dtype=np.uint8), False
        
        ratio = min(max_glyph_dim / w, max_glyph_dim / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # Use INTER_AREA for shrinking, INTER_CUBIC for enlarging
        interpolation = cv2.INTER_AREA if ratio < 1 else cv2.INTER_CUBIC
        resized_char = cv2.resize(character_crop, (new_w, new_h), interpolation=interpolation)
        
        # Create black canvas and paste character in the middle
        canvas = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        x_off, y_off = (self.target_size - new_w) // 2, (self.target_size - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized_char
        
        return canvas, True
    
    def clean_image(
        self, 
        image_path: Path,
        debug: bool = False
    ) -> Tuple[Optional[np.ndarray], dict]:
        """
        Complete cleaning pipeline.
        
        Args:
            image_path: Path to input image
            debug: Return debug info
            
        Returns:
            (cleaned_image, debug_info)
        """
        debug_info = {}
        
        # 1. Load Image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.error(f"Failed to load image: {image_path}")
            return None, debug_info
        
        debug_info['original_shape'] = img.shape
        
        # 2. Remove colored background and convert to grayscale
        gray = self.remove_colored_background(img)
        debug_info['has_color'] = len(img.shape) == 3 and img.shape[2] == 3
        
        # 3. Denoising
        if self.denoise_strength > 1:
            denoised = cv2.medianBlur(gray, self.denoise_strength)
        else:
            denoised = gray
        
        # 4. Detect background type
        bg_type = self.detect_background_color(denoised)
        debug_info['background_type'] = bg_type
        
        # If dark background, invert before processing
        if bg_type == 'dark':
            denoised = cv2.bitwise_not(denoised)
        
        # 5. Binarization
        if self.method == 'auto':
            binary = self.auto_select_method(denoised)
            debug_info['method_used'] = 'auto'
        elif self.method == 'otsu':
            binary = self.binarize_otsu(denoised)
            debug_info['method_used'] = 'otsu'
        elif self.method == 'adaptive':
            binary = self.binarize_adaptive(denoised)
            debug_info['method_used'] = 'adaptive'
        elif self.method == 'sauvola':
            binary = self.binarize_sauvola(denoised)
            debug_info['method_used'] = 'sauvola'
        else:
            logger.warning(f"Unknown method: {self.method}, using auto")
            binary = self.auto_select_method(denoised)
            debug_info['method_used'] = 'auto'
        
        # 6. Morphology Cleanup
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 7. Remove noise (small components)
        binary = self.remove_noise(binary, min_component_size=20)
        
        # 8. Normalize stroke thickness
        binary = self.normalize_stroke_thickness(binary)
        
        # 9. Center character
        centered, success = self.center_character(binary)
        
        if not success:
            logger.error(f"Failed to center character: {image_path}")
            return None, debug_info
        
        # 10. Final output: Invert so we have black strokes on white background
        final_output = cv2.bitwise_not(centered)
        
        debug_info['success'] = True
        debug_info['foreground_ratio'] = np.sum(centered > 0) / centered.size
        
        return final_output, debug_info


def visualize_cleaning_steps(
    image_path: Path,
    cleaner: CharacterImageCleaner,
    output_path: Optional[Path] = None
):
    """
    Visualize all cleaning steps for debugging.
    
    Args:
        image_path: Input image path
        cleaner: Cleaner instance
        output_path: Optional output path for visualization
    """
    import matplotlib.pyplot as plt
    
    # Load and process with debug info
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Failed to load: {image_path}")
        return
    
    # Manual step-by-step for visualization
    gray = cleaner.remove_colored_background(img)
    denoised = cv2.medianBlur(gray, cleaner.denoise_strength)
    
    bg_type = cleaner.detect_background_color(denoised)
    if bg_type == 'dark':
        denoised = cv2.bitwise_not(denoised)
    
    binary_otsu = cleaner.binarize_otsu(denoised)
    binary_adaptive = cleaner.binarize_adaptive(denoised)
    binary_sauvola = cleaner.binarize_sauvola(denoised)
    binary_auto = cleaner.auto_select_method(denoised)
    
    # Final cleaned
    final, _ = cleaner.clean_image(image_path)
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gray, cmap='gray')
    axes[0, 1].set_title('Grayscale (Color Removed)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(denoised, cmap='gray')
    axes[0, 2].set_title(f'Denoised ({bg_type} bg)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(binary_otsu, cmap='gray')
    axes[1, 0].set_title('Otsu Binarization')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(binary_adaptive, cmap='gray')
    axes[1, 1].set_title('Adaptive Binarization')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(binary_sauvola, cmap='gray')
    axes[1, 2].set_title('Sauvola Binarization')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(binary_auto, cmap='gray')
    axes[2, 0].set_title('Auto-selected Method')
    axes[2, 0].axis('off')
    
    if final is not None:
        axes[2, 1].imshow(final, cmap='gray')
        axes[2, 1].set_title('Final Output')
        axes[2, 1].axis('off')
    
    # Hide last subplot
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def batch_clean_images(
    input_dir: Path,
    output_dir: Path,
    cleaner: CharacterImageCleaner,
    verify: bool = False,
    verification_samples: int = 5
):
    """
    Batch clean all images in directory.
    
    Args:
        input_dir: Input directory
        output_dir: Output directory
        cleaner: Cleaner instance
        verify: Generate verification visualizations
        verification_samples: Number of samples to visualize
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = [f for f in input_dir.iterdir() if f.suffix.lower() in valid_exts]
    
    if not files:
        logger.error(f"No image files found in {input_dir}")
        return
    
    logger.info(f"Found {len(files)} images")
    
    # Process all images
    success_count = 0
    failed_files = []
    
    for i, image_path in enumerate(tqdm(files, desc="Cleaning images")):
        output_path = output_dir / f"{image_path.stem}_cleaned.png"
        
        cleaned_img, debug_info = cleaner.clean_image(image_path, debug=True)
        
        if cleaned_img is not None:
            cv2.imwrite(str(output_path), cleaned_img)
            success_count += 1
            
            # Generate verification visualization for first few images
            if verify and i < verification_samples:
                verify_output = output_dir / f"{image_path.stem}_verification.png"
                visualize_cleaning_steps(image_path, cleaner, verify_output)
        else:
            failed_files.append(image_path.name)
            logger.warning(f"Failed to clean: {image_path.name}")
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info(f"Cleaning Complete!")
    logger.info(f"{'='*60}")
    logger.info(f"Successfully cleaned: {success_count}/{len(files)} images")
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    if failed_files:
        logger.warning(f"\nFailed files ({len(failed_files)}):")
        for fname in failed_files[:10]:  # Show first 10
            logger.warning(f"  - {fname}")
        if len(failed_files) > 10:
            logger.warning(f"  ... and {len(failed_files) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Clean and prepare font style reference images for FontDiffuser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python clean_style_images.py input_folder --output cleaned
  
  # With auto method selection and verification
  python clean_style_images.py input_folder --method auto --verify
  
  # For degraded/historical documents
  python clean_style_images.py scanned_docs --method sauvola --denoise 5
  
  # Larger output size with more padding
  python clean_style_images.py input_folder --size 512 --padding 50
        """
    )
    
    parser.add_argument(
        "input_folder", 
        type=str, 
        help="Folder containing raw images"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="cleaned_images",
        help="Output folder name (default: cleaned_images)"
    )
    parser.add_argument(
        "--size", 
        type=int, 
        default=256,
        help="Target square size in pixels (default: 256)"
    )
    parser.add_argument(
        "--padding", 
        type=int, 
        default=30,
        help="Padding around character in pixels (default: 30)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=['auto', 'otsu', 'adaptive', 'sauvola'],
        default='auto',
        help="Binarization method (default: auto)"
    )
    parser.add_argument(
        "--denoise",
        type=int,
        default=3,
        help="Denoising strength (odd number, default: 3)"
    )
    parser.add_argument(
        "--min-stroke",
        type=int,
        default=2,
        help="Minimum stroke thickness in pixels (default: 2)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Generate verification visualizations for first few images"
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=5,
        help="Number of verification samples to generate (default: 5)"
    )
    
    args = parser.parse_args()
    
    # Validate input
    input_dir = Path(args.input_folder)
    if not input_dir.exists():
        logger.error(f"Input folder not found: {input_dir}")
        return
    
    output_dir = Path(args.output)
    
    # Create cleaner
    cleaner = CharacterImageCleaner(
        target_size=args.size,
        padding=args.padding,
        method=args.method,
        denoise_strength=args.denoise,
        min_stroke_thickness=args.min_stroke,
    )
    
    # Process images
    batch_clean_images(
        input_dir=input_dir,
        output_dir=output_dir,
        cleaner=cleaner,
        verify=args.verify,
        verification_samples=args.verify_samples
    )


if __name__ == '__main__':
    main()