"""
Robust Font Manager for FontDiffuser Batch Processing
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Set
from collections import defaultdict
from functools import lru_cache
import logging
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PIL import Image
import pygame
import pygame.freetype
from fontTools.ttLib import TTFont
import cv2
from utils import is_char_in_font, load_ttf


def ttf2im_robust(font, char, canvas_size=256):
    """
    Robust font rendering with proper centering and edge protection.

    Args:
        font: pygame.freetype.Font object
        char: Character to render
        canvas_size: Size of output square image

    Returns:
        PIL Image of the rendered character, or None on failure
    """
    try:
        # Try to render the character
        surface, _ = font.render(char, size=canvas_size)
    except Exception as e:
        print(f"Failed to render character '{char}': {e}")
        return None

    try:
        # Create white background
        bg = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

        # Get alpha channel and invert (black character on transparent background)
        imo = pygame.surfarray.pixels_alpha(surface).transpose(1, 0)
        imo = 255 - np.array(Image.fromarray(imo))

        # Make a copy for modification
        result = bg.copy()

        # Get original dimensions
        h, w = imo.shape[:2]

        # Resize only if character exceeds canvas - with safety
        if h > canvas_size or w > canvas_size:
            # Calculate scaling factor to fit within canvas
            scale = min(canvas_size / h, canvas_size / w)

            # Ensure minimum dimensions of 1 pixel
            new_h = max(1, int(h * scale))
            new_w = max(1, int(w * scale))

            # Resize with area interpolation (good for downscaling)
            imo = cv2.resize(imo, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w  # Update dimensions

        # Calculate centering position with bounds checking
        x = (canvas_size - w) // 2
        y = (canvas_size - h) // 2

        # Ensure positions are within valid range
        x = max(0, min(x, canvas_size - w))
        y = max(0, min(y, canvas_size - h))

        # Calculate end positions
        end_x = x + w
        end_y = y + h

        # Final bounds check
        if (
            x >= 0
            and y >= 0
            and end_x <= canvas_size
            and end_y <= canvas_size
            and w > 0
            and h > 0
        ):
            # CORRECT placement: im[y:y+h, x:x+w] = imo
            result[y:end_y, x:end_x] = imo
        else:
            # Fallback: Use the entire canvas if placement fails
            print(f"Warning: Placement error for '{char}', using center crop")
            if w <= canvas_size and h <= canvas_size:
                # Simple center placement
                center_x = (canvas_size - w) // 2
                center_y = (canvas_size - h) // 2
                result[center_y : center_y + h, center_x : center_x + w] = imo

        # Convert to PIL Image
        pil_image = Image.fromarray(result.astype("uint8")).convert("RGB")

        # Optional: Add a visual debug border for problematic characters
        if hasattr(font, "_debug") and font._debug:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(pil_image)
            draw.rectangle(
                [0, 0, canvas_size - 1, canvas_size - 1], outline="red", width=1
            )

        return pil_image

    except Exception as e:
        print(f"Error processing character '{char}': {e}")
        return None


class FontRenderer:
    """
    Robust font renderer with fixed ttf2im and additional safety checks
    """

    def __init__(self, font_size: int = 256, canvas_size: int = 256):
        """
        Args:
            font_size: Font size for rendering
            canvas_size: Size of output image (should be >= font_size)
        """
        self.font_size = font_size
        self.canvas_size = canvas_size

        # Initialize pygame for font rendering
        if not pygame.get_init():
            pygame.init()

    def ttf2im_fixed(
        self, font, char: str, debug: bool = False
    ) -> Optional[Image.Image]:
        """
        Fixed version of ttf2im that properly centers characters without cropping

        Args:
            font: pygame.freetype.Font object
            char: Character to render
            debug: Whether to save debug images

        Returns:
            PIL Image of rendered character or None on failure
        """
        return ttf2im_robust(font, char, self.canvas_size)

    def _create_default_x_marker(self, char: str) -> Image.Image:
        """Create a default X marker for invalid/unrenderable characters"""
        img = np.full((self.canvas_size, self.canvas_size, 3), 255, dtype=np.uint8)

        # Draw a red X
        cv2.line(
            img,
            (50, 50),
            (self.canvas_size - 50, self.canvas_size - 50),
            (255, 0, 0),
            5,
        )
        cv2.line(
            img,
            (self.canvas_size - 50, 50),
            (50, self.canvas_size - 50),
            (255, 0, 0),
            5,
        )

        # Add character label
        cv2.putText(
            img,
            char,
            (self.canvas_size // 2 - 20, self.canvas_size // 2 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )

        return Image.fromarray(img)

    def render_with_safety_check(self, font, char: str) -> Optional[Image.Image]:
        """
        Render character with additional safety checks for edge cases
        """
        result = self.ttf2im_fixed(font, char, debug=False)

        if result is None:
            return None

        # Convert to numpy for analysis
        img_array = np.array(result.convert("L"))  # Convert to grayscale

        # Check for edge artifacts (character touching edges)
        edge_thickness = 2
        top_edge = img_array[:edge_thickness, :]
        bottom_edge = img_array[-edge_thickness:, :]
        left_edge = img_array[:, :edge_thickness]
        right_edge = img_array[:, -edge_thickness:]

        # Check if character is too close to edges
        edge_threshold = 200  # Below this value indicates character pixel
        edges_touching = [
            np.any(top_edge < edge_threshold),
            np.any(bottom_edge < edge_threshold),
            np.any(left_edge < edge_threshold),
            np.any(right_edge < edge_threshold),
        ]

        if any(edges_touching):
            # Character is touching edges, render with more padding
            print(f"  Warning: Character '{char}' touches edges, adjusting...")
            # Re-render with larger canvas
            original_size = self.canvas_size
            self.canvas_size = int(original_size * 1.5)
            result = self.ttf2im_fixed(font, char, debug=False)
            self.canvas_size = original_size

            # Resize back to original size
            if result:
                result = result.resize((original_size, original_size), Image.LANCZOS)

        return result


class FontManager:
    """Manages multiple font files"""

    def __init__(self, ttf_path: str) -> None:
        """
        Initialize font manager

        Args:
            ttf_path: Path to a single font file or directory containing fonts
        """
        self.fonts: Dict[str, Dict[str, Any]] = {}
        self.font_paths: List[str] = []
        self._load_fonts(ttf_path)

    def _load_fonts(self, ttf_path: str) -> None:
        """Load font(s) from path"""
        if "*" in ttf_path:
            # Handle wildcard path
            import glob

            font_files: List[str] = glob.glob(ttf_path)
            if not font_files:
                raise ValueError(f"No font files found for pattern: {ttf_path}")

            self.font_paths = sorted(font_files)

            logging.info(f"{'=' * 60}")
            logging.info(f"Loading {len(font_files)} fonts from wildcard path...")
            logging.info("=" * 60)

            for font_path in self.font_paths:
                font_name: str = os.path.splitext(os.path.basename(font_path))[0]
                try:
                    self.fonts[font_name] = {
                        "path": font_path,
                        "font": load_ttf(font_path),
                        "name": font_name,
                    }
                    logging.info(f"✓ Loaded: {font_name}")
                except Exception as e:
                    logging.info(f"✗ Failed to load {font_name}: {e}")

            logging.info("=" * 60)
            logging.info(f"Successfully loaded {len(self.fonts)} fonts\n")

        elif os.path.isfile(ttf_path):
            # Single font file
            self.font_paths = [ttf_path]
            font_name: str = os.path.splitext(os.path.basename(ttf_path))[0]
            self.fonts[font_name] = {
                "path": ttf_path,
                "font": load_ttf(ttf_path),
                "name": font_name,
            }
            logging.info(f"✓ Loaded font: {font_name}")

        elif os.path.isdir(ttf_path):
            # Directory with multiple fonts
            font_extensions: Set[str] = {".ttf", ".otf", ".TTF", ".OTF"}
            font_files: List[str] = [
                os.path.join(ttf_path, f)
                for f in os.listdir(ttf_path)
                if os.path.splitext(f)[1] in font_extensions
            ]

            if not font_files:
                raise ValueError(f"No font files found in directory: {ttf_path}")

            self.font_paths = sorted(font_files)

            logging.info(f"{'=' * 60}")
            logging.info(f"Loading {len(font_files)} fonts from directory...")
            logging.info("=" * 60)

            for font_path in self.font_paths:
                font_name: str = os.path.splitext(os.path.basename(font_path))[0]
                try:
                    self.fonts[font_name] = {
                        "path": font_path,
                        "font": load_ttf(font_path),
                        "name": font_name,
                    }
                    logging.info(f"✓ Loaded: {font_name}")
                except Exception as e:
                    logging.info(f"✗ Failed to load {font_name}: {e}")

            logging.info("=" * 60)
            logging.info(f"Successfully loaded {len(self.fonts)} fonts\n")
        else:
            raise ValueError(f"Invalid ttf_path: {ttf_path}")

    def get_font_names(self) -> List[str]:
        """Get list of loaded font names"""
        return list(self.fonts.keys())

    def get_font(self, font_name: str) -> Any:
        """Get font object by name"""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        return self.fonts[font_name]["font"]

    def get_font_path(self, font_name: str) -> str:
        """Get font file path by name"""
        if font_name not in self.fonts:
            raise ValueError(f"Font not found: {font_name}")
        return self.fonts[font_name]["path"]

    def is_char_in_font(self, font_name: str, char: str) -> bool:
        """Check if character exists in font"""
        font_path: str = self.get_font_path(font_name)
        return is_char_in_font(font_path, char)

    def get_available_chars_for_font(
        self, font_name: str, characters: List[str]
    ) -> List[str]:
        """Get list of characters available in specific font"""
        return [char for char in characters if self.is_char_in_font(font_name, char)]
