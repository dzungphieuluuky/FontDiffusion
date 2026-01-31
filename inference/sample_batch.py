import os
import sys
import time
import json
import argparse
from huggingface_hub.utils import enable_progress_bars
import logging

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from argparse import Namespace, ArgumentParser

from src.dpm_solver.pipeline_dpm_solver import FontDiffuserDPMPipeline
from src.tools.utilities import HFTqdm
from src.tools.utils import (
    load_ttf,
    ttf2im,
    is_char_in_font,
)
from src.tools.filename_utils import (
    get_content_filename,
    get_target_filename,
    compute_file_hash,
)

# Import from same package (inference/)
from inference.sample_optimized import (
    load_fontdiffuser_pipeline,
    get_content_transform,
    get_style_transform,
)
from src.configs.fontdiffuser import get_parser

logger = logging.getLogger(__name__)
enable_progress_bars()

# Import evaluation metrics
try:
    import lpips

    LPIPS_AVAILABLE: bool = True
except ImportError:
    logger.info("Warning: lpips not available. Install with: pip install lpips")
    LPIPS_AVAILABLE: bool = False

try:
    from pytorch_fid import fid_score

    FID_AVAILABLE: bool = True
except ImportError:
    logger.info(
        "Warning: pytorch-fid not available. Install with: pip install pytorch-fid"
    )
    FID_AVAILABLE: bool = False

try:
    from skimage.metrics import structural_similarity as ssim

    SSIM_AVAILABLE: bool = True
except ImportError:
    logger.info(
        "Warning: scikit-image not available. Install with: pip install scikit-image"
    )
    SSIM_AVAILABLE: bool = False

try:
    import wandb

    WANDB_AVAILABLE: bool = True
except ImportError:
    logger.info("Warning: wandb not available. Install with: pip install wandb")
    WANDB_AVAILABLE: bool = False


class FontManager:
    """Manages multiple font files"""

    def __init__(self, ttf_path: str) -> None:
        """
        Initialize font manager

        Args:
            ttf_path: Path to a single font file or directory containing fonts
        """
        self.fonts: dict[str, dict[str]] = {}
        self.font_paths: list[str] = []
        self._load_fonts(ttf_path)

    def _load_fonts(self, ttf_path: str) -> None:
        """Load font(s) from path"""
        if "*" in ttf_path:
            # Handle wildcard path
            import glob

            font_files: list[str] = glob.glob(ttf_path)
            if not font_files:
                raise ValueError(f"No font files found for pattern: {ttf_path}")

            self.font_paths = sorted(font_files)

            logger.info(f"{'=' * 60}")
            logger.info(f"Loading {len(font_files)} fonts from wildcard path...")
            logger.info("=" * 60)

            for font_path in self.font_paths:
                font_name: str = os.path.splitext(os.path.basename(font_path))[0]
                try:
                    self.fonts[font_name] = {
                        "path": font_path,
                        "font": load_ttf(font_path),
                        "name": font_name,
                    }
                    logger.info(f"✓ Loaded: {font_name}")
                except Exception as e:
                    logger.info(f"✗ Failed to load {font_name}: {e}")

            logger.info("=" * 60)
            logger.info(f"Successfully loaded {len(self.fonts)} fonts\n")

        elif os.path.isfile(ttf_path):
            # Single font file
            self.font_paths = [ttf_path]
            font_name: str = os.path.splitext(os.path.basename(ttf_path))[0]
            self.fonts[font_name] = {
                "path": ttf_path,
                "font": load_ttf(ttf_path),
                "name": font_name,
            }
            logger.info(f"✓ Loaded font: {font_name}")

        elif os.path.isdir(ttf_path):
            # Directory with multiple fonts
            font_extensions: set[str] = {".ttf", ".otf", ".TTF", ".OTF"}
            font_files: list[str] = [
                os.path.join(ttf_path, f)
                for f in os.listdir(ttf_path)
                if os.path.splitext(f)[1] in font_extensions
            ]

            if not font_files:
                raise ValueError(f"No font files found in directory: {ttf_path}")

            self.font_paths = sorted(font_files)

            logger.info(f"{'=' * 60}")
            logger.info(f"Loading {len(font_files)} fonts from directory...")
            logger.info("=" * 60)

            for font_path in self.font_paths:
                font_name: str = os.path.splitext(os.path.basename(font_path))[0]
                try:
                    self.fonts[font_name] = {
                        "path": font_path,
                        "font": load_ttf(font_path),
                        "name": font_name,
                    }
                    logger.info(f"✓ Loaded: {font_name}")
                except Exception as e:
                    logger.info(f"✗ Failed to load {font_name}: {e}")

            logger.info("=" * 60)
            logger.info(f"Successfully loaded {len(self.fonts)} fonts\n")
        else:
            raise ValueError(f"Invalid ttf_path: {ttf_path}")

    def get_font_names(self) -> list[str]:
        """Get list of loaded font names"""
        return list(self.fonts.keys())

    def get_font(self, font_name: str):
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
        self, font_name: str, characters: list[str]
    ) -> list[str]:
        """Get list of characters available in specific font"""
        return [char for char in characters if self.is_char_in_font(font_name, char)]


class GenerationTracker:
    """
    ✅ Tracks which (character, style, font) combinations have been generated
    Uses hash-based checking for fast lookups
    """

    def __init__(self, checkpoint_path: str | None):
        """
        Initialize generation tracker

        Args:
            checkpoint_path: Path to results_checkpoint.json file
        """
        self.generated_hashes: set[str] = set()
        self.generations: list[dict[str, str]] = []

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_from_checkpoint(checkpoint_path)

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load existing generations from checkpoint"""
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results: dict[str, list[dict[str, str]]] = json.load(f)

            raw_generations: list[dict[str, str]] = results.get("generations", [])

            seen_hashes: set[str] = set()
            unique_generations: list[dict[str, str]] = []
            duplicate_count: int = 0

            # Build hash set for fast lookup and deduplicate
            for gen in raw_generations:
                target_hash: str = gen.get("target_hash")

                if not target_hash:
                    # Compute hash if not in checkpoint
                    char: str = gen.get("character", "")
                    style: str = gen.get("style", "")
                    font: str = gen.get("font", "")

                    # Skip invalid entries
                    if not char or not style:
                        continue

                    target_hash = compute_file_hash(char, style, font)

                if target_hash in seen_hashes:
                    duplicate_count += 1
                    continue  # Skip duplicate

                # Add to collections
                seen_hashes.add(target_hash)
                self.generated_hashes.add(target_hash)
                unique_generations.append(gen)

            self.generations = unique_generations

            logger.info(
                f"✓ Loaded checkpoint: {len(self.generations)} unique generations"
            )
            if duplicate_count > 0:
                logger.info(f"  ⚠️  Removed {duplicate_count} duplicate entries")
            logger.info(f"  Total raw entries: {len(raw_generations)}")

        except Exception as e:
            logger.info(f"⚠ Error loading checkpoint: {e}")
            import traceback

            traceback.print_exc()

    def is_generated(self, char: str, style: str, font: str = "") -> bool:
        """Check if (char, style, font) combination has been generated"""
        target_hash = compute_file_hash(char, style, font)
        return target_hash in self.generated_hashes

    def mark_generated(self, char: str, style: str, font: str = "") -> None:
        """Mark a (char, style, font) combination as generated"""
        target_hash = compute_file_hash(char, style, font)
        self.generated_hashes.add(target_hash)

    def add_generation(self, generation: dict[str, str]) -> None:
        """Add a generation record"""
        self.generations.append(generation)

        # Also add to hash set
        char = generation.get("character", "")
        style = generation.get("style", "")
        font = generation.get("font", "")
        self.mark_generated(char, style, font)


class QualityEvaluator:
    """Evaluates generated images using LPIPS, SSIM, and FID"""

    def __init__(self, device: str = "cuda:0") -> None:
        self.device: str = device

        # Initialize LPIPS
        if LPIPS_AVAILABLE:
            self.lpips_fn: lpips.LPIPS = lpips.LPIPS(net="alex").to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None

        self.transform_to_tensor: transforms.ToTensor = transforms.ToTensor()

    def compute_lpips(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute LPIPS between two images"""
        if not LPIPS_AVAILABLE or self.lpips_fn is None:
            return -1.0

        try:
            # Convert to tensors [-1, 1]
            img1_tensor: torch.Tensor = (
                self.transform_to_tensor(img1).unsqueeze(0).to(self.device) * 2 - 1
            )
            img2_tensor: torch.Tensor = (
                self.transform_to_tensor(img2).unsqueeze(0).to(self.device) * 2 - 1
            )

            with torch.inference_mode():
                lpips_value: float = self.lpips_fn(img1_tensor, img2_tensor).item()

            return lpips_value
        except Exception as e:
            logger.info(f"Error computing LPIPS: {e}")
            return -1.0

    def compute_ssim(self, img1: Image.Image, img2: Image.Image) -> float:
        """Compute SSIM between two images"""
        if not SSIM_AVAILABLE:
            return -1.0

        try:
            # Convert to grayscale numpy arrays
            img1_gray: np.ndarray = np.array(img1.convert("L"))
            img2_gray: np.ndarray = np.array(img2.convert("L"))

            ssim_value: float = ssim(img1_gray, img2_gray, data_range=255)
            return ssim_value
        except Exception as e:
            logger.info(f"Error computing SSIM: {e}")
            return -1.0

    def compute_fid(self, real_dir: str, fake_dir: str) -> float:
        """Compute FID between two directories of images"""
        if not FID_AVAILABLE:
            return -1.0

        try:
            fid_value: float = fid_score.calculate_fid_given_paths(
                [real_dir, fake_dir], batch_size=50, device=self.device, dims=2048
            )
            return fid_value
        except Exception as e:
            logger.info(f"Error computing FID: {e}")
            return -1.0

    def save_image(self, image: Image.Image, path: str) -> None:
        """Save PIL image to path"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            image.save(path)
        except Exception as e:
            logger.info(f"Error saving image to {path}: {e}")


def load_characters(
    characters_arg: str, start_line: int = 1, end_line: int = None
) -> list[str]:
    """Load characters from file or comma-separated string with line range support"""
    chars: list[str] = []
    if os.path.isfile(characters_arg):
        with open(characters_arg, "r", encoding="utf-8") as f:
            all_lines: list[str] = f.readlines()

        # Adjust for 1-indexed input
        start_idx: int = max(0, start_line - 1)
        end_idx: int = (
            len(all_lines) if end_line is None else min(len(all_lines), end_line)
        )

        # ✅ ADD VALIDATION
        if start_idx >= len(all_lines):
            raise ValueError(
                f"❌ start_line ({start_line}) exceeds file length ({len(all_lines)} lines)\n"
                f"   Your file only has {len(all_lines)} lines, but you're trying to start at line {start_line}."
            )

        if start_idx >= end_idx:
            raise ValueError(
                f"❌ Invalid line range: start_line={start_line}, end_line={end_line}\n"
                f"   File has {len(all_lines)} lines.\n"
                f"   Computed range [{start_idx}:{end_idx}] is empty.\n"
                f"   Make sure start_line <= end_line and both are within file bounds."
            )

        logger.info(f"📖 Loading characters from file: {characters_arg}")
        logger.info(
            f"   Lines {start_line} to {end_idx} (total file: {len(all_lines)} lines)"
        )
        logger.info(f"   Processing {end_idx - start_idx} lines...")

        for line_num, line in HFTqdm(
            enumerate(all_lines[start_idx:end_idx], start=start_line),
            total=(end_idx - start_idx),
            desc="📖 Reading character file",
            colour="green",
        ):
            char: str = line.strip()
            if not char:
                continue
            if len(char) != 1:
                logger.info(
                    f"Warning: Skipping line {line_num}: expected 1 char, got {len(char)}: '{char}'"
                )
                continue
            chars.append(char)

    else:
        for c in [x.strip() for x in characters_arg.split(",") if x.strip()]:
            if len(c) != 1:
                raise ValueError(
                    f"Invalid character in argument: '{c}' (must be single char)"
                )
            chars.append(c)

    # ✅ ADD FINAL CHECK
    if not chars:
        raise ValueError(
            f"❌ No valid characters loaded!\n"
            f"   Check your character file or line range (start={start_line}, end={end_line})"
        )

    logger.info(f"✅ Successfully loaded {len(chars)} single characters.")
    return chars


def load_style_images(style_images_arg: str) -> list[tuple[str, str]]:
    """
    Load style image paths and extract style names

    Supports:
    - Directory path: loads all images from directory
    - Glob pattern: e.g., "styles/*.png" or "styles/**/style_*.jpg"
    - Comma-separated paths: "style1.png,style2.png,/path/to/style3.png"
    - Single file path: "style.png"

    Returns: list of (style_path, style_name) tuples
    """
    import glob

    image_exts: set[str] = {".jpg", ".jpeg", ".png", ".bmp"}
    style_paths: list[str] = []

    # Case 1: Directory path
    if os.path.isdir(style_images_arg):
        logger.info(f"📂 Loading style images from directory: {style_images_arg}")

        style_paths: list[str] = [
            os.path.join(style_images_arg, f)
            for f in os.listdir(style_images_arg)
            if os.path.splitext(f)[1].lower() in image_exts
        ]
        style_paths.sort()
        logger.info(f"   Found {len(style_paths)} image files")

    # Case 2: Glob pattern (contains * or ?)
    elif "*" in style_images_arg or "?" in style_images_arg:
        logger.info(f"🔍 Loading style images using glob pattern: {style_images_arg}")

        style_paths = glob.glob(style_images_arg, recursive=True)

        # Filter by image extensions
        style_paths = [
            p
            for p in style_paths
            if os.path.splitext(p)[1].lower() in image_exts and os.path.isfile(p)
        ]

        if not style_paths:
            raise ValueError(
                f"❌ No image files found matching glob pattern: {style_images_arg}"
            )

        style_paths.sort()
        logger.info(f"   Found {len(style_paths)} matching image files")

    # Case 3: Comma-separated paths (files or mixed)
    else:
        raw_paths: list[str] = [p.strip() for p in style_images_arg.split(",")]
        logger.info(f"📋 Loading {len(raw_paths)} specified style image(s)")

        for path in raw_paths:
            if not path:
                continue

            if os.path.isfile(path):
                if os.path.splitext(path)[1].lower() in image_exts:
                    style_paths.append(path)
                else:
                    logger.warning(f"   ⚠️  Skipping unsupported file type: {path}")
            else:
                raise ValueError(f"❌ File not found: {path}")

    if not style_paths:
        raise ValueError("❌ No valid style images found!")

    # Verify and extract style names
    logger.info(f"📂 Verifying {len(style_paths)} style images...")
    verified_paths: list[tuple[str, str]] = []

    for path in HFTqdm(
        style_paths,
        desc="✓ Verifying style images",
        colour="green",
    ):
        if os.path.isfile(path):
            # Extract style name from filename (without extension)
            style_name = os.path.splitext(os.path.basename(path))[0]
            verified_paths.append((path, style_name))
            logger.info(f"   ✓ {style_name}: {path}")
        else:
            logger.warning(f"   ⚠️  File not found: {path}")

    if not verified_paths:
        raise ValueError("❌ No valid style images verified!")

    logger.info(f"✅ Successfully loaded {len(verified_paths)} style images\n")

    return verified_paths


def create_args_namespace(args: Namespace) -> Namespace:
    """Create args namespace for FontDiffuser pipeline"""

    try:
        from src.configs.fontdiffuser import get_parser

        parser: ArgumentParser = get_parser()
        default_args: Namespace = parser.parse_args([])
    except Exception:
        default_args: Namespace = Namespace()

    # Override with user arguments
    for key, value in vars(args).items():
        setattr(default_args, key, value)

    # Ensure image sizes are tuples
    if not hasattr(default_args, "style_image_size"):
        default_args.style_image_size = (96, 96)
    elif isinstance(default_args.style_image_size, int):
        default_args.style_image_size = (
            default_args.style_image_size,
            default_args.style_image_size,
        )

    if not hasattr(default_args, "content_image_size"):
        default_args.content_image_size = (96, 96)
    elif isinstance(default_args.content_image_size, int):
        default_args.content_image_size = (
            default_args.content_image_size,
            default_args.content_image_size,
        )

    # set required attributes
    default_args.demo = False
    default_args.character_input = True
    default_args.save_image = True
    default_args.cache_models = True
    default_args.controlnet = False
    default_args.resolution = 96
    default_args.ground_truth_dir = None

    # Generation parameters
    default_args.algorithm_type = getattr(default_args, "algorithm_type", "dpmsolver++")
    default_args.guidance_type = getattr(
        default_args, "guidance_type", "classifier-free"
    )
    default_args.method = getattr(default_args, "method", "multistep")
    default_args.order = getattr(default_args, "order", 2)
    default_args.model_type = getattr(default_args, "model_type", "noise")
    default_args.t_start = getattr(default_args, "t_start", 1.0)
    default_args.t_end = getattr(default_args, "t_end", 1e-3)
    default_args.skip_type = getattr(default_args, "skip_type", "time_uniform")
    default_args.correcting_x0_fn = getattr(default_args, "correcting_x0_fn", None)
    default_args.content_encoder_downsample_size = getattr(
        default_args, "content_encoder_downsample_size", 3
    )

    return default_args


def save_checkpoint(results: dict[str, str], output_dir: str) -> None:
    """
    ✅ Save results_checkpoint.json (single source of truth)
    """
    try:
        checkpoint_path: str = os.path.join(output_dir, "results_checkpoint.json")

        # Ensure metrics exist
        if "metrics" not in results:
            results["metrics"] = {"lpips": [], "ssim": [], "inference_times": []}

        # Save checkpoint
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        num_gens = len(results.get("generations", []))
        logger.info(f"  ✅ Saved results_checkpoint.json ({num_gens} generations)")

    except Exception as e:
        logger.info(f"  ⚠ Error saving checkpoint: {e}")


def generate_content_images(
    characters: list[str],
    font_manager: FontManager,
    output_dir: str,
    generation_tracker: GenerationTracker,
) -> dict[str, str]:
    """
    Generate and save content character images
    ✅ CORRECTED: Only generates if content image doesn't already exist
    Returns: char_paths dict mapping character to file path
    """
    content_dir: str = os.path.join(output_dir, "ContentImage")
    os.makedirs(content_dir, exist_ok=True)

    font_names: list[str] = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded")

    logger.info(f"{'=' * 60}")
    logger.info(f"Generating Content Images")
    logger.info(f"Using {len(font_names)} fonts")
    logger.info(f"Characters: {len(characters)}")
    logger.info("=" * 60)

    char_paths: dict[str, str] = {}
    chars_without_fonts: list[str] = []
    chars_already_exist: list[str] = []
    generated_new: int = 0

    for char in HFTqdm(
        characters,
        desc="📸 Generating content images",
        colour="magenta",
    ):
        found_font = None
        for font_name in font_names:
            if font_manager.is_char_in_font(font_name, char):
                found_font = font_name
                break

        if not found_font:
            logger.info(f"  ⚠ Warning: '{char}' not in any font, skipping...")
            chars_without_fonts.append(char)
            continue

        try:
            # ✅ Generate expected filename
            content_filename = get_content_filename(char)
            char_path: str = os.path.join(content_dir, content_filename)

            # ✅ Check if content image already exists
            if os.path.exists(char_path):
                logger.info(
                    f"  ✓ Content image already exists for '{char}' at {char_path}"
                )
                char_paths[char] = char_path
                chars_already_exist.append(char)
                continue

            # Generate new content image only if it doesn't exist
            font = font_manager.get_font(found_font)
            content_img: Image.Image = ttf2im(font=font, char=char)

            content_img.save(char_path)
            logger.info(f"  ✓ Generated new content image for '{char}' at {char_path}.")
            char_paths[char] = char_path
            generated_new += 1

        except Exception as e:
            logger.info(f"  ✗ Error generating '{char}': {e}")

    logger.info(f"{'=' * 60}")
    logger.info(f"Content Image Generation Summary:")
    logger.info(f"  Total characters:       {len(characters)}")
    logger.info(f"  Generated (new):        {generated_new}")
    logger.info(f"  Already exist (reused): {len(chars_already_exist)}")
    logger.info(f"  Not in any font:        {len(chars_without_fonts)}")
    logger.info(f"  Total usable:           {len(char_paths)}")
    logger.info("=" * 60)

    return char_paths


def batch_generate_images(
    pipe: FontDiffuserDPMPipeline,
    characters: list[str],
    style_paths_with_names: list[tuple[str, str]],
    output_dir: str,
    args: Namespace,
    evaluator: QualityEvaluator,
    font_manager: FontManager,
    generation_tracker: GenerationTracker,
) -> dict[str, str]:
    """
    ✅ Main batch generation with hash-based file naming
    """

    # Generate ALL content images first
    logger.info(f"{'=' * 60}")
    logger.info(f"{'GENERATING CONTENT IMAGES':^60}")
    logger.info("=" * 60)

    char_paths = generate_content_images(
        characters, font_manager, output_dir, generation_tracker
    )

    if not char_paths:
        raise ValueError("No content images generated!")

    # Extract ALL unique characters and styles from checkpoint
    all_chars_in_checkpoint: set[str] = set()
    all_styles_in_checkpoint: set[str] = set()

    for gen in generation_tracker.generations:
        all_chars_in_checkpoint.add(gen.get("character", ""))
        all_styles_in_checkpoint.add(gen.get("style", ""))

    # Add current session's chars
    all_chars_in_checkpoint.update(char_paths.keys())

    # Add current session's styles
    for style_path, style_name in style_paths_with_names:
        if any(
            gen.get("style") == style_name for gen in generation_tracker.generations
        ):
            all_styles_in_checkpoint.add(style_name)

    # Initialize results from tracker
    results = {
        "generations": generation_tracker.generations.copy(),
        "metrics": {"lpips": [], "ssim": [], "inference_times": []},
        "dataset_split": args.dataset_split,
        "fonts": font_manager.get_font_names(),
        "characters": sorted(list(all_chars_in_checkpoint)),
        "styles": sorted(list(all_styles_in_checkpoint)),
        "total_chars": len(all_chars_in_checkpoint),
        "total_styles": len(all_styles_in_checkpoint),
    }

    # Setup directories
    target_base_dir = os.path.join(output_dir, "TargetImage")
    os.makedirs(target_base_dir, exist_ok=True)

    # Print configuration
    logger.info(f"{'=' * 60}")
    logger.info(f"{'BATCH IMAGE GENERATION':^60}")
    logger.info("=" * 60)
    logger.info(f"Fonts:                {len(font_manager.get_font_names())}")
    logger.info(f"Styles:               {len(style_paths_with_names)}")
    logger.info(f"Characters (input):   {len(characters)}")
    logger.info(f"Characters (content): {len(char_paths)}")
    logger.info(f"Batch size:           {args.batch_size}")
    logger.info(
        f"Previously generated: {len(generation_tracker.generations)} unique pairs"
    )
    logger.info(f"Unique chars seen:    {len(all_chars_in_checkpoint)}")
    logger.info(f"Unique styles used:   {len(all_styles_in_checkpoint)}")
    logger.info(
        f"Style Transform:      {getattr(args, 'enable_style_transform', False)}"
    )  # ✅ ADD THIS
    logger.info("=" * 60 + "\n")

    # Use first font for all characters
    font_names = font_manager.get_font_names()
    if not font_names:
        raise ValueError("No fonts loaded!")

    primary_font = font_names[0]
    logger.info(f"Using font: {primary_font}")
    logger.info("=" * 60 + "\n")

    # Initialize counters
    generated_count = 0
    skipped_count = 0
    failed_count = 0
    generation_start_time = time.time()

    # Main generation loop
    for style_idx, (style_path, style_name) in HFTqdm(
        enumerate(style_paths_with_names),
        total=len(style_paths_with_names),
        desc="🎨 Generating styles",
    ):
        style_dir = os.path.join(target_base_dir, style_name)
        os.makedirs(style_dir, exist_ok=True)

        try:
            # Filter characters that haven't been generated yet
            chars_to_generate = [
                char
                for char in characters
                if not generation_tracker.is_generated(char, style_name, primary_font)
            ]

            if not chars_to_generate:
                logger.info(
                    f"  ⊘ {style_name}: All characters already generated, skipping"
                )
                skipped_count += len(characters)
                continue

            logger.info(
                f"  🔄 {style_name}: Generating {len(chars_to_generate)}/{len(characters)} new images"
            )

            # ✅ PASS STYLE TRANSFORM FLAG TO SAMPLING
            images, valid_chars, batch_time = sampling_batch_optimized(
                args,
                pipe,
                chars_to_generate,
                style_path,
                font_manager,
                primary_font,
                enable_style_transform=getattr(
                    args, "enable_style_transform", False
                ),  # ✅ ADD THIS
            )

            if images is None:
                logger.info(f"  ⚠️ {style_name}: No images generated")
                skipped_count += len(chars_to_generate)
                continue

            logger.info(f"  ✓ {style_name}: {len(images)} images in {batch_time:.2f}s")

            # Save images and metadata
            for char, img in zip(valid_chars, images):
                try:
                    if not font_manager.is_char_in_font(primary_font, char):
                        logger.error(
                            f"    ✗ Character '{char}' (U+{ord(char):04X}) not in font {primary_font}, skipping"
                        )
                        failed_count += 1
                        continue

                    target_filename = get_target_filename(char, style_name)

                    import re

                    expected_pattern = r".+\+.\.png"
                    if not re.match(expected_pattern, target_filename):
                        raise ValueError(
                            f"Invalid filename format: {target_filename}\n"
                            f"  Expected: U+XXXX_[optional_char]_style_hash.png"
                        )

                    img_path = os.path.join(style_dir, target_filename)
                    content_filename = get_content_filename(char)
                    content_path_rel = f"ContentImage/{content_filename}"
                    target_path_rel = f"TargetImage/{style_name}/{target_filename}"

                    evaluator.save_image(img, img_path)
                    logger.info(
                        f"    ✓ Saved generated image for '{char}' (U+{ord(char):04X}) at {img_path}."
                    )

                    generation_record = {
                        "character": char,
                        "char_code": f"U+{ord(char):04X}",
                        "style": style_name,
                        "font": primary_font,
                        "content_image_path": content_path_rel,
                        "target_image_path": target_path_rel,
                        "content_hash": compute_file_hash(char, "", primary_font),
                        "target_hash": compute_file_hash(
                            char, style_name, primary_font
                        ),
                        "content_filename": content_filename,
                        "target_filename": target_filename,
                    }

                    results["generations"].append(generation_record)
                    generation_tracker.add_generation(generation_record)

                    all_chars_in_checkpoint.add(char)
                    all_styles_in_checkpoint.add(style_name)
                    results["characters"] = sorted(list(all_chars_in_checkpoint))
                    results["styles"] = sorted(list(all_styles_in_checkpoint))
                    results["total_chars"] = len(all_chars_in_checkpoint)
                    results["total_styles"] = len(all_styles_in_checkpoint)

                    generated_count += 1

                except ValueError as e:
                    logger.error(f"    ✗ Invalid filename for '{char}': {e}")
                    failed_count += 1
                except Exception as e:
                    logger.error(f"    ✗ Error saving '{char}': {e}")
                    failed_count += 1

            # Track inference time
            results["metrics"]["inference_times"].append(
                {
                    "style": style_name,
                    "font": primary_font,
                    "total_time": batch_time,
                    "num_images": len(images),
                    "time_per_image": batch_time / len(images) if images else 0,
                }
            )

            # Save checkpoint periodically
            if args.save_interval > 0 and (style_idx + 1) % args.save_interval == 0:
                _print_checkpoint_status(
                    style_idx + 1,
                    len(style_paths_with_names),
                    generated_count,
                    skipped_count,
                    generation_start_time,
                )
                save_checkpoint(results, output_dir)

        except Exception as e:
            logger.info(f"  ✗ {style_name}: {e}")
            import traceback

            traceback.print_exc()
            failed_count += len(chars_to_generate)

    # Final statistics
    _print_generation_summary(
        generated_count,
        skipped_count,
        failed_count,
        len(characters) * len(style_paths_with_names),
        generation_start_time,
    )

    return results


def sampling_batch_optimized(
    args: Namespace,
    pipe: FontDiffuserDPMPipeline,
    characters: list[str],
    style_image_path: str | Image.Image,
    font_manager: FontManager,
    font_name: str,
    enable_style_transform: bool = False,
) -> tuple[list[Image.Image] | None, list[str] | None, float | None]:
    """Batch sampling for multiple characters with specific font"""

    # Get available characters for this font
    available_chars: list[str] = font_manager.get_available_chars_for_font(
        font_name, characters
    )

    if not available_chars:
        return None, None, None

    try:
        # Load style image
        if isinstance(style_image_path, str):
            style_image: Image.Image = Image.open(style_image_path).convert("RGB")
        else:
            style_image: Image.Image = style_image_path.convert("RGB")
        style_transform: transforms.Compose = get_style_transform(args.style_image_size)

        font = font_manager.get_font(font_name)
        content_transform: transforms.Compose = get_content_transform(
            args.content_image_size
        )

        # Generate content images
        content_images: list[torch.Tensor] = []
        content_images_pil: list[Image.Image] = []

        for char in HFTqdm(
            available_chars,
            desc=f"  📸 Preparing {font_name}",
            colour="cyan",
        ):
            try:
                content_image: Image.Image = ttf2im(font=font, char=char)
                content_images_pil.append(content_image.copy())
                content_images.append(content_transform(content_image))
            except Exception as e:
                logger.info(f"    ✗ Error processing '{char}': {e}")
                continue

        if not content_images:
            return None, None, None

        # Stack into batch
        content_batch: torch.Tensor = torch.stack(content_images)
        style_batch: torch.Tensor = style_transform(style_image)[None, :].repeat(
            len(content_images), 1, 1, 1
        )

        with torch.inference_mode():
            dtype: torch.dtype = torch.float16 if args.fp16 else torch.float32
            content_batch = content_batch.to(args.device, dtype=dtype)
            style_batch = style_batch.to(args.device, dtype=dtype)

            start: float = time.perf_counter()

            # Process in batches
            all_images: list[Image.Image] = []
            batch_size: int = args.batch_size

            num_batches = (len(content_batch) + batch_size - 1) // batch_size
            batch_pbar = HFTqdm(
                range(0, len(content_batch), batch_size),
                desc="    🚀 Batch Inference",
                colour="#1055C9",
            )
            for batch_idx, i in enumerate(batch_pbar):
                batch_content: torch.Tensor = content_batch[i : i + batch_size]
                batch_style: torch.Tensor = style_batch[i : i + batch_size]

                images: list[Image.Image] = pipe.generate(
                    content_images=batch_content,
                    style_images=batch_style,
                    batch_size=len(batch_content),
                    order=args.order,
                    num_inference_step=args.num_inference_steps,
                    content_encoder_downsample_size=args.content_encoder_downsample_size,
                    t_start=args.t_start,
                    t_end=args.t_end,
                    dm_size=args.content_image_size,
                    algorithm_type=args.algorithm_type,
                    skip_type=args.skip_type,
                    method=args.method,
                    correcting_x0_fn=args.correcting_x0_fn,
                    enable_style_transform=enable_style_transform,
                )

                all_images.extend(images)
                batch_pbar.update(1)

            end: float = time.perf_counter()
            total_time: float = end - start

            return all_images, available_chars, total_time

    except Exception as e:
        logger.info(f"    ✗ Error in batch sampling: {e}")
        import traceback

        traceback.print_exc()
        return None, None, None


def _print_checkpoint_status(
    current_style: int,
    total_styles: int,
    generated: int,
    skipped: int,
    start_time: float,
) -> None:
    """Print periodic checkpoint status"""
    elapsed = time.time() - start_time
    remaining = (
        elapsed * (total_styles - current_style) / current_style
        if current_style > 0
        else 0
    )

    logger.info(f"{'=' * 60}")
    logger.info(f"{'CHECKPOINT':^60}")
    logger.info("=" * 60)
    logger.info(f"Progress:           {current_style}/{total_styles} styles")
    logger.info(f"Generated:          {generated} pairs")
    logger.info(f"Skipped:            {skipped} pairs")
    logger.info(f"Elapsed time:       {elapsed / 60:.1f} minutes")
    logger.info(f"Est. remaining:     {remaining / 60:.1f} minutes")
    logger.info("=" * 60)


def _print_generation_summary(
    generated: int, skipped: int, failed: int, total: int, start_time: float
) -> None:
    """Print final generation summary"""
    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info(f"{'GENERATION COMPLETE':^60}")
    logger.info("=" * 60)
    logger.info(f"Pair Statistics:")
    logger.info(f"  Total possible:     {total}")
    logger.info(f"  Generated (new):    {generated}")
    logger.info(f"  Skipped (exist):    {skipped}")
    logger.info(f"  Failed (no font):   {failed}")
    logger.info(f"Timing:")
    logger.info(f"  Total time:         {elapsed / 60:.1f} minutes ({elapsed:.0f}s)")
    logger.info(
        f"  Avg per pair:       {elapsed / generated * 1000:.1f}ms"
        if generated > 0
        else "  Avg per pair:       N/A"
    )
    logger.info("=" * 60)


def evaluate_results(
    results: dict[str, str],
    evaluator: QualityEvaluator,
    ground_truth_dir: str = None,
    compute_fid: bool = False,
) -> dict[str, str]:
    """Evaluate generated images against ground truth"""

    if not ground_truth_dir or not os.path.exists(ground_truth_dir):
        logger.info(
            "\n⚠ No ground truth directory provided or not found, skipping evaluation"
        )
        return results

    logger.info("=" * 60)
    logger.info(f"{'EVALUATING GENERATED IMAGES':^60}")
    logger.info("=" * 60)

    lpips_scores: list[float] = []
    ssim_scores: list[float] = []
    evaluated_pairs: int = 0
    missing_gt: int = 0

    # Evaluate each generation
    for gen in HFTqdm(
        results["generations"],
        desc="📊 Evaluating",
        colour="green",
    ):
        char: str = gen["character"]
        style: str = gen["style"]
        font: str = gen.get("font", "")

        # Get generated image path
        target_path: str = gen["target_image_path"]
        generated_path: str = os.path.join(
            os.path.dirname(os.path.dirname(target_path)), target_path
        )

        if not os.path.exists(generated_path):
            continue

        # Find ground truth image
        gt_filename = get_target_filename(char, style)
        gt_path = os.path.join(ground_truth_dir, "TargetImage", style, gt_filename)

        if not os.path.exists(gt_path):
            # Try alternative naming
            gt_path = os.path.join(ground_truth_dir, style, gt_filename)

        if not os.path.exists(gt_path):
            missing_gt += 1
            continue

        try:
            # Load images
            generated_img: Image.Image = Image.open(generated_path).convert("RGB")
            gt_img: Image.Image = Image.open(gt_path).convert("RGB")

            # Compute metrics
            if LPIPS_AVAILABLE:
                lpips_score: float = evaluator.compute_lpips(generated_img, gt_img)
                if lpips_score >= 0:
                    lpips_scores.append(lpips_score)
                    gen["lpips"] = lpips_score

            if SSIM_AVAILABLE:
                ssim_score: float = evaluator.compute_ssim(generated_img, gt_img)
                if ssim_score >= 0:
                    ssim_scores.append(ssim_score)
                    gen["ssim"] = ssim_score

            evaluated_pairs += 1

        except Exception as e:
            logger.info(f"  ⚠ Error evaluating {char}/{style}: {e}")
            continue

    # Compute aggregate metrics
    if lpips_scores:
        results["metrics"]["lpips"] = {
            "mean": float(np.mean(lpips_scores)),
            "std": float(np.std(lpips_scores)),
            "min": float(np.min(lpips_scores)),
            "max": float(np.max(lpips_scores)),
            "median": float(np.median(lpips_scores)),
        }
        logger.info(f"📊 LPIPS Statistics:")
        logger.info(f"  Mean:   {results['metrics']['lpips']['mean']:.4f}")
        logger.info(f"  Std:    {results['metrics']['lpips']['std']:.4f}")
        logger.info(f"  Median: {results['metrics']['lpips']['median']:.4f}")
        logger.info(
            f"  Range:  [{results['metrics']['lpips']['min']:.4f}, {results['metrics']['lpips']['max']:.4f}]"
        )

    if ssim_scores:
        results["metrics"]["ssim"] = {
            "mean": float(np.mean(ssim_scores)),
            "std": float(np.std(ssim_scores)),
            "min": float(np.min(ssim_scores)),
            "max": float(np.max(ssim_scores)),
            "median": float(np.median(ssim_scores)),
        }
        logger.info(f"📊 SSIM Statistics:")
        logger.info(f"  Mean:   {results['metrics']['ssim']['mean']:.4f}")
        logger.info(f"  Std:    {results['metrics']['ssim']['std']:.4f}")
        logger.info(f"  Median: {results['metrics']['ssim']['median']:.4f}")
        logger.info(
            f"  Range:  [{results['metrics']['ssim']['min']:.4f}, {results['metrics']['ssim']['max']:.4f}]"
        )

    # Compute FID if requested
    if compute_fid and FID_AVAILABLE:
        logger.info("\n📊 Computing FID score...")
        try:
            # Create temporary directories for FID computation
            fake_dir = os.path.join(
                os.path.dirname(generated_path), "..", "TargetImage"
            )
            real_dir = os.path.join(ground_truth_dir, "TargetImage")

            if os.path.exists(fake_dir) and os.path.exists(real_dir):
                fid_value: float = evaluator.compute_fid(real_dir, fake_dir)
                if fid_value >= 0:
                    results["metrics"]["fid"] = fid_value
                    logger.info(f"  FID Score: {fid_value:.2f}")
            else:
                logger.info("  ⚠ Cannot compute FID: directories not found")
        except Exception as e:
            logger.info(f"  ⚠ Error computing FID: {e}")

    logger.info("=" * 60)
    logger.info(f"{'EVALUATION SUMMARY':^60}")
    logger.info("=" * 60)
    logger.info(f"Evaluated pairs:    {evaluated_pairs}")
    logger.info(f"Missing GT images:  {missing_gt}")
    logger.info(f"LPIPS samples:      {len(lpips_scores)}")
    logger.info(f"SSIM samples:       {len(ssim_scores)}")
    logger.info("=" * 60)

    return results


def log_to_wandb(results: dict, args: Namespace) -> None:
    """Log results to Weights & Biases"""

    if not WANDB_AVAILABLE:
        logger.info("\n⚠ Wandb not available, skipping logging")
        return

    try:
        logger.info("=" * 60)
        logger.info(f"{'Logging to Weights & Biases':^60}")
        logger.info("=" * 60)

        run_name: str | None = args.wandb_run_name

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "dataset_split": args.dataset_split,
                "num_characters": results.get("total_chars", 0),
                "num_styles": results.get("total_styles", 0),
                "num_fonts": len(results.get("fonts", [])),
                "batch_size": args.batch_size,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "fp16": args.fp16,
                "compile": args.compile,
                "xformers": args.enable_xformers,
            },
        )

        # Log generation statistics
        num_generations: int = len(results.get("generations", []))
        wandb.log(
            {
                "total_generations": num_generations,
                "num_characters": results.get("total_chars", 0),
                "num_styles": results.get("total_styles", 0),
                "num_fonts": len(results.get("fonts", [])),
            }
        )

        # Log metrics if available
        metrics: dict = results.get("metrics", {})

        if "lpips" in metrics and isinstance(metrics["lpips"], dict):
            wandb.log(
                {
                    "lpips/mean": metrics["lpips"]["mean"],
                    "lpips/std": metrics["lpips"]["std"],
                    "lpips/median": metrics["lpips"]["median"],
                    "lpips/min": metrics["lpips"]["min"],
                    "lpips/max": metrics["lpips"]["max"],
                }
            )

        if "ssim" in metrics and isinstance(metrics["ssim"], dict):
            wandb.log(
                {
                    "ssim/mean": metrics["ssim"]["mean"],
                    "ssim/std": metrics["ssim"]["std"],
                    "ssim/median": metrics["ssim"]["median"],
                    "ssim/min": metrics["ssim"]["min"],
                    "ssim/max": metrics["ssim"]["max"],
                }
            )

        if "fid" in metrics:
            wandb.log({"fid": metrics["fid"]})

        # Log inference timing
        if "inference_times" in metrics and metrics["inference_times"]:
            timing_data: list = metrics["inference_times"]

            # Only process dict entries
            total_times: list[float] = [
                t["total_time"]
                for t in timing_data
                if isinstance(t, dict) and "total_time" in t
            ]
            times_per_image: list[float] = [
                t["time_per_image"]
                for t in timing_data
                if isinstance(t, dict) and "time_per_image" in t
            ]

            if total_times:
                wandb.log(
                    {
                        "timing/mean_batch_time": float(np.mean(total_times)),
                        "timing/total_time": float(np.sum(total_times)),
                    }
                )

            if times_per_image:
                wandb.log(
                    {
                        "timing/mean_time_per_image": float(np.mean(times_per_image)),
                        "timing/median_time_per_image": float(
                            np.median(times_per_image)
                        ),
                    }
                )

        # Log sample images
        logger.info("\n📸 Logging sample images...")
        sample_generations: list[dict] = results.get("generations", [])[:20]

        sample_images: list = []
        for gen in sample_generations:
            target_path: str = gen.get("target_image_path", "")
            if target_path:
                full_path: str = os.path.join(args.output_dir, target_path)
                if os.path.exists(full_path):
                    try:
                        img = Image.open(full_path)
                        sample_images.append(
                            wandb.Image(
                                img,
                                caption=f"{gen['character']} - {gen['style']} ({gen.get('font', '')})",
                            )
                        )
                    except Exception as e:
                        logger.info(f"  ⚠ Error loading image {full_path}: {e}")

        if sample_images:
            wandb.log({"sample_images": sample_images})
            logger.info(f"✓ Logged {len(sample_images)} sample images")

        # Create summary table
        generation_table = wandb.Table(
            columns=[
                "Character",
                "Style",
                "Font",
                "LPIPS",
                "SSIM",
                "Content Path",
                "Target Path",
            ]
        )

        for gen in results.get("generations", [])[:100]:
            generation_table.add_data(
                gen.get("character", ""),
                gen.get("style", ""),
                gen.get("font", ""),
                gen.get("lpips", -1),
                gen.get("ssim", -1),
                gen.get("content_image_path", ""),
                gen.get("target_image_path", ""),
            )

        wandb.log({"generations": generation_table})

        # Finish run
        wandb.finish()

        logger.info("\n✓ Successfully logged to Weights & Biases")
        logger.info(f"  Project: {args.wandb_project}")
        logger.info(f"  Run: {run_name}")
        logger.info("=" * 60)

    except Exception as e:
        logger.info(f"⚠ Error logging to wandb: {e}")
        import traceback

        traceback.print_exc()


def main() -> None:
    """Main function"""
    parser = get_parser()
    args = parser.parse_args()

    # Validate required arguments
    if not args.characters:
        raise ValueError("--characters is required")
    if not args.style_images:
        raise ValueError("--style_images is required")
    if not args.ckpt_dir:
        raise ValueError("--ckpt_dir is required")
    if not args.ttf_path:
        raise ValueError("--ttf_path is required")

    # Convert image sizes to tuples (centralized parser uses int)
    if isinstance(args.style_image_size, int):
        args.style_image_size = (args.style_image_size, args.style_image_size)
    if isinstance(args.content_image_size, int):
        args.content_image_size = (args.content_image_size, args.content_image_size)

    # Set derived defaults if not provided
    if not args.output_dir:
        args.output_dir = "my_dataset/train_original"

    results: dict[str, str] = {}

    logger.info("=" * 60)
    logger.info("FontDiffuser Batch Sampling")
    logger.info("=" * 60)

    try:
        # Load characters
        characters: list[str] = load_characters(
            args.characters, args.start_line, args.end_line
        )

        # Load style images with names
        style_paths_with_names: list[tuple[str, str]] = load_style_images(
            args.style_images
        )

        logger.info(f"Initializing font manager...")
        font_manager: FontManager = FontManager(args.ttf_path)
        logger.info(f"✓ Loaded {len(font_manager.get_font_names())} fonts.")

        logger.info(f"📊 Configuration:")
        logger.info(f"  Dataset split: {args.dataset_split}")
        logger.info(
            f"  Characters: {len(characters)} (lines {args.start_line}-{args.end_line or 'end'})"
        )
        logger.info(f"  Styles: {len(style_paths_with_names)}")
        logger.info(f"  Output Directory: {args.output_dir}")
        logger.info(f"  Checkpoint Directory: {args.ckpt_dir}")
        logger.info(f"  Device: {args.device}")
        logger.info(f"  Batch Size: {args.batch_size}")
        logger.info(
            f"  Results checkpoint path: {os.path.join(args.output_dir, 'results_checkpoint.json')}"
        )

        os.makedirs(args.output_dir, exist_ok=True)

        # Initialize generation tracker
        checkpoint_path = os.path.join(args.output_dir, "results_checkpoint.json")
        generation_tracker = GenerationTracker(
            checkpoint_path if os.path.exists(checkpoint_path) else None
        )

        # Create args namespace for pipeline
        pipeline_args: Namespace = create_args_namespace(args)

        logger.info("\nLoading FontDiffuser pipeline...")
        pipe: FontDiffuserDPMPipeline = load_fontdiffuser_pipeline(pipeline_args)

        # Add this block to enable torch.compile if requested
        if getattr(args, "compile", False):
            import torch

            logger.info("🔧 Compiling model components with torch.compile...")
            try:
                if hasattr(pipe.model.config, "unet"):
                    pipe.model.config.unet = torch.compile(pipe.model.config.unet)
                if hasattr(pipe.model.config, "style_encoder"):
                    pipe.model.config.style_encoder = torch.compile(
                        pipe.model.config.style_encoder
                    )
                if hasattr(pipe.model.config, "content_encoder"):
                    pipe.model.config.content_encoder = torch.compile(
                        pipe.model.config.content_encoder
                    )
                logger.info("✓ Compilation complete.")
            except Exception as e:
                logger.info(f"⚠ Compilation failed: {e}")

        evaluator: QualityEvaluator = QualityEvaluator(device=args.device)

        # Generate images
        results: dict[str, str] = batch_generate_images(
            pipe,
            characters,
            style_paths_with_names,
            args.output_dir,
            pipeline_args,
            evaluator,
            font_manager,
            generation_tracker,
        )

        # Evaluate if requested
        if args.evaluate and args.ground_truth_dir:
            results = evaluate_results(
                results, evaluator, args.ground_truth_dir, args.compute_fid
            )

        # Save final checkpoint
        logger.info("\n💾 Saving final checkpoint...")
        save_checkpoint(results, args.output_dir)

        if args.use_wandb:
            log_to_wandb(results, args)

        logger.info("=" * 60)
        logger.info(" Generation done!")
        logger.info("=" * 60)
        logger.info(f"Output structure:")
        logger.info(f"  {args.output_dir}/")
        logger.info(f"    ├── ContentImage/")
        logger.info(f"    │   ├── U+XXXX_char_hash.png")
        logger.info(f"    │   └── ...")
        logger.info(f"    ├── TargetImage/")
        logger.info(f"    │   ├── style0/")
        logger.info(f"    │   │   ├── U+XXXX_char_style0_hash.png")
        logger.info(f"    │   │   └── ...")
        logger.info(f"    │   └── ...")
        logger.info(f"    └── results_checkpoint.json")

    except KeyboardInterrupt:
        logger.info("\n\n⚠ Generation interrupted by user!")
        logger.info("💾 Saving emergency checkpoint...")
        if "results" in locals() and results:
            save_checkpoint(results, args.output_dir)
            logger.info("✓ Latest state saved to results_checkpoint.json")
        sys.exit(1)

    except Exception as e:
        logger.info(f"✗ Fatal error: {e}")
        import traceback

        traceback.print_exc()

        if "results" in locals() and results:
            save_checkpoint(results, args.output_dir)
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Example usage
python sample_batch.py \
    --characters chars.txt \
    --style_images styles/ \
    --enable_style_transform \
    --output_dir output/
"""
