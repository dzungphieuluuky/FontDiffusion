"""
Compare content, style, and target images side-by-side with multiprocessing.

Usage:
    python tools/compare_images.py --content_dir ContentImage/ --style_dir style_images/ --target_dir TargetImage/ --output_dir comparison_output/
"""

import argparse
import os
from pathlib import Path
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare content, style, and target images side-by-side"
    )
    parser.add_argument(
        "--content_dir",
        type=str,
        required=True,
        help="Path to directory containing content images",
    )
    parser.add_argument(
        "--style_dir",
        type=str,
        required=True,
        help="Path to directory containing style images",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Path to directory containing target images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="comparison_output",
        help="Path to output directory for comparison images",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=256,
        help="Height to resize all images to (maintains aspect ratio)",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=10,
        help="Spacing between images in pixels",
    )
    parser.add_argument(
        "--match_by",
        type=str,
        default="name",
        choices=["name", "index"],
        help="How to match images: 'name' (by filename) or 'index' (by sorted order)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of worker processes (default: CPU count)",
    )
    return parser.parse_args()


def resize_image(image: Image.Image, target_height: int) -> Image.Image:
    """Resize image to target height while maintaining aspect ratio."""
    aspect_ratio = image.width / image.height
    new_width = int(target_height * aspect_ratio)
    return image.resize((new_width, target_height), Image.Resampling.LANCZOS)


def get_image_files(directory: Path) -> list[Path]:
    """Get all image files from directory, sorted by name."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    files = [
        f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in extensions
    ]
    return sorted(files)


def match_images_by_name(
    content_files: list[Path],
    style_files: list[Path],
    target_files: list[Path],
) -> list[tuple[Path, Path, Path]]:
    """
    Match images by content and style filename convention:
    - content: char.png
    - style: style.png
    - target: style+char.png
    """
    content_dict = {f.stem: f for f in content_files}
    style_dict = {f.stem: f for f in style_files}
    target_dict = {f.stem: f for f in target_files}

    matched = []
    for char_stem, content_path in content_dict.items():
        for style_stem, style_path in style_dict.items():
            target_stem = f"{style_stem}+{char_stem}"
            if target_stem in target_dict:
                matched.append((content_path, style_path, target_dict[target_stem]))
    return matched


def match_images_by_index(
    content_files: list[Path],
    style_files: list[Path],
    target_files: list[Path],
) -> list[tuple[Path, Path, Path]]:
    """Match images by index in sorted lists."""
    min_len = min(len(content_files), len(style_files), len(target_files))
    return list(
        zip(content_files[:min_len], style_files[:min_len], target_files[:min_len])
    )


def create_comparison_worker(args_tuple):
    """
    Worker function for creating a comparison image.
    Designed to be used with multiprocessing.

    Args:
        args_tuple: Tuple of (content_path, style_path, target_path, output_path, resize_height, spacing)

    Returns:
        Tuple of (success: bool, output_path: Path, error_msg: Optional[str])
    """
    content_path, style_path, target_path, output_path, resize_height, spacing = (
        args_tuple
    )

    try:
        # Load images
        content_img = Image.open(content_path).convert("RGB")
        style_img = Image.open(style_path).convert("RGB")
        target_img = Image.open(target_path).convert("RGB")

        # Resize to same height
        content_img = resize_image(content_img, resize_height)
        style_img = resize_image(style_img, resize_height)
        target_img = resize_image(target_img, resize_height)

        # Calculate total width
        total_width = (
            content_img.width + style_img.width + target_img.width + 2 * spacing
        )
        total_height = resize_height

        # Create output image
        comparison = Image.new(
            "RGB", (total_width, total_height), color=(255, 255, 255)
        )

        # Paste images
        x_offset = 0
        comparison.paste(content_img, (x_offset, 0))
        x_offset += content_img.width + spacing
        comparison.paste(style_img, (x_offset, 0))
        x_offset += style_img.width + spacing
        comparison.paste(target_img, (x_offset, 0))

        # Save
        comparison.save(output_path)
        return (True, output_path, None)

    except Exception as e:
        return (False, output_path, str(e))


def main():
    """Main entry point."""
    args = parse_args()

    # Convert to Path objects
    content_dir = Path(args.content_dir)
    style_dir = Path(args.style_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)

    # Validate directories exist
    for dir_path, name in [
        (content_dir, "content"),
        (style_dir, "style"),
        (target_dir, "target"),
    ]:
        if not dir_path.exists():
            logger.error(f"{name} directory does not exist: {dir_path}")
            return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get image files
    content_files = get_image_files(content_dir)
    style_files = get_image_files(style_dir)
    target_files = []

    style_images_list = [
        "thanh1",
        "thanh2",
        "thanh3",
        "thanh4",
        "thanh5",
        "thanh6",
        "thanh7",
        "thanh8",
    ]
    for style_image in style_images_list:
        style_subdir = target_dir / style_image
        if not style_subdir.exists():
            logger.error(f"Style sub-directory does not exist: {style_subdir}")
            return
        target_files.extend(get_image_files(style_subdir))

    logger.info(f"Found {len(content_files)} content images")
    logger.info(f"Found {len(style_files)} style images")
    logger.info(f"Found {len(target_files)} target images")

    # Match images
    if args.match_by == "name":
        matched = match_images_by_name(content_files, style_files, target_files)
    else:
        matched = match_images_by_index(content_files, style_files, target_files)

    if not matched:
        logger.error("No matching images found")
        return

    logger.info(f"Matched {len(matched)} image sets")

    # Prepare work items
    work_items = []
    for content_path, style_path, target_path in matched:
        output_name = f"comp_{target_path.stem}.png"
        output_path = output_dir / output_name
        work_items.append(
            (
                content_path,
                style_path,
                target_path,
                output_path,
                args.resize_height,
                args.spacing,
            )
        )

    # Determine number of workers
    n_workers = args.workers if args.workers else mp.cpu_count()
    logger.info(f"Processing with {n_workers} workers")

    # Process with multiprocessing
    success_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(create_comparison_worker, item): item for item in work_items
        }

        # Process results as they complete
        for future in as_completed(futures):
            success, output_path, error_msg = future.result()
            if success:
                logger.info(f"Created comparison: {output_path.name}")
                success_count += 1
            else:
                logger.error(
                    f"Failed to create comparison for {output_path.name}: {error_msg}"
                )
                error_count += 1

    logger.info(f"Done! Created {success_count} comparison images in {output_dir}")
    if error_count > 0:
        logger.warning(f"{error_count} comparisons failed")


if __name__ == "__main__":
    main()


"""Example Usage
# Match by filename (default)
python tools/compare_images.py --content_dir ContentImage/ --style_dir style_images/ --target_dir TargetImage/

# Match by index with custom output directory and 8 workers
python tools/compare_images.py --content_dir ContentImage/ --style_dir style_images/ --target_dir TargetImage/ --output_dir my_comparisons/ --match_by index --workers 8

# Custom resize height, spacing, and automatic worker count
python tools/compare_images.py --content_dir ContentImage/ --style_dir style_images/ --target_dir TargetImage/ --resize_height 512 --spacing 20
"""
