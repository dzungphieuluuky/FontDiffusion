"""
Generate results_checkpoint.json from existing ContentImage and TargetImage folders
✅ Rescans directories and recalculates all paths and hashes
✅ Handles new simplified filename format: {char}.png and {style}+{char}.png
"""

import json
import logging
from pathlib import Path
from typing import Any
from filename_utils import (
    get_content_filename,
    get_target_filename,
    parse_content_filename,
    parse_target_filename,
    compute_file_hash,
)

logger = logging.getLogger("GenerateMetadata")


def generate_checkpoint_from_disk(data_root: str) -> dict[str, Any]:
    """
    Generate complete checkpoint from disk structure.

    Parameters
    ----------
    data_root : str
        Path to directory containing ContentImage/ and TargetImage/

    Returns
    -------
    dict[str, Any]
        Dictionary with checkpoint data
    """
    data_root = Path(data_root)
    content_dir = data_root / "ContentImage"
    target_base_dir = data_root / "TargetImage"

    # Validate directories exist
    if not content_dir.exists():
        raise FileNotFoundError(f"ContentImage directory not found: {content_dir}")
    if not target_base_dir.exists():
        raise FileNotFoundError(f"TargetImage directory not found: {target_base_dir}")

    # Collections
    generations: list[dict[str, Any]] = []
    characters_set: set[str] = set()
    styles_set: set[str] = set()
    fonts_set: list[str] = ["NomNaTong-Regular"]

    # Track which content/target pairs exist
    content_chars: dict[str, str] = {}  # char -> filepath
    valid_pairs: set[tuple[str, str]] = set()  # (char, style) pairs

    # ========== PHASE 1: Scan ContentImage directory ==========
    content_files = list(content_dir.glob("*.png"))
    logger.info(f"Phase 1: Found {len(content_files)} content images")

    for content_file in sorted(content_files):
        char = parse_content_filename(content_file.name)
        if char:
            content_chars[char] = str(content_file.relative_to(data_root))
            characters_set.add(char)
        else:
            logger.warning(f"Could not parse content filename: {content_file.name}")

    # ========== PHASE 2: Scan TargetImage directory ==========
    target_style_dirs = [d for d in target_base_dir.iterdir() if d.is_dir()]
    logger.info(f"Phase 2: Found {len(target_style_dirs)} style directories")

    for style_dir in sorted(target_style_dirs):
        style_name = style_dir.name
        styles_set.add(style_name)
        target_files = list(style_dir.glob("*.png"))

        for target_file in sorted(target_files):
            char, parsed_style = parse_target_filename(target_file.name)

            if char is None:
                logger.warning(f"Could not parse target filename: {target_file.name}")
                continue

            if parsed_style != style_name:
                logger.warning(
                    f"Style mismatch in {target_file.name}: expected {style_name}, got {parsed_style}"
                )
                continue

            if char not in content_chars:
                logger.warning(f"No content image for character {repr(char)}")
                continue

            valid_pairs.add((char, style_name))

    # ========== PHASE 3: Build generation records ==========
    logger.info(f"Phase 3: Building {len(valid_pairs)} generation records")

    for char, style in sorted(valid_pairs):
        content_hash = compute_file_hash(char, "", fonts_set[0])
        target_hash = compute_file_hash(char, style, fonts_set[0])

        generation = {
            "character": char,
            "style": style,
            "font": "NomNaTong-Regular",
            "content_image_path": f"ContentImage/{get_content_filename(char)}",
            "target_image_path": f"TargetImage/{style}/{get_target_filename(char, style)}",
            "content_hash": content_hash,
            "target_hash": target_hash,
        }
        generations.append(generation)

    # ========== PHASE 4: Build final checkpoint ==========
    checkpoint = {
        "generations": generations,
        "characters": sorted(list(characters_set)),
        "styles": sorted(list(styles_set)),
        "fonts": sorted(list(fonts_set)),
        "total_chars": len(characters_set),
        "total_styles": len(styles_set),
    }

    logger.info(
        f"Checkpoint summary: {len(generations)} generations, "
        f"{checkpoint['total_chars']} chars, {checkpoint['total_styles']} styles"
    )

    return checkpoint


def save_checkpoint(checkpoint: dict[str, Any], output_path: str) -> None:
    """
    Save checkpoint to JSON file.

    Parameters
    ----------
    checkpoint : dict[str, Any]
        Checkpoint dictionary
    output_path : str
        Destination file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved checkpoint to: {output_path}")


def parse_args():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate results_checkpoint.json from ContentImage and TargetImage folders"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="my_dataset/handwritten_original",
        help="Path to dataset root containing ContentImage/ and TargetImage/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="my_dataset/handwritten_original/results_checkpoint.json",
        help="Path to output results_checkpoint.json file",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point"""
    logger.info("=" * 60)
    logger.info("Regenerating results_checkpoint.json from disk")
    logger.info("=" * 60)

    args = parse_args()

    try:
        checkpoint = generate_checkpoint_from_disk(args.data_root)
        save_checkpoint(checkpoint, args.output)
        logger.info("=" * 60)
        logger.info("✅ Checkpoint generation completed successfully")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.error("Expected directory structure:")
        logger.error("  data_root/")
        logger.error("  ├── ContentImage/  (char images)")
        logger.error("  └── TargetImage/   (style subdirs)")

    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)


if __name__ == "__main__":
    main()