"""
Create Hugging Face dataset from generated FontDiffusion images.

This module builds datasets from FontDiffusion outputs, using results_checkpoint.json
as the single source of truth for generation metadata. Includes comparison image generation.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image

from utilities import HFTqdm
from filename_utils import compute_file_hash

logger = logging.getLogger("DatasetCreator")


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""

    data_dir: Path
    style_images_dir: Path
    repo_id: str
    split: str = "train"
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None
    resize_height: int = 256
    spacing: int = 10

    def __post_init__(self):
        """Convert paths to Path if they're strings."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)


class DatasetBuilder:
    """Build FontDiffusion dataset in Hugging Face format."""

    REQUIRED_DIRS = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE = "results_checkpoint.json"

    def __init__(self, config: DatasetConfig):
        """Initialize the dataset builder.

        Args:
            config: Dataset configuration

        Raises:
            ValueError: If directory structure is invalid
        """
        self.config = config
        self.data_dir = config.data_dir
        self.style_images_dir = config.style_images_dir
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate that all required directories and files exist.

        Raises:
            ValueError: If any required directory or file is missing
        """
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                raise ValueError(f"Required directory not found: {dir_path}")

        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

        if not self.style_images_dir.exists():
            raise ValueError(f"Style images directory not found: {self.style_images_dir}")

        logger.info("Directory structure validated successfully")

    def _load_checkpoint(self) -> dict:
        """Load and validate results checkpoint.

        Returns:
            Checkpoint data dictionary

        Raises:
            ValueError: If checkpoint is invalid or empty
        """
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE

        with checkpoint_path.open("r", encoding="utf-8") as f:
            data: dict = json.load(f)

        generations: list = data.get("generations", [])
        if not generations:
            raise ValueError("No generations found in checkpoint")

        logger.info(
            f"Loaded checkpoint: {len(generations)} generations, "
            f"{len(data.get('characters', []))} characters, "
            f"{len(data.get('styles', []))} styles"
        )

        return data

    def _resize_image(self, image: Image.Image, target_height: int) -> Image.Image:
        """Resize image to target height while maintaining aspect ratio.

        Args:
            image: PIL Image to resize
            target_height: Target height in pixels

        Returns:
            Resized PIL Image
        """
        aspect_ratio = image.width / image.height
        new_width = int(target_height * aspect_ratio)
        return image.resize((new_width, target_height), Image.Resampling.LANCZOS)

    def _create_comparison_image(
        self,
        content_img: Image.Image,
        style_img: Image.Image,
        target_img: Image.Image,
    ) -> Optional[Image.Image]:
        """Create side-by-side comparison image (content | style | target).

        Args:
            content_img: Content/character image
            style_img: Style image
            target_img: Target image

        Returns:
            Comparison PIL Image or None if creation fails
        """
        try:
            content_resized = self._resize_image(content_img, self.config.resize_height)
            style_resized = self._resize_image(style_img, self.config.resize_height)
            target_resized = self._resize_image(target_img, self.config.resize_height)

            total_width = (
                content_resized.width
                + style_resized.width
                + target_resized.width
                + 2 * self.config.spacing
            )
            total_height = self.config.resize_height

            comparison = Image.new(
                "RGB", (total_width, total_height), color=(255, 255, 255)
            )

            x_offset = 0
            comparison.paste(content_resized, (x_offset, 0))
            x_offset += content_resized.width + self.config.spacing
            comparison.paste(style_resized, (x_offset, 0))
            x_offset += style_resized.width + self.config.spacing
            comparison.paste(target_resized, (x_offset, 0))

            return comparison

        except Exception as e:
            logger.warning(f"Failed to create comparison image: {e}")
            return None

    def _find_style_image(self, style: str) -> Optional[Path]:
        """Find style image in the style images directory.

        Args:
            style: Style name

        Returns:
            Path to style image or None if not found
        """
        for ext in [".png", ".jpg", ".jpeg"]:
            style_path = self.style_images_dir / f"{style}{ext}"
            if style_path.exists():
                return style_path

        return None

    def build(self) -> Dataset:
        """Build the dataset from checkpoint data with comparison images.

        Returns:
            HuggingFace Dataset with image pairs, comparison, and metadata

        Raises:
            ValueError: If no valid samples are found
        """
        logger.info("Building dataset...")

        checkpoint: dict = self._load_checkpoint()
        generations: list = checkpoint["generations"]

        characters: list[str] = []
        styles: list[str] = []
        fonts: list[str] = []
        content_images: list[Image.Image] = []
        style_images: list[Image.Image] = []
        target_images: list[Image.Image] = []
        comparison_images: list[Image.Image] = []
        content_hashes: list[str] = []
        target_hashes: list[str] = []

        skipped: int = 0

        for gen in HFTqdm(generations, desc="Loading image pairs", unit="pair"):
            char: str = gen.get("character")
            style: str = gen.get("style")
            font: str = gen.get("font", "unknown")

            content_path: Path = self.data_dir / gen.get("content_image_path", "")
            target_path: Path = self.data_dir / gen.get("target_image_path", "")

            style_path: Optional[Path] = self._find_style_image(style)

            if not content_path.exists() or not target_path.exists():
                logger.debug(
                    f"Missing images for {char}/{style}: content={content_path.exists()}, target={target_path.exists()}"
                )
                skipped += 1
                continue

            if style_path is None or not style_path.exists():
                logger.debug(f"Style image not found for style={style}")
                skipped += 1
                continue

            try:
                content_img: Image.Image = Image.open(content_path).convert("RGB")
                style_img: Image.Image = Image.open(style_path).convert("RGB")
                target_img: Image.Image = Image.open(target_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load images for {char}/{style}: {e}")
                skipped += 1
                continue

            comparison_img: Optional[Image.Image] = self._create_comparison_image(
                content_img, style_img, target_img
            )

            if comparison_img is None:
                logger.debug(f"Skipping {char}/{style} - comparison image failed")
                skipped += 1
                continue

            characters.append(char)
            styles.append(style)
            fonts.append(font)
            content_images.append(content_img)
            style_images.append(style_img)
            target_images.append(target_img)
            comparison_images.append(comparison_img)
            content_hashes.append(compute_file_hash(char, "", font))
            target_hashes.append(compute_file_hash(char, style, font))

        if not characters:
            raise ValueError("No valid samples found")

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid samples")

        logger.info(f"Successfully loaded {len(characters)} samples")

        features = Features(
            {
                "character": Value("string"),
                "style": Value("string"),
                "font": Value("string"),
                "content_image": HFImage(),
                "style_image": HFImage(),
                "target_image": HFImage(),
                "comparison_image": HFImage(),
                "content_hash": Value("string"),
                "target_hash": Value("string"),
            }
        )

        dataset = Dataset.from_dict(
            {
                "character": characters,
                "style": styles,
                "font": fonts,
                "content_image": content_images,
                "style_image": style_images,
                "target_image": target_images,
                "comparison_image": comparison_images,
                "content_hash": content_hashes,
                "target_hash": target_hashes,
            },
            features=features,
        )

        return dataset

    def push(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub.

        Args:
            dataset: Dataset to push
        """
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return

        logger.info(f"Pushing dataset to {self.config.repo_id}...")

        dataset.push_to_hub(
            repo_id=self.config.repo_id,
            split=self.config.split,
            private=self.config.private,
            token=self.config.token,
        )

        logger.info(
            f"Successfully pushed to https://huggingface.co/datasets/{self.config.repo_id}"
        )

    def save_local(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk.

        Args:
            dataset: Dataset to save
            output_path: Local directory path
        """
        logger.info(f"Saving dataset to {output_path}...")
        dataset.save_to_disk(str(output_path))
        logger.info("Dataset saved successfully")


def create_dataset(
    data_dir: str | Path,
    style_images_dir: str | Path,
    repo_id: str,
    split: str = "train",
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str | Path] = None,
    resize_height: int = 256,
    spacing: int = 10,
) -> Dataset:
    """Create and optionally push dataset to Hub with comparison images.

    Args:
        data_dir: Path to data directory containing ContentImage/ and TargetImage/
        style_images_dir: Path to directory containing style images
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        split: Dataset split name (default: 'train')
        push_to_hub: Whether to push to HuggingFace Hub (default: True)
        private: Whether to make the repository private (default: False)
        token: HuggingFace API token (optional)
        local_save_path: Local path to save dataset (optional)
        resize_height: Height for comparison images (default: 256)
        spacing: Spacing between images in comparison (default: 10)

    Returns:
        Created Dataset object

    Raises:
        ValueError: If data directory structure is invalid or no samples found
    """
    config = DatasetConfig(
        data_dir=Path(data_dir),
        style_images_dir=Path(style_images_dir),
        repo_id=repo_id,
        split=split,
        push_to_hub=push_to_hub,
        private=private,
        token=token,
        resize_height=resize_height,
        spacing=spacing,
    )

    builder = DatasetBuilder(config)
    dataset = builder.build()

    if local_save_path:
        builder.save_local(dataset, Path(local_save_path))

    if push_to_hub:
        builder.push(dataset)

    return dataset


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create HuggingFace dataset from FontDiffusion images"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory (with ContentImage/ and TargetImage/)",
    )
    parser.add_argument(
        "--style-images-dir",
        type=str,
        required=True,
        help="Path to directory containing style images",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (username/dataset-name)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (default: train)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to Hub",
    )
    parser.add_argument(
        "--local-save",
        type=str,
        help="Save dataset locally to this path",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=256,
        help="Height for comparison images (default: 256)",
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=10,
        help="Spacing between images in comparison (default: 10)",
    )

    args = parser.parse_args()

    try:
        create_dataset(
            data_dir=args.data_dir,
            style_images_dir=args.style_images_dir,
            repo_id=args.repo_id,
            split=args.split,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
            resize_height=args.resize_height,
            spacing=args.spacing,
        )
        logger.info("Dataset creation completed successfully")

    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()