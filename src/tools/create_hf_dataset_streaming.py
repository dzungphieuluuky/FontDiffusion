"""
Create Hugging Face dataset from generated FontDiffusion images with streaming support.

This module builds datasets from FontDiffusion outputs using streaming to prevent
RAM overflow, especially useful in constrained environments like Colab or Kaggle.
Includes comparison image generation for visual inspection.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image

from utilities import HFTqdm
from filename_utils import compute_file_hash

logger = logging.getLogger("DatasetCreator")


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""

    data_dir: Path
    repo_id: str
    split: str = "train"
    push_to_hub: bool = True
    private: bool = False
    token: str = None
    batch_size: int = 100  # Process in batches to limit memory usage
    resize_height: int = 256  # Height for comparison images
    spacing: int = 10  # Spacing between images in comparison

    def __post_init__(self):
        """Convert data_dir to Path if it's a string."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)


class DatasetBuilder:
    """Build FontDiffusion dataset in Hugging Face format with streaming."""

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

        logger.info("Directory structure validated successfully")

    def _load_checkpoint(self) -> dict[str, list[dict[str, str]]]:
        """Load and validate results checkpoint.

        Returns:
            Checkpoint data dictionary

        Raises:
            ValueError: If checkpoint is invalid or empty
        """
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE

        with checkpoint_path.open("r", encoding="utf-8") as f:
            data: dict[str, list[dict[str, str]]] = json.load(f)

        generations = data.get("generations", [])
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
    ) -> Image.Image:
        """Create side-by-side comparison image (content | style | target).

        Args:
            content_img: Content/character image
            style_img: Style image
            target_img: Target image

        Returns:
            Comparison PIL Image
        """
        try:
            # Resize all images to same height
            content_resized = self._resize_image(content_img, self.config.resize_height)
            style_resized = self._resize_image(style_img, self.config.resize_height)
            target_resized = self._resize_image(target_img, self.config.resize_height)

            # Calculate total width
            total_width = (
                content_resized.width
                + style_resized.width
                + target_resized.width
                + 2 * self.config.spacing
            )
            total_height = self.config.resize_height

            # Create comparison image
            comparison = Image.new(
                "RGB", (total_width, total_height), color=(255, 255, 255)
            )

            # Paste images left to right
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

    def _generate_samples(self) -> Generator[dict[str, Any], None, None]:
        """Generate dataset samples one at a time.

        Yields:
            Dictionary containing sample data with individual images and comparison

        This generator loads images one at a time to minimize memory usage.
        """
        checkpoint: dict[str, list[dict[str, str]]] = self._load_checkpoint()
        generations: list[dict[str, str]] = checkpoint["generations"]

        skipped: int = 0
        processed: int = 0

        for gen in generations:
            char: str = gen.get("character", "")
            style: str = gen.get("style", "")
            font: str = gen.get("font", "unknown")

            # Construct paths
            content_path: Path = self.data_dir / gen.get("content_image_path", "")
            target_path: Path = self.data_dir / gen.get("target_image_path", "")

            # Try to find style image - search in style_images directory
            style_path: Optional[Path] = self._find_style_image(style)

            # Validate paths exist
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

            # Load images
            try:
                content_img: Image.Image = Image.open(content_path).convert("RGB")
                style_img: Image.Image = Image.open(style_path).convert("RGB")
                target_img: Image.Image = Image.open(target_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to load images for {char}/{style}: {e}")
                skipped += 1
                continue

            # Create comparison image
            comparison_img = self._create_comparison_image(
                content_img, style_img, target_img
            )

            # Yield sample immediately to avoid holding in memory
            sample_dict = {
                "character": char,
                "style": style,
                "font": font,
                "content_image": content_img,
                "style_image": style_img,
                "target_image": target_img,
                "content_hash": compute_file_hash(char, "", font),
                "target_hash": compute_file_hash(char, style, font),
            }

            # Add comparison image if created successfully
            if comparison_img is not None:
                sample_dict["comparison_image"] = comparison_img

            yield sample_dict

            processed += 1

            # Log progress periodically
            if processed % 100 == 0:
                logger.info(f"Processed {processed} samples...")

        if processed == 0:
            raise ValueError("No valid samples found")

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid samples")

        logger.info(f"Successfully processed {processed} samples")

    def _find_style_image(self, style: str) -> Optional[Path]:
        """Find style image in the data directory.

        Args:
            style: Style name

        Returns:
            Path to style image or None if not found
        """
        # Look in style_images directory if it exists
        style_images_dir = self.data_dir / "style_images"
        if style_images_dir.exists():
            for ext in [".png", ".jpg", ".jpeg"]:
                style_path = style_images_dir / f"{style}{ext}"
                if style_path.exists():
                    return style_path

        # Fallback: look in root data directory
        for ext in [".png", ".jpg", ".jpeg"]:
            style_path = self.data_dir / f"{style}{ext}"
            if style_path.exists():
                return style_path

        return None

    def build_streaming(self) -> Dataset:
        """Build dataset using streaming to minimize memory usage.

        Returns:
            HuggingFace Dataset created from generator

        Raises:
            ValueError: If no valid samples are found
        """
        logger.info("Building dataset with streaming...")

        # Define explicit features for better type safety
        features = Features(
            {
                "character": Value("string"),
                "style": Value("string"),
                "font": Value("string"),
                "content_image": HFImage(),
                "style_image": HFImage(),
                "target_image": HFImage(),
                "comparison_image": HFImage(),  # Side-by-side comparison
                "content_hash": Value("string"),
                "target_hash": Value("string"),
            }
        )

        # Create dataset from generator for memory efficiency
        dataset = Dataset.from_generator(
            self._generate_samples,
            features=features,
        )

        return dataset

    def build_batched(self) -> Dataset:
        """Build dataset in batches for better control over memory usage.

        Returns:
            HuggingFace Dataset created from batched processing

        Raises:
            ValueError: If no valid samples are found
        """
        logger.info("Building dataset with batching...")

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

        batch_datasets = []
        current_batch = {
            "character": [],
            "style": [],
            "font": [],
            "content_image": [],
            "style_image": [],
            "target_image": [],
            "comparison_image": [],
            "content_hash": [],
            "target_hash": [],
        }

        sample_count = 0

        for sample in self._generate_samples():
            # Add to current batch
            for key, value in sample.items():
                current_batch[key].append(value)

            sample_count += 1

            # When batch is full, create dataset and clear batch
            if sample_count % self.config.batch_size == 0:
                batch_ds = Dataset.from_dict(current_batch, features=features)
                batch_datasets.append(batch_ds)

                # Clear batch to free memory
                current_batch = {key: [] for key in current_batch}

                logger.info(f"Completed batch {len(batch_datasets)}")

        # Handle remaining samples
        if current_batch["character"]:
            batch_ds = Dataset.from_dict(current_batch, features=features)
            batch_datasets.append(batch_ds)

        # Concatenate all batches
        if not batch_datasets:
            raise ValueError("No valid samples found")

        logger.info(f"Concatenating {len(batch_datasets)} batches...")
        from datasets import concatenate_datasets

        dataset = concatenate_datasets(batch_datasets)

        return dataset

    def push_streaming(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub with streaming.

        Args:
            dataset: Dataset to push
        """
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return

        logger.info(f"Pushing dataset to {self.config.repo_id} with streaming...")

        # Use max_shard_size to control memory usage during upload
        dataset.push_to_hub(
            repo_id=self.config.repo_id,
            split=self.config.split,
            private=self.config.private,
            token=self.config.token,
            max_shard_size="500MB",  # Split into smaller shards
        )

        logger.info(
            f"Successfully pushed to https://huggingface.co/datasets/{self.config.repo_id}"
        )

    def save_local_streaming(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk with streaming.

        Args:
            dataset: Dataset to save
            output_path: Local directory path
        """
        logger.info(f"Saving dataset to {output_path} with streaming...")

        # Use max_shard_size to split into smaller files
        dataset.save_to_disk(
            str(output_path),
            max_shard_size="500MB",
        )

        logger.info("Dataset saved successfully")


def create_dataset(
    data_dir: str | Path,
    repo_id: str,
    split: str = "train",
    push_to_hub: bool = True,
    private: bool = False,
    token: str = None,
    local_save_path: str | Path = None,
    batch_size: int = 100,
    use_streaming: bool = True,
    resize_height: int = 256,
    spacing: int = 10,
) -> Dataset:
    """Create and optionally push dataset to Hub with streaming support.

    Args:
        data_dir: Path to data directory containing ContentImage/ and TargetImage/
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        split: Dataset split name (default: 'train')
        push_to_hub: Whether to push to HuggingFace Hub (default: True)
        private: Whether to make the repository private (default: False)
        token: HuggingFace API token (optional)
        local_save_path: Local path to save dataset (optional)
        batch_size: Number of samples per batch (default: 100)
        use_streaming: Use streaming mode (True) or batched mode (False)
        resize_height: Height for comparison images (default: 256)
        spacing: Spacing between images in comparison (default: 10)

    Returns:
        Created Dataset object

    Raises:
        ValueError: If data directory structure is invalid or no samples found
    """
    config = DatasetConfig(
        data_dir=Path(data_dir),
        repo_id=repo_id,
        split=split,
        push_to_hub=push_to_hub,
        private=private,
        token=token,
        batch_size=batch_size,
        resize_height=resize_height,
        spacing=spacing,
    )

    builder = DatasetBuilder(config)

    # Choose streaming or batched approach
    if use_streaming:
        dataset = builder.build_streaming()
    else:
        dataset = builder.build_batched()

    if local_save_path:
        builder.save_local_streaming(dataset, Path(local_save_path))

    if push_to_hub:
        builder.push_streaming(dataset)

    return dataset


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create HuggingFace dataset from FontDiffusion images with streaming"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to data directory (with ContentImage/ and TargetImage/)",
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
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (default: 100)",
    )
    parser.add_argument(
        "--use-batched",
        action="store_true",
        default=False,
        help="Use batched mode instead of streaming",
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
            repo_id=args.repo_id,
            split=args.split,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
            batch_size=args.batch_size,
            use_streaming=not args.use_batched,
            resize_height=args.resize_height,
            spacing=args.spacing,
        )
        logger.info("Dataset creation completed successfully")

    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()