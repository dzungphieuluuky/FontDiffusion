"""
Create Hugging Face dataset from generated FontDiffusion images with streaming support.

This module builds datasets from FontDiffusion outputs using streaming to prevent
RAM overflow, especially useful in constrained environments like Colab or Kaggle.
Includes comparison image generation for visual inspection.

Enhanced with multiprocessing for faster image processing and parallel upload.
"""

import json
import logging
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Generator, Optional
from multiprocessing import Pool, cpu_count
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image
from huggingface_hub import HfApi

from utilities import HFTqdm
from filename_utils import compute_file_hash

logger = logging.getLogger(__name__)


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
    batch_size: int = 100
    resize_height: int = 256
    spacing: int = 10
    num_workers: int = None  # None = auto-detect
    config_name: str = None
    max_shard_size: str = "500MB"
    parallel_upload_workers: int = 4  # Parallel shard upload workers

    def __post_init__(self):
        """Convert paths to Path if they're strings."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)
        if self.num_workers is None:
            self.num_workers = max(1, cpu_count() - 1)


def _resize_image(image: Image.Image, target_height: int) -> Image.Image:
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
    content_img: Image.Image,
    style_img: Image.Image,
    target_img: Image.Image,
    resize_height: int,
    spacing: int,
) -> Optional[Image.Image]:
    """Create side-by-side comparison image (content | style | target).

    Args:
        content_img: Content/character image
        style_img: Style image
        target_img: Target image
        resize_height: Target height for resizing
        spacing: Spacing between images

    Returns:
        Comparison PIL Image or None if creation fails
    """
    try:
        content_resized = _resize_image(content_img, resize_height)
        style_resized = _resize_image(style_img, resize_height)
        target_resized = _resize_image(target_img, resize_height)

        total_width = (
            content_resized.width
            + style_resized.width
            + target_resized.width
            + 2 * spacing
        )
        total_height = resize_height

        comparison = Image.new(
            "RGB", (total_width, total_height), color=(255, 255, 255)
        )

        x_offset = 0
        comparison.paste(content_resized, (x_offset, 0))
        x_offset += content_resized.width + spacing
        comparison.paste(style_resized, (x_offset, 0))
        x_offset += style_resized.width + spacing
        comparison.paste(target_resized, (x_offset, 0))

        return comparison

    except Exception as e:
        logger.warning(f"Failed to create comparison image: {e}")
        return None


def _find_style_image(style_images_dir: Path, style: str) -> Optional[Path]:
    """Find style image in the style images directory.

    Args:
        style_images_dir: Directory containing style images
        style: Style name

    Returns:
        Path to style image or None if not found
    """
    for ext in [".png", ".jpg", ".jpeg"]:
        style_path = style_images_dir / f"{style}{ext}"
        if style_path.exists():
            return style_path
    return None


def _process_single_sample(
    gen: dict,
    data_dir: Path,
    style_images_dir: Path,
    resize_height: int,
    spacing: int,
) -> Optional[dict[str, Any]]:
    """Process a single sample (load images, create comparison).

    This function is designed to be called in parallel via multiprocessing.

    Args:
        gen: Generation metadata dictionary
        data_dir: Data directory path
        style_images_dir: Style images directory path
        resize_height: Height for comparison images
        spacing: Spacing between images

    Returns:
        Sample dictionary or None if processing fails
    """
    char: str = gen.get("character", "")
    style: str = gen.get("style", "")
    font: str = gen.get("font", "unknown")

    content_path: Path = data_dir / gen.get("content_image_path", "")
    target_path: Path = data_dir / gen.get("target_image_path", "")
    style_path: Optional[Path] = _find_style_image(style_images_dir, style)

    # Validate paths
    if not content_path.exists() or not target_path.exists():
        return None
    if style_path is None or not style_path.exists():
        return None

    try:
        # Load images
        content_img: Image.Image = Image.open(content_path).convert("RGB")
        style_img: Image.Image = Image.open(style_path).convert("RGB")
        target_img: Image.Image = Image.open(target_path).convert("RGB")

        # Create comparison
        comparison_img: Optional[Image.Image] = _create_comparison_image(
            content_img, style_img, target_img, resize_height, spacing
        )

        # Build sample dict
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

        if comparison_img is not None:
            sample_dict["comparison_image"] = comparison_img

        return sample_dict

    except Exception as e:
        logger.warning(f"Failed to process sample {char}/{style}: {e}")
        return None


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
            raise ValueError(
                f"Style images directory not found: {self.style_images_dir}"
            )

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

    def _generate_samples_parallel(self) -> Generator[dict[str, Any], None, None]:
        """Generate dataset samples using multiprocessing.

        Yields:
            Dictionary containing sample data with individual images and comparison

        This generator uses multiprocessing to parallelize image loading and processing.
        """
        checkpoint: dict = self._load_checkpoint()
        generations: list = checkpoint["generations"]

        logger.info(
            f"Processing {len(generations)} samples with {self.config.num_workers} workers..."
        )

        # Create partial function with fixed parameters
        process_func = partial(
            _process_single_sample,
            data_dir=self.data_dir,
            style_images_dir=self.style_images_dir,
            resize_height=self.config.resize_height,
            spacing=self.config.spacing,
        )

        skipped: int = 0
        processed: int = 0

        # Process in batches using multiprocessing
        batch_size = self.config.batch_size

        with Pool(processes=self.config.num_workers) as pool:
            for i in range(0, len(generations), batch_size):
                batch = generations[i : i + batch_size]

                # Process batch in parallel
                results = pool.map(process_func, batch)

                # Yield valid results
                for sample in results:
                    if sample is not None:
                        yield sample
                        processed += 1
                    else:
                        skipped += 1

                if processed % 100 == 0 and processed > 0:
                    logger.info(f"Processed {processed} samples...")

        if processed == 0:
            raise ValueError("No valid samples found")

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid samples")

        logger.info(f"Successfully processed {processed} samples")

    def _generate_samples(self) -> Generator[dict[str, Any], None, None]:
        """Generate dataset samples one at a time (single-threaded fallback).

        Yields:
            Dictionary containing sample data with individual images and comparison
        """
        checkpoint: dict = self._load_checkpoint()
        generations: list = checkpoint["generations"]

        skipped: int = 0
        processed: int = 0

        for gen in generations:
            sample = _process_single_sample(
                gen,
                self.data_dir,
                self.style_images_dir,
                self.config.resize_height,
                self.config.spacing,
            )

            if sample is not None:
                yield sample
                processed += 1
            else:
                skipped += 1

            if processed % 100 == 0 and processed > 0:
                logger.info(f"Processed {processed} samples...")

        if processed == 0:
            raise ValueError("No valid samples found")

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid samples")

        logger.info(f"Successfully processed {processed} samples")

    def build_streaming(self, use_multiprocessing: bool = True) -> Dataset:
        """Build dataset using streaming to minimize memory usage.

        Args:
            use_multiprocessing: Whether to use multiprocessing for speed

        Returns:
            HuggingFace Dataset created from generator

        Raises:
            ValueError: If no valid samples are found
        """
        logger.info(
            f"Building dataset with streaming (multiprocessing={use_multiprocessing})..."
        )

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

        generator_func = (
            self._generate_samples_parallel
            if use_multiprocessing
            else self._generate_samples
        )

        dataset = Dataset.from_generator(
            generator_func,
            features=features,
        )

        return dataset

    def build_batched(self, use_multiprocessing: bool = True) -> Dataset:
        """Build dataset in batches for better control over memory usage.

        Args:
            use_multiprocessing: Whether to use multiprocessing for speed

        Returns:
            HuggingFace Dataset created from batched processing

        Raises:
            ValueError: If no valid samples are found
        """
        logger.info(
            f"Building dataset with batching (multiprocessing={use_multiprocessing})..."
        )

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
        generator_func = (
            self._generate_samples_parallel
            if use_multiprocessing
            else self._generate_samples
        )

        for sample in generator_func():
            for key, value in sample.items():
                current_batch[key].append(value)

            sample_count += 1

            if sample_count % self.config.batch_size == 0:
                batch_ds = Dataset.from_dict(current_batch, features=features)
                batch_datasets.append(batch_ds)
                current_batch = {key: [] for key in current_batch}
                logger.info(f"Completed batch {len(batch_datasets)}")

        if current_batch["character"]:
            batch_ds = Dataset.from_dict(current_batch, features=features)
            batch_datasets.append(batch_ds)

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

        dataset.push_to_hub(
            repo_id=self.config.repo_id,
            split=self.config.split,
            config_name=self.config.config_name,
            private=self.config.private,
            token=self.config.token,
            max_shard_size=self.config.max_shard_size,
            num_proc=self.config.num_workers,  # Parallel shard writing
        )

        logger.info(
            f"Successfully pushed to https://huggingface.co/datasets/{self.config.repo_id}"
        )

    def push_parallel(self, dataset: Dataset) -> None:
        """Push dataset to Hub with parallel shard uploads for maximum speed.

        This method manually manages shard creation and parallel uploads using
        concurrent.futures for improved upload performance.

        Args:
            dataset: Dataset to push
        """
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return

        logger.info(
            f"Pushing dataset to {self.config.repo_id} with parallel uploads "
            f"({self.config.parallel_upload_workers} workers)..."
        )

        api = HfApi(token=self.config.token)

        # Create repository if it doesn't exist
        try:
            api.create_repo(
                repo_id=self.config.repo_id,
                repo_type="dataset",
                private=self.config.private,
                exist_ok=True,
            )
        except Exception as e:
            logger.warning(f"Repository creation warning: {e}")

        # Save shards to temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            logger.info("Saving dataset shards locally...")
            dataset.save_to_disk(
                str(tmp_path),
                max_shard_size=self.config.max_shard_size,
                num_proc=self.config.num_workers,
            )

            # Get all shard files
            shard_files = list(tmp_path.glob("**/*"))
            shard_files = [f for f in shard_files if f.is_file()]

            logger.info(
                f"Uploading {len(shard_files)} files in parallel "
                f"(max {self.config.parallel_upload_workers} concurrent uploads)..."
            )

            # Upload shards in parallel using ThreadPoolExecutor
            def upload_file(file_path: Path) -> tuple[str, bool]:
                """Upload a single file to the Hub.

                Args:
                    file_path: Path to file to upload

                Returns:
                    Tuple of (relative_path, success)
                """
                try:
                    relative_path = file_path.relative_to(tmp_path)
                    path_in_repo = f"{self.config.split}/{relative_path}"

                    api.upload_file(
                        path_or_fileobj=str(file_path),
                        path_in_repo=path_in_repo,
                        repo_id=self.config.repo_id,
                        repo_type="dataset",
                        token=self.config.token,
                    )
                    return str(relative_path), True
                except Exception as e:
                    logger.error(f"Failed to upload {file_path.name}: {e}")
                    return str(file_path.name), False

            # Use ThreadPoolExecutor for parallel uploads
            uploaded = 0
            failed = 0

            with ThreadPoolExecutor(
                max_workers=self.config.parallel_upload_workers
            ) as executor:
                # Submit all upload tasks
                futures = {
                    executor.submit(upload_file, shard_path): shard_path
                    for shard_path in shard_files
                }

                # Process completed uploads
                for future in as_completed(futures):
                    relative_path, success = future.result()
                    if success:
                        uploaded += 1
                        if uploaded % 10 == 0 or uploaded == len(shard_files):
                            logger.info(
                                f"Upload progress: {uploaded}/{len(shard_files)} "
                                f"({uploaded * 100 // len(shard_files)}%)"
                            )
                    else:
                        failed += 1

        if failed > 0:
            logger.warning(f"Failed to upload {failed} files")

        logger.info(
            f"Successfully pushed {uploaded} files to "
            f"https://huggingface.co/datasets/{self.config.repo_id}"
        )

    def save_local_streaming(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk with streaming.

        Args:
            dataset: Dataset to save
            output_path: Local directory path
        """
        logger.info(f"Saving dataset to {output_path} with streaming...")

        dataset.save_to_disk(
            str(output_path),
            max_shard_size=self.config.max_shard_size,
            num_proc=self.config.num_workers,
        )

        logger.info("Dataset saved successfully")


def create_dataset(
    data_dir: str | Path,
    style_images_dir: str | Path,
    repo_id: str,
    split: str = "train",
    push_to_hub: bool = True,
    config_name: str = "streaming",
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str | Path] = None,
    batch_size: int = 100,
    use_streaming: bool = True,
    resize_height: int = 256,
    spacing: int = 10,
    num_workers: Optional[int] = None,
    use_multiprocessing: bool = True,
    max_shard_size: str = "500MB",
    parallel_upload: bool = True,
    parallel_upload_workers: int = 4,
) -> Dataset:
    """Create and optionally push dataset to Hub with streaming support.

    Args:
        data_dir: Path to data directory containing ContentImage/ and TargetImage/
        style_images_dir: Path to directory containing style images
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        split: Dataset split name (default: 'train')
        push_to_hub: Whether to push to HuggingFace Hub (default: True)
        config_name: Dataset configuration name (default: 'streaming')
        private: Whether to make the repository private (default: False)
        token: HuggingFace API token (optional)
        local_save_path: Local path to save dataset (optional)
        batch_size: Number of samples per batch (default: 100)
        use_streaming: Use streaming mode (True) or batched mode (False)
        resize_height: Height for comparison images (default: 256)
        spacing: Spacing between images in comparison (default: 10)
        num_workers: Number of worker processes (default: CPU count - 1)
        use_multiprocessing: Whether to use multiprocessing (default: True)
        max_shard_size: Maximum shard size (default: '500MB')
        parallel_upload: Use parallel shard uploads (default: True)
        parallel_upload_workers: Number of parallel upload workers (default: 4)

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
        batch_size=batch_size,
        resize_height=resize_height,
        spacing=spacing,
        num_workers=num_workers,
        config_name=config_name,
        max_shard_size=max_shard_size,
        parallel_upload_workers=parallel_upload_workers,
    )

    builder = DatasetBuilder(config)

    if use_streaming:
        dataset = builder.build_streaming(use_multiprocessing=use_multiprocessing)
    else:
        dataset = builder.build_batched(use_multiprocessing=use_multiprocessing)

    if local_save_path:
        builder.save_local_streaming(dataset, Path(local_save_path))

    if push_to_hub:
        if parallel_upload:
            builder.push_parallel(dataset)
        else:
            builder.push_streaming(dataset)

    return dataset


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create HuggingFace dataset from FontDiffusion images with streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard streaming with parallel upload
  python create_hf_dataset_streaming.py --data-dir my_dataset/train \\
    --style-images-dir style_images/ --repo-id user/dataset \\
    --parallel-upload --parallel-upload-workers 4

  # High-speed processing and upload
  python create_hf_dataset_streaming.py --data-dir my_dataset/train \\
    --style-images-dir style_images/ --repo-id user/dataset \\
    --num-workers 12 --batch-size 200 --parallel-upload-workers 6
  
  # Local save only (no upload)
  python create_hf_dataset_streaming.py --data-dir my_dataset/train \\
    --style-images-dir style_images/ --repo-id user/dataset \\
    --no-push --local-save ./output
        """,
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
        default=False,
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count - 1)",
    )
    parser.add_argument(
        "--no-multiprocessing",
        action="store_true",
        default=False,
        help="Disable multiprocessing (use single-threaded processing)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="streaming",
        help="Dataset config name (default: streaming)",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="500MB",
        help="Maximum shard size (default: 500MB)",
    )
    parser.add_argument(
        "--parallel-upload",
        action="store_true",
        default=True,
        help="Use parallel shard uploads for faster pushing",
    )
    parser.add_argument(
        "--parallel-upload-workers",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of parallel upload workers (default: CPU count - 1)",
    )

    args = parser.parse_args()

    try:
        create_dataset(
            data_dir=args.data_dir,
            style_images_dir=args.style_images_dir,
            repo_id=args.repo_id,
            split=args.split,
            push_to_hub=not args.no_push,
            config_name=args.config_name,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
            batch_size=args.batch_size,
            use_streaming=not args.use_batched,
            resize_height=args.resize_height,
            spacing=args.spacing,
            num_workers=args.num_workers,
            use_multiprocessing=not args.no_multiprocessing,
            max_shard_size=args.max_shard_size,
            parallel_upload=args.parallel_upload,
            parallel_upload_workers=args.parallel_upload_workers,
        )
        logger.info("Dataset creation completed successfully")

    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()