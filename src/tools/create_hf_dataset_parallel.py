"""
Create Hugging Face dataset from generated FontDiffusion images with streaming support.

Enhanced with two-phase parallelization: I/O-bound image loading (ThreadPoolExecutor)
followed by CPU-bound image processing (ProcessPoolExecutor or threads).
Maximizes throughput by separating concerns and avoiding context switching.
"""

import json
import logging
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional, NamedTuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial, lru_cache

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image

from utilities import HFTqdm
from filename_utils import compute_file_hash

logger = logging.getLogger(__name__)


class LoadedImages(NamedTuple):
    """Container for loaded images with metadata."""
    content_img: Image.Image
    style_img: Image.Image
    target_img: Image.Image
    character: str
    style: str
    font: str
    content_path: str
    target_path: str


@dataclass
class DatasetConfig:
    """Configuration for dataset creation."""

    data_dir: Path
    style_images_dir: Path
    repo_id: str
    split: str = "train"
    config_name: Optional[str] = None
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None
    batch_size: int = 500
    resize_height: int = 256
    spacing: int = 10
    io_workers: int = None  # None = auto-detect (CPU count * 2)
    cpu_workers: int = None  # None = auto-detect (CPU count)

    def __post_init__(self):
        """Convert paths to Path if they're strings and set worker counts."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)
        
        cpu_count = os.cpu_count() or 4
        if self.io_workers is None:
            # I/O-bound: use many threads (2x CPU count, can go higher)
            self.io_workers = cpu_count * 2
        if self.cpu_workers is None:
            # CPU-bound: use CPU count (diminishing returns beyond this)
            self.cpu_workers = cpu_count


# Cache style images to avoid re-loading
@lru_cache(maxsize=256)
def _load_cached_style_image(style_path: str) -> Image.Image:
    """Load and cache style images to avoid repeated disk reads.
    
    Args:
        style_path: String path to style image (must be hashable for lru_cache)
    
    Returns:
        Loaded PIL Image
    """
    return Image.open(style_path).convert("RGB")


def _load_image_file(path: Path) -> Optional[Image.Image]:
    """Load a single image file with error handling.
    
    Args:
        path: Path to image file
    
    Returns:
        Loaded PIL Image or None if loading fails
    """
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
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


def _load_images_for_sample(
    gen: dict,
    data_dir: Path,
    style_images_dir: Path,
) -> Optional[LoadedImages]:
    """PHASE 1 (I/O-BOUND): Load all images for a single sample.
    
    This function only performs I/O operations - no CPU-intensive processing.
    Designed to be called with high parallelism (many threads).

    Args:
        gen: Generation metadata dictionary
        data_dir: Data directory path
        style_images_dir: Style images directory path

    Returns:
        LoadedImages container or None if loading fails
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
        # Load all three images (pure I/O, no processing)
        content_img = _load_image_file(content_path)
        target_img = _load_image_file(target_path)
        style_img = _load_cached_style_image(str(style_path))

        # Validate all images loaded successfully
        if content_img is None or target_img is None or style_img is None:
            return None

        return LoadedImages(
            content_img=content_img,
            style_img=style_img,
            target_img=target_img,
            character=char,
            style=style,
            font=font,
            content_path=gen.get("content_image_path", ""),
            target_path=gen.get("target_image_path", ""),
        )

    except Exception as e:
        logger.warning(f"Failed to load images for {char}/{style}: {e}")
        return None


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


def _process_loaded_sample(
    loaded: LoadedImages,
    resize_height: int,
    spacing: int,
) -> Optional[dict[str, Any]]:
    """PHASE 2 (CPU-BOUND): Process pre-loaded images into final sample.
    
    This function only performs CPU-intensive operations - no I/O.
    Designed to be called with moderate parallelism (CPU count).

    Args:
        loaded: LoadedImages container with pre-loaded images
        resize_height: Height for comparison images
        spacing: Spacing between images

    Returns:
        Sample dictionary or None if processing fails
    """
    try:
        # Create comparison (CPU-bound: resizing and compositing)
        comparison_img: Optional[Image.Image] = _create_comparison_image(
            loaded.content_img,
            loaded.style_img,
            loaded.target_img,
            resize_height,
            spacing,
        )

        # Build sample dict
        sample_dict = {
            "character": loaded.character,
            "style": loaded.style,
            "font": loaded.font,
            "content_image": loaded.content_img,
            "style_image": loaded.style_img,
            "target_image": loaded.target_img,
            "content_hash": compute_file_hash(loaded.character, "", loaded.font),
            "target_hash": compute_file_hash(loaded.character, loaded.style, loaded.font),
        }

        if comparison_img is not None:
            sample_dict["comparison_image"] = comparison_img

        return sample_dict

    except Exception as e:
        logger.warning(f"Failed to process sample {loaded.character}/{loaded.style}: {e}")
        return None


class DatasetBuilder:
    """Build FontDiffusion dataset in Hugging Face format with two-phase parallel processing."""

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

    def _generate_samples_two_phase(self) -> Generator[dict[str, Any], None, None]:
        """Generate dataset samples using two-phase parallel processing.
        
        Phase 1 (I/O-bound): Load images with high parallelism (ThreadPoolExecutor)
        Phase 2 (CPU-bound): Process images with moderate parallelism (ThreadPoolExecutor or ProcessPoolExecutor)

        Yields:
            Dictionary containing sample data with individual images and comparison

        This approach maximizes throughput by:
        1. Saturating I/O bandwidth with many concurrent reads
        2. Avoiding context switching between I/O and CPU work
        3. Using optimal worker count for each phase
        """
        checkpoint: dict = self._load_checkpoint()
        generations: list = checkpoint["generations"]

        logger.info(
            f"Processing {len(generations)} samples with two-phase parallelization:\n"
            f"  Phase 1 (I/O): {self.config.io_workers} threads\n"
            f"  Phase 2 (CPU): {self.config.cpu_workers} threads"
        )

        skipped: int = 0
        processed: int = 0
        batch_size = self.config.batch_size
        total_batches = (len(generations) + batch_size - 1) // batch_size

        start_time = time.time()

        # Process in batches
        for batch_idx, i in enumerate(range(0, len(generations), batch_size), 1):
            batch = generations[i : i + batch_size]
            
            logger.info(
                f"Processing batch {batch_idx}/{total_batches} ({len(batch)} samples)..."
            )
            batch_start = time.time()

            # ===== PHASE 1: I/O-BOUND IMAGE LOADING (HIGH PARALLELISM) =====
            logger.info(f"  Phase 1: Loading images with {self.config.io_workers} I/O threads...")
            io_start = time.time()

            loaded_samples: list[LoadedImages] = []
            load_func = partial(
                _load_images_for_sample,
                data_dir=self.data_dir,
                style_images_dir=self.style_images_dir,
            )

            # Use ThreadPoolExecutor with high worker count for I/O
            with ThreadPoolExecutor(max_workers=self.config.io_workers) as io_executor:
                io_futures = {io_executor.submit(load_func, gen): gen for gen in batch}
                
                for future in as_completed(io_futures):
                    loaded = future.result()
                    if loaded is not None:
                        loaded_samples.append(loaded)
                    else:
                        skipped += 1

            io_time = time.time() - io_start
            logger.info(
                f"  Phase 1 completed: {len(loaded_samples)}/{len(batch)} loaded in {io_time:.2f}s "
                f"({io_time/len(batch):.3f}s/sample)"
            )

            if not loaded_samples:
                logger.warning(f"  Batch {batch_idx}: No valid samples loaded, skipping processing phase")
                continue

            # ===== PHASE 2: CPU-BOUND IMAGE PROCESSING (MODERATE PARALLELISM) =====
            logger.info(f"  Phase 2: Processing images with {self.config.cpu_workers} CPU threads...")
            cpu_start = time.time()

            process_func = partial(
                _process_loaded_sample,
                resize_height=self.config.resize_height,
                spacing=self.config.spacing,
            )

            batch_valid = 0
            # Use ThreadPoolExecutor with CPU count for processing
            # Note: Could use ProcessPoolExecutor here, but ThreadPoolExecutor avoids
            # pickling overhead and is sufficient for PIL operations
            with ThreadPoolExecutor(max_workers=self.config.cpu_workers) as cpu_executor:
                cpu_futures = {cpu_executor.submit(process_func, loaded): loaded for loaded in loaded_samples}
                
                for future in as_completed(cpu_futures):
                    sample = future.result()
                    if sample is not None:
                        yield sample
                        processed += 1
                        batch_valid += 1
                    else:
                        skipped += 1

            cpu_time = time.time() - cpu_start
            logger.info(
                f"  Phase 2 completed: {batch_valid}/{len(loaded_samples)} processed in {cpu_time:.2f}s "
                f"({cpu_time/len(loaded_samples):.3f}s/sample)"
            )

            # Log batch summary
            batch_time = time.time() - batch_start
            logger.info(
                f"Batch {batch_idx}/{total_batches} completed in {batch_time:.2f}s "
                f"(I/O: {io_time:.2f}s, CPU: {cpu_time:.2f}s, total: {batch_valid} valid)"
            )

            # Overall progress
            if processed > 0:
                elapsed = time.time() - start_time
                rate = (processed + skipped) / elapsed
                progress_pct = (processed + skipped) / len(generations) * 100
                eta = (len(generations) - processed - skipped) / rate if rate > 0 else 0
                logger.info(
                    f"Overall progress: {processed}/{len(generations)} "
                    f"({progress_pct:.1f}%, {rate:.1f} samples/s, ETA: {eta:.0f}s)"
                )

        if processed == 0:
            raise ValueError("No valid samples found")

        if skipped > 0:
            logger.warning(f"Skipped {skipped}/{len(generations)} invalid samples")

        total_time = time.time() - start_time
        logger.info(
            f"Successfully processed {processed} samples in {total_time:.2f}s "
            f"({processed/total_time:.1f} samples/s)"
        )

    def _generate_samples_single_threaded(self) -> Generator[dict[str, Any], None, None]:
        """Generate dataset samples one at a time (single-threaded fallback).

        Yields:
            Dictionary containing sample data with individual images and comparison
        """
        checkpoint: dict = self._load_checkpoint()
        generations: list = checkpoint["generations"]

        logger.info(f"Processing {len(generations)} samples (single-threaded)...")

        skipped: int = 0
        processed: int = 0
        start_time = time.time()

        for gen in generations:
            # Phase 1: Load images
            loaded = _load_images_for_sample(
                gen,
                self.data_dir,
                self.style_images_dir,
            )

            if loaded is None:
                skipped += 1
                continue

            # Phase 2: Process images
            sample = _process_loaded_sample(
                loaded,
                self.config.resize_height,
                self.config.spacing,
            )

            if sample is not None:
                yield sample
                processed += 1
            else:
                skipped += 1

            if (processed + skipped) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (processed + skipped) / elapsed
                logger.info(
                    f"Processed {processed} samples ({skipped} skipped) "
                    f"at {rate:.1f} samples/s"
                )

        if processed == 0:
            raise ValueError("No valid samples found")

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid samples")

        total_time = time.time() - start_time
        logger.info(
            f"Successfully processed {processed} samples in {total_time:.2f}s "
            f"({processed/total_time:.1f} samples/s)"
        )

    def build_streaming(self, use_parallel: bool = True) -> Dataset:
        """Build dataset using streaming to minimize memory usage.

        Args:
            use_parallel: Whether to use two-phase parallel processing (default: True)

        Returns:
            HuggingFace Dataset created from generator

        Raises:
            ValueError: If no valid samples are found
        """
        mode = "two-phase parallel" if use_parallel else "single-threaded"
        logger.info(f"Building dataset with streaming ({mode})...")

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

        # Select generator based on parallelization strategy
        generator_func = (
            self._generate_samples_two_phase
            if use_parallel
            else self._generate_samples_single_threaded
        )

        logger.info("Converting samples to Arrow format (this may take a while)...")
        conversion_start = time.time()

        dataset = Dataset.from_generator(
            generator_func,
            features=features,
        )

        conversion_time = time.time() - conversion_start
        logger.info(
            f"Dataset conversion completed in {conversion_time:.2f}s "
            f"({len(dataset)} samples)"
        )

        return dataset

    def build_batched(self, use_parallel: bool = True) -> Dataset:
        """Build dataset in batches for better control over memory usage.

        Args:
            use_parallel: Whether to use two-phase parallel processing

        Returns:
            HuggingFace Dataset created from batched processing

        Raises:
            ValueError: If no valid samples are found
        """
        mode = "two-phase parallel" if use_parallel else "single-threaded"
        logger.info(f"Building dataset with batching ({mode})...")

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
        
        # Select generator based on parallelization strategy
        generator_func = (
            self._generate_samples_two_phase
            if use_parallel
            else self._generate_samples_single_threaded
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
        """Push dataset to Hugging Face Hub with streaming and detailed logging.

        Args:
            dataset: Dataset to push
        """
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return

        logger.info(f"Pushing dataset to {self.config.repo_id} with streaming...")
        logger.info("Authenticating with Hugging Face Hub...")
        
        push_start = time.time()

        try:
            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                config_name=self.config.config_name,
                private=self.config.private,
                token=self.config.token,
            )

            push_time = time.time() - push_start
            logger.info(
                f"Successfully pushed to https://huggingface.co/datasets/{self.config.repo_id} "
                f"in {push_time:.2f}s"
            )
        except Exception as e:
            logger.error(f"Push failed after {time.time() - push_start:.2f}s: {e}")
            raise

    def save_local_streaming(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk with streaming.

        Args:
            dataset: Dataset to save
            output_path: Local directory path
        """
        logger.info(f"Saving dataset to {output_path} with streaming...")
        save_start = time.time()

        dataset.save_to_disk(
            str(output_path),
        )

        save_time = time.time() - save_start
        logger.info(f"Dataset saved successfully in {save_time:.2f}s")


def create_dataset(
    data_dir: str | Path,
    style_images_dir: str | Path,
    repo_id: str,
    split: str = "train",
    config_name: Optional[str] = None,
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str | Path] = None,
    batch_size: int = 500,
    use_streaming: bool = True,
    resize_height: int = 256,
    spacing: int = 10,
    io_workers: Optional[int] = None,
    cpu_workers: Optional[int] = None,
    use_parallel: bool = True,
) -> Dataset:
    """Create and optionally push dataset to Hub with streaming support.

    Args:
        data_dir: Path to data directory containing ContentImage/ and TargetImage/
        style_images_dir: Path to directory containing style images
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        split: Dataset split name (default: 'train')
        config_name: Dataset configuration name (e.g., 'streaming', 'default')
        push_to_hub: Whether to push to HuggingFace Hub (default: True)
        private: Whether to make the repository private (default: False)
        token: HuggingFace API token (optional)
        local_save_path: Local path to save dataset (optional)
        batch_size: Number of samples per batch (default: 500)
        use_streaming: Use streaming mode (True) or batched mode (False)
        resize_height: Height for comparison images (default: 256)
        spacing: Spacing between images in comparison (default: 10)
        io_workers: Number of I/O threads (default: CPU count * 2)
        cpu_workers: Number of CPU threads (default: CPU count)
        use_parallel: Whether to use two-phase parallel processing (default: True)

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
        config_name=config_name,
        push_to_hub=push_to_hub,
        private=private,
        token=token,
        batch_size=batch_size,
        resize_height=resize_height,
        spacing=spacing,
        io_workers=io_workers,
        cpu_workers=cpu_workers,
    )

    builder = DatasetBuilder(config)

    if use_streaming:
        dataset = builder.build_streaming(use_parallel=use_parallel)
    else:
        dataset = builder.build_batched(use_parallel=use_parallel)

    if local_save_path:
        builder.save_local_streaming(dataset, Path(local_save_path))

    if push_to_hub:
        builder.push_streaming(dataset)

    return dataset


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Create HuggingFace dataset from FontDiffusion images with two-phase parallel processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard (recommended - automatic worker tuning)
  python tools/create_hf_dataset_parallel.py --data-dir my_dataset/train \\
    --style-images-dir style_images/ --repo-id user/dataset

  # High-performance mode (manual tuning)
  python tools/create_hf_dataset_parallel.py --data-dir my_dataset/train \\
    --style-images-dir style_images/ --repo-id user/dataset \\
    --io-workers 32 --cpu-workers 16 --batch-size 1000

  # Single-threaded (debugging)
  python tools/create_hf_dataset_parallel.py --data-dir my_dataset/train \\
    --style-images-dir style_images/ --repo-id user/dataset \\
    --no-parallel

Two-phase processing maximizes throughput by:
1. Phase 1 (I/O): Saturate disk bandwidth with many concurrent reads
2. Phase 2 (CPU): Process images with optimal CPU utilization
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
        "--config-name",
        type=str,
        default=None,
        help="Dataset configuration name (e.g., 'streaming', 'default')",
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
        default=500,
        help="Batch size for processing (default: 500)",
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
        "--io-workers",
        type=int,
        default=os.cpu_count() * 2,
        help="Number of I/O threads (default: CPU count * 2)",
    )
    parser.add_argument(
        "--cpu-workers",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of CPU threads (default: CPU count - 1)",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel processing (use single-threaded)",
    )
    args = parser.parse_args()

    try:
        create_dataset(
            data_dir=args.data_dir,
            style_images_dir=args.style_images_dir,
            repo_id=args.repo_id,
            split=args.split,
            config_name=args.config_name,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
            batch_size=args.batch_size,
            use_streaming=not args.use_batched,
            resize_height=args.resize_height,
            spacing=args.spacing,
            io_workers=args.io_workers,
            cpu_workers=args.cpu_workers,
            use_parallel=not args.no_parallel,
        )
        logger.info("Dataset creation completed successfully")

    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()