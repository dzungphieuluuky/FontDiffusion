"""
Create Hugging Face dataset from generated FontDiffusion images with streaming support.

This module builds datasets from FontDiffusion outputs using streaming to prevent
RAM overflow, especially useful in constrained environments like Colab or Kaggle.
Includes comparison image generation for visual inspection.

Enhanced with concurrent.futures for better parallel processing control.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional, List, Dict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import threading
from functools import partial

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
    batch_size: int = 100
    resize_height: int = 256
    spacing: int = 10
    num_workers: Optional[int] = None  # None = auto-detect
    config_name: str = "streaming"
    max_workers_per_cpu: int = 2  # Workers per CPU core
    io_max_workers: int = 32  # Max I/O workers for image loading
    
    def __post_init__(self):
        """Convert paths to Path if they're strings."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)
        
        # Auto-tune workers based on available resources
        import os
        cpu_count = os.cpu_count() or 4
        
        if self.num_workers is None:
            # Use all cores for CPU-bound tasks
            self.num_workers = cpu_count
        
        # Limit I/O workers to prevent excessive file descriptor usage
        if self.io_max_workers is None:
            self.io_max_workers = min(64, cpu_count * 4)


class ImageCache:
    """Thread-local cache for frequently accessed images."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._thread_local = threading.local()
    
    @property
    def cache(self) -> Dict[Path, Image.Image]:
        """Get thread-local cache."""
        if not hasattr(self._thread_local, 'cache'):
            self._thread_local.cache = {}
        return self._thread_local.cache
    
    def get(self, path: Path) -> Optional[Image.Image]:
        """Get image from cache if exists."""
        cache = self.cache
        if path in cache:
            return cache[path]
        return None
    
    def set(self, path: Path, image: Image.Image):
        """Add image to cache."""
        cache = self.cache
        if len(cache) >= self.max_size:
            # Remove oldest item (simplistic LRU)
            cache.pop(next(iter(cache)))
        cache[path] = image


def _resize_image(image: Image.Image, target_height: int) -> Image.Image:
    """Resize image to target height while maintaining aspect ratio.

    Args:
        image: PIL Image to resize
        target_height: Target height in pixels

    Returns:
        Resized PIL Image
    """
    if image.height == target_height:
        return image
    
    aspect_ratio = image.width / image.height
    new_width = int(target_height * aspect_ratio)
    
    # Use optimized resize method based on size difference
    if image.height > target_height * 2:
        # Large downscaling, use high quality
        return image.resize((new_width, target_height), Image.Resampling.LANCZOS)
    else:
        # Small scaling, use faster method
        return image.resize((new_width, target_height), Image.Resampling.BILINEAR)


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
        # Resize all images to same height
        content_resized = _resize_image(content_img, resize_height)
        style_resized = _resize_image(style_img, resize_height)
        target_resized = _resize_image(target_img, resize_height)

        # Calculate total dimensions
        total_width = (
            content_resized.width
            + style_resized.width
            + target_resized.width
            + 2 * spacing
        )
        total_height = resize_height

        # Create comparison image with white background
        comparison = Image.new(
            "RGB", (total_width, total_height), color=(255, 255, 255)
        )

        # Paste images with spacing
        x_offset = 0
        comparison.paste(content_resized, (x_offset, 0))
        x_offset += content_resized.width + spacing
        comparison.paste(style_resized, (x_offset, 0))
        x_offset += style_resized.width + spacing
        comparison.paste(target_resized, (x_offset, 0))

        return comparison

    except Exception as e:
        logger.debug(f"Failed to create comparison image: {e}")
        return None


def _find_style_image(style_images_dir: Path, style: str) -> Optional[Path]:
    """Find style image in the style images directory.

    Args:
        style_images_dir: Directory containing style images
        style: Style name

    Returns:
        Path to style image or None if not found
    """
    # Try common extensions in order of preference
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        style_path = style_images_dir / f"{style}{ext}"
        if style_path.exists():
            return style_path
    
    # Try case-insensitive search
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        for file_path in style_images_dir.glob(f"*{ext}"):
            if file_path.stem.lower() == style.lower():
                return file_path
    
    return None


def _load_image_cached(path: Path, cache: Optional[ImageCache] = None) -> Optional[Image.Image]:
    """Load image with optional caching.
    
    Args:
        path: Path to image file
        cache: Optional image cache
        
    Returns:
        PIL Image or None if loading fails
    """
    if cache:
        cached = cache.get(path)
        if cached:
            return cached.copy()  # Return copy to avoid mutation issues
    
    try:
        img = Image.open(path).convert("RGB")
        if cache:
            cache.set(path, img.copy())  # Store copy in cache
        return img
    except Exception as e:
        logger.debug(f"Failed to load image {path}: {e}")
        return None


def _process_single_sample(
    gen: dict,
    data_dir: Path,
    style_images_dir: Path,
    resize_height: int,
    spacing: int,
    image_cache: Optional[ImageCache] = None,
) -> Optional[dict[str, Any]]:
    """Process a single sample (load images, create comparison).

    This function is designed to be called in parallel via multiprocessing.

    Args:
        gen: Generation metadata dictionary
        data_dir: Data directory path
        style_images_dir: Style images directory path
        resize_height: Height for comparison images
        spacing: Spacing between images
        image_cache: Optional image cache for frequently accessed images

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
        # Load images (with caching)
        content_img = _load_image_cached(content_path, image_cache)
        style_img = _load_image_cached(style_path, image_cache)
        target_img = _load_image_cached(target_path, image_cache)
        
        if not all([content_img, style_img, target_img]):
            return None

        # Create comparison
        comparison_img: Optional[Image.Image] = _create_comparison_image(
            content_img, style_img, target_img, resize_height, spacing
        )

        if comparison_img is None:
            return None

        # Build sample dict
        sample_dict = {
            "character": char,
            "style": style,
            "font": font,
            "content_image": content_img,
            "style_image": style_img,
            "target_image": target_img,
            "comparison_image": comparison_img,
            "content_hash": compute_file_hash(char, "", font),
            "target_hash": compute_file_hash(char, style, font),
        }

        return sample_dict

    except Exception as e:
        logger.debug(f"Failed to process sample {char}/{style}: {e}")
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
        self.image_cache = ImageCache(max_size=50)  # Cache for frequently accessed images
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

    def _process_batch_with_futures(
        self,
        batch: List[dict],
        executor: ProcessPoolExecutor,
        image_cache: Optional[ImageCache] = None
    ) -> List[dict]:
        """Process a batch of samples using concurrent.futures.
        
        Args:
            batch: List of generation metadata dictionaries
            executor: ProcessPoolExecutor for parallel processing
            image_cache: Optional image cache
            
        Returns:
            List of processed samples (excluding failures)
        """
        # Create partial function with fixed parameters
        process_func = partial(
            _process_single_sample,
            data_dir=self.data_dir,
            style_images_dir=self.style_images_dir,
            resize_height=self.config.resize_height,
            spacing=self.config.spacing,
            image_cache=image_cache,
        )
        
        # Submit all tasks
        futures = {executor.submit(process_func, gen): i for i, gen in enumerate(batch)}
        
        # Collect results in order
        results = [None] * len(batch)
        for future in as_completed(futures):
            idx = futures[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.debug(f"Failed to process sample: {e}")
                results[idx] = None
        
        # Filter out None results
        return [r for r in results if r is not None]

    def _generate_samples_concurrent(self) -> Generator[dict[str, Any], None, None]:
        """Generate dataset samples using concurrent.futures with better control.

        Yields:
            Dictionary containing sample data with individual images and comparison
        """
        checkpoint: dict = self._load_checkpoint()
        generations: list = checkpoint["generations"]

        logger.info(f"Processing {len(generations)} samples with {self.config.num_workers} workers...")

        processed: int = 0
        skipped: int = 0
        
        # Use ProcessPoolExecutor for CPU-bound image processing
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Process in manageable batches
            for i in range(0, len(generations), self.config.batch_size):
                batch = generations[i:i + self.config.batch_size]
                
                # Process batch in parallel
                batch_results = self._process_batch_with_futures(
                    batch, executor, self.image_cache
                )
                
                # Yield results
                for sample in batch_results:
                    yield sample
                    processed += 1
                
                skipped += len(batch) - len(batch_results)
                
                # Progress logging
                if processed % 500 == 0 or i + self.config.batch_size >= len(generations):
                    progress_pct = min(100, (i + len(batch)) / len(generations) * 100)
                    logger.info(
                        f"Progress: {progress_pct:.1f}% ({processed + skipped}/{len(generations)}) | "
                        f"Valid: {processed} | Skipped: {skipped}"
                    )

        if processed == 0:
            raise ValueError("No valid samples found")

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid samples")

        logger.info(f"Successfully processed {processed} samples")

    def _generate_samples_single(self) -> Generator[dict[str, Any], None, None]:
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
                self.image_cache,
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

    def build_streaming(self, use_concurrent: bool = True) -> Dataset:
        """Build dataset using streaming to minimize memory usage.

        Args:
            use_concurrent: Whether to use concurrent.futures for parallel processing

        Returns:
            HuggingFace Dataset created from generator

        Raises:
            ValueError: If no valid samples are found
        """
        logger.info(f"Building dataset with streaming (concurrent={use_concurrent})...")

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

        generator_func = self._generate_samples_concurrent if use_concurrent else self._generate_samples_single

        dataset = Dataset.from_generator(
            generator_func,
            features=features,
        )

        return dataset

    def build_batched_concurrent(self) -> Dataset:
        """Build dataset in batches using concurrent processing.

        Returns:
            HuggingFace Dataset created from batched processing

        Raises:
            ValueError: If no valid samples are found
        """
        logger.info("Building dataset with concurrent batching...")

        checkpoint: dict = self._load_checkpoint()
        generations: list = checkpoint["generations"]

        # Use ThreadPoolExecutor for I/O operations and ProcessPoolExecutor for CPU work
        all_samples = []
        skipped = 0
        
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # Process all generations in parallel batches
            batch_futures = []
            for i in range(0, len(generations), self.config.batch_size):
                batch = generations[i:i + self.config.batch_size]
                future = executor.submit(
                    self._process_batch_with_futures,
                    batch,
                    executor,
                    self.image_cache
                )
                batch_futures.append(future)
            
            # Collect results
            with HFTqdm(total=len(generations), desc="Processing samples") as pbar:
                for future in as_completed(batch_futures):
                    try:
                        batch_results = future.result()
                        all_samples.extend(batch_results)
                        processed = len(batch_results)
                        skipped += self.config.batch_size - processed
                        pbar.update(self.config.batch_size)
                    except Exception as e:
                        logger.error(f"Batch processing failed: {e}")
                        skipped += self.config.batch_size
                        pbar.update(self.config.batch_size)

        if not all_samples:
            raise ValueError("No valid samples found")

        if skipped > 0:
            logger.warning(f"Skipped {skipped} invalid samples")

        logger.info(f"Successfully processed {len(all_samples)} samples")

        # Convert to dataset
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

        # Organize samples into columns
        columns = {
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
        
        for sample in all_samples:
            for key, value in sample.items():
                columns[key].append(value)

        dataset = Dataset.from_dict(columns, features=features)
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
            max_shard_size="500MB",
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

        dataset.save_to_disk(
            str(output_path),
            max_shard_size="500MB",
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
    use_concurrent: bool = True,
    max_workers_per_cpu: int = 2,
    io_max_workers: int = 32,
) -> Dataset:
    """Create and optionally push dataset to Hub with streaming support.

    Args:
        data_dir: Path to data directory containing ContentImage/ and TargetImage/
        style_images_dir: Path to directory containing style images
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
        num_workers: Number of worker processes (default: CPU count)
        use_concurrent: Whether to use concurrent processing (default: True)
        max_workers_per_cpu: Workers per CPU core for auto-tuning
        io_max_workers: Maximum I/O workers for image loading

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
        max_workers_per_cpu=max_workers_per_cpu,
        io_max_workers=io_max_workers,
    )

    builder = DatasetBuilder(config)

    if use_streaming:
        dataset = builder.build_streaming(use_concurrent=use_concurrent)
    else:
        dataset = builder.build_batched_concurrent()

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
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--no-concurrent",
        action="store_true",
        default=False,
        help="Disable concurrent processing (use single-threaded)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="streaming",
        help="Dataset config name (default: streaming)",
    )
    parser.add_argument(
        "--max-workers-per-cpu",
        type=int,
        default=2,
        help="Maximum workers per CPU core (default: 2)",
    )
    parser.add_argument(
        "--io-max-workers",
        type=int,
        default=32,
        help="Maximum I/O workers for image loading (default: 32)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        dataset = create_dataset(
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
            use_concurrent=not args.no_concurrent,
            max_workers_per_cpu=args.max_workers_per_cpu,
            io_max_workers=args.io_max_workers,
        )
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"✅ Dataset creation completed successfully!")
        print(f"📊 Dataset size: {len(dataset)} samples")
        print(f"⚡ Processing mode: {'concurrent' if not args.no_concurrent else 'single-threaded'}")
        print(f"📁 Local save: {args.local_save or 'No'}")
        print(f"☁️  Pushed to Hub: {not args.no_push}")
        if not args.no_push:
            print(f"🔗 Hub URL: https://huggingface.co/datasets/{args.repo_id}")
        print(f"{'='*60}")
        
        logger.info("Dataset creation completed successfully")

    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()