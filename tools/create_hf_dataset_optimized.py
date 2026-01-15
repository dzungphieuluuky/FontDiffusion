"""
High‑performance Hugging Face dataset builder for FontDiffusion images.
Optimized with parallel processing, caching, and efficient memory management.
"""

from __future__ import annotations

import hashlib
import json
import logging
import multiprocessing as mp
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import psutil
from datasets import Dataset, Features, Image as HFImage, Value
from huggingface_hub.utils import tqdm
from PIL import Image

# Suppress PIL warnings for better performance
Image.warnings.simplefilter("ignore", Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger("DatasetCreator")


# --------------------------------------------------------------------------- #
# Configuration dataclass
# --------------------------------------------------------------------------- #
@dataclass
class DatasetConfig:
    """
    Configuration for high‑performance dataset creation.
    """

    data_dir: Path
    repo_id: str
    split: str = "train"
    push_to_hub: bool = True
    private: bool = False
    token: str | None = None
    num_workers: int = field(default_factory=lambda: min(16, mp.cpu_count()))
    max_memory_usage_mb: int = 4096
    use_cache: bool = True
    cache_dir: Path | None = None
    compression_level: int = 3  # 0‑9, higher = more compression, slower

    def __post_init__(self) -> None:
        """Post‑initialization configuration."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)

        if self.cache_dir is None:
            self.cache_dir = self.data_dir / ".dataset_cache"

        # Limit workers based on available memory
        available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # GB
        if available_memory_gb < 4:
            self.num_workers = max(1, self.num_workers // 2)
            logger.warning(
                f"Limited workers to {self.num_workers} due to low memory ({available_memory_gb:.1f}GB)"
            )

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Processed sample container
# --------------------------------------------------------------------------- #
@dataclass
class ProcessedSample:
    """
    Container for processed sample data with memory optimization.
    """

    character: str
    style: str
    font: str
    content_path: str
    target_path: str
    content_hash: str
    target_hash: str
    content_size: tuple[int, int]  # For validation
    target_size: tuple[int, int]

    def to_dict(self) -> dict:
        """Convert to dictionary for dataset creation."""
        return {
            "character": self.character,
            "style": self.style,
            "font": self.font,
            "content_path": self.content_path,
            "target_path": self.target_path,
            "content_hash": self.content_hash,
            "target_hash": self.target_hash,
        }


# --------------------------------------------------------------------------- #
# Memory‑aware thread pool executor
# --------------------------------------------------------------------------- #
class MemoryAwareExecutor:
    """
    Memory‑aware thread pool executor with adaptive concurrency.
    """

    def __init__(self, max_workers: int, memory_limit_mb: int = 4096) -> None:
        self.max_workers = max_workers
        self.memory_limit_mb = memory_limit_mb
        self.executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> ThreadPoolExecutor:
        """Dynamically adjust workers based on current memory usage."""
        current_memory_mb = psutil.Process().memory_info().rss / (1024 ** 2)  # MB
        available_memory_mb = self.memory_limit_mb - current_memory_mb

        # Estimate memory per worker (approx 50MB for PIL + processing)
        memory_per_worker = 50
        safe_workers = max(1, int(available_memory_mb / memory_per_worker))
        workers = min(self.max_workers, safe_workers)

        logger.info(
            f"Memory‑aware executor: {workers} workers "
            f"(available: {available_memory_mb:.0f}MB, "
            f"current: {current_memory_mb:.0f}MB)"
        )

        self.executor = ThreadPoolExecutor(max_workers=workers)
        return self.executor

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.executor:
            self.executor.shutdown(wait=True)


# --------------------------------------------------------------------------- #
# Cache manager
# --------------------------------------------------------------------------- #
class CacheManager:
    """Efficient caching system for processed metadata."""

    def __init__(self, cache_dir: Path, compression_level: int = 3) -> None:
        self.cache_dir = cache_dir
        self.compression_level = compression_level
        self._cache: dict[Path, list[ProcessedSample]] = {}

    def _get_cache_key(self, checkpoint_path: Path) -> str:
        """Generate cache key from file stats."""
        stat = checkpoint_path.stat()
        key = f"{checkpoint_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(key.encode()).hexdigest()

    def get_cache_path(self, checkpoint_path: Path) -> Path:
        """Get path for cached data."""
        cache_key = self._get_cache_key(checkpoint_path)
        return self.cache_dir / f"{cache_key}.pkl"

    def load_from_cache(self, checkpoint_path: Path) -> list[ProcessedSample] | None:
        """Load processed samples from cache if available."""
        cache_path = self.get_cache_path(checkpoint_path)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)

            # Validate cache by checking if source files still exist
            if self._validate_cache(cached_data):
                logger.info(f"Loaded {len(cached_data)} samples from cache")
                return cached_data
            else:
                logger.info("Cache invalidated - source files changed")
                cache_path.unlink(missing_ok=True)
                return None

        except Exception as e:
            logger.warning(f"Cache loading failed: {e}")
            cache_path.unlink(missing_ok=True)
            return None

    def save_to_cache(self, checkpoint_path: Path, samples: list[ProcessedSample]) -> None:
        """Save processed samples to cache."""
        cache_path = self.get_cache_path(checkpoint_path)

        try:
            # Use highest protocol for speed
            with open(cache_path, "wb") as f:
                pickle.dump(samples, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Cached {len(samples)} samples to {cache_path}")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _validate_cache(self, cached_samples: list[ProcessedSample]) -> bool:
        """Validate cache by checking file existence and sizes."""
        for sample in cached_samples[:10]:  # Check first 10 samples as representative
            if not Path(sample.content_path).exists() or not Path(sample.target_path).exists():
                return False
        return True


# --------------------------------------------------------------------------- #
# Dataset builder
# --------------------------------------------------------------------------- #
class DatasetBuilder:
    """High‑performance FontDiffusion dataset builder."""

    REQUIRED_DIRS: list[str] = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE: str = "results_checkpoint.json"

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the dataset builder."""
        self.config = config
        self.data_dir = config.data_dir
        self.cache_manager = CacheManager(config.cache_dir, config.compression_level)
        self._validate_structure()

    # ----------------------------------------------------------------------- #
    # Directory & checkpoint validation
    # ----------------------------------------------------------------------- #
    def _validate_structure(self) -> None:
        """Fast directory structure validation."""
        missing_dirs: list[str] = []
        for dir_name in self.REQUIRED_DIRS:
            if not (self.data_dir / dir_name).exists():
                missing_dirs.append(dir_name)

        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            missing_dirs.append(self.CHECKPOINT_FILE)

        if missing_dirs:
            raise ValueError(f"Missing required directories/files: {missing_dirs}")

        logger.debug("Directory structure validated")

    # ----------------------------------------------------------------------- #
    # Checkpoint loading
    # ----------------------------------------------------------------------- #
    def _load_checkpoint(self) -> dict:
        """Efficient checkpoint loading with progress indicator."""
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE

        file_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
        if file_size_mb > 100:
            logger.info(f"Loading large checkpoint file ({file_size_mb:.1f} MB)...")

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        generations = data.get("generations", [])
        if not generations:
            raise ValueError("No generations found in checkpoint")

        # Pre‑compute stats
        num_characters = len({gen.get("character", "") for gen in generations})
        num_styles = len({gen.get("style", "") for gen in generations})

        logger.info(
            f"Checkpoint loaded: {len(generations):,} generations, "
            f"{num_characters:,} characters, "
            f"{num_styles:,} styles"
        )

        return data

    # ----------------------------------------------------------------------- #
    # Sample processing
    # ----------------------------------------------------------------------- #
    def _process_single_sample(self, gen: dict) -> ProcessedSample | None:
        """Process a single generation with error handling."""
        try:
            char = gen.get("character", "").strip()
            style = gen.get("style", "").strip()
            font = gen.get("font", "unknown").strip()

            # Construct absolute paths
            content_path = self.data_dir / gen.get("content_image_path", "")
            target_path = self.data_dir / gen.get("target_image_path", "")

            # Fast path validation
            if not (content_path.exists() and target_path.exists()):
                return None

            # Quick image validation without full loading
            try:
                with Image.open(content_path) as img:
                    content_size = img.size
                    if img.mode != "RGB":
                        img.convert("RGB")

                with Image.open(target_path) as img:
                    target_size = img.size
                    if img.mode != "RGB":
                        img.convert("RGB")
            except Exception as e:
                logger.debug(f"Image validation failed for {char}/{style}: {e}")
                return None

            # Compute hashes (can be optimized further with file hashing)
            content_hash = self._compute_fast_hash(char, "", font, content_path)
            target_hash = self._compute_fast_hash(char, style, font, target_path)

            return ProcessedSample(
                character=char,
                style=style,
                font=font,
                content_path=str(content_path),
                target_path=str(target_path),
                content_hash=content_hash,
                target_hash=target_hash,
                content_size=content_size,
                target_size=target_size,
            )

        except Exception as e:
            logger.debug(f"Skipping invalid sample: {e}")
            return None

    def _compute_fast_hash(self, char: str, style: str, font: str, path: Path) -> str:
        """Fast hash computation combining metadata and file stats."""
        try:
            stat = path.stat()
            hash_input = f"{char}:{style}:{font}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(hash_input.encode()).hexdigest()
        except Exception:
            # Fallback to simple string hash
            return hashlib.md5(f"{char}:{style}:{font}".encode()).hexdigest()

    # ----------------------------------------------------------------------- #
    # Parallel image loading
    # ----------------------------------------------------------------------- #
    def _load_images_parallel(self, samples: list[ProcessedSample]) -> list[dict]:
        """Parallel image loading with memory management."""
        processed_samples: list[dict] = []
        failed_samples = 0

        # Group samples to batch processing
        batch_size = max(1, len(samples) // (self.config.num_workers * 10))

        with MemoryAwareExecutor(self.config.num_workers, self.config.max_memory_usage_mb) as executor:
            futures = []

            # Submit batches for processing
            for i in range(0, len(samples), batch_size):
                batch = samples[i : i + batch_size]
                future = executor.submit(self._load_image_batch, batch)
                futures.append(future)

            # Collect results with progress bar
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Loading images",
                unit="batch",
            ):
                try:
                    batch_results, batch_failed = future.result()
                    processed_samples.extend(batch_results)
                    failed_samples += batch_failed
                except Exception as e:
                    logger.warning(f"Batch processing failed: {e}")
                    failed_samples += batch_size

        if failed_samples > 0:
            logger.warning(f"Failed to load {failed_samples} images")

        return processed_samples

    def _load_image_batch(self, batch: list[ProcessedSample]) -> tuple[list[dict], int]:
        """Load a batch of images with error handling."""
        batch_results: list[dict] = []
        failed = 0

        for sample in batch:
            try:
                # Load and convert images
                content_img = Image.open(sample.content_path).convert("RGB")
                target_img = Image.open(sample.target_path).convert("RGB")

                batch_results.append(
                    {
                        "character": sample.character,
                        "style": sample.style,
                        "font": sample.font,
                        "content_image": content_img,
                        "target_image": target_img,
                        "content_hash": sample.content_hash,
                        "target_hash": sample.target_hash,
                    }
                )
            except Exception as e:
                logger.debug(f"Failed to load images for {sample.character}/{sample.style}: {e}")
                failed += 1

        return batch_results, failed

    # ----------------------------------------------------------------------- #
    # Sample validation & deduplication
    # ----------------------------------------------------------------------- #
    def _validate_and_clean_samples(self, samples: list[ProcessedSample]) -> list[ProcessedSample]:
        """Validate samples and remove duplicates/invalid entries."""
        if not samples:
            return []

        # Remove duplicates based on hash
        seen_hashes: set[str] = set()
        unique_samples: list[ProcessedSample] = []

        for sample in samples:
            sample_hash = f"{sample.content_hash}:{sample.target_hash}"
            if sample_hash not in seen_hashes:
                seen_hashes.add(sample_hash)
                unique_samples.append(sample)

        if len(unique_samples) < len(samples):
            logger.info(f"Removed {len(samples) - len(unique_samples)} duplicate samples")

        # Validate file existence one more time
        valid_samples: list[ProcessedSample] = []
        for sample in unique_samples:
            if Path(sample.content_path).exists() and Path(sample.target_path).exists():
                valid_samples.append(sample)

        return valid_samples

    # ----------------------------------------------------------------------- #
    # Main build routine
    # ----------------------------------------------------------------------- #
    def build(self) -> Dataset:
        """Build the dataset with caching and parallel processing."""
        logger.info("Building dataset...")

        # Load checkpoint
        checkpoint = self._load_checkpoint()
        generations = checkpoint["generations"]

        # Try to load from cache first
        cached_samples: list[ProcessedSample] | None = None
        if self.config.use_cache:
            cached_samples = self.cache_manager.load_from_cache(
                self.data_dir / self.CHECKPOINT_FILE
            )

        if cached_samples is not None:
            # Use cached samples
            processed_samples = cached_samples
        else:
            # Process samples in parallel
            processed_samples: list[ProcessedSample] = []

            with MemoryAwareExecutor(self.config.num_workers) as executor:
                futures = []

                # Submit processing tasks
                for gen in generations:
                    future = executor.submit(self._process_single_sample, gen)
                    futures.append(future)

                # Collect results with progress bar
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing metadata",
                    unit="sample",
                ):
                    try:
                        sample = future.result()
                        if sample is not None:
                            processed_samples.append(sample)
                    except Exception as e:
                        logger.debug(f"Sample processing failed: {e}")

            # Validate and clean samples
            processed_samples = self._validate_and_clean_samples(processed_samples)

            # Cache processed samples
            if self.config.use_cache and processed_samples:
                self.cache_manager.save_to_cache(
                    self.data_dir / self.CHECKPOINT_FILE,
                    processed_samples,
                )

        if not processed_samples:
            raise ValueError("No valid samples found")

        logger.info(f"Processing {len(processed_samples)} unique samples")

        # Load images in parallel
        loaded_samples = self._load_images_parallel(processed_samples)

        if not loaded_samples:
            raise ValueError("Failed to load any images")

        # Create dataset with explicit features for better performance
        features = Features(
            {
                "character": Value("string"),
                "style": Value("string"),
                "font": Value("string"),
                "content_image": HFImage(),
                "target_image": HFImage(),
                "content_hash": Value("string"),
                "target_hash": Value("string"),
            }
        )

        # Convert to columnar format for faster dataset creation
        dataset_dict: dict = {
            "character": [s["character"] for s in loaded_samples],
            "style": [s["style"] for s in loaded_samples],
            "font": [s["font"] for s in loaded_samples],
            "content_image": [s["content_image"] for s in loaded_samples],
            "target_image": [s["target_image"] for s in loaded_samples],
            "content_hash": [s["content_hash"] for s in loaded_samples],
            "target_hash": [s["target_hash"] for s in loaded_samples],
        }

        dataset = Dataset.from_dict(dataset_dict, features=features)

        # Add dataset info
        dataset.info.description = "FontDiffusion generated dataset"
        dataset.info.splits = {self.config.split: len(dataset)}

        logger.info(f"Dataset built successfully: {len(dataset)} samples")

        return dataset

    # ----------------------------------------------------------------------- #
    # Push to Hub
    # ----------------------------------------------------------------------- #
    def push(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub with retry logic."""
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return

        logger.info(f"Pushing dataset to {self.config.repo_id}...")

        try:
            # Use efficient push with multiple retries
            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                private=self.config.private,
                token=self.config.token,
                max_shard_size="500MB",  # Optimal shard size for HF
                num_proc=self.config.num_workers,
            )

            logger.info(
                f"Successfully pushed to https://huggingface.co/datasets/{self.config.repo_id}"
            )

        except Exception as e:
            logger.error(f"Failed to push dataset: {e}")
            raise

    # ----------------------------------------------------------------------- #
    # Save locally
    # ----------------------------------------------------------------------- #
    def save_local(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk with compression."""
        logger.info(f"Saving dataset to {output_path}...")

        output_path.mkdir(parents=True, exist_ok=True)

        # Save with multiple workers for better performance
        dataset.save_to_disk(
            str(output_path),
            num_proc=self.config.num_workers,
            max_shard_size="500MB",
        )

        logger.info(f"Dataset saved to {output_path} ({len(dataset):,} samples)")


# --------------------------------------------------------------------------- #
# Public helper function
# --------------------------------------------------------------------------- #
def create_dataset(
    data_dir: str | Path,
    repo_id: str,
    split: str = "train",
    push_to_hub: bool = True,
    private: bool = False,
    token: str | None = None,
    local_save_path: str | Path | None = None,
    num_workers: int | None = None,
    max_memory_usage_mb: int = 4096,
    use_cache: bool = True,
) -> Dataset:
    """
    Create and optionally push dataset to Hub with performance optimizations.

    Args:
        data_dir: Path to data directory
        repo_id: HuggingFace repository ID
        split: Dataset split name
        push_to_hub: Whether to push to HuggingFace Hub
        private: Whether to make the repository private
        token: HuggingFace API token
        local_save_path: Local path to save dataset
        num_workers: Number of parallel workers (auto‑detected if None)
        max_memory_usage_mb: Maximum memory usage limit
        use_cache: Whether to use caching for faster rebuilds

    Returns:
        Created Dataset object
    """
    config = DatasetConfig(
        data_dir=Path(data_dir),
        repo_id=repo_id,
        split=split,
        push_to_hub=push_to_hub,
        private=private,
        token=token,
        num_workers=num_workers or min(16, mp.cpu_count()),
        max_memory_usage_mb=max_memory_usage_mb,
        use_cache=use_cache,
    )

    builder = DatasetBuilder(config)
    dataset = builder.build()

    if local_save_path:
        builder.save_local(dataset, Path(local_save_path))

    if push_to_hub:
        builder.push(dataset)

    return dataset


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """CLI entry point with performance options."""
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
        "--num-workers",
        type=int,
        help="Number of parallel workers (default: auto‑detect)",
    )
    parser.add_argument(
        "--max-memory",
        type=int,
        default=4096,
        help="Maximum memory usage in MB (default: 4096)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        create_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            split=args.split,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
            num_workers=args.num_workers,
            max_memory_usage_mb=args.max_memory,
            use_cache=not args.no_cache,
        )
        logger.info("Dataset creation completed successfully")

    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()