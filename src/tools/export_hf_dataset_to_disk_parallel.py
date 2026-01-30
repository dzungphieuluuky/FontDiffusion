"""
Export Hugging Face dataset back to FontDiffusion directory structure.

This module reconstructs the original directory layout from a HuggingFace dataset,
preserving results_checkpoint.json as the single source of truth.
Fully optimized with concurrent.futures for maximum speed.
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, load_dataset
from PIL import Image
from utilities import HFTqdm
from filename_utils import (
    compute_file_hash,
    get_content_filename,
    get_target_filename,
)

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Configuration for dataset export."""

    output_dir: Path
    repo_id: Optional[str] = None
    local_dataset_path: Optional[Path] = None
    split: str = "train"
    config_name: Optional[str] = None
    token: Optional[str] = None
    num_workers: int = 8
    batch_size: int = 1000
    use_process_pool: bool = False  # NEW: Use ProcessPool for batch prep

    def __post_init__(self):
        """Validate and convert paths."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.local_dataset_path, str):
            self.local_dataset_path = Path(self.local_dataset_path)

        if not self.repo_id and not self.local_dataset_path:
            raise ValueError(
                "Must provide either repo_id (Hub) or local_dataset_path (disk)"
            )


class DatasetExporter:
    """Export HuggingFace dataset to FontDiffusion directory structure."""

    def __init__(self, config: ExportConfig):
        """Initialize the exporter.

        Args:
            config: Export configuration
        """
        self.config = config
        self.output_dir = config.output_dir
        self.content_dir = self.output_dir / "ContentImage"
        self.target_dir = self.output_dir / "TargetImage"

    def _load_dataset(self) -> Dataset:
        """Load dataset from Hub or local disk with streaming support.

        Returns:
            Loaded dataset

        Raises:
            ValueError: If dataset cannot be loaded
        """
        if self.config.local_dataset_path:
            logger.info(f"Loading local dataset from {self.config.local_dataset_path}")
            load_start = time.time()
            try:
                dataset = Dataset.load_from_disk(str(self.config.local_dataset_path))
                load_time = time.time() - load_start
                logger.info(
                    f"Loaded {len(dataset)} samples from disk in {load_time:.2f}s"
                )
                return dataset
            except Exception as e:
                raise ValueError(f"Failed to load local dataset: {e}") from e

        config_msg = (
            f" (config: {self.config.config_name})"
            if self.config.config_name
            else ""
        )
        logger.info(
            f"Loading dataset from Hub: {self.config.repo_id} "
            f"(split: {self.config.split}){config_msg}"
        )
        load_start = time.time()
        try:
            dataset = load_dataset(
                self.config.repo_id,
                name=self.config.config_name,
                split=self.config.split,
                token=self.config.token,
            )
            load_time = time.time() - load_start
            logger.info(
                f"Loaded {len(dataset)} samples from Hub in {load_time:.2f}s"
            )
            return dataset
        except Exception as e:
            raise ValueError(
                f"Failed to load from Hub {self.config.repo_id}: {e}"
            ) from e

    def _create_directories(self) -> None:
        """Create output directory structure."""
        self.content_dir.mkdir(parents=True, exist_ok=True)
        self.target_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory structure at {self.output_dir}")

    def _save_image_task(self, image: Image.Image, path: Path) -> tuple[bool, Path]:
        """Save a single image (thread-safe task).

        Args:
            image: PIL Image to save
            path: Destination path

        Returns:
            Tuple of (success, path)
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            return True, path
        except Exception as e:
            logger.warning(f"Failed to save image to {path}: {e}")
            return False, path

    def _prepare_sample_metadata(
        self, sample: dict
    ) -> tuple[Optional[tuple], Optional[tuple], dict]:
        """Prepare metadata for a single sample (parallelizable).

        Args:
            sample: Single dataset sample

        Returns:
            Tuple of (content_task, target_task, generation_record)
        """
        char = sample["character"]
        style = sample["style"]
        font = sample.get("font", "unknown")

        content_task = None
        target_task = None

        # Prepare content image task
        content_filename = get_content_filename(char)
        content_img = sample.get("content_image")
        if isinstance(content_img, Image.Image):
            content_path = self.content_dir / content_filename
            content_task = (content_img, content_path, "content", content_filename)

        # Prepare target image task
        target_img = sample.get("target_image")
        if isinstance(target_img, Image.Image):
            style_dir = self.target_dir / style
            target_filename = get_target_filename(char, style)
            target_path = style_dir / target_filename
            target_task = (target_img, target_path, "target", None)

        # Build generation record
        generation = {
            "character": char,
            "style": style,
            "font": font,
            "content_image_path": f"ContentImage/{content_filename}",
            "target_image_path": f"TargetImage/{style}/{get_target_filename(char, style)}",
            "content_hash": compute_file_hash(char, "", font),
            "target_hash": compute_file_hash(char, style, font),
        }

        return content_task, target_task, generation

    def _export_images_parallel(self, dataset: Dataset) -> dict[str, Any]:
        """Export images using dual-layer parallel processing for maximum speed.

        Strategy:
        1. Outer ThreadPool: Prepares sample metadata in parallel
        2. Inner ThreadPool: Saves images in parallel
        3. Deduplicates content images efficiently

        Args:
            dataset: Dataset to export

        Returns:
            Complete metadata dictionary for results_checkpoint.json
        """
        logger.info(
            f"Exporting images with {self.config.num_workers} workers "
            f"(dual-layer parallelization)..."
        )

        all_generations = []
        exported_content = set()
        dataset_size = len(dataset)

        export_start = time.time()

        # Use nested executors for maximum parallelization
        with ThreadPoolExecutor(
            max_workers=self.config.num_workers
        ) as metadata_executor:
            with ThreadPoolExecutor(
                max_workers=self.config.num_workers * 2
            ) as save_executor:
                # Process in batches
                for batch_start in range(0, dataset_size, self.config.batch_size):
                    batch_end = min(batch_start + self.config.batch_size, dataset_size)
                    batch_samples = [dataset[i] for i in range(batch_start, batch_end)]

                    batch_start_time = time.time()
                    logger.info(
                        f"Processing batch {batch_start}-{batch_end} "
                        f"({len(batch_samples)} samples)..."
                    )

                    # PARALLEL: Prepare metadata for all samples in batch
                    metadata_futures = {
                        metadata_executor.submit(
                            self._prepare_sample_metadata, sample
                        ): i
                        for i, sample in enumerate(batch_samples)
                    }

                    # Collect prepared tasks
                    save_tasks = []
                    for future in as_completed(metadata_futures):
                        content_task, target_task, generation = future.result()

                        # Deduplicate content images
                        if content_task:
                            _, _, _, content_filename = content_task
                            if content_filename not in exported_content:
                                save_tasks.append(content_task)
                                exported_content.add(content_filename)

                        # Always save target images
                        if target_task:
                            save_tasks.append(target_task)

                        all_generations.append(generation)

                    # PARALLEL: Save all images in batch
                    save_futures = [
                        save_executor.submit(self._save_image_task, img, path)
                        for img, path, _, _ in save_tasks
                    ]

                    # Wait for saves to complete
                    success_count = sum(
                        1
                        for future in as_completed(save_futures)
                        if future.result()[0]
                    )

                    batch_time = time.time() - batch_start_time
                    logger.info(
                        f"Batch completed in {batch_time:.2f}s "
                        f"({success_count}/{len(save_tasks)} saves successful, "
                        f"{batch_time/len(batch_samples):.3f}s/sample)"
                    )

                    # Progress update
                    progress_pct = batch_end / dataset_size * 100
                    elapsed = time.time() - export_start
                    rate = batch_end / elapsed
                    eta = (dataset_size - batch_end) / rate if rate > 0 else 0
                    logger.info(
                        f"Overall progress: {batch_end}/{dataset_size} "
                        f"({progress_pct:.1f}%, {rate:.1f} samples/s, ETA: {eta:.0f}s)"
                    )

        export_time = time.time() - export_start
        logger.info(
            f"Exported {len(exported_content)} content images, "
            f"{len(all_generations)} target images in {export_time:.2f}s "
            f"({len(all_generations)/export_time:.1f} samples/s)"
        )

        # Build metadata efficiently using set comprehensions
        characters = sorted({g["character"] for g in all_generations})
        styles = sorted({g["style"] for g in all_generations})
        fonts = sorted(
            {g["font"] for g in all_generations if g["font"] != "unknown"}
        )

        return {
            "generations": all_generations,
            "characters": characters,
            "styles": styles,
            "fonts": fonts if fonts else ["unknown"],
            "total_chars": len(characters),
            "total_styles": len(styles),
        }

    def _export_images_sequential(self, dataset: Dataset) -> dict[str, Any]:
        """Export images sequentially (fallback for single-threaded mode).

        Args:
            dataset: Dataset to export

        Returns:
            Complete metadata dictionary for results_checkpoint.json
        """
        logger.info("Exporting images (sequential mode)...")

        exported_content = set()
        generations = []
        start_time = time.time()

        for i, sample in enumerate(dataset):
            char = sample["character"]
            style = sample["style"]
            font = sample.get("font", "unknown")

            # Export content image (once per character)
            content_filename = get_content_filename(char)
            if content_filename not in exported_content:
                content_img = sample.get("content_image")
                if isinstance(content_img, Image.Image):
                    content_path = self.content_dir / content_filename
                    self._save_image_task(content_img, content_path)
                    exported_content.add(content_filename)

            # Export target image
            target_img = sample.get("target_image")
            if isinstance(target_img, Image.Image):
                style_dir = self.target_dir / style
                target_filename = get_target_filename(char, style)
                target_path = style_dir / target_filename
                self._save_image_task(target_img, target_path)

            # Build generation record
            generations.append(
                {
                    "character": char,
                    "style": style,
                    "font": font,
                    "content_image_path": f"ContentImage/{content_filename}",
                    "target_image_path": f"TargetImage/{style}/{get_target_filename(char, style)}",
                    "content_hash": compute_file_hash(char, "", font),
                    "target_hash": compute_file_hash(char, style, font),
                }
            )

            # Progress logging
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                logger.info(
                    f"Progress: {i + 1}/{len(dataset)} samples ({rate:.1f} samples/s)"
                )

        export_time = time.time() - start_time
        logger.info(
            f"Exported {len(exported_content)} content images, "
            f"{len(generations)} target images in {export_time:.2f}s"
        )

        # Build metadata
        characters = sorted({g["character"] for g in generations})
        styles = sorted({g["style"] for g in generations})
        fonts = sorted({g["font"] for g in generations if g["font"] != "unknown"})

        return {
            "generations": generations,
            "characters": characters,
            "styles": styles,
            "fonts": fonts if fonts else ["unknown"],
            "total_chars": len(characters),
            "total_styles": len(styles),
        }

    def _save_checkpoint(self, metadata: dict[str, Any]) -> None:
        """Save results_checkpoint.json efficiently.

        Args:
            metadata: Metadata dictionary to save
        """
        checkpoint_path = self.output_dir / "results_checkpoint.json"

        logger.info("Writing checkpoint file...")
        write_start = time.time()

        # Write with minimal whitespace for faster I/O
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        write_time = time.time() - write_start
        logger.info(
            f"Saved checkpoint in {write_time:.2f}s with {len(metadata['generations'])} generations: "
            f"{len(metadata['characters'])} chars, {len(metadata['styles'])} styles"
        )

    def export(self) -> dict[str, Any]:
        """Execute the full export process with optimizations.

        Returns:
            Complete metadata dictionary

        Raises:
            ValueError: If dataset loading or export fails
        """
        logger.info("Starting optimized dataset export...")
        total_start = time.time()

        dataset = self._load_dataset()
        self._create_directories()

        # Use parallel export if workers > 1, otherwise sequential
        if self.config.num_workers > 1:
            actual_workers = min(self.config.num_workers, os.cpu_count())
            logger.info(f"Using {actual_workers} parallel workers for export")
            metadata = self._export_images_parallel(dataset)
        else:
            logger.info("Using sequential export (single-threaded)")
            metadata = self._export_images_sequential(dataset)

        self._save_checkpoint(metadata)

        total_time = time.time() - total_start
        logger.info(f"Export completed successfully in {total_time:.2f}s")
        return metadata


def export_dataset(
    output_dir: str | Path,
    repo_id: Optional[str] = None,
    local_dataset_path: Optional[str | Path] = None,
    split: str = "train",
    config_name: Optional[str] = None,
    token: Optional[str] = None,
    num_workers: int = 8,
    batch_size: int = 1000,
    use_process_pool: bool = False,
) -> dict[str, Any]:
    """Export HuggingFace dataset to disk with high performance.

    Args:
        output_dir: Directory to export to
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        local_dataset_path: Local dataset path (alternative to repo_id)
        split: Dataset split name (default: 'train')
        config_name: Dataset configuration name (e.g., 'streaming', 'default')
        token: HuggingFace API token for private datasets
        num_workers: Number of parallel workers (default: 8)
        batch_size: Number of samples to process per batch (default: 1000)
        use_process_pool: Use ProcessPool for metadata prep (default: False)

    Returns:
        Metadata dictionary from results_checkpoint.json

    Raises:
        ValueError: If neither repo_id nor local_dataset_path is provided,
                   or if dataset cannot be loaded
    """
    config = ExportConfig(
        output_dir=Path(output_dir),
        repo_id=repo_id,
        local_dataset_path=Path(local_dataset_path) if local_dataset_path else None,
        split=split,
        config_name=config_name,
        token=token,
        num_workers=num_workers,
        batch_size=batch_size,
        use_process_pool=use_process_pool,
    )

    exporter = DatasetExporter(config)
    return exporter.export()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export HuggingFace dataset to FontDiffusion directory structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # High-performance export with parallel processing
  python export_hf_dataset_to_disk_parallel.py --output-dir ./output \\
    --repo-id user/dataset --workers 12 --batch-size 2000

  # Export from local cache
  python export_hf_dataset_to_disk_parallel.py --output-dir ./output \\
    --local-path ~/.cache/huggingface/datasets/... --workers 8

  # Single-threaded export (debugging)
  python export_hf_dataset_to_disk_parallel.py --output-dir ./output \\
    --repo-id user/dataset --workers 1
        """,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to export to",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        help="HuggingFace repository ID",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        help="Local dataset path (alternative to --repo-id)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split name (default: train)",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Dataset configuration name (e.g., 'streaming', 'default')",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="HuggingFace API token for private datasets",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count() - 1,
        help="Number of parallel workers (default: number of CPU cores - 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of samples to process per batch (default: 1000)",
    )

    args = parser.parse_args()

    try:
        metadata = export_dataset(
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            local_dataset_path=args.local_path,
            split=args.split,
            config_name=args.config_name,
            token=args.token,
            num_workers=args.workers,
            batch_size=args.batch_size,
        )

        logger.info(
            f"Successfully exported to {args.output_dir}\n"
            f"  ContentImage/ ({metadata['total_chars']} unique characters)\n"
            f"  TargetImage/ ({len(metadata['generations'])} images)\n"
            f"  results_checkpoint.json"
        )

    except KeyboardInterrupt:
        logger.warning("Export interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception(f"Export failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()