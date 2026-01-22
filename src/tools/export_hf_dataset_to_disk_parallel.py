"""
Export Hugging Face dataset back to FontDiffusion directory structure.

This module reconstructs the original directory layout from a HuggingFace dataset,
preserving results_checkpoint.json as the single source of truth.
Optimized for high performance with parallel processing and efficient I/O.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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

logger = logging.getLogger("DatasetExporter")


@dataclass
class ExportConfig:
    """Configuration for dataset export."""

    output_dir: Path
    repo_id: Optional[str] = None
    local_dataset_path: Optional[Path] = None
    split: str = "train"
    token: Optional[str] = None
    num_workers: int = 4  # Parallel workers for image saving
    batch_size: int = 1000  # Process images in batches

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
            try:
                dataset = Dataset.load_from_disk(str(self.config.local_dataset_path))
                logger.info(f"Loaded {len(dataset)} samples from disk")
                return dataset
            except Exception as e:
                raise ValueError(f"Failed to load local dataset: {e}") from e

        logger.info(
            f"Loading dataset from Hub: {self.config.repo_id} (split: {self.config.split})"
        )
        try:
            dataset = load_dataset(
                self.config.repo_id,
                split=self.config.split,
                token=self.config.token,
            )
            logger.info(f"Loaded {len(dataset)} samples from Hub")
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

    def _save_image_task(self, image: Image.Image, path: Path) -> bool:
        """Save a single image (thread-safe task).

        Args:
            image: PIL Image to save
            path: Destination path

        Returns:
            True if successful, False otherwise
        """
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path)
            return True
        except Exception as e:
            logger.warning(f"Failed to save image to {path}: {e}")
            return False

    def _prepare_export_batch(self, samples: list[dict]) -> tuple[list, list, set]:
        """Prepare a batch of samples for export.

        Args:
            samples: List of dataset samples

        Returns:
            Tuple of (save_tasks, generations, content_filenames)
        """
        save_tasks = []
        generations = []
        content_filenames = set()

        for sample in samples:
            char = sample["character"]
            style = sample["style"]
            font = sample.get("font", "unknown")

            # Prepare content image save task
            content_filename = get_content_filename(char)
            content_img = sample.get("content_image")

            if isinstance(content_img, Image.Image):
                content_path = self.content_dir / content_filename
                save_tasks.append(
                    ("content", content_img, content_path, content_filename)
                )
                content_filenames.add(content_filename)

            # Prepare target image save task
            target_img = sample.get("target_image")
            if isinstance(target_img, Image.Image):
                style_dir = self.target_dir / style
                target_filename = get_target_filename(char, style)
                target_path = style_dir / target_filename
                save_tasks.append(("target", target_img, target_path, None))

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

        return save_tasks, generations, content_filenames

    def _export_images_parallel(self, dataset: Dataset) -> dict[str, Any]:
        """Export images using parallel processing for maximum speed.

        Args:
            dataset: Dataset to export

        Returns:
            Complete metadata dictionary for results_checkpoint.json
        """
        logger.info(f"Exporting images with {self.config.num_workers} workers...")

        all_generations = []
        exported_content = set()
        total_content_saves = 0
        total_target_saves = 0

        # Process dataset in batches
        dataset_size = len(dataset)
        batch_size = self.config.batch_size

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            for batch_start in range(0, dataset_size, batch_size):
                batch_end = min(batch_start + batch_size, dataset_size)
                batch_samples = [dataset[i] for i in range(batch_start, batch_end)]

                # Prepare batch
                save_tasks, generations, content_filenames = self._prepare_export_batch(
                    batch_samples
                )

                # Filter out already-saved content images
                filtered_tasks = []
                for task_type, img, path, filename in save_tasks:
                    if task_type == "content":
                        if filename not in exported_content:
                            filtered_tasks.append((img, path))
                            exported_content.add(filename)
                    else:  # target
                        filtered_tasks.append((img, path))

                # Submit save tasks in parallel
                futures = [
                    executor.submit(self._save_image_task, img, path)
                    for img, path in filtered_tasks
                ]

                # Wait for completion
                success_count = sum(
                    1 for future in as_completed(futures) if future.result()
                )

                # Count saves
                content_in_batch = len([t for t in save_tasks if t[0] == "content"])
                target_in_batch = len([t for t in save_tasks if t[0] == "target"])

                total_content_saves += min(
                    success_count,
                    len([t for t in filtered_tasks if t[1].parent == self.content_dir]),
                )
                total_target_saves += min(
                    success_count,
                    len([t for t in filtered_tasks if t[1].parent != self.content_dir]),
                )

                all_generations.extend(generations)

                # Log progress
                if batch_end % 1000 == 0 or batch_end == dataset_size:
                    logger.info(
                        f"Progress: {batch_end}/{dataset_size} samples "
                        f"({batch_end * 100 // dataset_size}%)"
                    )

        logger.info(
            f"Exported {len(exported_content)} content images, "
            f"{len(all_generations)} target images"
        )

        # Build metadata efficiently using set comprehensions
        characters = sorted({g["character"] for g in all_generations})
        styles = sorted({g["style"] for g in all_generations})
        fonts = sorted({g["font"] for g in all_generations if g["font"] != "unknown"})

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

        for sample in HFTqdm(dataset, desc="Exporting images", unit="sample"):
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

        logger.info(
            f"Exported {len(exported_content)} content images, "
            f"{len(generations)} target images"
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

        # Write with minimal whitespace for faster I/O
        with checkpoint_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(
            f"Saved checkpoint with {len(metadata['generations'])} generations: "
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

        dataset = self._load_dataset()
        self._create_directories()

        # Use parallel export if workers > 1, otherwise sequential
        if self.config.num_workers > 1:
            self.config.num_workers = min(self.config.num_workers, os.cpu_count())
            print(f"Using {self.config.num_workers} parallel workers for export")
            metadata = self._export_images_parallel(dataset)
        else:
            metadata = self._export_images_sequential(dataset)

        self._save_checkpoint(metadata)

        logger.info("Export completed successfully")
        return metadata


def export_dataset(
    output_dir: str | Path,
    repo_id: Optional[str] = None,
    local_dataset_path: Optional[str | Path] = None,
    split: str = "train",
    token: Optional[str] = None,
    num_workers: int = 4,
    batch_size: int = 1000,
) -> dict[str, Any]:
    """Export HuggingFace dataset to disk with high performance.

    Args:
        output_dir: Directory to export to
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        local_dataset_path: Local dataset path (alternative to repo_id)
        split: Dataset split name (default: 'train')
        token: HuggingFace API token for private datasets
        num_workers: Number of parallel workers for image saving (default: 8)
        batch_size: Number of samples to process per batch (default: 1000)

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
        token=token,
        num_workers=num_workers,
        batch_size=batch_size,
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
  # Export from Hub with parallel processing
  python export_hf_dataset.py --output-dir ./output --repo-id user/dataset --workers 8
  
  # Export from local cache with custom batch size
  python export_hf_dataset.py --output-dir ./output --local-path ~/.cache/... --batch-size 500
  
  # Single-threaded export
  python export_hf_dataset.py --output-dir ./output --repo-id user/dataset --workers 1
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
        "--token",
        type=str,
        help="HuggingFace API token for private datasets",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of parallel workers for image saving (default: 12)",
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
            token=args.token,
            num_workers=args.workers,
            batch_size=args.batch_size,
        )

        logger.info(
            f"Successfully exported to {args.output_dir}\n"
            f"  ContentImage/\n"
            f"  TargetImage/\n"
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
