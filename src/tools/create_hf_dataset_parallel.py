"""
Create Hugging Face dataset from FontDiffusion images with ultra-parallel processing.

Auto-tuned for maximum speed - just provide essential arguments!
"""

import json
import logging
import time
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial, lru_cache

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image

from utilities import HFTqdm
from filename_utils import compute_file_hash

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset creation - simplified!"""
    data_dir: Path
    style_images_dir: Path
    repo_id: str
    split: str = "train"
    config_name: Optional[str] = None
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None

    def __post_init__(self):
        """Convert paths to Path if they're strings."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)


# Cache style images to avoid re-loading
@lru_cache(maxsize=256)
def _load_cached_style_image(style_path: str) -> Image.Image:
    """Load and cache style images to avoid repeated disk reads."""
    return Image.open(style_path).convert("RGB")


def _load_image_file(path: Path) -> Optional[Image.Image]:
    """Load a single image file with error handling."""
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return None


def _find_style_image(style_images_dir: Path, style: str) -> Optional[Path]:
    """Find style image in the style images directory."""
    for ext in [".png", ".jpg", ".jpeg"]:
        style_path = style_images_dir / f"{style}{ext}"
        if style_path.exists():
            return style_path
    return None


class UltraParallelDatasetBuilder:
    """Ultra-parallel dataset builder with auto-tuned performance."""
    
    REQUIRED_DIRS = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE = "results_checkpoint.json"
    
    def __init__(self, config: DatasetConfig):
        """Initialize with auto-tuned performance settings."""
        self.config = config
        self.data_dir = config.data_dir
        self.style_images_dir = config.style_images_dir
        
        # Auto-tune performance parameters
        self.cpu_count = os.cpu_count() or 4
        self.io_workers = max(1, self.cpu_count * 4)  # Many I/O threads
        self.cpu_workers = max(1, self.cpu_count)     # CPU count for processing
        self.batch_size = 1000  # Optimal batch size
        self.resize_height = 256
        self.spacing = 10
        
        logger.info(f"Auto-tuned: {self.io_workers} I/O workers, "
                   f"{self.cpu_workers} CPU workers, batch_size={self.batch_size}")
        
        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate that all required directories and files exist."""
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
        """Load and validate results checkpoint."""
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE

        with checkpoint_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        generations = data.get("generations", [])
        if not generations:
            raise ValueError("No generations found in checkpoint")

        logger.info(f"Loaded checkpoint: {len(generations)} generations, "
                   f"{len(data.get('characters', []))} characters, "
                   f"{len(data.get('styles', []))} styles")

        return data

    def _load_and_process_sample(self, gen: dict) -> Optional[dict]:
        """Load and process a single sample - optimized for parallel execution."""
        try:
            # Extract metadata
            char = gen.get("character", "")
            style = gen.get("style", "")
            font = gen.get("font", "unknown")
            
            # Build paths
            content_path = self.data_dir / gen.get("content_image_path", "")
            target_path = self.data_dir / gen.get("target_image_path", "")
            style_path = _find_style_image(self.style_images_dir, style)
            
            # Validate paths
            if not content_path.exists() or not target_path.exists() or not style_path:
                return None
            
            # Load images
            content_img = _load_image_file(content_path)
            target_img = _load_image_file(target_path)
            style_img = _load_cached_style_image(str(style_path))
            
            if not all([content_img, target_img, style_img]):
                return None
            
            # Create comparison image
            def resize_image(img, height):
                aspect = img.width / img.height
                new_width = int(height * aspect)
                return img.resize((new_width, height), Image.Resampling.LANCZOS)
            
            content_resized = resize_image(content_img, self.resize_height)
            style_resized = resize_image(style_img, self.resize_height)
            target_resized = resize_image(target_img, self.resize_height)
            
            total_width = content_resized.width + style_resized.width + target_resized.width + 2 * self.spacing
            comparison = Image.new("RGB", (total_width, self.resize_height), color=(255, 255, 255))
            
            x_offset = 0
            comparison.paste(content_resized, (x_offset, 0))
            x_offset += content_resized.width + self.spacing
            comparison.paste(style_resized, (x_offset, 0))
            x_offset += style_resized.width + self.spacing
            comparison.paste(target_resized, (x_offset, 0))
            
            # Build sample
            return {
                "character": char,
                "style": style,
                "font": font,
                "content_image": content_img,
                "style_image": style_img,
                "target_image": target_img,
                "comparison_image": comparison,
                "content_hash": compute_file_hash(char, "", font),
                "target_hash": compute_file_hash(char, style, font),
            }
            
        except Exception as e:
            logger.debug(f"Failed to process {gen.get('character', '')}/{gen.get('style', '')}: {e}")
            return None

    def _generate_samples_ultra_parallel(self) -> Generator[dict, None, None]:
        """Generate samples using ultra-parallel processing."""
        checkpoint = self._load_checkpoint()
        generations = checkpoint["generations"]
        
        logger.info(f"Processing {len(generations)} samples with ultra-parallel mode...")
        logger.info(f"Using {self.io_workers} workers for maximum throughput")
        
        start_time = time.time()
        total_batches = (len(generations) + self.batch_size - 1) // self.batch_size
        
        # Process batches in parallel
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.batch_size
            batch_end = min(batch_start + self.batch_size, len(generations))
            batch = generations[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch)} samples)...")
            batch_start_time = time.time()
            
            # Process entire batch in parallel
            with ProcessPoolExecutor(max_workers=self.io_workers) as executor:
                futures = {executor.submit(self._load_and_process_sample, gen): gen for gen in batch}
                
                valid_samples = 0
                for future in as_completed(futures):
                    sample = future.result()
                    if sample:
                        yield sample
                        valid_samples += 1
            
            batch_time = time.time() - batch_start_time
            elapsed = time.time() - start_time
            processed = batch_end
            rate = processed / elapsed if elapsed > 0 else 0
            progress_pct = processed / len(generations) * 100
            eta = (len(generations) - processed) / rate if rate > 0 else 0
            
            logger.info(f"Batch {batch_idx + 1}: {valid_samples}/{len(batch)} valid in {batch_time:.2f}s")
            logger.info(f"Overall: {progress_pct:.1f}%, {rate:.1f} samples/s, ETA: {eta:.0f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Processing completed in {total_time:.2f}s")

    def build(self) -> Dataset:
        """Build dataset with ultra-parallel processing."""
        logger.info("Building dataset with ultra-parallel processing...")
        
        features = Features({
            "character": Value("string"),
            "style": Value("string"),
            "font": Value("string"),
            "content_image": HFImage(),
            "style_image": HFImage(),
            "target_image": HFImage(),
            "comparison_image": HFImage(),
            "content_hash": Value("string"),
            "target_hash": Value("string"),
        })
        
        logger.info("Converting samples to Arrow format...")
        start_time = time.time()
        
        dataset = Dataset.from_generator(
            self._generate_samples_ultra_parallel,
            features=features,
        )
        
        conversion_time = time.time() - start_time
        logger.info(f"Dataset created: {len(dataset)} samples in {conversion_time:.2f}s")
        
        return dataset

    def push_to_hub(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub."""
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return
        
        logger.info(f"Pushing dataset to {self.config.repo_id}...")
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
            logger.info(f"Successfully pushed in {push_time:.2f}s")
            logger.info(f"Dataset available at: https://huggingface.co/datasets/{self.config.repo_id}")
            
        except Exception as e:
            logger.error(f"Push failed: {e}")
            raise

    def save_local(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk."""
        logger.info(f"Saving dataset to {output_path}...")
        save_start = time.time()
        
        dataset.save_to_disk(str(output_path))
        
        save_time = time.time() - save_start
        logger.info(f"Dataset saved in {save_time:.2f}s")


def create_dataset_ultra_fast(
    data_dir: str | Path,
    style_images_dir: str | Path,
    repo_id: str,
    split: str = "train",
    config_name: Optional[str] = None,
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str | Path] = None,
) -> Dataset:
    """Create dataset with ultra-fast parallel processing.
    
    Just provide essential arguments - performance is auto-tuned!
    
    Args:
        data_dir: Path to data directory (must contain ContentImage/ and TargetImage/)
        style_images_dir: Path to style images directory
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        split: Dataset split name
        config_name: Dataset configuration name
        push_to_hub: Whether to push to HuggingFace Hub
        private: Whether to make repository private
        token: HuggingFace API token
        local_save_path: Local path to save dataset
    
    Returns:
        Created Dataset object
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
    )
    
    builder = UltraParallelDatasetBuilder(config)
    dataset = builder.build()
    
    if local_save_path:
        builder.save_local(dataset, Path(local_save_path))
    
    if push_to_hub:
        builder.push_to_hub(dataset)
    
    return dataset


def main():
    """Simple CLI - just provide essential arguments!"""
    parser = argparse.ArgumentParser(
        description="Ultra-fast HuggingFace dataset creator from FontDiffusion images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ultra-fast dataset creation (auto-tuned for max speed)
  python create_ultra_fast.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name
  
  # Create but don't push to Hub
  python create_ultra_fast.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name --no-push
  
  # Create private dataset
  python create_ultra_fast.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name --private
  
  # Save locally as well
  python create_ultra_fast.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name \\
    --local-save ./local_dataset
  
  # Use specific split and config
  python create_ultra_fast.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name \\
    --split test --config-name v2

Performance features:
  • Auto-tuned parallelization based on CPU cores
  • Intelligent batching for optimal throughput
  • Combined I/O and CPU processing in single pass
  • No manual tuning needed - just works fast!
        """,
    )
    
    # Essential arguments
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to data directory (must contain ContentImage/ and TargetImage/)"
    )
    parser.add_argument(
        "--style-images-dir",
        required=True,
        help="Path to directory containing style images"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repository ID (username/dataset-name)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split name (default: train)"
    )
    parser.add_argument(
        "--config-name",
        help="Dataset configuration name"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to HuggingFace Hub"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make repository private"
    )
    parser.add_argument(
        "--local-save",
        help="Save dataset locally to this path"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token for private datasets"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Run ultra-fast dataset creation
        dataset = create_dataset_ultra_fast(
            data_dir=args.data_dir,
            style_images_dir=args.style_images_dir,
            repo_id=args.repo_id,
            split=args.split,
            config_name=args.config_name,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
        )
        
        # Success summary
        print(f"\n✅ Dataset creation completed successfully!")
        print(f"📊 Samples: {len(dataset)}")
        print(f"🔤 Characters: {len(set(dataset['character']))}")
        print(f"🎨 Styles: {len(set(dataset['style']))}")
        
        if not args.no_push:
            print(f"🌐 Uploaded to: https://huggingface.co/datasets/{args.repo_id}")
        
        if args.local_save:
            print(f"💾 Local copy: {args.local_save}")
        
    except KeyboardInterrupt:
        logger.warning("Dataset creation interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()