"""
Create Hugging Face dataset from FontDiffusion images with three-stage pipeline.

Stage 1: Bulk I/O (ThreadPoolExecutor, high parallelism)
Stage 2: CPU processing (ProcessPoolExecutor/ThreadPool, moderate parallelism)
Stage 3: Streaming upload (as samples are ready)
"""

import json
import logging
import time
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Optional, NamedTuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image

from utilities import HFTqdm
from filename_utils import compute_file_hash

logger = logging.getLogger(__name__)


class LoadedImages(NamedTuple):
    """Container for loaded images (I/O stage output)."""
    content_img: Image.Image
    style_img: Image.Image
    target_img: Image.Image
    char: str
    style: str
    font: str


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

    def __post_init__(self):
        """Convert paths to Path."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)


class ThreeStageBuilder:
    """Three-stage pipeline: I/O → CPU → Yield."""
    
    REQUIRED_DIRS = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE = "results_checkpoint.json"
    
    def __init__(self, config: DatasetConfig):
        """Initialize with optimized three-stage pipeline."""
        self.config = config
        self.data_dir = config.data_dir
        self.style_images_dir = config.style_images_dir
        
        # Auto-tune workers
        self.cpu_count = os.cpu_count() or 4
        self.io_workers = self.cpu_count * 8   # Saturate I/O
        self.cpu_workers = self.cpu_count      # CPU-bound work
        
        # Pipeline batch sizes
        self.io_batch_size = 500    # I/O stage (large batches)
        self.cpu_batch_size = 100   # CPU stage (smaller for pipelining)
        
        # Image params
        self.resize_height = 256
        self.spacing = 10
        
        logger.info(f"Three-stage pipeline: I/O workers={self.io_workers}, CPU workers={self.cpu_workers}")
        
        # Pre-build caches (runs once at startup)
        self._validate_structure()
        self._preload_styles()
        self._build_path_cache()

    def _validate_structure(self) -> None:
        """Validate directory structure."""
        for dir_name in self.REQUIRED_DIRS:
            if not (self.data_dir / dir_name).exists():
                raise ValueError(f"Required directory not found: {self.data_dir / dir_name}")
        if not (self.data_dir / self.CHECKPOINT_FILE).exists():
            raise ValueError(f"Checkpoint not found: {self.data_dir / self.CHECKPOINT_FILE}")
        if not self.style_images_dir.exists():
            raise ValueError(f"Style images directory not found: {self.style_images_dir}")
        logger.info("Directory structure validated")

    def _preload_styles(self) -> None:
        """Preload ALL style images into memory (runs once)."""
        logger.info("Preloading style images...")
        start = time.time()
        
        self.style_cache: dict[str, Image.Image] = {}
        for ext in [".png", ".jpg", ".jpeg"]:
            for style_file in self.style_images_dir.glob(f"*{ext}"):
                try:
                    self.style_cache[style_file.stem] = Image.open(style_file).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load style {style_file}: {e}")
        
        elapsed = time.time() - start
        logger.info(f"Preloaded {len(self.style_cache)} style images in {elapsed:.2f}s")

    def _build_path_cache(self) -> None:
        """Build path lookup cache (runs once)."""
        logger.info("Building path cache...")
        start = time.time()
        
        # Content paths: char → path
        self.content_paths: dict[str, Path] = {}
        content_dir = self.data_dir / "ContentImage"
        for img_file in content_dir.glob("*"):
            if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                self.content_paths[img_file.stem] = img_file
        
        # Target paths: style → {char → path}
        self.target_paths: dict[str, dict[str, Path]] = {}
        target_dir = self.data_dir / "TargetImage"
        for style_dir in target_dir.iterdir():
            if not style_dir.is_dir():
                continue
            
            style = style_dir.name
            style_paths = {}
            for img_file in style_dir.glob("*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Extract char from filename (assumes format: char_style.ext or char.ext)
                    char = img_file.stem.split('_')[0]
                    style_paths[char] = img_file
            self.target_paths[style] = style_paths
        
        elapsed = time.time() - start
        total_targets = sum(len(v) for v in self.target_paths.values())
        logger.info(f"Built path cache: {len(self.content_paths)} content, {total_targets} target in {elapsed:.2f}s")

    def _load_checkpoint(self) -> list[dict]:
        """Load checkpoint generations."""
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        generations = data.get("generations", [])
        if not generations:
            raise ValueError("No generations found in checkpoint")
        
        logger.info(f"Loaded {len(generations)} generations from checkpoint")
        return generations

    # ===== STAGE 1: I/O (Bulk Image Loading) =====
    def _load_images_io_stage(self, gen: dict) -> Optional[LoadedImages]:
        """Stage 1: Load images only (pure I/O, no CPU work)."""
        char = gen.get("character", "")
        style = gen.get("style", "")
        font = gen.get("font", "unknown")
        
        # Fast path lookup (no disk I/O)
        content_path = self.content_paths.get(char)
        target_path = self.target_paths.get(style, {}).get(char)
        style_img = self.style_cache.get(style)
        
        if not all([content_path, target_path, style_img]):
            return None
        
        try:
            # Only I/O here - no resizing
            content_img = Image.open(content_path).convert("RGB")
            target_img = Image.open(target_path).convert("RGB")
            
            return LoadedImages(
                content_img=content_img,
                style_img=style_img,
                target_img=target_img,
                char=char,
                style=style,
                font=font,
            )
        except Exception as e:
            logger.debug(f"Failed to load {char}/{style}: {e}")
            return None

    # ===== STAGE 2: CPU (Image Processing) =====
    def _process_images_cpu_stage(self, loaded: LoadedImages) -> Optional[dict]:
        """Stage 2: Process loaded images (CPU-bound: resize, composite)."""
        try:
            # Resize helper
            def resize_img(img: Image.Image, height: int) -> Image.Image:
                aspect = img.width / img.height
                new_width = int(height * aspect)
                return img.resize((new_width, height), Image.Resampling.LANCZOS)
            
            # CPU-bound resizing
            content_resized = resize_img(loaded.content_img, self.resize_height)
            style_resized = resize_img(loaded.style_img, self.resize_height)
            target_resized = resize_img(loaded.target_img, self.resize_height)
            
            # CPU-bound compositing
            total_width = (content_resized.width + style_resized.width + 
                          target_resized.width + 2 * self.spacing)
            comparison = Image.new("RGB", (total_width, self.resize_height), color=(255, 255, 255))
            
            comparison.paste(content_resized, (0, 0))
            comparison.paste(style_resized, (content_resized.width + self.spacing, 0))
            comparison.paste(target_resized, 
                           (content_resized.width + style_resized.width + 2 * self.spacing, 0))
            
            return {
                "character": loaded.char,
                "style": loaded.style,
                "font": loaded.font,
                "content_image": loaded.content_img,
                "style_image": loaded.style_img,
                "target_image": loaded.target_img,
                "comparison_image": comparison,
                "content_hash": compute_file_hash(loaded.char, "", loaded.font),
                "target_hash": compute_file_hash(loaded.char, loaded.style, loaded.font),
            }
        except Exception as e:
            logger.debug(f"Failed to process {loaded.char}/{loaded.style}: {e}")
            return None

    def _generate_samples_pipeline(self) -> Generator[dict, None, None]:
        """Three-stage pipeline with pipelining."""
        generations = self._load_checkpoint()
        total = len(generations)
        
        logger.info(f"Starting three-stage pipeline for {total} samples")
        start_time = time.time()
        
        # Create executors
        io_executor = ThreadPoolExecutor(max_workers=self.io_workers)
        cpu_executor = ThreadPoolExecutor(max_workers=self.cpu_workers)
        
        try:
            processed = 0
            valid = 0
            
            # Process in I/O batches
            for io_batch_start in range(0, total, self.io_batch_size):
                io_batch_end = min(io_batch_start + self.io_batch_size, total)
                io_batch = generations[io_batch_start:io_batch_end]
                
                logger.info(f"I/O batch {io_batch_start}-{io_batch_end} ({len(io_batch)} samples)")
                io_start = time.time()
                
                # ===== STAGE 1: Bulk I/O =====
                io_futures = {io_executor.submit(self._load_images_io_stage, gen): gen 
                             for gen in io_batch}
                
                loaded_samples: list[LoadedImages] = []
                for future in as_completed(io_futures):
                    loaded = future.result()
                    if loaded:
                        loaded_samples.append(loaded)
                
                io_time = time.time() - io_start
                logger.info(f"  I/O stage: {len(loaded_samples)}/{len(io_batch)} loaded in {io_time:.2f}s")
                
                if not loaded_samples:
                    continue
                
                # ===== STAGE 2: CPU Processing (in smaller batches for pipelining) =====
                cpu_start = time.time()
                
                for cpu_batch_start in range(0, len(loaded_samples), self.cpu_batch_size):
                    cpu_batch_end = min(cpu_batch_start + self.cpu_batch_size, len(loaded_samples))
                    cpu_batch = loaded_samples[cpu_batch_start:cpu_batch_end]
                    
                    cpu_futures = {cpu_executor.submit(self._process_images_cpu_stage, loaded): loaded 
                                  for loaded in cpu_batch}
                    
                    # ===== STAGE 3: Yield as ready =====
                    for future in as_completed(cpu_futures):
                        sample = future.result()
                        if sample:
                            yield sample
                            valid += 1
                
                cpu_time = time.time() - cpu_start
                processed = io_batch_end
                
                # Progress
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                progress_pct = processed / total * 100
                eta = (total - processed) / rate if rate > 0 else 0
                
                logger.info(f"  CPU stage: {valid} valid samples processed in {cpu_time:.2f}s")
                logger.info(f"Progress: {progress_pct:.1f}%, {rate:.1f} samples/s, "
                          f"Valid: {valid}/{processed}, ETA: {eta:.0f}s")
            
            # Final stats
            total_time = time.time() - start_time
            if valid > 0:
                logger.info(f"Pipeline completed: {valid} samples in {total_time:.2f}s "
                          f"({valid/total_time:.1f} samples/s)")
            else:
                raise ValueError(
                    f"No valid samples from {total} attempts. Check:\n"
                    f"  - ContentImage/ and TargetImage/ have images\n"
                    f"  - Style images exist\n"
                    f"  - results_checkpoint.json matches actual files"
                )
        
        finally:
            io_executor.shutdown(wait=True)
            cpu_executor.shutdown(wait=True)

    def build(self) -> Dataset:
        """Build dataset with three-stage pipeline."""
        logger.info("Building dataset with three-stage pipeline...")
        
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
        
        logger.info("Converting to Arrow format...")
        start = time.time()
        
        dataset = Dataset.from_generator(
            self._generate_samples_pipeline,
            features=features,
        )
        
        elapsed = time.time() - start
        
        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Check logs for errors.")
        
        logger.info(f"Dataset created: {len(dataset)} samples in {elapsed:.2f}s")
        return dataset

    def push_to_hub(self, dataset: Dataset) -> None:
        """Push to HuggingFace Hub."""
        if not self.config.push_to_hub:
            return
        
        logger.info(f"Pushing to {self.config.repo_id}...")
        start = time.time()
        
        try:
            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                config_name=self.config.config_name,
                private=self.config.private,
                token=self.config.token,
            )
            
            elapsed = time.time() - start
            logger.info(f"Pushed in {elapsed:.2f}s")
            logger.info(f"Dataset: https://huggingface.co/datasets/{self.config.repo_id}")
        except Exception as e:
            logger.error(f"Push failed: {e}")
            raise

    def save_local(self, dataset: Dataset, output_path: Path) -> None:
        """Save locally."""
        logger.info(f"Saving to {output_path}...")
        start = time.time()
        dataset.save_to_disk(str(output_path))
        elapsed = time.time() - start
        logger.info(f"Saved in {elapsed:.2f}s")


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
    """Create dataset with three-stage pipeline (auto-tuned)."""
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
    
    builder = ThreeStageBuilder(config)
    dataset = builder.build()
    
    if local_save_path:
        builder.save_local(dataset, Path(local_save_path))
    
    if push_to_hub:
        builder.push_to_hub(dataset)
    
    return dataset


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ultra-fast three-stage HuggingFace dataset creator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--data-dir", required=True, help="Data directory")
    parser.add_argument("--style-images-dir", required=True, help="Style images directory")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--config-name", help="Dataset config name")
    parser.add_argument("--no-push", action="store_true", help="Skip push to Hub")
    parser.add_argument("--private", action="store_true", help="Private repo")
    parser.add_argument("--local-save", help="Save locally")
    parser.add_argument("--token", help="HuggingFace token")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
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
        
        print(f"\n✅ Success: {len(dataset)} samples")
        print(f"🔤 Characters: {len(set(dataset['character']))}")
        print(f"🎨 Styles: {len(set(dataset['style']))}")
        
        if not args.no_push:
            print(f"🌐 Hub: https://huggingface.co/datasets/{args.repo_id}")
        
    except Exception as e:
        logger.exception(f"Failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()