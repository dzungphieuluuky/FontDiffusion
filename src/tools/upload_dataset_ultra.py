"""
Create Hugging Face dataset from FontDiffusion images with three-stage parallel processing.

Stage 1: ThreadPoolExecutor for I/O (image loading)
Stage 2: ProcessPoolExecutor for CPU (image resizing)
Stage 3: Parallel uploading to HuggingFace Hub
"""

import json
import logging
import time
import os
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Optional, List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import lru_cache
import queue
from queue import Queue
from threading import Lock
import multiprocessing as mp

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image, ImageFile

# Enable PIL optimizations
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# Import local utilities
try:
    from utilities import HFTqdm
    from filename_utils import compute_file_hash
except ImportError:
    def compute_file_hash(char: str, style: str, font: str) -> str:
        import hashlib
        return hashlib.md5(f"{char}_{style}_{font}".encode()).hexdigest()
    
    HFTqdm = None

logger = logging.getLogger(__name__)


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
        """Convert paths to Path if they're strings."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)


class ThreeStageParallelBuilder:
    """Three-stage parallel builder with specialized executors for each stage."""
    
    REQUIRED_DIRS = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE = "results_checkpoint.json"
    
    def __init__(self, config: DatasetConfig):
        """Initialize with optimized three-stage pipeline."""
        self.config = config
        self.data_dir = config.data_dir
        self.style_images_dir = config.style_images_dir
        
        # Auto-tune performance parameters
        self.cpu_count = os.cpu_count() or 4
        
        # Stage 1: I/O - many threads for disk operations
        self.io_workers = max(1, self.cpu_count * 8)  # Lots of threads for I/O
        
        # Stage 2: CPU - processes for heavy computation
        self.cpu_workers = max(1, self.cpu_count)  # One process per CPU core
        
        # Stage 3: Upload - threads for network I/O
        self.upload_workers = max(1, self.cpu_count * 4)  # Many threads for network
        
        # Batch sizes
        self.load_batch_size = 500   # Images to load per batch
        self.process_batch_size = 100  # Images to process per batch
        self.upload_batch_size = 50   # Samples to upload per batch
        
        # Image parameters
        self.resize_height = 256
        self.spacing = 10
        
        # Cache for style images (loaded once, used many times)
        self.style_cache: Dict[str, Image.Image] = {}
        self._preload_style_images()
        
        # Path cache
        self.path_cache: Dict[str, Dict[str, Path]] = {}
        self._build_path_cache()
        
        logger.info(f"Three-stage pipeline initialized:")
        logger.info(f"  Stage 1 (I/O): {self.io_workers} threads for loading")
        logger.info(f"  Stage 2 (CPU): {self.cpu_workers} processes for processing")
        logger.info(f"  Stage 3 (Upload): {self.upload_workers} threads for upload")
        
        self._validate_structure()

    def _preload_style_images(self):
        """Preload all style images into memory."""
        logger.info("Preloading style images...")
        for ext in [".png", ".jpg", ".jpeg"]:
            for style_file in self.style_images_dir.glob(f"*{ext}"):
                style_name = style_file.stem
                try:
                    self.style_cache[style_name] = Image.open(style_file).convert("RGB")
                except Exception as e:
                    logger.warning(f"Failed to load style image {style_file}: {e}")
        logger.info(f"Preloaded {len(self.style_cache)} style images")

    def _build_path_cache(self):
        """Build cache of all image paths for fast lookup."""
        logger.info("Building path cache...")
        
        # Content images
        content_dir = self.data_dir / "ContentImage"
        content_paths = {}
        if content_dir.exists():
            for img_file in content_dir.glob("*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    char = img_file.stem
                    content_paths[char] = img_file
        
        # Target images by style
        target_paths = {}
        target_dir = self.data_dir / "TargetImage"
        if target_dir.exists():
            for style_dir in target_dir.iterdir():
                if style_dir.is_dir():
                    style = style_dir.name
                    style_paths = {}
                    for img_file in style_dir.glob("*"):
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            # Extract character from filename (format: char_style.ext)
                            filename_parts = img_file.stem.split('_')
                            if filename_parts:
                                char = filename_parts[0]
                                style_paths[char] = img_file
                    target_paths[style] = style_paths
        
        self.path_cache = {
            'content': content_paths,
            'target': target_paths
        }
        
        total_targets = sum(len(v) for v in target_paths.values())
        logger.info(f"Path cache built: {len(content_paths)} content, {total_targets} target paths")

    def _validate_structure(self) -> None:
        """Validate directory structure."""
        for dir_name in self.REQUIRED_DIRS:
            dir_path = self.data_dir / dir_name
            if not dir_path.exists():
                raise ValueError(f"Required directory not found: {dir_path}")

        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

        if not self.style_images_dir.exists():
            raise ValueError(f"Style images directory not found: {self.style_images_dir}")

        logger.info("Directory structure validated")

    def _load_checkpoint(self) -> Tuple[List[Dict], Dict]:
        """Load checkpoint with metadata."""
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        generations = data.get("generations", [])
        if not generations:
            raise ValueError("No generations found in checkpoint")
        
        metadata = {
            'characters': data.get('characters', []),
            'styles': data.get('styles', []),
            'fonts': data.get('fonts', ['unknown']),
            'total': len(generations)
        }
        
        logger.info(f"Loaded {len(generations)} generations from checkpoint")
        return generations, metadata

    # ===== STAGE 1: I/O - Image Loading =====
    def _load_images_batch(self, batch: List[Dict]) -> List[Optional[Dict]]:
        """Load images for a batch of samples (I/O-bound)."""
        loaded_batch = []
        
        for gen in batch:
            char = gen.get("character", "")
            style = gen.get("style", "")
            
            # Get paths from cache
            content_path = self.path_cache['content'].get(char)
            target_path = self.path_cache['target'].get(style, {}).get(char)
            style_img = self.style_cache.get(style)
            
            if not all([content_path, target_path, style_img]):
                loaded_batch.append(None)
                continue
            
            try:
                # Load images
                content_img = Image.open(content_path).convert("RGB")
                target_img = Image.open(target_path).convert("RGB")
                
                loaded_batch.append({
                    'char': char,
                    'style': style,
                    'font': gen.get("font", "unknown"),
                    'content_img': content_img,
                    'target_img': target_img,
                    'style_img': style_img,
                    'content_path': str(content_path.relative_to(self.data_dir)),
                    'target_path': str(target_path.relative_to(self.data_dir)),
                })
            except Exception as e:
                logger.debug(f"Failed to load images for {char}/{style}: {e}")
                loaded_batch.append(None)
        
        return loaded_batch

    # ===== STAGE 2: CPU - Image Processing =====
    def _process_images_batch(self, batch: List[Dict]) -> List[Optional[Dict]]:
        """Process a batch of loaded images (CPU-bound)."""
        processed_batch = []
        
        for loaded in batch:
            if loaded is None:
                processed_batch.append(None)
                continue
            
            try:
                # Extract data
                char = loaded['char']
                style = loaded['style']
                font = loaded['font']
                content_img = loaded['content_img']
                target_img = loaded['target_img']
                style_img = loaded['style_img']
                
                # Calculate dimensions for resizing
                c_width, c_height = content_img.size
                s_width, s_height = style_img.size
                t_width, t_height = target_img.size
                
                # Resize images
                scale = self.resize_height / c_height
                c_new_width = int(c_width * scale)
                s_new_width = int(s_width * (self.resize_height / s_height))
                t_new_width = int(t_width * (self.resize_height / t_height))
                
                content_resized = content_img.resize((c_new_width, self.resize_height), 
                                                    Image.Resampling.LANCZOS)
                style_resized = style_img.resize((s_new_width, self.resize_height), 
                                                Image.Resampling.LANCZOS)
                target_resized = target_img.resize((t_new_width, self.resize_height), 
                                                  Image.Resampling.LANCZOS)
                
                # Create comparison image
                total_width = c_new_width + s_new_width + t_new_width + 2 * self.spacing
                comparison = Image.new("RGB", (total_width, self.resize_height), 
                                      color=(255, 255, 255))
                
                comparison.paste(content_resized, (0, 0))
                comparison.paste(style_resized, (c_new_width + self.spacing, 0))
                comparison.paste(target_resized, (c_new_width + s_new_width + 2 * self.spacing, 0))
                
                # Build final sample
                processed_batch.append({
                    "character": char,
                    "style": style,
                    "font": font,
                    "content_image": content_img,
                    "style_image": style_img,
                    "target_image": target_img,
                    "comparison_image": comparison,
                    "content_hash": compute_file_hash(char, "", font),
                    "target_hash": compute_file_hash(char, style, font),
                })
                
            except Exception as e:
                logger.debug(f"Failed to process {loaded.get('char', '')}/{loaded.get('style', '')}: {e}")
                processed_batch.append(None)
        
        return processed_batch

    def _generate_samples_pipeline(self) -> Generator[Dict, None, None]:
        """Generate samples using three-stage pipeline."""
        generations, metadata = self._load_checkpoint()
        total_samples = len(generations)
        
        logger.info(f"Starting three-stage pipeline for {total_samples} samples")
        
        # Create executors for each stage
        io_executor = ThreadPoolExecutor(max_workers=self.io_workers)
        cpu_executor = ProcessPoolExecutor(max_workers=self.cpu_workers)
        
        try:
            # Statistics
            start_time = time.time()
            loaded_count = 0
            processed_count = 0
            valid_count = 0
            
            # Process in batches through the pipeline
            for batch_start in range(0, total_samples, self.load_batch_size):
                batch_end = min(batch_start + self.load_batch_size, total_samples)
                generation_batch = generations[batch_start:batch_end]
                
                logger.info(f"Processing batch {batch_start}-{batch_end} ({len(generation_batch)} samples)")
                batch_start_time = time.time()
                
                # ===== STAGE 1: Load images (I/O-bound) =====
                io_start = time.time()
                io_future = io_executor.submit(self._load_images_batch, generation_batch)
                loaded_batch = io_future.result()
                loaded_count += len([x for x in loaded_batch if x is not None])
                io_time = time.time() - io_start
                
                # ===== STAGE 2: Process images (CPU-bound) =====
                cpu_start = time.time()
                
                # Split loaded batch into smaller chunks for CPU processing
                processed_results = []
                for chunk_start in range(0, len(loaded_batch), self.process_batch_size):
                    chunk_end = min(chunk_start + self.process_batch_size, len(loaded_batch))
                    chunk = loaded_batch[chunk_start:chunk_end]
                    
                    if any(x is not None for x in chunk):  # Only submit if there's work
                        cpu_future = cpu_executor.submit(self._process_images_batch, chunk)
                        processed_results.append(cpu_future)
                
                # Collect processed results
                processed_batch = []
                for future in as_completed(processed_results):
                    processed_chunk = future.result()
                    processed_batch.extend(processed_chunk)
                
                cpu_time = time.time() - cpu_start
                
                # ===== Yield results =====
                for sample in processed_batch:
                    if sample is not None:
                        yield sample
                        valid_count += 1
                
                processed_count += len(generation_batch)
                
                # Log batch progress
                batch_time = time.time() - batch_start_time
                elapsed_total = time.time() - start_time
                progress_pct = processed_count / total_samples * 100
                rate = processed_count / elapsed_total if elapsed_total > 0 else 0
                eta = (total_samples - processed_count) / rate if rate > 0 else 0
                
                logger.info(f"Batch completed in {batch_time:.2f}s (I/O: {io_time:.2f}s, CPU: {cpu_time:.2f}s)")
                logger.info(f"Progress: {progress_pct:.1f}%, {rate:.1f} samples/s, "
                          f"Valid: {valid_count}/{processed_count}, ETA: {eta:.0f}s")
            
            # Final statistics
            total_time = time.time() - start_time
            logger.info(f"Pipeline completed in {total_time:.2f}s")
            logger.info(f"Statistics: {valid_count} valid samples, "
                      f"{loaded_count} images loaded, {total_time/valid_count:.3f}s/sample")
            
        finally:
            # Cleanup
            io_executor.shutdown(wait=True)
            cpu_executor.shutdown(wait=True)

    def build(self) -> Dataset:
        """Build dataset using three-stage pipeline."""
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
        
        logger.info("Converting samples to Arrow format...")
        start_time = time.time()
        
        dataset = Dataset.from_generator(
            self._generate_samples_pipeline,
            features=features,
            keep_in_memory=False,
        )
        
        conversion_time = time.time() - start_time
        logger.info(f"Dataset created: {len(dataset)} samples in {conversion_time:.2f}s")
        
        return dataset

    # ===== STAGE 3: Parallel Upload =====
    def _upload_chunk(self, chunk: List[Dict], chunk_id: int, repo_id: str, token: Optional[str]) -> Tuple[int, bool]:
        """Upload a chunk of samples in parallel."""
        try:
            # Create a temporary dataset from the chunk
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
            
            chunk_dataset = Dataset.from_list(chunk, features=features)
            
            # Upload with a unique configuration name to avoid conflicts
            temp_config = f"chunk_{chunk_id}"
            chunk_dataset.push_to_hub(
                repo_id=repo_id,
                split=self.config.split,
                config_name=temp_config,
                private=self.config.private,
                token=token,
            )
            
            return len(chunk), True
        except Exception as e:
            logger.error(f"Failed to upload chunk {chunk_id}: {e}")
            return len(chunk), False

    def push_to_hub_parallel(self, dataset: Dataset) -> None:
        """Push dataset to HuggingFace Hub with parallel upload."""
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return
        
        logger.info(f"Pushing dataset to {self.config.repo_id} with parallel upload...")
        push_start = time.time()
        
        # Convert dataset to list for chunking
        logger.info("Converting dataset to list for chunked upload...")
        dataset_list = []
        for i in range(len(dataset)):
            dataset_list.append(dataset[i])
        
        total_samples = len(dataset_list)
        logger.info(f"Prepared {total_samples} samples for parallel upload")
        
        # Upload in parallel chunks
        upload_start = time.time()
        chunk_size = 500  # Samples per upload chunk
        total_chunks = (total_samples + chunk_size - 1) // chunk_size
        
        logger.info(f"Uploading {total_chunks} chunks in parallel...")
        
        completed = 0
        successful = 0
        
        with ThreadPoolExecutor(max_workers=self.upload_workers) as upload_executor:
            futures = []
            
            # Submit all chunks for upload
            for chunk_id in range(total_chunks):
                chunk_start = chunk_id * chunk_size
                chunk_end = min(chunk_start + chunk_size, total_samples)
                chunk = dataset_list[chunk_start:chunk_end]
                
                future = upload_executor.submit(
                    self._upload_chunk,
                    chunk,
                    chunk_id,
                    self.config.repo_id,
                    self.config.token
                )
                futures.append((future, chunk_id, len(chunk)))
            
            # Monitor progress
            for future, chunk_id, chunk_size in futures:
                try:
                    uploaded_count, success = future.result(timeout=300)  # 5-minute timeout
                    completed += uploaded_count
                    
                    if success:
                        successful += 1
                        logger.info(f"Chunk {chunk_id} uploaded successfully ({chunk_size} samples)")
                    else:
                        logger.warning(f"Chunk {chunk_id} failed")
                    
                    # Progress update
                    progress_pct = completed / total_samples * 100
                    elapsed = time.time() - upload_start
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_samples - completed) / rate if rate > 0 else 0
                    
                    logger.info(f"Upload progress: {progress_pct:.1f}%, "
                              f"{rate:.1f} samples/s, ETA: {eta:.0f}s")
                    
                except Exception as e:
                    logger.error(f"Chunk {chunk_id} timed out or failed: {e}")
        
        # Final merge of chunks
        logger.info(f"Merging {successful}/{total_chunks} successful chunks...")
        try:
            # Create final dataset from all chunks
            final_dataset = Dataset.from_list(dataset_list)
            
            # Push final merged dataset
            final_dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                config_name=self.config.config_name,
                private=self.config.private,
                token=self.config.token,
            )
            
            push_time = time.time() - push_start
            logger.info(f"Successfully pushed {total_samples} samples in {push_time:.2f}s")
            logger.info(f"Dataset available at: https://huggingface.co/datasets/{self.config.repo_id}")
            
        except Exception as e:
            logger.error(f"Final merge/push failed: {e}")
            # If merge fails, at least individual chunks might be uploaded
            logger.info(f"Partial upload completed: {successful}/{total_chunks} chunks uploaded")

    def save_local(self, dataset: Dataset, output_path: Path) -> None:
        """Save dataset to local disk."""
        logger.info(f"Saving dataset to {output_path}...")
        save_start = time.time()
        
        dataset.save_to_disk(str(output_path))
        
        save_time = time.time() - save_start
        logger.info(f"Dataset saved in {save_time:.2f}s")


def create_dataset_three_stage(
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
    """Create dataset with three-stage parallel processing.
    
    Stage 1: ThreadPoolExecutor for I/O (image loading)
    Stage 2: ProcessPoolExecutor for CPU (image processing)
    Stage 3: Parallel uploading to HuggingFace Hub
    
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
    
    builder = ThreeStageParallelBuilder(config)
    dataset = builder.build()
    
    if local_save_path:
        builder.save_local(dataset, Path(local_save_path))
    
    if push_to_hub:
        builder.push_to_hub_parallel(dataset)
    
    return dataset


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Three-stage parallel HuggingFace dataset creator from FontDiffusion images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Three-stage parallel processing
  python create_three_stage.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name
  
  # Create but don't push to Hub
  python create_three_stage.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name --no-push
  
  # Create private dataset
  python create_three_stage.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name --private
  
  # Save locally as well
  python create_three_stage.py --data-dir ./my_dataset \\
    --style-images-dir ./style_images --repo-id username/dataset-name \\
    --local-save ./local_dataset

Three-stage pipeline:
  Stage 1: ThreadPoolExecutor for I/O (image loading)
  Stage 2: ProcessPoolExecutor for CPU (image resizing)
  Stage 3: Parallel uploading to HuggingFace Hub
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
        # Run three-stage dataset creation
        start_time = time.time()
        
        dataset = create_dataset_three_stage(
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
        
        total_time = time.time() - start_time
        
        # Success summary
        print(f"\n✅ Dataset creation completed in {total_time:.2f}s!")
        print(f"📊 Samples: {len(dataset)}")
        print(f"⚡ Speed: {len(dataset)/total_time:.1f} samples/second")
        print(f"🔤 Unique characters: {len(set(dataset['character']))}")
        print(f"🎨 Unique styles: {len(set(dataset['style']))}")
        
        if not args.no_push:
            print(f"🌐 Uploaded to: https://huggingface.co/datasets/{args.repo_id}")
        
        if args.local_save:
            print(f"💾 Local copy saved to: {args.local_save}")
        
    except KeyboardInterrupt:
        logger.warning("Dataset creation interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()