import json
import logging
import time
import os
import argparse
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from datasets import Dataset, Features, Image as HFImage, Value
from PIL import Image, ImageFile
import cv2
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

try:
    from filename_utils import compute_file_hash
except ImportError:
    def compute_file_hash(char: str, style: str, font: str) -> str:
        import hashlib
        return hashlib.md5(f"{char}_{style}_{font}".encode()).hexdigest()

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    data_dir: Path
    style_images_dir: Path
    repo_id: str
    split: str = "train"
    config_name: Optional[str] = None
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None
    resize_height: int = 256
    spacing: int = 10

    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.style_images_dir, str):
            self.style_images_dir = Path(self.style_images_dir)


class UltraFastDatasetBuilder:
    REQUIRED_DIRS = ["ContentImage", "TargetImage"]
    CHECKPOINT_FILE = "results_checkpoint.json"

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data_dir = config.data_dir
        self.style_images_dir = config.style_images_dir
        self.resize_height = config.resize_height
        self.spacing = config.spacing

        self.cpu_count = os.cpu_count() or 4
        self.num_proc = min(self.cpu_count, 8)
        self.process_batch_size = 1000

        self.style_paths: dict[str, Path] = {}
        self.path_cache: dict[str, dict] = {}

        self._validate_structure()
        self._build_style_path_index()
        self._build_path_cache_with_dims()
        self.generations = self._load_checkpoint()

        logger.info("Ultra-fast pipeline initialized:")
        logger.info(f"  Total generations: {len(self.generations)}")
        logger.info(f"  CPU workers: {self.num_proc} processes")
        logger.info(f"  Process batch size: {self.process_batch_size}")
        logger.info(f"  Style images: {len(self.style_paths)}")

    def _validate_structure(self) -> None:
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

    def _build_style_path_index(self) -> None:
        """Build index of style image paths for direct loading."""
        logger.info("Building style image path index...")
        self.style_paths = {}
        for ext in [".png", ".jpg", ".jpeg"]:
            for style_file in self.style_images_dir.glob(f"*{ext}"):
                style_name = style_file.stem
                self.style_paths[style_name] = style_file
        logger.info(f"Indexed {len(self.style_paths)} style images")

    def _build_path_cache_with_dims(self) -> None:
        logger.info("Building path cache with dimensions...")
        content_dir = self.data_dir / "ContentImage"
        content_paths = {}
        if content_dir.exists():
            for img_file in content_dir.glob("*"):
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    char = img_file.stem
                    try:
                        with Image.open(img_file) as img:
                            width, height = img.size
                        content_paths[char] = {
                            'path': str(img_file),
                            'width': width,
                            'height': height,
                        }
                    except Exception as e:
                        logger.debug(f"Failed to read dimensions for {img_file}: {e}")

        target_paths = {}
        target_dir = self.data_dir / "TargetImage"
        if target_dir.exists():
            for style_dir in target_dir.iterdir():
                if style_dir.is_dir():
                    style = style_dir.name
                    style_paths = {}
                    for img_file in style_dir.glob("*"):
                        if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                            filename_parts = img_file.stem.split('+')
                            if len(filename_parts) >= 2:
                                char = filename_parts[1]
                                try:
                                    with Image.open(img_file) as img:
                                        width, height = img.size
                                    style_paths[char] = {
                                        'path': str(img_file),
                                        'width': width,
                                        'height': height,
                                    }
                                except Exception as e:
                                    logger.debug(f"Failed to read dimensions for {img_file}: {e}")
                    target_paths[style] = style_paths

        # Build style dimensions from direct file access
        style_dims = {}
        for style_name, style_path in self.style_paths.items():
            try:
                with Image.open(style_path) as img:
                    style_dims[style_name] = img.size
            except Exception as e:
                logger.debug(f"Failed to read style dimensions for {style_path}: {e}")

        self.path_cache = {
            'content': content_paths,
            'target': target_paths,
            'style_dims': style_dims,
            'style_paths': {k: str(v) for k, v in self.style_paths.items()},
        }
        total_targets = sum(len(v) for v in target_paths.values())
        logger.info(f"Path cache built: {len(content_paths)} content, {total_targets} target paths")

    def _load_checkpoint(self) -> list[dict]:
        checkpoint_path = self.data_dir / self.CHECKPOINT_FILE
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        generations = data.get("generations", [])
        if not generations:
            raise ValueError("No generations found in checkpoint")
        logger.info(f"Loaded {len(generations)} generations from checkpoint")
        return generations

    @staticmethod
    def _resize_image_opencv(img: Image.Image, new_width: int, new_height: int) -> Image.Image:
        img_array = np.asarray(img)
        resized = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return Image.fromarray(resized)

    @staticmethod
    def _encode_image_to_bytes(img: Image.Image) -> bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _process_batch_parallel(
        batch: dict,
        path_cache: dict,
        resize_height: int,
        spacing: int,
    ) -> dict:
        batch_size = len(batch["character"])
        results = {
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

        style_paths = path_cache.get('style_paths', {})

        for i in range(batch_size):
            char = batch["character"][i]
            style = batch["style"][i]
            font = batch["font"][i]

            content_info = path_cache['content'].get(char)
            target_info = path_cache['target'].get(style, {}).get(char)
            style_dims = path_cache['style_dims'].get(style)
            style_path = style_paths.get(style)

            if not all([content_info, target_info, style_dims, style_path]):
                continue

            try:
                content_img = Image.open(content_info['path']).convert("RGB")
                target_img = Image.open(target_info['path']).convert("RGB")
                style_img = Image.open(style_path).convert("RGB")

                c_width, c_height = content_info['width'], content_info['height']
                t_width, t_height = target_info['width'], target_info['height']
                s_width, s_height = style_dims

                c_new_width = int(c_width * (resize_height / c_height))
                s_new_width = int(s_width * (resize_height / s_height))
                t_new_width = int(t_width * (resize_height / t_height))

                content_resized = UltraFastDatasetBuilder._resize_image_opencv(
                    content_img, c_new_width, resize_height
                )
                style_resized = UltraFastDatasetBuilder._resize_image_opencv(
                    style_img, s_new_width, resize_height
                )
                target_resized = UltraFastDatasetBuilder._resize_image_opencv(
                    target_img, t_new_width, resize_height
                )

                total_width = c_new_width + s_new_width + t_new_width + 2 * spacing
                comparison = Image.new("RGB", (total_width, resize_height), color=(255, 255, 255))
                comparison.paste(content_resized, (0, 0))
                comparison.paste(style_resized, (c_new_width + spacing, 0))
                comparison.paste(target_resized, (c_new_width + s_new_width + 2 * spacing, 0))

                results["character"].append(char)
                results["style"].append(style)
                results["font"].append(font)
                results["content_image"].append({
                    "bytes": UltraFastDatasetBuilder._encode_image_to_bytes(content_img)
                })
                results["style_image"].append({
                    "bytes": UltraFastDatasetBuilder._encode_image_to_bytes(style_img)
                })
                results["target_image"].append({
                    "bytes": UltraFastDatasetBuilder._encode_image_to_bytes(target_img)
                })
                results["comparison_image"].append({
                    "bytes": UltraFastDatasetBuilder._encode_image_to_bytes(comparison)
                })
                results["content_hash"].append(compute_file_hash(char, "", font))
                results["target_hash"].append(compute_file_hash(char, style, font))
            except Exception as e:
                logger.debug(f"Failed to process {char}/{style}: {e}")
                continue

        return results

    def build(self) -> Dataset:
        logger.info("Building dataset with batched map() pipeline...")
        start_time = time.time()

        metadata = {
            "character": [g.get("character", "") for g in self.generations],
            "style": [g.get("style", "") for g in self.generations],
            "font": [g.get("font", "unknown") for g in self.generations],
        }
        thin_dataset = Dataset.from_dict(metadata)

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

        logger.info(f"Processing with {self.num_proc} workers (batch_size={self.process_batch_size})...")

        dataset = thin_dataset.map(
            self._process_batch_parallel,
            batched=True,
            batch_size=self.process_batch_size,
            num_proc=self.num_proc,
            features=features,
            remove_columns=thin_dataset.column_names,
            desc="Processing images",
            fn_kwargs={
                "path_cache": self.path_cache,
                "resize_height": self.resize_height,
                "spacing": self.spacing,
            },
        )

        build_time = time.time() - start_time
        logger.info(f"Dataset built: {len(dataset)} samples in {build_time:.2f}s")
        logger.info(f"Processing speed: {len(dataset)/build_time:.1f} samples/s")
        return dataset

    def push_to_hub_streaming(self, dataset: Dataset) -> None:
        if not self.config.push_to_hub:
            logger.info("Skipping push to Hub")
            return

        logger.info(f"Streaming dataset to {self.config.repo_id}...")
        start_time = time.time()

        try:
            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                config_name=self.config.config_name,
                private=self.config.private,
                token=self.config.token,
                embed_external_files=False,
                commit_message="Dataset upload",
            )
            upload_time = time.time() - start_time
            logger.info(f"Upload completed in {upload_time:.2f}s ({len(dataset)/upload_time:.1f} samples/s)")
            logger.info(f"Dataset: https://huggingface.co/datasets/{self.config.repo_id}")
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise

    def save_local(self, dataset: Dataset, output_path: Path) -> None:
        logger.info(f"Saving dataset to {output_path}...")
        start_time = time.time()
        dataset.save_to_disk(str(output_path))
        save_time = time.time() - start_time
        logger.info(f"Dataset saved in {save_time:.2f}s")


def create_dataset_ultra(
    data_dir: str | Path,
    style_images_dir: str | Path,
    repo_id: str,
    split: str = "train",
    config_name: Optional[str] = None,
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str | Path] = None,
    resize_height: int = 256,
    spacing: int = 10,
) -> Dataset:
    config = DatasetConfig(
        data_dir=Path(data_dir),
        style_images_dir=Path(style_images_dir),
        repo_id=repo_id,
        split=split,
        config_name=config_name,
        push_to_hub=push_to_hub,
        private=private,
        token=token,
        resize_height=resize_height,
        spacing=spacing,
    )

    builder = UltraFastDatasetBuilder(config)
    dataset = builder.build()

    if local_save_path:
        builder.save_local(dataset, Path(local_save_path))

    if push_to_hub:
        builder.push_to_hub_streaming(dataset)

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Ultra-fast HuggingFace dataset creator with map() pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/upload_dataset_ultra.py \\
    --data-dir my_dataset \\
    --style-images-dir style_images \\
    --repo-id username/fontdiffusion-dataset
        """,
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Path to data directory (must contain ContentImage/ and TargetImage/)"
    )
    parser.add_argument(
        "--style-images-dir",
        required=True,
        help="Path to style images directory"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repository ID (username/dataset-name)"
    )
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
        help="HuggingFace API token"
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=256,
        help="Height for comparison images (default: 256)"
    )
    parser.add_argument(
        "--spacing",
        type=int,
        default=10,
        help="Spacing between images in comparison (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        start_time = time.time()

        dataset = create_dataset_ultra(
            data_dir=args.data_dir,
            style_images_dir=args.style_images_dir,
            repo_id=args.repo_id,
            split=args.split,
            config_name=args.config_name,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
            resize_height=args.resize_height,
            spacing=args.spacing,
        )

        total_time = time.time() - start_time
        print(f"\nDataset creation completed in {total_time:.2f}s")
        print(f"Samples: {len(dataset)}")
        print(f"Speed: {len(dataset)/total_time:.1f} samples/second")
        print(f"Unique characters: {len(set(dataset['character']))}")
        print(f"Unique styles: {len(set(dataset['style']))}")

        if not args.no_push:
            print(f"Uploaded to: https://huggingface.co/datasets/{args.repo_id}")
        if args.local_save:
            print(f"Local copy saved to: {args.local_save}")

    except KeyboardInterrupt:
        logger.warning("Dataset creation interrupted by user")
        raise SystemExit(130)
    except Exception as e:
        logger.exception(f"Dataset creation failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()