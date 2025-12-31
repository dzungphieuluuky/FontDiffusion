"""
Create Hugging Face dataset from generated FontDiffusion images
✅ FIXED: Dynamically discovers images instead of using checkpoint paths
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import torch
from datasets import Dataset, DatasetDict, Image as HFImage
from PIL import Image as PILImage
import pyarrow.parquet as pq
from tqdm import tqdm


@dataclass
class FontDiffusionDatasetConfig:
    """Configuration for dataset creation"""

    data_dir: str
    repo_id: str
    split: str = "train"
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None


class FontDiffusionDatasetBuilder:
    """Build FontDiffusion dataset in Hugging Face format"""

    def __init__(self, config: FontDiffusionDatasetConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.content_dir = self.data_dir / "ContentImage"
        self.target_dir = self.data_dir / "TargetImage"

        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate directory structure"""
        if not self.content_dir.exists():
            raise ValueError(f"ContentImage directory not found: {self.content_dir}")
        if not self.target_dir.exists():
            raise ValueError(f"TargetImage directory not found: {self.target_dir}")

        print(f"✓ Validated directory structure")
        print(f"  Content images: {self.content_dir}")
        print(f"  Target images: {self.target_dir}")

    def _discover_content_images(self) -> Dict[int, str]:
        """
        ✅ DISCOVER from filesystem (not from checkpoint)
        Returns mapping: char_index -> image_path
        """
        char_images = {}
        
        for img_file in sorted(self.content_dir.glob("char*.png")):
            try:
                char_idx = int(img_file.stem.replace("char", ""))
                char_images[char_idx] = str(img_file)
            except ValueError:
                continue
        
        if not char_images:
            raise ValueError(f"No content images found in {self.content_dir}")
        
        print(f"  ✓ Found {len(char_images)} content images")
        return char_images

    def _discover_target_images(self) -> Dict[Tuple[int, int], str]:
        """
        ✅ DISCOVER from filesystem (not from checkpoint)
        Returns mapping: (char_index, style_index) -> image_path
        """
        target_images = {}
        
        for style_dir in sorted(self.target_dir.glob("style*")):
            if not style_dir.is_dir():
                continue
            
            try:
                style_idx = int(style_dir.name.replace("style", ""))
            except ValueError:
                continue
            
            for img_file in sorted(style_dir.glob("style*+char*.png")):
                try:
                    filename = img_file.stem
                    parts = filename.split("+")
                    if len(parts) != 2:
                        continue
                    
                    char_idx = int(parts[1].replace("char", ""))
                    target_images[(char_idx, style_idx)] = str(img_file)
                except ValueError:
                    continue
        
        if not target_images:
            raise ValueError(f"No target images found in {self.target_dir}")
        
        print(f"  ✓ Found {len(target_images)} target images")
        return target_images

    def build_dataset(self) -> Dataset:
        """
        Build dataset by discovering images from filesystem
        ✅ Does NOT rely on checkpoint paths
        """
        print("\n" + "=" * 60)
        print("BUILDING DATASET")
        print("=" * 60)

        print(f"\n🖼️  Discovering images from disk...")
        content_images = self._discover_content_images()
        target_images = self._discover_target_images()

        dataset_rows: List[Dict[str, Any]] = []

        print(f"\n📋 Loading {len(target_images)} image pairs...")

        for (char_idx, style_idx), target_path in tqdm(
            target_images.items(),
            desc="Loading images",
            ncols=100,
            unit="pair"
        ):
            # Get content image path
            if char_idx not in content_images:
                tqdm.write(f"⚠ Missing content for char{char_idx}")
                continue
            
            content_path = content_images[char_idx]
            
            # Load images
            try:
                content_image = PILImage.open(content_path).convert("RGB")
                target_image = PILImage.open(target_path).convert("RGB")
            except Exception as e:
                tqdm.write(f"⚠ Error loading pair ({char_idx}, {style_idx}): {e}")
                continue
            
            # Extract metadata from filenames
            target_filename = Path(target_path).stem
            parts = target_filename.split("+")
            
            style_name = parts[0] if parts else f"style{style_idx}"
            character = f"char{char_idx}"
            
            row = {
                "character": character,
                "char_index": char_idx,
                "style": style_name,
                "style_index": style_idx,
                "content_image": content_image,
                "target_image": target_image,
                "font": "unknown",  # Not in filesystem, use default
            }
            
            dataset_rows.append(row)

        print(f"✓ Loaded {len(dataset_rows)} samples")

        if not dataset_rows:
            raise ValueError("No samples loaded!")

        # Create HuggingFace dataset
        return (
            Dataset.from_dict(
                {
                    "character": [r["character"] for r in dataset_rows],
                    "char_index": [r["char_index"] for r in dataset_rows],
                    "style": [r["style"] for r in dataset_rows],
                    "style_index": [r["style_index"] for r in dataset_rows],
                    "content_image": [r["content_image"] for r in dataset_rows],
                    "target_image": [r["target_image"] for r in dataset_rows],
                    "font": [r["font"] for r in dataset_rows],
                }
            )
            .cast_column("content_image", HFImage())
            .cast_column("target_image", HFImage())
        )

    def push_to_hub(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub"""
        if not self.config.push_to_hub:
            print("\n⊘ Skipping push to Hub")
            return

        print("\n" + "=" * 60)
        print("PUSHING TO HUB")
        print("=" * 60)

        try:
            print(f"Repository: {self.config.repo_id}")
            print(f"Split: {self.config.split}")

            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                private=self.config.private,
                token=self.config.token,
            )

            print(f"\n✓ Successfully pushed to Hub!")
            print(f"  URL: https://huggingface.co/datasets/{self.config.repo_id}")

        except Exception as e:
            print(f"\n✗ Error: {e}")
            raise

    def save_locally(self, output_path: str) -> None:
        """Save dataset locally"""
        print(f"\nSaving dataset to {output_path}")
        dataset = self.build_dataset()
        dataset.save_to_disk(output_path)
        print(f"✓ Saved!")


def create_and_push_dataset(
    data_dir: str,
    repo_id: str,
    split: str = "train",
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str] = None,
) -> Dataset:
    """Create and optionally push dataset to Hub"""

    config = FontDiffusionDatasetConfig(
        data_dir=data_dir,
        repo_id=repo_id,
        split=split,
        push_to_hub=push_to_hub,
        private=private,
        token=token,
    )

    builder = FontDiffusionDatasetBuilder(config)
    dataset = builder.build_dataset()

    if local_save_path:
        builder.save_locally(local_save_path)

    if push_to_hub:
        builder.push_to_hub(dataset)

    return dataset


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Create HF dataset from FontDiffusion images")

    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to data directory (with ContentImage/ and TargetImage/)")
    parser.add_argument("--repo_id", type=str, required=True,
                       help="HuggingFace repo ID")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split name")
    parser.add_argument("--private", action="store_true", default=False,
                       help="Make repo private")
    parser.add_argument("--no-push", action="store_true", default=False,
                       help="Don't push to Hub")
    parser.add_argument("--local-save", type=str, default=None,
                       help="Also save locally to this path")
    parser.add_argument("--token", type=str, default=None,
                       help="HF token")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FONTDIFFUSION DATASET CREATOR")
    print("=" * 60)
    print(f"\nData dir: {args.data_dir}")
    print(f"Repo: {args.repo_id}")
    print(f"Push to Hub: {not args.no_push}")

    try:
        create_and_push_dataset(
            data_dir=args.data_dir,
            repo_id=args.repo_id,
            split=args.split,
            push_to_hub=not args.no_push,
            private=args.private,
            token=args.token,
            local_save_path=args.local_save,
        )

        print("\n✅ COMPLETE!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)