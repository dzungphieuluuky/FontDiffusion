"""
Create Hugging Face dataset from generated FontDiffusion images and push to Hub
"""

import os
import json
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
    data_dir: str  # Path to data_examples/train
    repo_id: str  # huggingface repo id (e.g., "username/fontdiffusion-dataset")
    split: str = "train"
    push_to_hub: bool = True
    private: bool = False
    token: Optional[str] = None  # HF token for private repos


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
    
    def _load_results_metadata(self) -> Optional[Dict[str, Any]]:
        """Load results.json metadata if available"""
        results_path = self.data_dir / "results.json"
        if not results_path.exists():
            return None
        
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"\n⚠ Warning: results.json is corrupted ({e})")
            print(f"  Proceeding without metadata...")
            return None
        except Exception as e:
            print(f"\n⚠ Warning: Could not load results.json ({e})")
            print(f"  Proceeding without metadata...")
            return None 
           
    def build_dataset(self) -> Dataset:
        """
        Build dataset with structure:
        {
            'character': str,
            'char_index': int,
            'style': str,
            'style_index': int,
            'content_image': PIL.Image,
            'target_image': PIL.Image,
            'font': str (optional)
        }
        """
        print("\n" + "="*60)
        print("BUILDING DATASET")
        print("="*60)
        
        # Load metadata
        metadata = self._load_results_metadata()
        
        dataset_rows: List[Dict[str, Any]] = []
        
        # Load all content and target images
        content_dir = self.data_dir / "ContentImage"
        target_base_dir = self.data_dir / "TargetImage"
        
        if not content_dir.exists() or not target_base_dir.exists():
            raise ValueError(f"Missing required directories in {self.data_dir}")
        
        # Build mapping from results.json
        gen_map: Dict[Tuple[int, int], Dict[str, Any]] = {}  # (char_idx, style_idx) -> gen_info
        
        if metadata and 'generations' in metadata:
            for gen_info in metadata['generations']:
                char_idx = gen_info.get('char_index')
                style_idx = gen_info.get('style_index')
                
                if char_idx is not None and style_idx is not None:
                    gen_map[(char_idx, style_idx)] = gen_info
        
        # Iterate through target images
        for style_dir in sorted(target_base_dir.iterdir()):
            if not style_dir.is_dir():
                continue
            
            style_name = style_dir.name  # e.g., "style0"
            style_idx = int(style_name.replace('style', ''))
            
            for target_img_path in sorted(style_dir.glob("*.png")):
                # Parse filename: style0+char5.png
                filename = target_img_path.stem
                parts = filename.split('+')
                
                if len(parts) != 2:
                    continue
                
                char_idx_str = parts[1].replace('char', '')
                
                try:
                    char_idx = int(char_idx_str)
                except ValueError:
                    continue
                
                # Get content image path
                content_img_path = content_dir / f"char{char_idx}.png"
                
                if not content_img_path.exists():
                    print(f"⚠ Missing content image: {content_img_path}")
                    continue
                
                # Load images
                try:
                    content_image = PILImage.open(content_img_path).convert('RGB')
                    target_image = PILImage.open(target_img_path).convert('RGB')
                except Exception as e:
                    print(f"⚠ Error loading images for {filename}: {e}")
                    continue
                
                # Get metadata for this pair
                gen_info = gen_map.get((char_idx, style_idx), {})
                
                # Extract information - use gen_info first, then fallback
                character = gen_info.get('character', '?')
                font_name = gen_info.get('font', 'unknown')  # Now gets from gen_info!
                
                row = {
                    'character': character,
                    'char_index': char_idx,
                    'style': style_name,
                    'style_index': style_idx,
                    'content_image': content_image,
                    'target_image': target_image,
                    'font': font_name
                }
                
                dataset_rows.append(row)
        
        print(f"✓ Loaded {len(dataset_rows)} samples")
        
        if not dataset_rows:
            raise ValueError("No samples loaded!")
        
        return Dataset.from_dict({
            'character': [r['character'] for r in dataset_rows],
            'char_index': [r['char_index'] for r in dataset_rows],
            'style': [r['style'] for r in dataset_rows],
            'style_index': [r['style_index'] for r in dataset_rows],
            'content_image': [r['content_image'] for r in dataset_rows],
            'target_image': [r['target_image'] for r in dataset_rows],
            'font': [r['font'] for r in dataset_rows],
        }).cast_column('content_image', HFImage()).cast_column('target_image', HFImage())
    
    def _load_content_images(self) -> List[Dict[str, Any]]:
        """Load all content images and their metadata"""
        content_images: List[Dict[str, Any]] = []
        
        content_files = sorted(self.content_dir.glob("char*.png"))
        
        if not content_files:
            raise ValueError(f"No content images found in {self.content_dir}")
        
        metadata = self._load_results_metadata()
        
        for img_path in content_files:
            try:
                # Parse filename: char0.png
                char_idx = int(img_path.stem.replace('char', ''))
                
                # Get character from metadata or use placeholder
                character = '?'
                
                if metadata:
                    for gen_info in metadata.get('generations', []):
                        if gen_info.get('char_index') == char_idx:
                            character = gen_info.get('character', '?')
                            break
                
                content_images.append({
                    'index': char_idx,
                    'character': character,
                    'path': str(img_path)
                })
            except Exception as e:
                print(f"⚠ Error processing {img_path}: {e}")
                continue
        
        # Sort by index
        content_images.sort(key=lambda x: x['index'])
        
        print(f"✓ Loaded {len(content_images)} content images")
        
        return content_images    
    def push_to_hub(self, dataset: Dataset) -> None:
        """Push dataset to Hugging Face Hub"""
        if not self.config.push_to_hub:
            print("\n⊘ Skipping push to Hub (push_to_hub=False)")
            return
        
        print("\n" + "="*60)
        print("PUSHING TO HUB")
        print("="*60)
        
        try:
            print(f"\nRepository: {self.config.repo_id}")
            print(f"Split: {self.config.split}")
            print(f"Private: {self.config.private}")
            
            dataset.push_to_hub(
                repo_id=self.config.repo_id,
                split=self.config.split,
                private=self.config.private,
                token=self.config.token
            )
            
            print(f"\n✓ Successfully pushed to Hub!")
            print(f"  Dataset URL: https://huggingface.co/datasets/{self.config.repo_id}")
            
        except Exception as e:
            print(f"\n✗ Error pushing to Hub: {e}")
            raise
    
    def save_locally(self, output_path: str) -> None:
        """Save dataset locally for inspection"""
        print(f"\nSaving dataset locally to {output_path}")
        dataset = self.build_dataset()
        dataset.save_to_disk(output_path)
        print(f"✓ Dataset saved to {output_path}")


def create_and_push_dataset(
    data_dir: str,
    repo_id: str,
    split: str = "train",
    push_to_hub: bool = True,
    private: bool = False,
    token: Optional[str] = None,
    local_save_path: Optional[str] = None
) -> Dataset:
    """
    Create FontDiffusion dataset and optionally push to Hub
    
    Args:
        data_dir: Path to data_examples/train directory
        repo_id: Hugging Face repo ID (e.g., "username/fontdiffusion-dataset")
        split: Dataset split name (default: "train")
        push_to_hub: Whether to push to Hub
        private: Whether repo should be private
        token: HF token (if None, uses HUGGINGFACE_TOKEN env var)
        local_save_path: Path to save dataset locally
    
    Returns:
        Dataset object
    """
    
    config = FontDiffusionDatasetConfig(
        data_dir=data_dir,
        repo_id=repo_id,
        split=split,
        push_to_hub=push_to_hub,
        private=private,
        token=token
    )
    
    builder = FontDiffusionDatasetBuilder(config)
    
    # Build dataset
    dataset = builder.build_dataset()
    
    # Save locally if requested
    if local_save_path:
        builder.save_locally(local_save_path)
    
    # Push to Hub if requested
    if push_to_hub:
        builder.push_to_hub(dataset)
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create and push FontDiffusion dataset")
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to data_examples/train directory')
    parser.add_argument('--repo_id', type=str, required=True,
                       help='Hugging Face repo ID (e.g., username/fontdiffusion-dataset)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split name')
    parser.add_argument('--private', action='store_true', default=False,
                       help='Make repository private')
    parser.add_argument('--no-push', action='store_true', default=False,
                       help='Do not push to Hub')
    parser.add_argument('--local-save', type=str, default=None,
                       help='Also save dataset locally to this path')
    parser.add_argument('--token', type=str, default=None,
                       help='Hugging Face token')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FONTDIFFUSION DATASET CREATOR")
    print("="*60)
    
    dataset = create_and_push_dataset(
        data_dir=args.data_dir,
        repo_id=args.repo_id,
        split=args.split,
        push_to_hub=not args.no_push,
        private=args.private,
        token=args.token,
        local_save_path=args.local_save
    )
    
    print("\n✓ Done!")
"""Example
python sample_batch.py \
  --characters "chars.txt" \
  --style_images "styles/" \
  --ckpt_dir "checkpoints/phase1" \
  --ttf_path "fonts/" \
  --output_dir "data_examples/train_original"
  
python create_hf_dataset.py \
  --data_dir "my_dataset/train" \
  --repo_id "HF_USERNAME/font-diffusion-generated-data" \
  --split "train" \
  --private \
  --token "HF_TOKEN"""