"""
Export Hugging Face dataset back to original FontDiffusion directory structure
Useful for inspection, backup, or sharing the generated data
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

from datasets import Dataset, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm


@dataclass
class ExportConfig:
    """Configuration for dataset export"""
    output_dir: str  # Root output directory (data_examples/train)
    repo_id: Optional[str] = None  # HF repo ID to load from
    local_dataset_path: Optional[str] = None  # Local dataset path to load from
    split: str = "train"
    create_metadata: bool = True  # Create results.json metadata
    token: Optional[str] = None  # HF token for private repos


class DatasetExporter:
    """Export Hugging Face dataset to original directory structure"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.content_dir = self.output_dir / "ContentImage"
        self.target_dir = self.output_dir / "TargetImage"
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.content_dir.mkdir(exist_ok=True)
        self.target_dir.mkdir(exist_ok=True)
        
        print(f"✓ Created output directories:")
        print(f"  Root: {self.output_dir}")
        print(f"  Content: {self.content_dir}")
        print(f"  Target: {self.target_dir}")
    
    def load_dataset(self) -> Dataset:
        """Load dataset from Hub or local path"""
        print("\n" + "="*60)
        print("LOADING DATASET")
        print("="*60)
        
        if self.config.repo_id:
            print(f"\nLoading from Hub: {self.config.repo_id}")
            # ✅ Pass token for private repos
            dataset = load_dataset(
                self.config.repo_id,
                split=self.config.split,
                token=self.config.token)  # ← Add token here
        elif self.config.local_dataset_path:
            print(f"\nLoading from local path: {self.config.local_dataset_path}")
            dataset = Dataset.load_from_disk(self.config.local_dataset_path)
        else:
            raise ValueError("Either repo_id or local_dataset_path must be provided")
        
        print(f"✓ Loaded dataset with {len(dataset)} samples")
        print(f"  Columns: {dataset.column_names}")
        
        return dataset
    
    def export_to_disk(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Export dataset to directory structure:
        output_dir/
        ├── ContentImage/
        │   ├── char0.png
        │   ├── char1.png
        │   └── ...
        └── TargetImage/
            ├── style0/
            │   ├── style0+char0.png
            │   ├── style0+char1.png
            │   └── ...
            ├── style1/
            │   ├── style1+char0.png
            │   └── ...
            └── ...
        """
        print("\n" + "="*60)
        print("EXPORTING DATASET TO DISK")
        print("="*60)
        
        # Get unique styles and characters
        unique_styles = sorted(set(dataset['style']))
        unique_chars = sorted(set(dataset['character']))
        unique_char_indices = sorted(set(dataset['char_index']))
        
        print(f"\nFound:")
        print(f"  {len(unique_styles)} unique styles")
        print(f"  {len(unique_chars)} unique characters")
        print(f"  {len(unique_char_indices)} character indices")
        
        # Track exported data for metadata
        exported_data = {
            'styles': unique_styles,
            'characters': unique_chars,
            'total_samples': len(dataset),
            'generations': []
        }
        
        # Create style directories
        style_dirs = {}
        for style in unique_styles:
            style_path = self.target_dir / style
            style_path.mkdir(exist_ok=True)
            style_dirs[style] = style_path
        
        print(f"\n✓ Created {len(style_dirs)} style directories")
        
        # Export content images (one per character index)
        print("\n" + "-"*60)
        print("Exporting content images...")
        print("-"*60)
        
        exported_content = {}
        content_iterator = tqdm(dataset, desc="🖼️  Processing", ncols=80)
        
        for sample in content_iterator:
            char_idx = sample['char_index']
            
            # Save content image (once per unique char_index)
            if char_idx not in exported_content:
                content_image = sample['content_image']
                content_path = self.content_dir / f"char{char_idx}.png"
                
                if isinstance(content_image, PILImage.Image):
                    content_image.save(content_path)
                else:
                    # Handle case where it's already a path string
                    content_image_obj = PILImage.open(content_image).convert('RGB')
                    content_image_obj.save(content_path)
                
                exported_content[char_idx] = str(content_path)
        
        print(f"✓ Exported {len(exported_content)} content images")
        
        # Export target images
        print("\n" + "-"*60)
        print("Exporting target images...")
        print("-"*60)
        
        target_iterator = tqdm(dataset, desc="🎨 Styles", ncols=80)
        exported_targets = 0
        
        for sample in target_iterator:
            char_idx = sample['char_index']
            style = sample['style']
            character = sample['character']
            font = sample.get('font', 'unknown')
            
            # Save target image
            target_image = sample['target_image']
            target_path = style_dirs[style] / f"{style}+char{char_idx}.png"
            
            if isinstance(target_image, PILImage.Image):
                target_image.save(target_path)
            else:
                # Handle case where it's already a path string
                target_image_obj = PILImage.open(target_image).convert('RGB')
                target_image_obj.save(target_path)
            
            # Add to metadata
            exported_data['generations'].append({
                'char_index': int(char_idx),
                'character': character,
                'style': style,
                'style_index': sample.get('style_index', -1),
                'font': font,
                'target_image_path': str(target_path),
                'content_image_path': exported_content[char_idx]
            })
            
            exported_targets += 1
        
        print(f"✓ Exported {exported_targets} target images")
        
        return exported_data
    
    def save_metadata(self, exported_data: Dict[str, Any]) -> None:
        """Save metadata to results.json"""
        if not self.config.create_metadata:
            print("\n⊘ Skipping metadata creation")
            return
        
        print("\n" + "-"*60)
        print("Saving metadata...")
        print("-"*60)
        
        metadata_path = self.output_dir / "results.json"
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(exported_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Saved metadata to {metadata_path}")
        print(f"  Total generations: {len(exported_data['generations'])}")
        print(f"  Styles: {len(exported_data['styles'])}")
        print(f"  Characters: {len(exported_data['characters'])}")
    
    def generate_summary(self, exported_data: Dict[str, Any]) -> None:
        """Generate and print summary report"""
        print("\n" + "="*60)
        print("EXPORT SUMMARY")
        print("="*60)
        
        summary = {
            'root_directory': str(self.output_dir),
            'total_samples': exported_data['total_samples'],
            'styles': {
                'count': len(exported_data['styles']),
                'list': exported_data['styles']
            },
            'characters': {
                'count': len(exported_data['characters']),
                'sample': exported_data['characters'][:5] + (['...'] if len(exported_data['characters']) > 5 else [])
            },
            'content_images': len([g for g in exported_data['generations'] if g['char_index'] == 0]),
            'target_images': len(exported_data['generations'])
        }
        
        print(f"\n✓ Export Statistics:")
        print(f"  Root: {summary['root_directory']}")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Styles: {summary['styles']['count']}")
        print(f"  Characters: {summary['characters']['count']}")
        print(f"  Target images: {summary['target_images']}")
        
        print(f"\n✓ Directory Structure:")
        print(f"  {self.output_dir}/")
        print(f"  ├── ContentImage/ ({len([g for g in exported_data['generations'] if g['char_index'] == 0])} images)")
        print(f"  ├── TargetImage/")
        for style in summary['styles']['list'][:3]:
            count = len([g for g in exported_data['generations'] if g['style'] == style])
            print(f"  │   ├── {style}/ ({count} images)")
        if len(summary['styles']['list']) > 3:
            print(f"  │   ├── ... ({len(summary['styles']['list']) - 3} more styles)")
        print(f"  └── results.json")
        
        # Save summary to file
        summary_path = self.output_dir / "export_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Summary saved to {summary_path}")


def export_dataset(
    output_dir: str,
    repo_id: Optional[str] = None,
    local_dataset_path: Optional[str] = None,
    split: str = "train",
    create_metadata: bool = True,
    token: Optional[str] = None  # ← Add token parameter
) -> None:
    """
    Export dataset to original directory structure
    
    Args:
        output_dir: Root output directory (data_examples/train)
        repo_id: Hugging Face repo ID to load from
        local_dataset_path: Local dataset path to load from
        split: Dataset split name
        create_metadata: Whether to create results.json
        token: HF token for private repos
    """
    
    if not repo_id and not local_dataset_path:
        raise ValueError("Either repo_id or local_dataset_path must be provided")
    
    config = ExportConfig(
        output_dir=output_dir,
        repo_id=repo_id,
        local_dataset_path=local_dataset_path,
        split=split,
        create_metadata=create_metadata,
        token=token  # ← Pass token
    )
    
    exporter = DatasetExporter(config)
    
    # Load dataset
    dataset = exporter.load_dataset()
    
    # Export to disk
    exported_data = exporter.export_to_disk(dataset)
    
    # Save metadata
    exporter.save_metadata(exported_data)
    
    # Generate summary
    exporter.generate_summary(exported_data)
    
    print("\n✓ Export completed successfully!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export dataset to original directory structure")
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Root output directory (e.g., data_examples/train)')
    parser.add_argument('--repo_id', type=str, default=None,
                       help='Hugging Face repo ID to load from')
    parser.add_argument('--local_dataset_path', type=str, default=None,
                       help='Local dataset path to load from')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split name')
    parser.add_argument('--no-metadata', action='store_true', default=False,
                       help='Do not create results.json metadata')
    parser.add_argument('--token', type=str, default=None,  # ← Add this
                       help='Hugging Face token for private repos')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("FONTDIFFUSION DATASET EXPORTER")
    print("="*60)
    
    export_dataset(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        local_dataset_path=args.local_dataset_path,
        split=args.split,
        create_metadata=not args.no_metadata,
        token=args.token  # ← Pass token
    )

"""Example
python export_dataset_to_disk.py \
  --output_dir "data_examples/train" \
  --repo_id "your_username/fontdiffusion-dataset" \
  --split "train"""