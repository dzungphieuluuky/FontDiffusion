"""
Export Hugging Face dataset back to original FontDiffusion directory structure
Preserves original results.json AND results_checkpoint.json from pipeline generation
"""
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import json
import os
import shutil

from datasets import Dataset, load_dataset
from PIL import Image as PILImage
from tqdm import tqdm


@dataclass
class ExportConfig:
    output_dir: str
    repo_id: Optional[str] = None
    local_dataset_path: Optional[str] = None
    split: str = "train"
    create_metadata: bool = True
    token: Optional[str] = None
    preserve_original_metadata: bool = True


class DatasetExporter:
    """Export Hugging Face dataset to disk, preserving original metadata files"""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export(self) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Export dataset from Hub to disk
        
        Returns:
            Tuple of (results_metadata, checkpoint_metadata)
        """
        print("\n" + "="*60)
        print("EXPORTING DATASET TO DISK")
        print("="*60)
        
        # Load dataset
        if self.config.local_dataset_path:
            print(f"Loading local dataset from {self.config.local_dataset_path}...")
            try:
                dataset = Dataset.load_from_disk(self.config.local_dataset_path)
                print(f"✓ Loaded dataset with {len(dataset)} samples")
            except Exception as e:
                raise ValueError(f"Failed to load local dataset: {e}")
        
        elif self.config.repo_id:
            print(f"\n📥 Loading dataset from Hub...")
            print(f"   Repository: {self.config.repo_id}")
            print(f"   Split: {self.config.split}")
            
            try:
                # ✅ FIXED: Correct way to load from Hub
                dataset = load_dataset(
                    self.config.repo_id,
                    split=self.config.split,
                    trust_remote_code=True,
                    token=self.config.token
                )
                print(f"✓ Loaded dataset with {len(dataset)} samples from Hub")
            
            except Exception as e:
                print(f"\n❌ Error loading from Hub:")
                print(f"   Repository: {self.config.repo_id}")
                print(f"   Split: {self.config.split}")
                print(f"   Error: {type(e).__name__}: {e}")
                print(f"\n⚠️  Troubleshooting:")
                print(f"   1. Check repository exists: https://huggingface.co/datasets/{self.config.repo_id}")
                print(f"   2. Check split name is correct (available splits: train, validation, test)")
                print(f"   3. Check token has access to private datasets")
                print(f"   4. Try using --local_dataset_path instead if you have dataset cached locally")
                raise
        
        else:
            raise ValueError(
                "❌ Must provide either:\n"
                "   --repo_id (load from Hub)\n"
                "   --local_dataset_path (load from disk)"
            )
        
        # Create directory structure
        content_dir = self.output_dir / "ContentImage"
        target_base_dir = self.output_dir / "TargetImage"
        
        content_dir.mkdir(parents=True, exist_ok=True)
        target_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load both original metadata files
        original_results, original_checkpoint = self._try_load_original_metadata()
        
        if (original_results or original_checkpoint) and self.config.preserve_original_metadata:
            print("\n✅ Found original metadata - preserving it")
            return self._export_with_original_metadata(
                dataset, original_results, original_checkpoint
            )
        else:
            print("\n⚠ Original metadata not found - reconstructing from dataset")
            return self._export_with_reconstructed_metadata(dataset), None    
    def _try_load_original_metadata(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Try to load both results.json and results_checkpoint.json from various sources
        
        PRIORITY:
        1. results_checkpoint.json (complete, saved during training)
        2. results.json (may be incomplete if training was interrupted)
        
        Returns:
            Tuple of (results_data, checkpoint_data)
        """
        results_data: Optional[Dict[str, Any]] = None
        checkpoint_data: Optional[Dict[str, Any]] = None
        
        try:
            # Method 1: Try to download from Hub repository
            if self.config.repo_id:
                from huggingface_hub import hf_hub_download
                
                print("\n📥 Attempting to load metadata from Hub...")
                
                # ✅ PRIORITY 1: Try results_checkpoint.json FIRST (it's more complete)
                try:
                    checkpoint_path = hf_hub_download(
                        repo_id=self.config.repo_id,
                        filename="results_checkpoint.json",
                        repo_type="dataset",
                        token=self.config.token
                    )
                    
                    with open(checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    
                    print(f"  ✓ Loaded results_checkpoint.json from Hub ({len(checkpoint_data.get('generations', []))} generations)")
                    
                except Exception as e:
                    print(f"  ⚠ results_checkpoint.json not found on Hub: {type(e).__name__}")
                
                # ✅ PRIORITY 2: Try results.json (fallback only)
                try:
                    results_path = hf_hub_download(
                        repo_id=self.config.repo_id,
                        filename="results.json",
                        repo_type="dataset",
                        token=self.config.token
                    )
                    
                    with open(results_path, 'r', encoding='utf-8') as f:
                        results_data = json.load(f)
                    
                    num_gens = len(results_data.get('generations', []))
                    print(f"  ✓ Loaded results.json from Hub ({num_gens} generations)")
                    
                    # ⚠️ VALIDATE: Check if results.json is incomplete
                    if checkpoint_data and len(results_data.get('generations', [])) < len(checkpoint_data.get('generations', [])):
                        print(f"  ⚠ WARNING: results.json ({num_gens} gens) is incomplete!")
                        print(f"           Using results_checkpoint.json instead ({len(checkpoint_data.get('generations', []))} gens)")
                        results_data = None  # Don't use incomplete file
                    
                except Exception as e:
                    print(f"  ⚠ results.json not found on Hub: {type(e).__name__}")
                
                # If we found at least one file from Hub, return it
                if results_data or checkpoint_data:
                    return results_data, checkpoint_data
                
                print("  ⚠ No metadata files found on Hub")
            
            # Method 2: Try local cache/parent directory
            print("\n📁 Attempting to load metadata from local cache...")
            
            # ✅ PRIORITY 1: Try results_checkpoint.json FIRST
            local_checkpoint_path = self.output_dir.parent / "results_checkpoint.json"
            if local_checkpoint_path.exists():
                try:
                    with open(local_checkpoint_path, 'r', encoding='utf-8') as f:
                        checkpoint_data = json.load(f)
                    print(f"  ✓ Loaded results_checkpoint.json from local ({len(checkpoint_data.get('generations', []))} generations)")
                except Exception as e:
                    print(f"  ⚠ Error loading local results_checkpoint.json: {e}")
            
            # ✅ PRIORITY 2: Try results.json ONLY if checkpoint not found
            local_results_path = self.output_dir.parent / "results.json"
            if local_results_path.exists() and not checkpoint_data:
                try:
                    with open(local_results_path, 'r', encoding='utf-8') as f:
                        results_data = json.load(f)
                    
                    num_gens = len(results_data.get('generations', []))
                    print(f"  ✓ Loaded results.json from local ({num_gens} generations)")
                    
                    # ⚠️ VALIDATE: Check completeness
                    if checkpoint_data and num_gens < len(checkpoint_data.get('generations', [])):
                        print(f"  ⚠ WARNING: results.json is incomplete, using checkpoint instead")
                        results_data = None
                        
                except Exception as e:
                    print(f"  ⚠ Error loading local results.json: {e}")
            elif local_results_path.exists() and checkpoint_data:
                # We have checkpoint, so results.json is optional
                try:
                    with open(local_results_path, 'r', encoding='utf-8') as f:
                        potential_results = json.load(f)
                    
                    # Only use results.json if it has more data than checkpoint
                    if len(potential_results.get('generations', [])) > len(checkpoint_data.get('generations', [])):
                        results_data = potential_results
                        print(f"  ✓ Loaded results.json from local ({len(results_data.get('generations', []))} generations)")
                except Exception as e:
                    pass  # Silently skip if there's an issue
            
            if results_data or checkpoint_data:
                return results_data, checkpoint_data
            
            print(f"  ⚠ No metadata files found locally")
            return None, None
            
        except Exception as e:
            print(f"\n⚠ Error loading original metadata: {e}")
            return None, None    
        
    def _export_with_original_metadata(self, 
                                    dataset: Dataset, 
                                    original_results: Optional[Dict[str, Any]],
                                    original_checkpoint: Optional[Dict[str, Any]]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        Export dataset while preserving original results.json and results_checkpoint.json
        
        ✅ Uses results_checkpoint.json as SOURCE OF TRUTH (most complete)
        ✅ Falls back to results.json if checkpoint missing
        """
        print("\n" + "="*60)
        print("EXPORTING WITH ORIGINAL METADATA")
        print("="*60)
        
        # ✅ PRIORITY: Use checkpoint if available (it's more complete)
        metadata_source = original_checkpoint if original_checkpoint else original_results
        
        if not metadata_source:
            print("⚠️  WARNING: No metadata source found, reconstructing from dataset")
            return self._export_with_reconstructed_metadata(dataset), None
        
        # Show which source we're using
        source_name = "results_checkpoint.json" if original_checkpoint else "results.json"
        num_generations = len(metadata_source.get('generations', []))
        print(f"📋 Using {source_name} as metadata source ({num_generations} generations)")
        
        content_dir = self.output_dir / "ContentImage"
        target_base_dir = self.output_dir / "TargetImage"
        
        # Export content images
        print("\nExporting content images...")
        content_samples_exported = 0
        
        for idx, sample in enumerate(tqdm(dataset, desc="📝 Exporting content images", ncols=80)):
            if 'content_image' in sample:
                char_idx = sample.get('char_index', idx)
                content_img = sample['content_image']
                
                if isinstance(content_img, PILImage.Image):
                    content_path = content_dir / f"char{char_idx}.png"
                    content_img.save(str(content_path))
                    content_samples_exported += 1
        
        print(f"✓ Exported {content_samples_exported} content images")
        
        # Export target images organized by style
        print("\nExporting target images...")
        target_samples_exported = 0
        
        for sample in tqdm(dataset, desc="🎨 Exporting target images", ncols=80):
            if 'target_image' in sample:
                char_idx = sample.get('char_index', 0)
                style = sample.get('style', 'style0')
                
                style_dir = target_base_dir / style
                style_dir.mkdir(parents=True, exist_ok=True)
                
                target_img = sample['target_image']
                
                if isinstance(target_img, PILImage.Image):
                    target_path = style_dir / f"{style}+char{char_idx}.png"
                    target_img.save(str(target_path))
                    target_samples_exported += 1
        
        print(f"✓ Exported {target_samples_exported} target images")
        
        # ✅ Save metadata files
        print("\nSaving metadata files...")
        
        updated_results = None
        updated_checkpoint = None
        
        # ✅ ALWAYS save the source metadata (checkpoint or results)
        if original_checkpoint:
            updated_checkpoint = self._update_metadata_paths(original_checkpoint)
            checkpoint_path = self.output_dir / "results_checkpoint.json"
            with open(checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(updated_checkpoint, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved results_checkpoint.json ({len(updated_checkpoint.get('generations', []))} generations)")
        
        # ✅ ALSO save results.json if available (even if incomplete, preserve it)
        if original_results:
            updated_results = self._update_metadata_paths(original_results)
            results_path = self.output_dir / "results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(updated_results, f, indent=2, ensure_ascii=False)
            num_gen = len(updated_results.get('generations', []))
            print(f"  ✓ Saved results.json ({num_gen} generations)")
            
            # ⚠️ WARN if results.json is incomplete
            if original_checkpoint and num_gen < len(original_checkpoint.get('generations', [])):
                print(f"  ⚠️  WARNING: results.json has {num_gen} generations but checkpoint has {len(original_checkpoint.get('generations', []))}")
                print(f"             This is expected if training was interrupted. Using checkpoint as source of truth.")
        
        # Log metadata statistics from the COMPLETE source
        if updated_checkpoint:
            self._log_metadata_stats(updated_checkpoint)
        elif updated_results:
            self._log_metadata_stats(updated_results)
        
        return updated_results or {}, updated_checkpoint    
    def _update_metadata_paths(self, original_metadata: Dict[str, Any]) -> Dict[str, Any]:
        metadata = json.loads(json.dumps(original_metadata))
        
        for generation in metadata.get('generations', []):
            char_idx = generation.get('char_index', 0)
            style = generation.get('style', 'style0')
            
            # Update ALL path fields consistently
            generation['output_path'] = f"{self.output_dir}/TargetImage/{style}/{style}+char{char_idx}.png"
            generation['content_image_path'] = f"{self.output_dir}/ContentImage/char{char_idx}.png"
            generation['target_image_path'] = f"{self.output_dir}/TargetImage/{style}/{style}+char{char_idx}.png"
        
        return metadata
    
    def _log_metadata_stats(self, metadata: Dict[str, Any]) -> None:
        """Log metadata statistics"""
        try:
            num_generations = len(metadata.get('generations', []))
            num_styles = len(metadata.get('styles', []))
            num_chars = len(metadata.get('characters', []))
            fonts = metadata.get('fonts', [])
            
            print(f"\n📊 Metadata Statistics:")
            print(f"  Total generations: {num_generations}")
            print(f"  Total characters: {num_chars}")
            print(f"  Total styles: {num_styles}")
            print(f"  Fonts: {', '.join(fonts) if fonts else 'unknown'}")
        except Exception as e:
            print(f"⚠ Could not log metadata stats: {e}")
    
    def _export_with_reconstructed_metadata(self, dataset: Dataset) -> Dict[str, Any]:
        """
        Reconstruct metadata from dataset if original not available
        ⚠ Use only as fallback
        """
        print("\n" + "="*60)
        print("RECONSTRUCTING METADATA FROM DATASET")
        print("="*60)
        
        content_dir = self.output_dir / "ContentImage"
        target_base_dir = self.output_dir / "TargetImage"
        
        print("\nExporting images...")
        content_samples_exported = 0
        target_samples_exported = 0
        
        metadata: Dict[str, Any] = {
            'generations': [],
            'metrics': {
                'lpips': [],
                'ssim': [],
                'inference_times': []
            },
            'styles': [],
            'characters': [],
            'fonts': [],
            'total_chars': 0,
            'total_styles': 0
        }
        
        unique_chars = set()
        unique_styles = set()
        unique_fonts = set()
        
        for sample in tqdm(dataset, desc="📊 Exporting samples", ncols=80):
            char_idx = sample.get('char_index', 0)
            character = sample.get('character', '?')
            style = sample.get('style', 'style0')
            style_idx = sample.get('style_index', 0)
            font = sample.get('font', 'unknown')
            
            unique_chars.add(character)
            unique_styles.add(style)
            unique_fonts.add(font)
            
            # Export content image
            if 'content_image' in sample:
                content_img = sample['content_image']
                if isinstance(content_img, PILImage.Image):
                    content_path = content_dir / f"char{char_idx}.png"
                    content_img.save(str(content_path))
                    content_samples_exported += 1
            
            # Export target image
            if 'target_image' in sample:
                target_img = sample['target_image']
                if isinstance(target_img, PILImage.Image):
                    style_dir = target_base_dir / style
                    style_dir.mkdir(parents=True, exist_ok=True)
                    
                    target_path = style_dir / f"{style}+char{char_idx}.png"
                    target_img.save(str(target_path))
                    target_samples_exported += 1
            
            # Add to metadata
            metadata['generations'].append({
                'character': character,
                'char_index': char_idx,
                'style': style,
                'style_index': style_idx,
                'font': font,
                'content_image_path': f"{self.config.output_dir}/ContentImage/char{char_idx}.png",
                'target_image_path': f"{self.config.output_dir}/TargetImage/{style}/{style}+char{char_idx}.png"
            })
        
        # Update metadata summary
        metadata['characters'] = sorted(list(unique_chars))
        metadata['styles'] = sorted(list(unique_styles))
        metadata['fonts'] = sorted(list(unique_fonts))
        metadata['total_chars'] = len(unique_chars)
        metadata['total_styles'] = len(unique_styles)
        
        print(f"✓ Exported {content_samples_exported} content images")
        print(f"✓ Exported {target_samples_exported} target images")
        
        # Save reconstructed metadata
        print("\nSaving reconstructed metadata...")
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved reconstructed results.json")
        
        # Log metadata statistics
        self._log_metadata_stats(metadata)
        
        return metadata


def export_dataset(
    output_dir: str,
    repo_id: Optional[str] = None,
    local_dataset_path: Optional[str] = None,
    split: str = "train",
    create_metadata: bool = True,
    token: Optional[str] = None,
    preserve_original: bool = True
) -> None:
    """
    Export dataset from Hub or local path to disk
    Handles both results.json and results_checkpoint.json
    
    Args:
        output_dir: Directory to export to
        repo_id: Hub repo ID
        local_dataset_path: Local dataset path
        split: Dataset split
        create_metadata: Whether to create metadata
        token: HF token
        preserve_original: Try to preserve original metadata files
    """
    
    config = ExportConfig(
        output_dir=output_dir,
        repo_id=repo_id,
        local_dataset_path=local_dataset_path,
        split=split,
        create_metadata=create_metadata,
        token=token,
        preserve_original_metadata=preserve_original
    )
    
    exporter = DatasetExporter(config)
    results_metadata, checkpoint_metadata = exporter.export()
    
    print("\n" + "="*60)
    print("✓ EXPORT COMPLETE!")
    print("="*60)
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"    ├── ContentImage/")
    print(f"    │   ├── char0.png")
    print(f"    │   ├── char1.png")
    print(f"    │   └── ...")
    print(f"    ├── TargetImage/")
    print(f"    │   ├── style0/")
    print(f"    │   ├── style1/")
    print(f"    │   └── ...")
    
    if results_metadata:
        print(f"    ├── results.json ✅ ({len(results_metadata.get('generations', []))} generations)")
    
    if checkpoint_metadata:
        print(f"    └── results_checkpoint.json ✅ ({len(checkpoint_metadata.get('generations', []))} generations)")
    
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export Hugging Face dataset to disk")
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to export to')
    parser.add_argument('--repo_id', type=str, default=None,
                       help='Hugging Face repo ID')
    parser.add_argument('--local_dataset_path', type=str, default=None,
                       help='Local dataset path')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split')
    parser.add_argument('--token', type=str, default=None,
                       help='Hugging Face token')
    parser.add_argument('--preserve_original', action='store_true', default=True,
                       help='Preserve original results.json and results_checkpoint.json')
    parser.add_argument('--no_preserve_original', action='store_false', dest='preserve_original',
                       help='Do not preserve original metadata files')
    
    args = parser.parse_args()
    
    export_dataset(
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        local_dataset_path=args.local_dataset_path,
        split=args.split,
        token=args.token,
        preserve_original=args.preserve_original
    )


"""
USAGE EXAMPLES:

# Export with both metadata files preserved from Hub
python export_hf_dataset_to_disk.py \
  --output_dir "my_dataset/train_original" \
  --repo_id "dzungpham/font-diffusion-generated-data" \
  --split "train_original" \
  --token "hf_xxxxx" \
  --preserve_original

# Export from local dataset
python export_hf_dataset_to_disk.py \
  --output_dir "my_dataset/train" \
  --local_dataset_path "cached_dataset/train" \
  --preserve_original

# Export without preserving metadata (reconstruct from dataset)
python export_hf_dataset_to_disk.py \
  --output_dir "my_dataset/val" \
  --repo_id "dzungpham/font-diffusion-generated-data" \
  --split "val" \
  --no_preserve_original

OUTPUT FILES:
  ✅ results.json          - Main results with all generations
  ✅ results_checkpoint.json - Checkpoint saved during generation
  ✅ ContentImage/         - Original character images
  ✅ TargetImage/          - Generated styled character images
"""