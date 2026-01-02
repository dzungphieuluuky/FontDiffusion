"""
Generate results_checkpoint.json from existing ContentImage and TargetImage folders
✅ Rescans directories and recalculates all paths and hashes
✅ Handles new simplified filename format: {char}.png and {style}+{char}.png
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Set
import hashlib
from filename_utils import (
    get_content_filename,
    get_target_filename,
    parse_content_filename,
    parse_target_filename,
    compute_file_hash
)


def generate_checkpoint_from_disk(data_root: str) -> Dict[str, Any]:
    """
    Generate complete checkpoint from disk structure
    
    Args:
        data_root: Path to directory containing ContentImage/ and TargetImage/
    
    Returns:
        Dictionary with checkpoint data
    """
    
    data_root = Path(data_root)
    content_dir = data_root / "ContentImage"
    target_base_dir = data_root / "TargetImage"
    
    print(f"📂 Scanning directory: {data_root}")
    print(f"   Content images: {content_dir}")
    print(f"   Target images: {target_base_dir}")
    
    # Validate directories exist
    if not content_dir.exists():
        raise FileNotFoundError(f"❌ ContentImage directory not found: {content_dir}")
    if not target_base_dir.exists():
        raise FileNotFoundError(f"❌ TargetImage directory not found: {target_base_dir}")
    
    # Collections
    generations: List[Dict[str, Any]] = []
    characters_set: Set[str] = set()
    styles_set: Set[str] = set()
    fonts_set: List[str] = ["NomNaTong-Regular"]
    
    # Track which content/target pairs exist
    content_chars: Dict[str, str] = {}  # char -> filepath
    valid_pairs: Set[tuple] = set()  # (char, style) pairs
    
    # ========== PHASE 1: Scan ContentImage directory ==========
    print(f"\n📝 Phase 1: Scanning ContentImage/...")
    content_files = list(content_dir.glob("*.png"))
    print(f"   Found {len(content_files)} content images")
    
    for content_file in sorted(content_files):
        char = parse_content_filename(content_file.name)
        
        if char:
            content_chars[char] = str(content_file.relative_to(data_root))
            characters_set.add(char)
            print(f"   ✓ {content_file.name} -> char: {repr(char)}")
        else:
            print(f"   ✗ Could not parse: {content_file.name}")
    
    # ========== PHASE 2: Scan TargetImage directory ==========
    print(f"\n🎨 Phase 2: Scanning TargetImage/...")
    target_style_dirs = [d for d in target_base_dir.iterdir() if d.is_dir()]
    print(f"   Found {len(target_style_dirs)} style directories")
    
    for style_dir in sorted(target_style_dirs):
        style_name = style_dir.name
        styles_set.add(style_name)
        
        target_files = list(style_dir.glob("*.png"))
        print(f"\n   {style_name}: {len(target_files)} images")
        
        for target_file in sorted(target_files):
            char, parsed_style = parse_target_filename(target_file.name)
            
            if char is None:
                print(f"      ✗ Could not parse: {target_file.name}")
                continue
            
            # Verify style matches directory
            if parsed_style != style_name:
                print(f"      ⚠️  Style mismatch in {target_file.name}: expected {style_name}, got {parsed_style}")
                continue
            
            # Check if content image exists for this character
            if char not in content_chars:
                print(f"      ⚠️  No content image for char {repr(char)} (target: {target_file.name})")
                continue
            
            valid_pairs.add((char, style_name))
            print(f"      ✓ {target_file.name} -> char: {repr(char)}, style: {style_name}")
    
    # ========== PHASE 3: Build generation records ==========
    print(f"\n⚙️  Phase 3: Building generation records...")
    print(f"   Valid char-style pairs: {len(valid_pairs)}")
    
    for char, style in sorted(valid_pairs):
        content_path = content_dir / get_content_filename(char)
        target_path = target_base_dir / style / get_target_filename(char, style)
        
        # Compute hashes
        print(f"\n   Processing: char={repr(char)}, style={style}, font={fonts_set[0]}")
        content_hash = compute_file_hash(char, "", fonts_set[0])
        target_hash = compute_file_hash(char, style, fonts_set[0])
        print(f"      content_hash: {content_hash}")
        print(f"      target_hash: {target_hash}")
        
        generation = {
            "character": char,
            "style": style,
            "font": "NomNaTong-Regular",
            "content_image_path": f"ContentImage/{get_content_filename(char)}",
            "target_image_path": f"TargetImage/{style}/{get_target_filename(char, style)}",
            "content_hash": content_hash,
            "target_hash": target_hash,
        }
        generations.append(generation)
    
    # ========== PHASE 4: Build final checkpoint ==========
    print(f"\n📊 Phase 4: Building checkpoint...")
    
    checkpoint = {
        "generations": generations,
        "characters": sorted(list(characters_set)),
        "styles": sorted(list(styles_set)),
        "fonts": sorted(list(fonts_set)),
        "total_chars": len(characters_set),
        "total_styles": len(styles_set),
    }
    
    print(f"\n✅ Checkpoint Summary:")
    print(f"   Total generations: {len(generations)}")
    print(f"   Total unique characters: {checkpoint['total_chars']}")
    print(f"   Total unique styles: {checkpoint['total_styles']}")
    print(f"   Fonts: {checkpoint['fonts']}")
    
    return checkpoint


def save_checkpoint(checkpoint: Dict[str, Any], output_path: str) -> None:
    """Save checkpoint to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved checkpoint to: {output_path}")


def main():
    """Main entry point"""
    
    print("=" * 70)
    print("REGENERATING RESULTS_CHECKPOINT.JSON FROM DISK")
    print("=" * 70)
    
    # Configure paths
    data_root = "my_dataset/train_original"
    checkpoint_output = "my_dataset/train_original/results_checkpoint.json"
    
    try:
        # Generate checkpoint
        checkpoint = generate_checkpoint_from_disk(data_root)
        
        # Save checkpoint
        save_checkpoint(checkpoint, checkpoint_output)
        
        print("\n" + "=" * 70)
        print("✅ SUCCESSFULLY GENERATED CHECKPOINT!")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure your directory structure is:")
        print("   data_root/")
        print("   ├── ContentImage/")
        print("   │   ├── {char1}.png")
        print("   │   ├── {char2}.png")
        print("   │   └── ...")
        print("   └── TargetImage/")
        print("       ├── {style1}/")
        print("       │   ├── {style1}+{char1}.png")
        print("       │   ├── {style1}+{char2}.png")
        print("       │   └── ...")
        print("       ├── {style2}/")
        print("       │   └── ...")
        print("       └── ...")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()