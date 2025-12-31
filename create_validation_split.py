"""
Create validation/test splits from training data
✅ FIXED: Ensures proper matching of content, style, and target images
Supports multiple validation scenarios:
1. Seen char + Unseen style
2. Unseen char + Seen style
3. Unseen char + Unseen style
"""

import os
import shutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import random

from tqdm import tqdm


@dataclass
class ValidationSplitConfig:
    """Configuration for validation split creation"""

    data_root: str  # e.g., "data_examples"
    val_split_ratio: float = 0.2  # 20% for validation
    test_split_ratio: float = 0.1  # 10% for test
    random_seed: int = 42
    create_scenarios: bool = True  # Create multiple validation scenarios


class ValidationSplitCreator:
    """Create train/val/test splits with different scenarios"""

    def __init__(self, config: ValidationSplitConfig):
        self.config = config
        self.data_root = Path(config.data_root)

        # Use separate directory for original data
        self.original_train_dir = self.data_root / "train_original"

        # Split directories
        self.train_dir = self.data_root / "train"
        self.val_dir = self.data_root / "val"
        self.test_dir = self.data_root / "test"

        # Set random seed
        random.seed(config.random_seed)

        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate training directory structure"""
        # Check if original exists, otherwise check train
        source_dir = (
            self.original_train_dir
            if self.original_train_dir.exists()
            else self.train_dir
        )

        if not (source_dir / "TargetImage").exists():
            raise ValueError(f"TargetImage not found in {source_dir}")
        if not (source_dir / "ContentImage").exists():
            raise ValueError(f"ContentImage not found in {source_dir}")

        self.source_train_dir = source_dir
        print(f"✓ Using source directory: {self.source_train_dir}")

    def analyze_data(self) -> Tuple[List[str], List[str], Dict[str, List[str]], Dict[Tuple[str, str], bool]]:
        """
        ✅ ENHANCED: Analyze training data and create a map of valid (char, style) pairs
        
        Returns:
        - All styles
        - All characters  
        - Character->Style mapping
        - Valid (char, style) pairs that have target images
        """
        print("\n" + "=" * 60)
        print("ANALYZING TRAINING DATA")
        print("=" * 60)

        styles = set()
        characters = set()
        char_to_styles = defaultdict(set)  # char -> set of styles
        valid_pairs = set()  # (char, style) pairs that exist as target images

        target_dir = self.source_train_dir / "TargetImage"
        content_dir = self.source_train_dir / "ContentImage"

        # ✅ Scan all style directories
        for style_folder in sorted(target_dir.iterdir()):
            if not style_folder.is_dir():
                continue

            style_name = style_folder.name
            styles.add(style_name)

            # Scan images: style0+char0.png
            for img_file in sorted(style_folder.glob("*.png")):
                filename = img_file.stem  # Remove .png
                if "+" not in filename:
                    continue

                try:
                    char_part = filename.split("+")[1]  # Get "char0"
                    characters.add(char_part)
                    char_to_styles[char_part].add(style_name)
                    valid_pairs.add((char_part, style_name))
                except IndexError:
                    continue

        styles_list = sorted(list(styles))
        chars_list = sorted(list(characters))

        # ✅ Validate that all characters have content images
        print(f"\n✓ Found:")
        print(f"  Styles: {len(styles_list)}")
        print(f"  Characters: {len(chars_list)}")
        print(f"  Valid (char, style) pairs: {len(valid_pairs)}")

        # Check for missing content images
        missing_content = []
        for char in chars_list:
            content_path = content_dir / f"{char}.png"
            if not content_path.exists():
                missing_content.append(char)

        if missing_content:
            print(f"\n⚠️  WARNING: {len(missing_content)} characters missing content images:")
            print(f"  {missing_content[:10]}{'...' if len(missing_content) > 10 else ''}")
            print(f"  These will be excluded from splits")
            
            # Remove characters with missing content images
            for char in missing_content:
                del char_to_styles[char]
                characters.discard(char)
                # Remove all pairs with this character
                valid_pairs = {(c, s) for c, s in valid_pairs if c != char}
            
            chars_list = sorted(list(characters))

        print(f"\n✓ After validation:")
        print(f"  Characters with content images: {len(chars_list)}")
        print(f"  Valid (char, style) pairs: {len(valid_pairs)}")

        return styles_list, chars_list, dict(char_to_styles), valid_pairs

    def create_validation_scenarios(
        self,
        styles: List[str],
        characters: List[str],
        char_to_styles: Dict[str, List[str]],
    ) -> Dict[str, Dict]:
        """
        Create 4 validation scenarios:
        1. Seen char + Seen style (control - should be easy)
        2. Seen char + Unseen style (test generalization to new styles)
        3. Unseen char + Seen style (test generalization to new chars)
        4. Unseen char + Unseen style (test full generalization)
        """
        print("\n" + "=" * 60)
        print("CREATING VALIDATION SCENARIOS")
        print("=" * 60)

        num_styles = len(styles)
        num_chars = len(characters)

        # Split indices
        num_val_styles = max(1, int(num_styles * self.config.val_split_ratio))
        num_test_styles = max(1, int(num_styles * self.config.test_split_ratio))
        num_train_styles = num_styles - num_val_styles - num_test_styles

        num_val_chars = max(1, int(num_chars * self.config.val_split_ratio))
        num_test_chars = max(1, int(num_chars * self.config.test_split_ratio))
        num_train_chars = num_chars - num_val_chars - num_test_chars

        # Randomly split
        shuffled_styles = styles.copy()
        random.shuffle(shuffled_styles)

        shuffled_chars = characters.copy()
        random.shuffle(shuffled_chars)

        train_styles = set(shuffled_styles[:num_train_styles])
        val_styles = set(
            shuffled_styles[num_train_styles : num_train_styles + num_val_styles]
        )
        test_styles = set(shuffled_styles[num_train_styles + num_val_styles :])

        train_chars = set(shuffled_chars[:num_train_chars])
        val_chars = set(
            shuffled_chars[num_train_chars : num_train_chars + num_val_chars]
        )
        test_chars = set(shuffled_chars[num_train_chars + num_val_chars :])

        scenarios = {
            "train": {
                "styles": list(train_styles),
                "characters": list(train_chars),
                "description": "Seen styles + Seen characters (training data)",
            },
            "val_seen_style_unseen_char": {
                "styles": list(train_styles),
                "characters": list(val_chars),
                "description": "Seen styles + Unseen characters",
            },
            "val_unseen_style_seen_char": {
                "styles": list(val_styles),
                "characters": list(train_chars),
                "description": "Unseen styles + Seen characters",
            },
            "val_unseen_both": {
                "styles": list(val_styles),
                "characters": list(val_chars),
                "description": "Unseen styles + Unseen characters",
            },
            "test": {
                "styles": list(test_styles),
                "characters": list(test_chars),
                "description": "Test set (hold-out)",
            },
        }

        print("\n📊 Split Statistics:")
        print(
            f"  Styles: {num_train_styles} train + {num_val_styles} val + {num_test_styles} test"
        )
        print(
            f"  Chars:  {num_train_chars} train + {num_val_chars} val + {num_test_chars} test"
        )

        print("\n📋 Validation Scenarios:")
        for scenario_name, scenario_data in scenarios.items():
            print(f"\n  {scenario_name}:")
            print(f"    Description: {scenario_data['description']}")
            print(f"    Styles: {scenario_data['styles']}")
            print(
                f"    Chars: {scenario_data['characters'][:5]}{'...' if len(scenario_data['characters']) > 5 else ''}"
            )

        return scenarios

    def copy_images_for_split(
        self, 
        split_name: str, 
        split_dir: Path, 
        scenarios: Dict[str, Dict],
        valid_pairs: Set[Tuple[str, str]]
    ) -> Tuple[int, int, int]:
        """
        ✅ ENHANCED: Copy images for a specific split with validation
        
        Returns:
            (content_copied, target_copied, skipped_pairs)
        """
        split_config = scenarios[split_name]
        allowed_styles = set(split_config["styles"])
        allowed_chars = set(split_config["characters"])

        # Create directories
        split_content_dir = split_dir / "ContentImage"
        split_target_dir = split_dir / "TargetImage"
        split_content_dir.mkdir(parents=True, exist_ok=True)
        split_target_dir.mkdir(parents=True, exist_ok=True)

        # Create style subdirectories
        for style in allowed_styles:
            (split_target_dir / style).mkdir(exist_ok=True)

        # ============================================================================
        # Step 1: Identify valid pairs for this split
        # ============================================================================
        split_valid_pairs = set()
        for char, style in valid_pairs:
            if char in allowed_chars and style in allowed_styles:
                split_valid_pairs.add((char, style))

        if not split_valid_pairs:
            print(f"  ⚠️  No valid pairs found for {split_name}")
            return 0, 0, 0

        print(f"  Valid pairs: {len(split_valid_pairs)}")

        # Extract unique chars and styles from valid pairs
        chars_in_split = {char for char, style in split_valid_pairs}
        styles_in_split = {style for char, style in split_valid_pairs}

        print(f"  Unique chars in valid pairs: {len(chars_in_split)}")
        print(f"  Unique styles in valid pairs: {len(styles_in_split)}")

        # ============================================================================
        # Step 2: Copy content images for characters that have target images
        # ============================================================================
        source_content_dir = self.source_train_dir / "ContentImage"
        content_copied = 0

        print(f"\n  📥 Copying content images...")
        for char in tqdm(chars_in_split, desc="  Content", ncols=80, unit="char", leave=False):
            src_path = source_content_dir / f"{char}.png"
            dst_path = split_content_dir / f"{char}.png"

            if not src_path.exists():
                tqdm.write(f"    ⚠️  Missing content image: {src_path}")
                continue

            if src_path.resolve() != dst_path.resolve():
                shutil.copy2(src_path, dst_path)
                content_copied += 1

        # ============================================================================
        # Step 3: Copy target images only for valid (char, style) pairs
        # ============================================================================
        source_target_dir = self.source_train_dir / "TargetImage"
        target_copied = 0
        skipped_pairs = 0

        print(f"  📥 Copying target images...")
        for char, style in tqdm(split_valid_pairs, desc="  Target", ncols=80, unit="pair", leave=False):
            style_dir = source_target_dir / style
            if not style_dir.exists():
                skipped_pairs += 1
                continue

            # Find the target image: style0+charX.png
            target_filename = f"{style}+{char}.png"
            src_path = style_dir / target_filename

            if not src_path.exists():
                tqdm.write(f"    ⚠️  Missing target image: {src_path}")
                skipped_pairs += 1
                continue

            dst_path = split_target_dir / style / target_filename

            if src_path.resolve() != dst_path.resolve():
                shutil.copy2(src_path, dst_path)
                target_copied += 1

        # ============================================================================
        # Step 4: Validate split (optional but recommended)
        # ============================================================================
        print(f"\n  🔍 Validating split...")
        self._validate_split(split_dir)

        return content_copied, target_copied, skipped_pairs

    def _validate_split(self, split_dir: Path) -> None:
        """
        ✅ NEW: Validate that every target image has corresponding content image
        """
        content_dir = split_dir / "ContentImage"
        target_dir = split_dir / "TargetImage"

        missing_pairs = 0
        total_targets = 0

        for style_folder in target_dir.iterdir():
            if not style_folder.is_dir():
                continue

            for target_file in style_folder.glob("*.png"):
                total_targets += 1
                filename = target_file.stem
                
                if "+" not in filename:
                    continue

                char_part = filename.split("+")[1]
                content_path = content_dir / f"{char_part}.png"

                if not content_path.exists():
                    missing_pairs += 1
                    tqdm.write(f"    ❌ Validation failed: {target_file.name} missing {char_part}.png")

        if missing_pairs > 0:
            print(f"  ⚠️  VALIDATION ERROR: {missing_pairs}/{total_targets} targets missing content images!")
            raise ValueError(
                f"Split validation failed: {missing_pairs} target images have no matching content images"
            )
        else:
            print(f"  ✓ Validation passed: All {total_targets} targets have matching content images")

    def create_splits(self) -> None:
        """Create all splits"""
        print("\n" + "=" * 60)
        print("CREATING DATA SPLITS")
        print("=" * 60)

        # ✅ Analyze data and get valid pairs
        styles, characters, char_to_styles, valid_pairs = self.analyze_data()

        # Create scenarios
        scenarios = self.create_validation_scenarios(styles, characters, char_to_styles)

        # Create directory structure
        print("\n🔧 Creating directory structure...")

        # Train split
        print("\n📁 Train split:")
        train_content, train_target, train_skipped = self.copy_images_for_split(
            "train", self.train_dir, scenarios, valid_pairs
        )
        print(f"  ✓ Copied {train_content} content + {train_target} target images (skipped {train_skipped})")

        # Validation splits
        if self.config.create_scenarios:
            val_scenarios = [
                "val_seen_style_unseen_char",
                "val_unseen_style_seen_char",
                "val_unseen_both",
            ]

            for val_scenario in val_scenarios:
                print(f"\n📁 {val_scenario}:")
                scenario_dir = self.data_root / val_scenario
                val_content, val_target, val_skipped = self.copy_images_for_split(
                    val_scenario, scenario_dir, scenarios, valid_pairs
                )
                print(f"  ✓ Copied {val_content} content + {val_target} target images (skipped {val_skipped})")
        else:
            # Create simple val directory (combination of all unseen)
            print(f"\n📁 val (all unseen):")
            val_combined_scenarios = {
                "val": {
                    "styles": scenarios["val_unseen_both"]["styles"]
                    + scenarios["val_unseen_style_seen_char"]["styles"],
                    "characters": scenarios["val_unseen_both"]["characters"]
                    + scenarios["val_unseen_style_seen_char"]["characters"],
                }
            }
            val_content, val_target, val_skipped = self.copy_images_for_split(
                "val", self.val_dir, val_combined_scenarios, valid_pairs
            )
            print(f"  ✓ Copied {val_content} content + {val_target} target images (skipped {val_skipped})")

        # Test split
        print(f"\n📁 test:")
        test_content, test_target, test_skipped = self.copy_images_for_split(
            "test", self.test_dir, scenarios, valid_pairs
        )
        print(f"  ✓ Copied {test_content} content + {test_target} target images (skipped {test_skipped})")

        # Save scenario metadata
        self._save_scenario_metadata(scenarios)

    def _save_scenario_metadata(self, scenarios: Dict[str, Dict]) -> None:
        """Save scenario information to JSON"""
        metadata_path = self.data_root / "validation_scenarios.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(scenarios, f, indent=2)

        print(f"\n✓ Saved scenario metadata to {metadata_path}")


def create_validation_split(
    data_root: str,
    val_split_ratio: float = 0.2,
    test_split_ratio: float = 0.1,
    create_scenarios: bool = True,
    random_seed: int = 42,
) -> None:
    """
    Create validation splits with proper validation
    ✅ Ensures every (char, style) pair has both content and target images

    Args:
        data_root: Root data directory
        val_split_ratio: Fraction of data for validation
        test_split_ratio: Fraction of data for testing
        create_scenarios: Create separate scenario folders
        random_seed: Random seed for reproducibility
    """

    config = ValidationSplitConfig(
        data_root=data_root,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio,
        random_seed=random_seed,
        create_scenarios=create_scenarios,
    )

    creator = ValidationSplitCreator(config)
    creator.create_splits()

    print("\n" + "=" * 60)
    print("✓ VALIDATION SPLIT CREATION COMPLETE")
    print("=" * 60)

    if create_scenarios:
        print("\n✅ Created directories with validated pairs:")
        print("  📁 train/ - Training data (matched content + targets)")
        print("  📁 val_seen_style_unseen_char/ - Test new characters")
        print("  📁 val_unseen_style_seen_char/ - Test new styles")
        print("  📁 val_unseen_both/ - Test full generalization")
        print("  📁 test/ - Hold-out test set")
    else:
        print("\n✅ Created directories with validated pairs:")
        print("  📁 train/ - Training data (matched content + targets)")
        print("  📁 val/ - Validation data (matched content + targets)")
        print("  📁 test/ - Test data (matched content + targets)")
    
    print("\n💡 Each folder guarantees:")
    print("  ✓ For every charX+styleY.png target, charX.png content exists")
    print("  ✓ No orphaned target images without content")
    print("  ✓ No unused content images without targets")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create validation splits with proper matching")
    parser.add_argument(
        "--data_root", type=str, default="data_examples", help="Root data directory"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.1, help="Test split ratio"
    )
    parser.add_argument(
        "--scenarios",
        action="store_true",
        default=True,
        help="Create separate scenario folders",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("FONTDIFFUSION VALIDATION SPLIT CREATOR")
    print("=" * 60)

    try:
        create_validation_split(
            data_root=args.data_root,
            val_split_ratio=args.val_ratio,
            test_split_ratio=args.test_ratio,
            create_scenarios=args.scenarios,
            random_seed=args.seed,
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

"""
Example usage:
python create_validation_split.py \\
  --data_root data_examples \\
  --val_ratio 0.2 \\
  --test_ratio 0.1 \\
  --seed 42
"""