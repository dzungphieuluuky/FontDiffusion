"""
Create train/test splits for font generation evaluation.

Produces 4 splits:
1. train: seen_char x seen_style (training data)
2. test_seen_style_unseen_char: unseen_char x seen_style (content generalization)
3. test_unseen_style_seen_char: seen_char x unseen_style (style generalization)
4. test_unseen_style_unseen_char: unseen_char x unseen_style (full generalization)

This is the standard evaluation protocol for font style transfer.
"""

import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from filename_utils import (
    parse_content_filename,
    parse_target_filename,
)

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for dataset split creation."""

    data_root: str
    original_split: str = "total"
    train_char_ratio: float = 0.8
    train_style_ratio: float = 0.8
    random_seed: int = 42


@dataclass
class SplitInfo:
    """Information about a single split."""

    name: str
    description: str
    characters: list[str]
    styles: list[str]
    num_pairs: int = 0


class DatasetSplitCreator:
    """Create train/test splits with proper disjoint character and style sets."""

    SPLIT_NAMES = {
        "train": "train",
        "seen_style_unseen_char": "test_seen_style_unseen_char",
        "unseen_style_seen_char": "test_unseen_style_seen_char",
        "unseen_style_unseen_char": "test_unseen_style_unseen_char",
    }

    def __init__(self, config: SplitConfig):
        self.config = config
        self.data_root = Path(config.data_root)
        self.original_train_dir = self.data_root / config.original_split
        self.train_dir = self.data_root / "train"

        random.seed(config.random_seed)

        self._validate_structure()

    def _validate_structure(self) -> None:
        """Validate training directory structure."""
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
        logger.info(f"Using source directory: {self.source_train_dir}")

    def analyze_data(
        self,
        style_pattern: str = "*.png",
    ) -> tuple[dict[str, str], dict[tuple[str, str], str], dict[str, set[str]]]:
        """
        Analyze data by scanning actual files and matching content-target pairs.

        Returns:
            content_files: {char -> file_path}
            target_files: {(char, style) -> file_path}
            char_to_styles: {char -> set of styles}
        """
        logger.info("=" * 60)
        logger.info("ANALYZING TRAINING DATA")
        logger.info("=" * 60)

        content_dir = self.source_train_dir / "ContentImage"
        target_dir = self.source_train_dir / "TargetImage"

        content_files: dict[str, str] = {}
        target_files: dict[tuple[str, str], str] = {}
        char_to_styles: dict[str, set[str]] = defaultdict(set)

        # Scan content images
        logger.info("Scanning content images...")
        if content_dir.exists():
            for img_file in tqdm(
                list(content_dir.glob("*.png")),
                desc="Content images",
                unit="img",
            ):
                char = parse_content_filename(img_file.name)
                if char:
                    content_files[char] = str(img_file)

        logger.info(f"  Found {len(content_files)} content images")

        # Scan target images
        logger.info("Scanning target images...")
        total_targets = 0
        style_mismatch_count = 0
        parse_error_count = 0
        unparseable_files = []

        for style_folder in tqdm(
            sorted(target_dir.iterdir()),
            desc="Styles",
            unit="style",
        ):
            if not style_folder.is_dir():
                continue

            style_name = style_folder.name

            for img_file in style_folder.glob(style_pattern):
                parsed = parse_target_filename(img_file.name)

                if parsed is None:
                    parse_error_count += 1
                    unparseable_files.append(
                        {"folder": style_name, "filename": img_file.name}
                    )
                    continue

                char, style = parsed

                if style != style_name:
                    style_mismatch_count += 1
                    continue

                target_files[(char, style)] = str(img_file)
                char_to_styles[char].add(style)
                total_targets += 1

        logger.info(f"  Found {total_targets} valid target images")

        # Export unparseable files for diagnosis
        if parse_error_count > 0:
            logger.warning(f"  Parse errors: {parse_error_count}")
            unparseable_path = self.data_root / "unparseable_files.txt"
            with open(unparseable_path, "w", encoding="utf-8") as f:
                for item in unparseable_files:
                    abs_path = str(
                        (
                            self.source_train_dir
                            / "TargetImage"
                            / item["folder"]
                            / item["filename"]
                        ).resolve()
                    )
                    f.write(abs_path + "\n")
            logger.info(f"  Exported unparseable files to {unparseable_path}")

        if style_mismatch_count > 0:
            logger.warning(f"  Style mismatches: {style_mismatch_count}")

        # Validate content-target pairing
        logger.info("Validating content-target pairs...")
        valid_target_files: dict[tuple[str, str], str] = {}
        missing_content_count = 0

        for (char, style), path in target_files.items():
            if char in content_files:
                valid_target_files[(char, style)] = path
            else:
                missing_content_count += 1

        logger.info("=" * 60)
        logger.info("DATA ANALYSIS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Content images:     {len(content_files):,}")
        logger.info(f"Valid target pairs: {len(valid_target_files):,}")
        logger.info(f"Missing content:    {missing_content_count:,}")

        return content_files, valid_target_files, dict(char_to_styles)

    def create_splits(
        self,
        content_files: dict[str, str],
        target_files: dict[tuple[str, str], str],
        char_to_styles: dict[str, set[str]],
    ) -> dict[str, SplitInfo]:
        """
        Create train/test splits with disjoint character and style sets.

        Split strategy:
        - Characters: train_char_ratio go to seen, rest to unseen
        - Styles: train_style_ratio go to seen, rest to unseen

        Resulting splits:
        - train: seen_char x seen_style
        - test_seen_style_unseen_char: unseen_char x seen_style
        - test_unseen_style_seen_char: seen_char x unseen_style
        - test_unseen_style_unseen_char: unseen_char x unseen_style
        """
        logger.info("=" * 60)
        logger.info("CREATING TRAIN/TEST SPLITS")
        logger.info("=" * 60)

        # Get all unique characters and styles
        all_chars = sorted(content_files.keys())
        all_styles = sorted({style for (_, style) in target_files.keys()})

        num_chars = len(all_chars)
        num_styles = len(all_styles)

        # Calculate split sizes
        num_train_chars = max(1, int(num_chars * self.config.train_char_ratio))
        num_train_styles = max(1, int(num_styles * self.config.train_style_ratio))

        # Shuffle and split characters
        shuffled_chars = all_chars.copy()
        random.shuffle(shuffled_chars)
        seen_chars = set(shuffled_chars[:num_train_chars])
        unseen_chars = set(shuffled_chars[num_train_chars:])

        # Shuffle and split styles
        shuffled_styles = all_styles.copy()
        random.shuffle(shuffled_styles)
        seen_styles = set(shuffled_styles[:num_train_styles])
        unseen_styles = set(shuffled_styles[num_train_styles:])

        logger.info(f"Characters: {num_chars} total")
        logger.info(f"  Seen (train):   {len(seen_chars)}")
        logger.info(f"  Unseen (test):  {len(unseen_chars)}")
        logger.info(f"Styles: {num_styles} total")
        logger.info(f"  Seen (train):   {len(seen_styles)}")
        logger.info(f"  Unseen (test):  {len(unseen_styles)}")

        # Define splits
        splits = {
            "train": SplitInfo(
                name="train",
                description="Training: seen characters x seen styles",
                characters=sorted(seen_chars),
                styles=sorted(seen_styles),
            ),
            "seen_style_unseen_char": SplitInfo(
                name="test_seen_style_unseen_char",
                description="Test content generalization: unseen characters x seen styles",
                characters=sorted(unseen_chars),
                styles=sorted(seen_styles),
            ),
            "unseen_style_seen_char": SplitInfo(
                name="test_unseen_style_seen_char",
                description="Test style generalization: seen characters x unseen styles",
                characters=sorted(seen_chars),
                styles=sorted(unseen_styles),
            ),
            "unseen_style_unseen_char": SplitInfo(
                name="test_unseen_style_unseen_char",
                description="Test full generalization: unseen characters x unseen styles",
                characters=sorted(unseen_chars),
                styles=sorted(unseen_styles),
            ),
        }

        # Count valid pairs for each split
        for split_key, split_info in splits.items():
            char_set = set(split_info.characters)
            style_set = set(split_info.styles)
            pair_count = sum(
                1
                for (char, style) in target_files.keys()
                if char in char_set and style in style_set
            )
            split_info.num_pairs = pair_count

        # Log split statistics
        logger.info("=" * 60)
        logger.info("SPLIT STATISTICS")
        logger.info("=" * 60)
        for split_key, split_info in splits.items():
            logger.info(f"{split_info.name}:")
            logger.info(f"  {split_info.description}")
            logger.info(f"  Characters: {len(split_info.characters)}")
            logger.info(f"  Styles:     {len(split_info.styles)}")
            logger.info(f"  Pairs:      {split_info.num_pairs:,}")

        return splits

    def copy_images_for_split(
        self,
        split_key: str,
        split_info: SplitInfo,
        content_files: dict[str, str],
        target_files: dict[tuple[str, str], str],
    ) -> tuple[int, int, int]:
        """Copy images for a specific split."""
        split_dir = self.data_root / split_info.name
        allowed_chars = set(split_info.characters)
        allowed_styles = set(split_info.styles)

        # Create directories
        split_content_dir = split_dir / "ContentImage"
        split_target_dir = split_dir / "TargetImage"
        split_content_dir.mkdir(parents=True, exist_ok=True)
        split_target_dir.mkdir(parents=True, exist_ok=True)

        # Create style subdirectories
        for style in allowed_styles:
            (split_target_dir / style).mkdir(exist_ok=True)

        content_copied = 0
        target_copied = 0
        skipped = 0

        # Copy content images
        logger.info(f"  Copying content images for {split_info.name}...")
        for char in tqdm(
            sorted(allowed_chars),
            desc="  Content",
            unit="char",
            leave=False,
        ):
            if char not in content_files:
                skipped += 1
                continue

            src_path = Path(content_files[char])
            if not src_path.exists():
                skipped += 1
                continue

            dst_path = split_content_dir / src_path.name
            if src_path.resolve() != dst_path.resolve():
                try:
                    shutil.copy2(src_path, dst_path)
                    content_copied += 1
                except Exception as e:
                    logger.warning(f"Error copying {src_path}: {e}")
                    skipped += 1
            else:
                content_copied += 1

        # Copy target images
        logger.info(f"  Copying target images for {split_info.name}...")
        for (char, style), target_path_str in tqdm(
            sorted(target_files.items()),
            desc="  Target",
            unit="pair",
            leave=False,
        ):
            if char not in allowed_chars or style not in allowed_styles:
                continue

            src_path = Path(target_path_str)
            if not src_path.exists():
                skipped += 1
                continue

            dst_path = split_target_dir / style / src_path.name
            if src_path.resolve() != dst_path.resolve():
                try:
                    shutil.copy2(src_path, dst_path)
                    target_copied += 1
                except Exception as e:
                    logger.warning(f"Error copying {src_path}: {e}")
                    skipped += 1
            else:
                target_copied += 1

        logger.info(
            f"  {split_info.name}: {content_copied} content, "
            f"{target_copied} target (skipped: {skipped})"
        )

        return content_copied, target_copied, skipped

    def copy_and_filter_checkpoint(
        self,
        split_info: SplitInfo,
        target_files: dict[tuple[str, str], str],
    ) -> None:
        """Filter results_checkpoint.json for this split."""
        split_dir = self.data_root / split_info.name
        allowed_chars = set(split_info.characters)
        allowed_styles = set(split_info.styles)

        original_checkpoint_path = self.source_train_dir / "results_checkpoint.json"

        if not original_checkpoint_path.exists():
            logger.info(f"  No checkpoint found, skipping")
            return

        try:
            with open(original_checkpoint_path, "r", encoding="utf-8") as f:
                original_data = json.load(f)
        except Exception as e:
            logger.warning(f"  Error loading checkpoint: {e}")
            return

        original_generations = original_data.get("generations", [])
        filtered_generations = []

        for gen in original_generations:
            char = gen.get("character")
            style = gen.get("style")

            if (
                char in allowed_chars
                and style in allowed_styles
                and (char, style) in target_files
            ):
                filtered_generations.append(gen)

        split_checkpoint = {
            "split": split_info.name,
            "description": split_info.description,
            "num_characters": len(split_info.characters),
            "num_styles": len(split_info.styles),
            "num_generations": len(filtered_generations),
            "characters": split_info.characters,
            "styles": split_info.styles,
            "generations": filtered_generations,
            "fonts": original_data.get("fonts", []),
            "original_source": str(self.source_train_dir),
        }

        split_checkpoint_path = split_dir / "results_checkpoint.json"
        with open(split_checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(split_checkpoint, f, indent=2, ensure_ascii=False)

        logger.info(
            f"  Checkpoint: {len(filtered_generations):,}/{len(original_generations):,} generations"
        )

    def export_character_lists(
        self,
        splits: dict[str, SplitInfo],
    ) -> None:
        """Export character lists to txt files for each split."""
        logger.info("=" * 60)
        logger.info("EXPORTING CHARACTER LISTS")
        logger.info("=" * 60)

        for split_key, split_info in splits.items():
            split_dir = self.data_root / split_info.name
            split_dir.mkdir(parents=True, exist_ok=True)

            # Export characters
            char_file = split_dir / "characters.txt"
            with open(char_file, "w", encoding="utf-8") as f:
                for char in split_info.characters:
                    f.write(char + "\n")
            logger.info(
                f"  {split_info.name}/characters.txt: {len(split_info.characters)} characters"
            )

            # Export styles
            style_file = split_dir / "styles.txt"
            with open(style_file, "w", encoding="utf-8") as f:
                for style in split_info.styles:
                    f.write(style + "\n")
            logger.info(
                f"  {split_info.name}/styles.txt: {len(split_info.styles)} styles"
            )

        # Also export a summary file at the root
        summary_file = self.data_root / "split_characters_summary.txt"
        with open(summary_file, "w", encoding="utf-8") as f:
            for split_key, split_info in splits.items():
                f.write(f"=== {split_info.name} ===\n")
                f.write(f"Description: {split_info.description}\n")
                f.write(f"Characters ({len(split_info.characters)}): ")
                f.write(" ".join(split_info.characters[:50]))
                if len(split_info.characters) > 50:
                    f.write(f" ... and {len(split_info.characters) - 50} more")
                f.write("\n")
                f.write(f"Styles ({len(split_info.styles)}): ")
                f.write(", ".join(split_info.styles[:10]))
                if len(split_info.styles) > 10:
                    f.write(f" ... and {len(split_info.styles) - 10} more")
                f.write("\n\n")

        logger.info(f"  Summary: {summary_file}")

    def save_split_metadata(
        self,
        splits: dict[str, SplitInfo],
    ) -> None:
        """Save comprehensive split metadata to JSON."""
        metadata = {
            "config": {
                "train_char_ratio": self.config.train_char_ratio,
                "train_style_ratio": self.config.train_style_ratio,
                "random_seed": self.config.random_seed,
            },
            "splits": {},
        }

        for split_key, split_info in splits.items():
            metadata["splits"][split_key] = {
                "name": split_info.name,
                "description": split_info.description,
                "num_characters": len(split_info.characters),
                "num_styles": len(split_info.styles),
                "num_pairs": split_info.num_pairs,
                "characters": split_info.characters,
                "styles": split_info.styles,
            }

        metadata_path = self.data_root / "split_info.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved split metadata to {metadata_path}")

    def run(self, style_pattern: str = "*.png") -> None:
        """Main function to create all splits."""
        logger.info("=" * 60)
        logger.info("FONTDIFFUSION DATASET SPLIT CREATOR")
        logger.info("=" * 60)
        logger.info(f"Train char ratio:  {self.config.train_char_ratio}")
        logger.info(f"Train style ratio: {self.config.train_style_ratio}")
        logger.info(f"Random seed:       {self.config.random_seed}")

        # Step 1: Analyze data
        content_files, target_files, char_to_styles = self.analyze_data(
            style_pattern=style_pattern
        )

        if len(content_files) == 0 or len(target_files) == 0:
            raise ValueError("No valid data found. Check directory structure.")

        # Step 2: Create split definitions
        splits = self.create_splits(content_files, target_files, char_to_styles)

        # Step 3: Export character lists first (before copying files)
        self.export_character_lists(splits)

        # Step 4: Copy images and checkpoints for each split
        logger.info("=" * 60)
        logger.info("COPYING FILES TO SPLIT DIRECTORIES")
        logger.info("=" * 60)

        for split_key, split_info in splits.items():
            logger.info(f"\nProcessing {split_info.name}...")
            self.copy_images_for_split(
                split_key, split_info, content_files, target_files
            )
            self.copy_and_filter_checkpoint(split_info, target_files)

        # Step 5: Save metadata
        self.save_split_metadata(splits)

        # Step 6: Print summary
        self._print_summary(splits)

    def _print_summary(self, splits: dict[str, SplitInfo]) -> None:
        """Print final summary."""
        logger.info("\n" + "=" * 60)
        logger.info("SPLIT CREATION COMPLETE")
        logger.info("=" * 60)

        logger.info("\nCreated directories:")
        for split_key, split_info in splits.items():
            split_dir = self.data_root / split_info.name
            logger.info(f"  {split_dir}/")
            logger.info(f"    ContentImage/     ({len(split_info.characters)} chars)")
            logger.info(f"    TargetImage/      ({len(split_info.styles)} styles)")
            logger.info(f"    characters.txt    (character list)")
            logger.info(f"    styles.txt        (style list)")
            logger.info(f"    results_checkpoint.json")

        logger.info("\nSplit summary:")
        logger.info(f"  {'Split':<35} {'Chars':>8} {'Styles':>8} {'Pairs':>10}")
        logger.info(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*10}")
        for split_key, split_info in splits.items():
            logger.info(
                f"  {split_info.name:<35} "
                f"{len(split_info.characters):>8} "
                f"{len(split_info.styles):>8} "
                f"{split_info.num_pairs:>10,}"
            )

        logger.info("\nGuarantees:")
        logger.info("  - Character sets are disjoint between seen/unseen")
        logger.info("  - Style sets are disjoint between seen/unseen")
        logger.info("  - Every target image has matching content image")
        logger.info("  - Checkpoints contain only relevant generations")


def create_dataset_splits(
    data_root: str,
    train_char_ratio: float = 0.8,
    train_style_ratio: float = 0.8,
    random_seed: int = 42,
    style_pattern: str = "*.png",
    original_split: str = "total",
) -> None:
    """
    Create train/test splits for font generation evaluation.

    Args:
        data_root: Root data directory
        train_char_ratio: Fraction of characters for training (default 0.8)
        train_style_ratio: Fraction of styles for training (default 0.8)
        random_seed: Random seed for reproducibility
        style_pattern: Glob pattern for style images
        original_split: Name of the original training split directory
    """
    config = SplitConfig(
        data_root=data_root,
        original_split=original_split,
        train_char_ratio=train_char_ratio,
        train_style_ratio=train_style_ratio,
        random_seed=random_seed,
    )

    creator = DatasetSplitCreator(config)
    creator.run(style_pattern=style_pattern)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Create train/test splits for font generation evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data_examples",
        help="Root data directory",
    )
    parser.add_argument(
        "--train_char_ratio",
        type=float,
        default=0.8,
        help="Fraction of characters for training",
    )
    parser.add_argument(
        "--train_style_ratio",
        type=float,
        default=0.8,
        help="Fraction of styles for training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--style_pattern",
        type=str,
        default="*.png",
        help="Glob pattern for style images",
    )
    parser.add_argument(
        "--original_split",
        type=str,
        default="total",
        help="Name of the original training split directory",
    )

    args = parser.parse_args()

    try:
        create_dataset_splits(
            data_root=args.data_root,
            train_char_ratio=args.train_char_ratio,
            train_style_ratio=args.train_style_ratio,
            random_seed=args.seed,
            style_pattern=args.style_pattern,
            original_split=args.original_split,
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        import sys

        sys.exit(1)