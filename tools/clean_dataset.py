"""
Clean FontDiffusion dataset by removing partially covered characters.

Removes all characters that are not fully covered across all styles,
including their content images and all corresponding target images.
Updates results_checkpoint.json to reflect the cleaned dataset.
"""

import json
import logging
from pathlib import Path

from huggingface_hub.utils import tqdm

logger = logging.getLogger("DatasetCleaner")
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class DatasetCleaner:
    """Clean dataset by removing partially covered characters."""

    def __init__(self, dataset_dir: str | Path, dry_run: bool = True):
        """Initialize cleaner.
        
        Args:
            dataset_dir: Path to dataset root
            dry_run: If True, only simulate deletion without actually deleting
        """
        self.dataset_dir: Path = Path(dataset_dir)
        self.content_dir: Path = self.dataset_dir / "ContentImage"
        self.target_base_dir: Path = self.dataset_dir / "TargetImage"
        self.checkpoint_path: Path = self.dataset_dir / "results_checkpoint.json"
        self.checkpoint_backup_path: Path = self.dataset_dir / "results_checkpoint.json.backup"
        
        self.dry_run: bool = dry_run
        self.checkpoint_data: dict | None = None
        self.checkpoint_records: list[dict] = []
        
        # Statistics
        self.stats: dict = {
            "initial_records": 0,
            "initial_characters": 0,
            "initial_styles": 0,
            "fully_covered_chars": 0,
            "partially_covered_chars": 0,
            "removed_records": 0,
            "removed_content_images": 0,
            "removed_target_images": 0,
            "final_records": 0,
            "final_characters": 0,
        }

    def load_checkpoint(self) -> bool:
        """Load checkpoint file.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.checkpoint_path.exists():
            logger.error(f"❌ Checkpoint file not found: {self.checkpoint_path}")
            return False
        
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                self.checkpoint_data = json.load(f)
            
            self.checkpoint_records = self.checkpoint_data.get("generations", [])
            self.stats["initial_records"] = len(self.checkpoint_records)
            
            logger.info(f"✓ Loaded checkpoint with {len(self.checkpoint_records)} records")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            return False

    def identify_coverage(self) -> tuple[set[str], set[str], dict[str, dict[str, bool]]]:
        """Identify fully and partially covered characters.
        
        Returns:
            Tuple of (fully_covered_chars, partially_covered_chars, coverage_matrix)
        """
        logger.info("=" * 70)
        logger.info("IDENTIFYING CHARACTER COVERAGE")
        logger.info("=" * 70)
        
        # Extract unique styles and characters
        unique_styles: set[str] = set()
        unique_characters: set[str] = set()
        
        for record in self.checkpoint_records:
            char = record.get("character", "")
            style = record.get("style", "")
            if char:
                unique_characters.add(char)
            if style:
                unique_styles.add(style)
        
        self.stats["initial_characters"] = len(unique_characters)
        self.stats["initial_styles"] = len(unique_styles)
        
        logger.info(f"Found {len(unique_characters)} unique characters")
        logger.info(f"Found {len(unique_styles)} unique styles: {sorted(unique_styles)}")
        
        # Build coverage matrix
        coverage: dict[str, dict[str, bool]] = {}
        
        for char in unique_characters:
            coverage[char] = {}
            for style in unique_styles:
                has_pair: bool = False
                for record in self.checkpoint_records:
                    if record.get("character") == char and record.get("style") == style:
                        content_path = record.get("content_image_path", "")
                        target_path = record.get("target_image_path", "")
                        
                        content_exists = (self.dataset_dir / content_path).exists()
                        target_exists = (self.dataset_dir / target_path).exists()
                        
                        has_pair = content_exists and target_exists
                        break
                
                coverage[char][style] = has_pair
        
        # Classify characters
        fully_covered: set[str] = set()
        partially_covered: set[str] = set()
        
        for char in unique_characters:
            covered_styles = sum(1 for s in unique_styles if coverage[char].get(s, False))
            total_styles = len(unique_styles)
            
            if covered_styles == total_styles:
                fully_covered.add(char)
            else:
                partially_covered.add(char)
        
        self.stats["fully_covered_chars"] = len(fully_covered)
        self.stats["partially_covered_chars"] = len(partially_covered)
        
        logger.info(f"\n📊 Coverage Summary:")
        logger.info(f"  Fully covered (all {len(unique_styles)} styles):  {len(fully_covered)}")
        logger.info(f"  Partially covered (some styles):  {len(partially_covered)}")
        
        return fully_covered, partially_covered, coverage

    def get_files_to_remove(self, partially_covered_chars: set[str]) -> tuple[list[str], list[str]]:
        """Get list of content and target image files to remove.
        
        Args:
            partially_covered_chars: Set of characters to remove
            
        Returns:
            Tuple of (content_files_to_remove, target_files_to_remove)
        """
        content_files_to_remove: list[str] = []
        target_files_to_remove: list[str] = []
        
        for record in self.checkpoint_records:
            if record.get("character") in partially_covered_chars:
                content_path: str = record.get("content_image_path", "")
                target_path: str = record.get("target_image_path", "")
                
                if content_path:
                    content_files_to_remove.append(content_path)
                if target_path:
                    target_files_to_remove.append(target_path)
        
        return content_files_to_remove, target_files_to_remove

    def preview_deletion(
        self,
        partially_covered_chars: set[str],
        content_files: list[str],
        target_files: list[str],
    ) -> None:
        """Preview what will be deleted.
        
        Args:
            partially_covered_chars: Characters to remove
            content_files: Content image files to remove
            target_files: Target image files to remove
        """
        logger.info("=" * 70)
        logger.info("PREVIEW: FILES TO BE REMOVED")
        logger.info("=" * 70)
        
        logger.info(f"\n🗑️  Will remove {len(partially_covered_chars)} partially covered characters:")
        for char in sorted(partially_covered_chars)[:20]:
            logger.info(f"  • '{char}' (U+{ord(char):04X})")
        if len(partially_covered_chars) > 20:
            logger.info(f"  ... and {len(partially_covered_chars) - 20} more")
        
        logger.info(f"\n🗑️  Will remove {len(content_files)} content images")
        logger.info(f"\n🗑️  Will remove {len(target_files)} target images")
        
        logger.info(f"\n📊 Impact Summary:")
        logger.info(f"  Current records: {len(self.checkpoint_records)}")
        logger.info(f"  Records to remove: {len([r for r in self.checkpoint_records if r.get('character') in partially_covered_chars])}")
        logger.info(f"  Remaining records: {len(self.checkpoint_records) - len([r for r in self.checkpoint_records if r.get('character') in partially_covered_chars])}")

    def perform_deletion(self, partially_covered_chars: set[str]) -> bool:
        """Perform actual deletion of files and update checkpoint.
        
        Args:
            partially_covered_chars: Characters to remove
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("=" * 70)
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}PERFORMING DELETION")
        logger.info("=" * 70)
        
        # Get files to remove
        content_files, target_files = self.get_files_to_remove(partially_covered_chars)
        
        # Remove content images
        logger.info(f"\nRemoving content images...")
        for content_path in tqdm(content_files, desc="Content images", unit="file"):
            full_path = self.dataset_dir / content_path
            if full_path.exists():
                if not self.dry_run:
                    try:
                        full_path.unlink()
                        self.stats["removed_content_images"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove {full_path}: {e}")
                else:
                    self.stats["removed_content_images"] += 1
        
        # Remove target images
        logger.info(f"\nRemoving target images...")
        for target_path in tqdm(target_files, desc="Target images", unit="file"):
            full_path = self.dataset_dir / target_path
            if full_path.exists():
                if not self.dry_run:
                    try:
                        full_path.unlink()
                        self.stats["removed_target_images"] += 1
                    except Exception as e:
                        logger.warning(f"Failed to remove {full_path}: {e}")
                else:
                    self.stats["removed_target_images"] += 1
        
        # Update checkpoint records
        logger.info(f"\nUpdating checkpoint...")
        original_record_count = len(self.checkpoint_records)
        
        self.checkpoint_records = [
            record for record in self.checkpoint_records
            if record.get("character") not in partially_covered_chars
        ]
        
        self.stats["removed_records"] = original_record_count - len(self.checkpoint_records)
        self.stats["final_records"] = len(self.checkpoint_records)
        
        # Extract final statistics
        final_chars: set[str] = set()
        for record in self.checkpoint_records:
            char = record.get("character", "")
            if char:
                final_chars.add(char)
        self.stats["final_characters"] = len(final_chars)
        
        # Save backup and updated checkpoint
        if not self.dry_run:
            try:
                # Backup original
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    original_data = json.load(f)
                with open(self.checkpoint_backup_path, "w", encoding="utf-8") as f:
                    json.dump(original_data, f, indent=2, ensure_ascii=False)
                logger.info(f"✓ Backup saved to: {self.checkpoint_backup_path}")
                
                # Save updated checkpoint
                self.checkpoint_data["generations"] = self.checkpoint_records
                with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                    json.dump(self.checkpoint_data, f, indent=2, ensure_ascii=False)
                logger.info(f"✓ Updated checkpoint saved")
                
                return True
                
            except Exception as e:
                logger.error(f"❌ Error saving checkpoint: {e}")
                return False
        else:
            logger.info(f"[DRY RUN] Would backup to: {self.checkpoint_backup_path}")
            logger.info(f"[DRY RUN] Would update checkpoint with {len(self.checkpoint_records)} records")
            return True

    def print_summary(self) -> None:
        """Print deletion summary."""
        logger.info("=" * 70)
        logger.info("DELETION SUMMARY")
        logger.info("=" * 70)
        
        logger.info(f"\n📊 Statistics:")
        logger.info(f"  Initial records:              {self.stats['initial_records']}")
        logger.info(f"  Initial characters:           {self.stats['initial_characters']}")
        logger.info(f"  Initial styles:               {self.stats['initial_styles']}")
        logger.info(f"")
        logger.info(f"  Fully covered characters:     {self.stats['fully_covered_chars']}")
        logger.info(f"  Partially covered characters: {self.stats['partially_covered_chars']}")
        logger.info(f"")
        logger.info(f"  Records removed:              {self.stats['removed_records']}")
        logger.info(f"  Content images removed:       {self.stats['removed_content_images']}")
        logger.info(f"  Target images removed:        {self.stats['removed_target_images']}")
        logger.info(f"")
        logger.info(f"  Final records:                {self.stats['final_records']}")
        logger.info(f"  Final characters:             {self.stats['final_characters']}")
        
        if self.dry_run:
            logger.info(f"\n⚠️  This was a DRY RUN. No files were actually deleted.")
            logger.info(f"Run with --no-dry-run to perform actual deletion.")

    def run_full_cleaning(self) -> bool:
        """Run complete cleaning pipeline.
        
        Returns:
            True if successful, False otherwise
        """
        # Step 1: Load checkpoint
        if not self.load_checkpoint():
            return False
        
        # Step 2: Identify coverage
        fully_covered, partially_covered, coverage = self.identify_coverage()
        
        if not partially_covered:
            logger.info("\n✅ Dataset is already fully covered! No cleaning needed.")
            return True
        
        # Step 3: Preview deletion
        content_files, target_files = self.get_files_to_remove(partially_covered)
        self.preview_deletion(partially_covered, content_files, target_files)
        
        # Step 4: Perform deletion
        if not self.perform_deletion(partially_covered):
            return False
        
        # Step 5: Print summary
        self.print_summary()
        
        logger.info("\n" + "=" * 70)
        logger.info("CLEANING COMPLETE")
        logger.info("=" * 70)
        
        return True


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean FontDiffusion dataset by removing partially covered characters"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Actually perform deletion (by default, only simulates)",
    )
    
    args = parser.parse_args()
    
    dry_run = not args.no_dry_run
    
    if dry_run:
        logger.warning("=" * 70)
        logger.warning("DRY RUN MODE - No files will be deleted")
        logger.warning("Use --no-dry-run to perform actual deletion")
        logger.warning("=" * 70)
    else:
        logger.warning("=" * 70)
        logger.warning("⚠️  ACTUAL DELETION MODE - Files will be permanently deleted!")
        logger.warning("=" * 70)
        confirm = input("\nType 'DELETE' to confirm: ")
        if confirm != "DELETE":
            logger.info("Cancelled.")
            return
    
    # Run cleaning
    cleaner = DatasetCleaner(args.dataset_dir, dry_run=dry_run)
    success = cleaner.run_full_cleaning()
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()