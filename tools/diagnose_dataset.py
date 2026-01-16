"""
Diagnose FontDiffusion dataset structure and integrity.

Validates dataset consistency using results_checkpoint.json as the single source of truth.
Identifies missing pairs and generates statistics about character-style coverage.
Uses glob patterns for flexible style matching.
"""

import json
import logging
from collections import defaultdict
from pathlib import Path

from huggingface_hub.utils import tqdm

logger = logging.getLogger("DatasetDiagnostics")
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


class DatasetDiagnostics:
    """Diagnose FontDiffusion dataset structure and integrity."""

    def __init__(self, dataset_dir: str | Path):
        """Initialize diagnostics.
        
        Args:
            dataset_dir: Path to dataset root (contains ContentImage/, TargetImage/, results_checkpoint.json)
        """
        self.dataset_dir: Path = Path(dataset_dir)
        self.content_dir: Path = self.dataset_dir / "ContentImage"
        self.target_base_dir: Path = self.dataset_dir / "TargetImage"
        self.checkpoint_path: Path = self.dataset_dir / "results_checkpoint.json"
        
        # Statistics
        self.stats: dict = {
            "total_checkpoint_records": 0,
            "total_content_images_found": 0,
            "total_target_images_found": 0,
            "total_styles": 0,
            "total_characters": 0,
            "unique_characters": set(),
            "unique_styles": set(),
        }
        
        # Issues tracking
        self.issues: dict = {
            "missing_content_images": list(),  # [(char, style, expected_path), ...]
            "missing_target_images": list(),   # [(char, style, expected_path), ...]
            "extra_content_images": list(),    # Files in ContentImage not in checkpoint
            "extra_target_images": list(),     # Files in TargetImage not in checkpoint
            "checkpoint_records_missing_fields": list(),  # Invalid records in checkpoint
        }
        
        # Data loaded from checkpoint
        self.checkpoint_data: dict | None = None
        self.checkpoint_records: list[dict] = []

    def validate_structure(self) -> bool:
        """Validate basic directory structure and checkpoint file.
        
        Returns:
            True if structure is valid, False otherwise
        """
        logger.info("=" * 70)
        logger.info("VALIDATING DATASET STRUCTURE")
        logger.info("=" * 70)
        
        valid: bool = True
        
        # Check root directory
        if not self.dataset_dir.exists():
            logger.error(f"❌ Dataset directory not found: {self.dataset_dir}")
            return False
        
        logger.info(f"✓ Dataset directory exists: {self.dataset_dir}")
        
        # Check checkpoint file first (source of truth)
        if not self.checkpoint_path.exists():
            logger.error(f"❌ Checkpoint file not found: {self.checkpoint_path}")
            return False
        logger.info(f"✓ Checkpoint file found: {self.checkpoint_path}")
        
        # Check ContentImage directory
        if not self.content_dir.exists():
            logger.error(f"❌ ContentImage directory not found: {self.content_dir}")
            valid = False
        else:
            content_count = len(list(self.content_dir.glob("*.png")))
            logger.info(f"✓ ContentImage directory exists ({content_count} images)")
            self.stats["total_content_images_found"] = content_count
        
        # Check TargetImage directory
        if not self.target_base_dir.exists():
            logger.error(f"❌ TargetImage directory not found: {self.target_base_dir}")
            valid = False
        else:
            target_count = len(list(self.target_base_dir.rglob("*.png")))
            style_count = len([d for d in self.target_base_dir.iterdir() if d.is_dir()])
            logger.info(f"✓ TargetImage directory exists ({style_count} styles, {target_count} images)")
            self.stats["total_target_images_found"] = target_count
            self.stats["total_styles"] = style_count
        
        return valid

    def load_checkpoint(self) -> bool:
        """Load results_checkpoint.json as the source of truth.
        
        Returns:
            True if checkpoint loaded successfully, False otherwise
        """
        if not self.checkpoint_path.exists():
            logger.error(f"Checkpoint file not found: {self.checkpoint_path}")
            return False
        
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                self.checkpoint_data = json.load(f)
            
            self.checkpoint_records = self.checkpoint_data.get("generations", [])
            self.stats["total_checkpoint_records"] = len(self.checkpoint_records)
            
            logger.info(f"✓ Loaded checkpoint with {len(self.checkpoint_records)} generation records")
            
            # Validate record fields
            invalid_records: int = 0
            for idx, record in enumerate(self.checkpoint_records):
                required_fields = ["character", "style", "content_image_path", "target_image_path"]
                missing_fields = [f for f in required_fields if f not in record]
                if missing_fields:
                    invalid_records += 1
                    self.issues["checkpoint_records_missing_fields"].append({
                        "record_index": idx,
                        "missing_fields": missing_fields,
                        "record": record,
                    })
            
            if invalid_records > 0:
                logger.warning(f"⚠ Found {invalid_records} records with missing required fields")
            
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error parsing checkpoint JSON: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error loading checkpoint: {e}")
            return False

    def extract_unique_styles_and_chars(self) -> tuple[set[str], set[str]]:
        """Extract unique characters and styles from checkpoint records.
        
        Returns:
            Tuple of (unique_styles, unique_characters) sets
        """
        unique_styles: set[str] = set()
        unique_characters: set[str] = set()
        
        for record in self.checkpoint_records:
            char = record.get("character", "")
            style = record.get("style", "")
            
            if char:
                unique_characters.add(char)
            if style:
                unique_styles.add(style)
        
        self.stats["unique_characters"] = unique_characters
        self.stats["unique_styles"] = unique_styles
        self.stats["total_characters"] = len(unique_characters)
        self.stats["total_styles"] = len(unique_styles)
        
        return unique_styles, unique_characters

    def verify_checkpoint_files(self) -> None:
        """Verify that all files referenced in checkpoint exist on disk.
        
        Compares checkpoint records against actual files in ContentImage/ and TargetImage/.
        """
        logger.info("=" * 70)
        logger.info("VERIFYING CHECKPOINT FILES ON DISK")
        logger.info("=" * 70)
        
        missing_content: int = 0
        missing_target: int = 0
        
        for record in tqdm(self.checkpoint_records, desc="Checking files", unit="record"):
            char: str = record.get("character", "")
            style: str = record.get("style", "")
            content_path: str = record.get("content_image_path", "")
            target_path: str = record.get("target_image_path", "")
            
            # Check content image
            full_content_path = self.dataset_dir / content_path
            if not full_content_path.exists():
                missing_content += 1
                self.issues["missing_content_images"].append({
                    "character": char,
                    "style": style,
                    "expected_path": content_path,
                })
            
            # Check target image
            full_target_path = self.dataset_dir / target_path
            if not full_target_path.exists():
                missing_target += 1
                self.issues["missing_target_images"].append({
                    "character": char,
                    "style": style,
                    "expected_path": target_path,
                })
        
        logger.info(f"✓ Checkpoint verification complete")
        if missing_content > 0:
            logger.warning(f"⚠ Missing {missing_content} content images referenced in checkpoint")
        if missing_target > 0:
            logger.warning(f"⚠ Missing {missing_target} target images referenced in checkpoint")

    def find_extra_files(self) -> None:
        """Find files on disk that are not referenced in checkpoint.
        
        This identifies orphan images that may not be part of the dataset.
        """
        logger.info("=" * 70)
        logger.info("CHECKING FOR EXTRA FILES NOT IN CHECKPOINT")
        logger.info("=" * 70)
        
        # Build set of expected paths from checkpoint
        expected_content_files: set[str] = set()
        expected_target_files: set[str] = set()
        
        for record in self.checkpoint_records:
            content_path: str = record.get("content_image_path", "")
            target_path: str = record.get("target_image_path", "")
            
            if content_path:
                expected_content_files.add(Path(content_path).name)
            if target_path:
                expected_target_files.add(Path(target_path).name)
        
        # Find extra content images
        if self.content_dir.exists():
            actual_content_files = {f.name for f in self.content_dir.glob("*.png")}
            extra_content = actual_content_files - expected_content_files
            
            if extra_content:
                logger.warning(f"⚠ Found {len(extra_content)} extra content images not in checkpoint:")
                for filename in sorted(extra_content)[:20]:
                    logger.warning(f"  • {filename}")
                    self.issues["extra_content_images"].append(filename)
                if len(extra_content) > 20:
                    logger.warning(f"  ... and {len(extra_content) - 20} more")
        
        # Find extra target images
        if self.target_base_dir.exists():
            actual_target_files = {f.name for f in self.target_base_dir.rglob("*.png")}
            extra_target = actual_target_files - expected_target_files
            
            if extra_target:
                logger.warning(f"⚠ Found {len(extra_target)} extra target images not in checkpoint:")
                for filename in sorted(extra_target)[:20]:
                    logger.warning(f"  • {filename}")
                    self.issues["extra_target_images"].append(filename)
                if len(extra_target) > 20:
                    logger.warning(f"  ... and {len(extra_target) - 20} more")

    def diagnose_coverage(self, style_glob_pattern: str = "*") -> dict[str, dict[str, bool]]:
        """Diagnose character-style coverage from checkpoint records.
        
        Args:
            style_glob_pattern: Glob pattern to filter styles (e.g., "hand*" for handwritten)
            
        Returns:
            Coverage matrix: {character: {style: has_valid_pair}}
        """
        logger.info("=" * 70)
        logger.info("DIAGNOSING CHARACTER-STYLE COVERAGE (from checkpoint)")
        logger.info("=" * 70)
        
        # Extract unique values from checkpoint
        unique_styles, unique_characters = self.extract_unique_styles_and_chars()
        
        # Filter by glob pattern (for reference, but use all from checkpoint)
        logger.info(f"Found {len(unique_styles)} unique styles and {len(unique_characters)} unique characters")
        
        # Build coverage matrix
        coverage: dict[str, dict[str, bool]] = {}
        
        for char in sorted(unique_characters):
            coverage[char] = {}
            for style in sorted(unique_styles):
                # Check if record exists and both files are present
                has_valid_pair: bool = False
                
                for record in self.checkpoint_records:
                    if record.get("character") == char and record.get("style") == style:
                        content_path = record.get("content_image_path", "")
                        target_path = record.get("target_image_path", "")
                        
                        content_exists = (self.dataset_dir / content_path).exists()
                        target_exists = (self.dataset_dir / target_path).exists()
                        
                        has_valid_pair = content_exists and target_exists
                        break
                
                coverage[char][style] = has_valid_pair
        
        return coverage

    def print_coverage_summary(self, coverage: dict[str, dict[str, bool]]) -> None:
        """Print human-readable coverage summary.
        
        Args:
            coverage: Coverage matrix from checkpoint
        """
        styles: list[str] = sorted(self.stats["unique_styles"])
        
        logger.info("=" * 70)
        logger.info("COVERAGE SUMMARY (from checkpoint)")
        logger.info("=" * 70)
        
        # Per-style statistics
        logger.info(f"\n📊 Coverage by Style:\n")
        for style in styles:
            total_chars = len(coverage)
            covered_chars = sum(1 for char in coverage if coverage[char].get(style, False))
            coverage_pct = (covered_chars / total_chars * 100) if total_chars > 0 else 0
            logger.info(
                f"  {style:30s} {covered_chars:4d}/{total_chars:4d} ({coverage_pct:5.1f}%)"
            )
        
        # Per-character coverage
        logger.info(f"\n📊 Coverage by Character:\n")
        fully_covered = 0
        partially_covered = 0
        uncovered = 0
        
        for char in sorted(coverage.keys()):
            covered_styles = sum(1 for s in styles if coverage[char].get(s, False))
            total_styles = len(styles)
            
            if covered_styles == total_styles:
                fully_covered += 1
            elif covered_styles > 0:
                partially_covered += 1
            else:
                uncovered += 1
        
        logger.info(f"  Fully covered (all styles):    {fully_covered:4d}")
        logger.info(f"  Partially covered (some):       {partially_covered:4d}")
        logger.info(f"  Uncovered (no styles):         {uncovered:4d}")

    def print_issues(self) -> None:
        """Print all detected issues."""
        logger.info("=" * 70)
        logger.info("DETECTED ISSUES")
        logger.info("=" * 70)
        
        has_issues: bool = False
        
        # Records with missing fields
        if self.issues["checkpoint_records_missing_fields"]:
            has_issues = True
            logger.error(f"\n❌ {len(self.issues['checkpoint_records_missing_fields'])} checkpoint records have missing fields:")
            for issue in self.issues["checkpoint_records_missing_fields"][:5]:
                logger.error(f"  • Record {issue['record_index']}: missing {issue['missing_fields']}")
            if len(self.issues["checkpoint_records_missing_fields"]) > 5:
                logger.error(f"  ... and {len(self.issues['checkpoint_records_missing_fields']) - 5} more")
        
        # Missing content images
        if self.issues["missing_content_images"]:
            has_issues = True
            logger.warning(f"\n⚠ {len(self.issues['missing_content_images'])} missing content images:")
            for issue in self.issues["missing_content_images"][:10]:
                logger.warning(f"  • '{issue['character']}' / {issue['style']}: {issue['expected_path']}")
            if len(self.issues["missing_content_images"]) > 10:
                logger.warning(f"  ... and {len(self.issues['missing_content_images']) - 10} more")
        
        # Missing target images
        if self.issues["missing_target_images"]:
            has_issues = True
            logger.warning(f"\n⚠ {len(self.issues['missing_target_images'])} missing target images:")
            for issue in self.issues["missing_target_images"][:10]:
                logger.warning(f"  • '{issue['character']}' / {issue['style']}: {issue['expected_path']}")
            if len(self.issues["missing_target_images"]) > 10:
                logger.warning(f"  ... and {len(self.issues['missing_target_images']) - 10} more")
        
        # Extra files
        if self.issues["extra_content_images"]:
            has_issues = True
            logger.warning(f"\n⚠ {len(self.issues['extra_content_images'])} extra content images (not in checkpoint)")
        
        if self.issues["extra_target_images"]:
            has_issues = True
            logger.warning(f"\n⚠ {len(self.issues['extra_target_images'])} extra target images (not in checkpoint)")
        
        if not has_issues:
            logger.info("\n✅ No issues detected!")

    def generate_report(self, output_file: str | Path | None = None) -> dict:
        """Generate comprehensive diagnostic report.
        
        Args:
            output_file: Optional file to save JSON report
            
        Returns:
            Report dictionary
        """
        report: dict = {
            "dataset_path": str(self.dataset_dir),
            "checkpoint_file": str(self.checkpoint_path),
            "stats": {
                "total_checkpoint_records": self.stats["total_checkpoint_records"],
                "total_content_images_found": self.stats["total_content_images_found"],
                "total_target_images_found": self.stats["total_target_images_found"],
                "total_characters": self.stats["total_characters"],
                "total_styles": self.stats["total_styles"],
                "unique_characters": sorted(list(self.stats["unique_characters"])),
                "unique_styles": sorted(list(self.stats["unique_styles"])),
            },
            "issues": {
                "missing_content_images": self.issues["missing_content_images"],
                "missing_target_images": self.issues["missing_target_images"],
                "extra_content_images": self.issues["extra_content_images"],
                "extra_target_images": self.issues["extra_target_images"],
                "checkpoint_records_missing_fields": self.issues["checkpoint_records_missing_fields"],
            },
        }
        
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info(f"\n✓ Report saved to: {output_path}")
        
        return report

    def run_full_diagnosis(
        self, output_file: str | Path | None = None
    ) -> dict:
        """Run complete diagnosis pipeline using checkpoint as source of truth.
        
        Args:
            output_file: Optional file to save report
            
        Returns:
            Complete report
        """
        # Step 1: Validate structure
        if not self.validate_structure():
            logger.error("❌ Dataset structure invalid, cannot continue")
            return {}
        
        # Step 2: Load checkpoint (source of truth)
        if not self.load_checkpoint():
            logger.error("❌ Failed to load checkpoint, cannot continue")
            return {}
        
        # Step 3: Verify all checkpoint files exist on disk
        self.verify_checkpoint_files()
        
        # Step 4: Find extra files not in checkpoint
        self.find_extra_files()
        
        # Step 5: Diagnose coverage
        coverage = self.diagnose_coverage()
        
        # Step 6: Print summaries
        self.print_coverage_summary(coverage)
        self.print_issues()
        
        # Step 7: Generate report
        report = self.generate_report(output_file)
        
        logger.info("\n" + "=" * 70)
        logger.info("DIAGNOSIS COMPLETE")
        logger.info("=" * 70)
        
        return report


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Diagnose FontDiffusion dataset using checkpoint as source of truth"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to dataset directory (containing results_checkpoint.json, ContentImage/, TargetImage/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="diagnose_report.json",
        help="Path to save JSON report (default: diagnose_report.json)",
    )
    
    args = parser.parse_args()
    
    # Run diagnosis
    diagnostics = DatasetDiagnostics(args.dataset_dir)
    diagnostics.run_full_diagnosis(args.output)


if __name__ == "__main__":
    main()