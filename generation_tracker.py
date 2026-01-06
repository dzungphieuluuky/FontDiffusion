import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Set
import logging
from logging_utils import setup_logging
from filename_utils import compute_file_hash

logger = setup_logging(level=logging.INFO, name="GenerationTracker")
class GenerationTracker:
    """
    ✅ Tracks which (character, style, font) combinations have been generated
    Uses hash-based checking for fast lookups
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        """
        Initialize generation tracker

        Args:
            checkpoint_path: Path to results_checkpoint.json file
        """
        self.generated_hashes: Set[str] = set()
        self.generations: List[Dict[str, Any]] = []

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_from_checkpoint(checkpoint_path)

    def _load_from_checkpoint(self, checkpoint_path: str) -> None:
        """Load existing generations from checkpoint"""
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                results = json.load(f)

            raw_generations = results.get("generations", [])

            # ✅ Track duplicates
            seen_hashes: Set[str] = set()
            unique_generations: List[Dict[str, Any]] = []
            duplicate_count: int = 0

            # Build hash set for fast lookup and deduplicate
            for gen in raw_generations:
                target_hash = gen.get("target_hash")

                if not target_hash:
                    # Compute hash if not in checkpoint
                    char = gen.get("character", "")
                    style = gen.get("style", "")
                    font = gen.get("font", "")

                    # Skip invalid entries
                    if not char or not style:
                        continue

                    target_hash = compute_file_hash(char, style, font)

                # ✅ Check for duplicates
                if target_hash in seen_hashes:
                    duplicate_count += 1
                    continue  # Skip duplicate

                # Add to collections
                seen_hashes.add(target_hash)
                self.generated_hashes.add(target_hash)
                unique_generations.append(gen)

            # ✅ Store only unique generations
            self.generations = unique_generations

            logger.info(
                f"✓ Loaded checkpoint: {len(self.generations)} unique generations"
            )
            if duplicate_count > 0:
                logger.info(f"  ⚠️  Removed {duplicate_count} duplicate entries")
            logger.info(f"  Total raw entries: {len(raw_generations)}")

        except Exception as e:
            logger.info(f"⚠ Error loading checkpoint: {e}")
            import traceback

            traceback.print_exc()

    def is_generated(self, char: str, style: str, font: str = "") -> bool:
        """Check if (char, style, font) combination has been generated"""
        target_hash = compute_file_hash(char, style, font)
        return target_hash in self.generated_hashes

    def mark_generated(self, char: str, style: str, font: str = "") -> None:
        """Mark a (char, style, font) combination as generated"""
        target_hash = compute_file_hash(char, style, font)
        self.generated_hashes.add(target_hash)

    def add_generation(self, generation: Dict[str, Any]) -> None:
        """Add a generation record"""
        self.generations.append(generation)

        # Also add to hash set
        char = generation.get("character", "")
        style = generation.get("style", "")
        font = generation.get("font", "")
        self.mark_generated(char, style, font)
