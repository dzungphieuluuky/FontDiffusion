"""
HuggingFace Arrow-based FontDataset for FontDiffuserWithFST.
Operates on memory-mapped Arrow tables instead of filesystem I/O.
Maintains identical interface to original FontDataset for trainer compatibility.
"""

import logging
import random
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset, load_dataset, load_from_disk
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


def get_nonorm_transform(resolution: int):
    """Get transform without normalization."""
    return transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )


class HFFontDataset(TorchDataset):
    """
    Arrow-based font dataset wrapping HuggingFace Dataset.

    Maintains identical interface to FontDataset but operates on Arrow tables
    instead of filesystem. Supports FST and SCR modes.

    Args:
        args: Arguments containing resolution, num_neg, etc.
        phase: 'train' or 'test' (used for logging only in HF mode)
        transforms: List of [content_transform, style_transform, target_transform]
        scr: Whether to use SCR loss (loads negative samples)
        use_fst: Whether to use FST mode (loads source style images)
        style_source_same_prob: Probability of using same style for source/target
        repo_id: HuggingFace repository ID (e.g., 'username/dataset-name')
        local_dataset_path: Local path to Arrow dataset (alternative to repo_id)
        split: Dataset split name (default: 'train')
        config_name: Dataset configuration name
        token: HuggingFace API token for private datasets
    """

    def __init__(
        self,
        args,
        phase: str,
        transforms: Optional[list] = None,
        scr: bool = False,
        use_fst: bool = False,
        style_source_same_prob: float = 0.5,
        repo_id: Optional[str] = None,
        local_dataset_path: Optional[str | Path] = None,
        split: str = "train",
        config_name: Optional[str] = None,
        token: Optional[str] = None,
    ):
        super().__init__()

        self.phase = phase
        self.scr = scr
        self.use_fst = use_fst
        self.style_source_same_prob = style_source_same_prob
        self.resolution = args.resolution

        if self.scr:
            self.num_neg = args.num_neg

        # Load HuggingFace dataset
        self.dataset = self._load_dataset(
            repo_id=repo_id,
            local_dataset_path=local_dataset_path,
            split=split,
            config_name=config_name,
            token=token,
        )

        # Build internal indices for fast sampling
        self._build_indices()

        # Setup transforms
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

        logger.info(
            f"HFFontDataset initialized:\n"
            f"  Phase: {phase}\n"
            f"  Use_FST: {use_fst}\n"
            f"  SCR: {scr}\n"
            f"  Total samples: {len(self.dataset)}\n"
            f"  Unique styles: {len(self.style_to_indices)}\n"
            f"  Unique characters: {len(self.content_to_indices)}"
        )

    def _load_dataset(
        self,
        repo_id: Optional[str],
        local_dataset_path: Optional[str | Path],
        split: str,
        config_name: Optional[str],
        token: Optional[str],
    ) -> Dataset:
        """Load dataset from Hub or local disk."""
        dataset = None

        if local_dataset_path:
            logger.info(f"Loading dataset from disk: {local_dataset_path}")
            try:
                # load_from_disk might return Dataset or DatasetDict
                dataset = load_from_disk(str(local_dataset_path))
            except Exception as e:
                raise ValueError(f"Failed to load local dataset: {e}") from e

        elif repo_id:
            config_msg = f" (config: {config_name})" if config_name else ""
            logger.info(f"Loading from Hub: {repo_id} (split: {split}){config_msg}")
            try:
                # load_dataset usually handles splits via argument,
                # but explicit handling is safer for edge cases
                dataset = load_dataset(
                    repo_id,
                    name=config_name,
                    split=split,
                    token=token,
                )
            except Exception as e:
                raise ValueError(f"Failed to load from Hub: {e}") from e
        else:
            raise ValueError("Must provide either repo_id or local_dataset_path")

        # Handle DatasetDict (if 'split' wasn't applied or load_from_disk returned dict)
        if isinstance(dataset, dict) and not isinstance(dataset, Dataset):
            if split in dataset:
                logger.info(f"Selected split '{split}' from DatasetDict")
                return dataset[split]
            else:
                # Fallback: if 'train' requested but not found, try 'default' or first key
                keys = list(dataset.keys())
                logger.warning(
                    f"Split '{split}' not found. Available: {keys}. Using '{keys[0]}'"
                )
                return dataset[keys[0]]

        return dataset

    def _build_indices(self) -> None:
        """
        Build internal index mappings for fast sampling.
        Optimized to read ONLY metadata columns, avoiding image decoding.
        """
        logger.info("Building internal indices for fast sampling...")

        self.style_to_indices = {}
        self.content_to_indices = {}

        # OPTIMIZATION: Select only metadata columns to avoid loading images
        # This makes iteration 100x-1000x faster
        metadata_only = self.dataset.select_columns(["character", "style"])

        for idx, item in enumerate(metadata_only):
            character = item["character"]
            style = item["style"]

            # Build style_to_indices
            if style not in self.style_to_indices:
                self.style_to_indices[style] = []
            self.style_to_indices[style].append(idx)

            # Build content_to_indices
            if character not in self.content_to_indices:
                self.content_to_indices[character] = {}
            if style not in self.content_to_indices[character]:
                self.content_to_indices[character][style] = []
            self.content_to_indices[character][style].append(idx)

        logger.info(
            f"Built indices: {len(self.style_to_indices)} styles, "
            f"{len(self.content_to_indices)} characters"
        )

    def _get_pil_image(self, sample: dict, key: str) -> Image.Image:
        """
        Extract PIL Image from dataset sample.

        HuggingFace datasets auto-decode images to PIL when using Image feature.
        """
        img = sample.get(key)

        if isinstance(img, Image.Image):
            return img.convert("RGB")

        # Fallback: if bytes, decode manually
        if isinstance(img, bytes):
            from io import BytesIO

            return Image.open(BytesIO(img)).convert("RGB")

        raise ValueError(f"Unexpected image type for {key}: {type(img)}")

    def _sample_style_image(self, current_idx: int, style: str) -> Image.Image:
        """
        Sample a different image from the same style.

        Original logic: pick random image from same style, excluding current.
        """
        style_indices = self.style_to_indices[style].copy()

        # Remove current index
        if current_idx in style_indices:
            style_indices.remove(current_idx)

        if not style_indices:
            # Fallback: use current image if no other samples
            return self._get_pil_image(self.dataset[current_idx], "target_image")

        # Sample random different image
        sampled_idx = random.choice(style_indices)
        return self._get_pil_image(self.dataset[sampled_idx], "target_image")

    def _get_style_source_image(
        self,
        current_idx: int,
        target_style: str,
        content: str,
    ) -> Image.Image:
        """
        Get source style image for FST.

        Strategy:
        1. With probability style_source_same_prob: same style, different character
        2. Otherwise: different style (for style transformation learning)
        """
        use_same_style = random.random() < self.style_source_same_prob

        if use_same_style:
            # Same style, different character
            style_indices = self.style_to_indices[target_style].copy()
            if current_idx in style_indices:
                style_indices.remove(current_idx)

            if style_indices:
                source_idx = random.choice(style_indices)
            else:
                source_idx = current_idx

        else:
            # Different style for style transformation
            # Try same content, different style first
            if content in self.content_to_indices:
                available_styles = list(self.content_to_indices[content].keys())
                if target_style in available_styles:
                    available_styles.remove(target_style)

                if available_styles:
                    # Same content, different style
                    source_style = random.choice(available_styles)
                    source_candidates = self.content_to_indices[content][source_style]
                    source_idx = random.choice(source_candidates)
                else:
                    # Fallback: random different style
                    source_idx = self._sample_random_different_style(target_style)
            else:
                # Fallback: random different style
                source_idx = self._sample_random_different_style(target_style)

        return self._get_pil_image(self.dataset[source_idx], "target_image")

    def _sample_random_different_style(self, exclude_style: str) -> int:
        """Sample random index from a different style."""
        other_styles = [s for s in self.style_to_indices.keys() if s != exclude_style]

        if not other_styles:
            # Fallback: use any style
            other_styles = list(self.style_to_indices.keys())

        random_style = random.choice(other_styles)
        return random.choice(self.style_to_indices[random_style])

    def _get_negative_samples(self, content: str, target_style: str) -> torch.Tensor:
        """
        Get negative samples for SCR loss.

        Original logic: same character, different styles.
        """
        if content not in self.content_to_indices:
            # Fallback: return empty tensor
            logger.warning(
                f"No content mapping for {content}, returning empty negatives"
            )
            return torch.empty(0, 3, self.resolution, self.resolution)

        available_styles = list(self.content_to_indices[content].keys())
        if target_style in available_styles:
            available_styles.remove(target_style)

        if not available_styles:
            # Fallback: return empty tensor
            return torch.empty(0, 3, self.resolution, self.resolution)

        # Sample up to num_neg styles
        num_to_sample = min(self.num_neg, len(available_styles))
        sampled_styles = random.sample(available_styles, num_to_sample)

        neg_images_list = []
        for neg_style in sampled_styles:
            # Get random sample from this style with same content
            neg_candidates = self.content_to_indices[content][neg_style]
            neg_idx = random.choice(neg_candidates)

            try:
                neg_img = self._get_pil_image(self.dataset[neg_idx], "target_image")

                if self.transforms is not None:
                    neg_img = self.transforms[2](neg_img)

                neg_images_list.append(neg_img.unsqueeze(0))

            except Exception as e:
                logger.debug(f"Failed to load negative sample at index {neg_idx}: {e}")

        if neg_images_list:
            return torch.cat(neg_images_list, dim=0)

        # Fallback: return empty tensor
        return torch.empty(0, 3, self.resolution, self.resolution)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns dictionary with identical keys to FontDataset:
        - content_image: Character to generate
        - style_image: Target style reference
        - style_source_image: Source style reference (FST only)
        - target_image: Ground truth
        - nonorm_target_image: Target without normalization
        - neg_images: Negative samples (SCR only)
        - target_image_path: Synthesized path for logging
        """
        # Get sample from Arrow table
        sample = self.dataset[index]

        character = sample["character"]
        style = sample["style"]
        font = sample.get("font", "unknown")

        # Load images from Arrow format (auto-decoded to PIL)
        content_image = self._get_pil_image(sample, "content_image")
        target_image = self._get_pil_image(sample, "target_image")

        # Sample style image (different character, same style)
        style_image = self._sample_style_image(index, style)

        # Create nonorm target before transforms
        nonorm_target_image = self.nonorm_transforms(target_image)

        # Apply transforms
        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

        # Build sample dictionary
        result = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "nonorm_target_image": nonorm_target_image,
            "target_image_path": f"{style}/{style}+{character}.png",  # Synthesized
        }

        # Add style_source_image for FST
        if self.use_fst:
            style_source_image = self._get_style_source_image(
                current_idx=index,
                target_style=style,
                content=character,
            )

            if self.transforms is not None:
                style_source_image = self.transforms[1](style_source_image)

            result["style_source_image"] = style_source_image

        # Add negative samples for SCR
        if self.scr:
            neg_images = self._get_negative_samples(character, style)
            result["neg_images"] = neg_images

        return result

    def __len__(self) -> int:
        return len(self.dataset)


class HFFontDatasetDebug:
    """Debug utility to verify HF dataset structure and samples."""

    def __init__(self, dataset: HFFontDataset):
        self.dataset = dataset

    def check_structure(self) -> dict[str, any]:
        """Check dataset structure and return statistics."""
        stats = {
            "total_samples": len(self.dataset),
            "num_styles": len(self.dataset.style_to_indices),
            "num_contents": len(self.dataset.content_to_indices),
            "samples_per_style": {},
            "contents_per_style": {},
        }

        # Count samples per style
        for style, indices in self.dataset.style_to_indices.items():
            stats["samples_per_style"][style] = len(indices)

        # Count contents per style
        for content, styles_dict in self.dataset.content_to_indices.items():
            for style in styles_dict:
                if style not in stats["contents_per_style"]:
                    stats["contents_per_style"][style] = 0
                stats["contents_per_style"][style] += 1

        return stats

    def print_sample(self, index: int = 0):
        """Print detailed information about a sample."""
        sample = self.dataset[index]

        print(f"\n{'=' * 80}")
        print(f"HFFontDataset Sample {index} Information")
        print(f"{'=' * 80}")

        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(
                    f"{key:25s}: shape={tuple(value.shape)}, "
                    f"dtype={value.dtype}, "
                    f"range=[{value.min():.3f}, {value.max():.3f}]"
                )
            else:
                print(f"{key:25s}: {value}")

        print(f"{'=' * 80}\n")

    def verify_index_integrity(self) -> bool:
        """Verify internal index mappings are consistent."""
        print("\nVerifying index integrity...")

        issues = []

        # Check all indices are valid
        for style, indices in self.dataset.style_to_indices.items():
            for idx in indices:
                if idx >= len(self.dataset.dataset):
                    issues.append(f"Invalid index {idx} in style {style}")

        for char, styles_dict in self.dataset.content_to_indices.items():
            for style, indices in styles_dict.items():
                for idx in indices:
                    if idx >= len(self.dataset.dataset):
                        issues.append(
                            f"Invalid index {idx} in content {char}, style {style}"
                        )

        if issues:
            print("❌ Index integrity issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        print("✅ Index integrity verified")
        return True

    def compare_with_original(self, original_dataset_path: str):
        """Compare statistics with original filesystem-based dataset."""
        from .font_dataset_fst import FontDataset

        # Mock args
        class Args:
            data_root = original_dataset_path
            resolution = self.dataset.resolution
            num_neg = 3

        args = Args()

        print("\nComparing with original filesystem dataset...")

        try:
            original_dataset = FontDataset(
                args=args,
                phase=self.dataset.phase,
                scr=False,
                use_fst=False,
            )

            hf_stats = self.check_structure()

            print(f"\nOriginal Dataset:")
            print(f"  Total samples: {len(original_dataset)}")
            print(f"  Unique styles: {len(original_dataset.style_to_images)}")

            print(f"\nHF Arrow Dataset:")
            print(f"  Total samples: {hf_stats['total_samples']}")
            print(f"  Unique styles: {hf_stats['num_styles']}")

            # Check if counts match
            if len(original_dataset) == hf_stats["total_samples"]:
                print("✅ Sample counts match")
            else:
                print("❌ Sample counts differ")

            if len(original_dataset.style_to_images) == hf_stats["num_styles"]:
                print("✅ Style counts match")
            else:
                print("❌ Style counts differ")

        except Exception as e:
            print(f"⚠️ Could not load original dataset for comparison: {e}")


# Example usage and testing
if __name__ == "__main__":
    import argparse

    # Mock args for testing
    class Args:
        resolution = 128
        num_neg = 3

    args = Args()

    # Test HF dataset from local path
    print("Testing HFFontDataset from local path...")
    print("=" * 80)

    dataset_hf = HFFontDataset(
        args=args,
        phase="train",
        scr=False,
        use_fst=True,
        style_source_same_prob=0.5,
        local_dataset_path="path/to/your/hf_dataset",  # Replace with actual path
    )

    debug = HFFontDatasetDebug(dataset_hf)

    # Check structure
    stats = debug.check_structure()
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Number of styles: {stats['num_styles']}")
    print(f"  Number of characters: {stats['num_contents']}")

    # Verify indices
    debug.verify_index_integrity()

    # Print sample
    if len(dataset_hf) > 0:
        debug.print_sample(0)

    print("\n" + "=" * 80)
    print("Testing completed!")
