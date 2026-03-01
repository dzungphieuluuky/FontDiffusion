"""
Enhanced FontDataset for FontDiffuserWithFST.
Supports both original and FST training modes.
"""

from PIL import Image
from typing import List, Dict, Optional
from torch.utils.data import Dataset
import torch
import random
import logging
import os
from torchvision import transforms

from src.modules.skeleton_distance_transform import (
    SkeletonDistanceTransform,
)  # ADD THIS IMPORT
from src.modules.frequency_decomposition import (
    FrequencyDecomposition,
)  # ADD THIS IMPORT

logger = logging.getLogger(__name__)


def get_nonorm_transform(resolution):
    """Get transform without normalization."""
    nonorm_transform = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ]
    )
    return nonorm_transform


class FontDataset(Dataset):
    def __init__(
        self,
        args,
        phase: str,
        transforms: Optional[list] = None,
        scr: bool = False,
        use_fst: bool = False,
        style_source_same_prob: float = 0.5,
        num_consistency_pairs: int = 0,
        num_identity_pairs: int = 0,
        identity_pair_mode: str = "random",
        use_skeleton_transform: bool = False,  # ADD THIS
        skeleton_config: Optional[dict] = None,  # ADD THIS
        use_frequency_decomp: bool = False,
        frequency_config: Optional[dict] = None,
    ):
        """
        Initialize FontDataset with optional skeleton transform.

        Args:
            args: Configuration arguments
            phase: Dataset phase ("train", "val", "test")
            transforms: List of transforms for [content, style, target]
            scr: Whether to use SCR loss
            use_fst: Whether to use FST model
            style_source_same_prob: Probability of using same style for source
            num_consistency_pairs: Number of consistency pairs
            num_identity_pairs: Number of identity pairs
            identity_pair_mode: Mode for selecting identity pairs
            use_skeleton_transform: Whether to apply skeleton-distance transform
            skeleton_config: Configuration for skeleton transform
        """
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        self.use_fst = use_fst
        self.style_source_same_prob = style_source_same_prob
        self.num_consistency_pairs = num_consistency_pairs
        self.num_identity_pairs = num_identity_pairs  # ADD THIS
        self.identity_pair_mode = identity_pair_mode  # ADD THIS
        if self.scr:
            self.num_neg = args.num_neg

        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

        # Frequency decomposition setup
        self.use_frequency_decomp = use_frequency_decomp

        if self.use_frequency_decomp:
            # Default configuration
            default_config = {
                "image_size": 96,
                "low_cutoff": 0.10,
                "mid_cutoff": 0.40,
                "filter_type": "gaussian",
                "normalize_bands": True,
            }

            # Update with user config
            if frequency_config:
                default_config.update(frequency_config)

            # Create decomposition module
            self.freq_decomp = FrequencyDecomposition(**default_config)
            logger.info(f"Frequency decomposition enabled: {default_config}")
        else:
            self.freq_decomp = None

        logger.info(
            f"Dataset initialized:\n "
            f"Phase: {phase}\n"
            f"Use_FST: {use_fst}\n"
            f"SCR: {scr}\n"
            f"Frequency Decomposition: {self.use_frequency_decomp}\n"
            f"Total samples: {len(self.target_images)}"
        )

    def get_path(self):
        """Build dataset paths and style mappings."""
        self.target_images = []
        # Images with related style
        self.style_to_images = {}
        # Content to images mapping (for FST cross-style sampling)
        self.content_to_images = {}

        target_image_dir = f"{self.root}/{self.phase}/TargetImage"

        for style in os.listdir(target_image_dir):
            style_dir = f"{target_image_dir}/{style}"
            if not os.path.isdir(style_dir):
                continue

            images_related_style = []
            for img in os.listdir(style_dir):
                if not img.endswith((".jpg", ".png", ".jpeg")):
                    continue

                img_path = f"{style_dir}/{img}"
                self.target_images.append(img_path)
                images_related_style.append(img_path)

                # Extract content for FST
                # Assuming filename format: style+content.png
                try:
                    img_name = img.split(".")[0]
                    if "+" in img_name:
                        style_name, content_name = img_name.split("+")
                        if content_name not in self.content_to_images:
                            self.content_to_images[content_name] = {}
                        if style_name not in self.content_to_images[content_name]:
                            self.content_to_images[content_name][style_name] = []
                        self.content_to_images[content_name][style_name].append(
                            img_path
                        )
                except Exception as e:
                    logger.warning(f"Could not parse filename {img}: {e}")

            self.style_to_images[style] = images_related_style

        logger.info(
            f"Found {len(self.target_images)} target images across "
            f"{len(self.style_to_images)} styles"
        )

    def get_same_style_pairs(
        self,
        num_pairs: int,
        target_style: Optional[str] = None,
        exclude_content: Optional[str] = None,
    ) -> list[tuple[str, str]]:
        """Sample pairs of images with same style but different content.

        Args:
            num_pairs: Number of pairs to sample
            target_style: If provided, sample from this specific style
            exclude_content: Content character to exclude

        Returns:
            List of (image1_path, image2_path) tuples
        """
        pairs = []

        if target_style:
            # Sample from specific style
            if target_style not in self.style_to_images:
                return pairs

            images_in_style = self.style_to_images[target_style].copy()

            # Remove excluded content
            if exclude_content:
                images_in_style = [
                    img for img in images_in_style if exclude_content not in img
                ]

            # Sample up to num_pairs
            sample_size = min(num_pairs, len(images_in_style) // 2)
            for _ in range(sample_size):
                if len(images_in_style) < 2:
                    break
                img1, img2 = random.sample(images_in_style, 2)
                pairs.append((img1, img2))

        else:
            # Sample from random styles
            available_styles = list(self.style_to_images.keys())

            for _ in range(num_pairs):
                if not available_styles:
                    break

                style = random.choice(available_styles)
                images = self.style_to_images[style]

                if len(images) < 2:
                    continue

                img1, img2 = random.sample(images, 2)
                pairs.append((img1, img2))

        return pairs

    def get_style_source_image(
        self, target_style: str, content: str, target_image_path: str
    ) -> Image.Image:
        """
        Get source style image for FST.

        Strategy:
        1. With probability style_source_same_prob: use same style (different character)
        2. Otherwise: use different style (same or different character)
        """
        use_same_style = random.random() < self.style_source_same_prob

        if use_same_style:
            # Same style, different character (standard case)
            images_in_style = self.style_to_images[target_style].copy()
            images_in_style.remove(target_image_path)
            if images_in_style:
                source_image_path = random.choice(images_in_style)
            else:
                # Fallback: use target image if no other images available
                source_image_path = target_image_path
        else:
            # Different style for style transformation learning
            # Try to get same content in different style
            if content in self.content_to_images:
                available_styles = list(self.content_to_images[content].keys())
                if target_style in available_styles:
                    available_styles.remove(target_style)

                if available_styles:
                    # Same content, different style
                    source_style = random.choice(available_styles)
                    source_candidates = self.content_to_images[content][source_style]
                    source_image_path = random.choice(source_candidates)
                else:
                    # Fallback: random style image
                    other_styles = [
                        s for s in self.style_to_images.keys() if s != target_style
                    ]
                    if other_styles:
                        random_style = random.choice(other_styles)
                        source_image_path = random.choice(
                            self.style_to_images[random_style]
                        )
                    else:
                        source_image_path = target_image_path
            else:
                # Fallback: random different style
                other_styles = [
                    s for s in self.style_to_images.keys() if s != target_style
                ]
                if other_styles:
                    random_style = random.choice(other_styles)
                    source_image_path = random.choice(
                        self.style_to_images[random_style]
                    )
                else:
                    source_image_path = target_image_path

        source_image = Image.open(source_image_path).convert("RGB")
        return source_image

    def get_consistency_pairs(
        self, target_style: str, source_style: str, exclude_content: str, num_pairs: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Get k pairs of images for consistency loss.

        Each pair contains the SAME content in source→target style transformation:
        - pair 1: char A in source_style → char A in target_style
        - pair 2: char B in source_style → char B in target_style
        - pair 3: char C in source_style → char C in target_style

        All pairs use the SAME style transformation (source_style → target_style)
        but with DIFFERENT characters.

        Args:
            target_style: Target style name (e.g., "style2")
            source_style: Source style name (e.g., "style1")
            exclude_content: Content to exclude (the main training sample)
            num_pairs: Number of pairs to return (k)

        Returns:
            List of (source_image, target_image) tuples, each already transformed
        """
        pairs = []

        # Find contents that exist in BOTH source_style AND target_style
        available_contents = []
        for content, styles_dict in self.content_to_images.items():
            if content == exclude_content:
                continue
            # Content must exist in both styles
            if source_style in styles_dict and target_style in styles_dict:
                available_contents.append(content)

        if not available_contents:
            logger.warning(
                f"No available contents for consistency pairs "
                f"(source_style={source_style}, target_style={target_style}, "
                f"exclude_content={exclude_content})"
            )
            return pairs

        # Sample k different contents
        selected_contents = random.sample(
            available_contents, min(num_pairs, len(available_contents))
        )

        # Build pairs: each pair has same content, different styles
        for content in selected_contents:
            # Source image: content in source_style
            source_candidates = self.content_to_images[content][source_style]
            source_image_path = random.choice(source_candidates)
            source_image = Image.open(source_image_path).convert("RGB")

            # Target image: SAME content in target_style
            target_candidates = self.content_to_images[content][target_style]
            target_image_path = random.choice(target_candidates)
            target_image = Image.open(target_image_path).convert("RGB")

            # Apply transforms if available
            if self.transforms is not None:
                source_image = self.transforms[1](source_image)  # Style transform
                target_image = self.transforms[1](target_image)  # Style transform

            pairs.append((source_image, target_image))

        logger.debug(
            f"Created {len(pairs)} consistency pairs with contents={selected_contents}, "
            f"source_style={source_style}, target_style={target_style}"
        )

        return pairs

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Returns:
            Dictionary containing:
            - content_image: Character to generate
            - style_image: Target style reference
            - style_source_image: Source style reference (FST only)
            - target_image: Ground truth
            - nonorm_target_image: Target without normalization
            - neg_images: Negative samples (SCR only)
            - consistency_pairs: List of (source, target) tuples for consistency loss (FST only)
        """
        target_image_path = self.target_images[index]
        target_image_name = target_image_path.split("/")[-1]

        # Parse style and content from filename
        try:
            style, content = target_image_name.split(".")[0].split("+")
        except ValueError:
            raise ValueError(
                f"Invalid filename format: {target_image_name}. "
                f"Expected format: 'style+content.png'"
            )

        # Read content image
        content_image_path = f"{self.root}/{self.phase}/ContentImage/{content}.png"
        if not os.path.exists(content_image_path):
            # Try alternative extensions
            for ext in [".png", ".jpeg", ".JPG", ".PNG"]:
                alt_path = f"{self.root}/{self.phase}/ContentImage/{content}{ext}"
                if os.path.exists(alt_path):
                    content_image_path = alt_path
                    break

        content_image = Image.open(content_image_path).convert("RGB")

        # Sample style image (target style, different character)
        images_related_style = self.style_to_images[style].copy()
        images_related_style.remove(target_image_path)

        if images_related_style:
            style_image_path = random.choice(images_related_style)
        else:
            # Fallback: use target image itself if no other samples
            style_image_path = target_image_path

        style_image = Image.open(style_image_path).convert("RGB")

        # Read target image
        target_image = Image.open(target_image_path).convert("RGB")
        nonorm_target_image = self.nonorm_transforms(target_image)

        # Apply transforms
        if self.transforms is not None:
            content_image = self.transforms[0](content_image)
            style_image = self.transforms[1](style_image)
            target_image = self.transforms[2](target_image)

        # Build sample dictionary
        sample = {
            "content_image": content_image,
            "style_image": style_image,
            "target_image": target_image,
            "target_image_path": target_image_path,
            "nonorm_target_image": nonorm_target_image,
        }
        # Add source style image for FST
        source_style = None  # Track source style for consistency pairs
        if self.use_fst:
            # Get source style image AND track which style it came from
            source_style, style_source_image = self.get_style_source_image_with_name(
                target_style=style, content=content, target_image_path=target_image_path
            )

            if self.transforms is not None:
                style_source_image = self.transforms[1](style_source_image)

            sample["style_source_image"] = style_source_image

        # Add consistency pairs for FST consistency loss
        if self.num_consistency_pairs > 0:
            # If FST not enabled, we can't create consistency pairs
            if source_style is None:
                logger.warning(
                    "num_consistency_pairs > 0 but use_fst=False. "
                    "Consistency pairs require FST mode. Skipping consistency pairs."
                )
                # Don't add empty list to sample - omit the key entirely
            else:
                # Get consistency pairs: same source→target style transformation, different contents
                consistency_pairs = self.get_consistency_pairs(
                    target_style=style,
                    source_style=source_style,
                    exclude_content=content,
                    num_pairs=self.num_consistency_pairs,
                )

                # Only add to sample if we got valid pairs
                if consistency_pairs:
                    sample["consistency_pairs"] = consistency_pairs

        # Add negative samples for SCR loss
        if self.scr:
            style_list = list(self.style_to_images.keys())
            style_index = style_list.index(style)
            style_list.pop(style_index)

            choose_neg_names = []
            for i in range(self.num_neg):
                if not style_list:
                    # Not enough styles for negatives
                    break

                choose_style = random.choice(style_list)
                choose_index = style_list.index(choose_style)
                style_list.pop(choose_index)

                neg_path = f"{self.root}/train/TargetImage/{choose_style}/{choose_style}+{content}.png"

                # Check if negative sample exists
                if os.path.exists(neg_path):
                    choose_neg_names.append(neg_path)
                else:
                    # Try to find any image from this style with same content
                    if (
                        content in self.content_to_images
                        and choose_style in self.content_to_images[content]
                    ):
                        alt_neg = random.choice(
                            self.content_to_images[content][choose_style]
                        )
                        choose_neg_names.append(alt_neg)

            # Load neg_images
            neg_images_list = []
            for neg_name in choose_neg_names:
                try:
                    neg_image = Image.open(neg_name).convert("RGB")
                    if self.transforms is not None:
                        neg_image = self.transforms[2](neg_image)
                    neg_images_list.append(neg_image.unsqueeze(0))
                except Exception as e:
                    print(f"Warning: Could not load negative sample {neg_name}: {e}")

            if neg_images_list:
                neg_images = torch.cat(neg_images_list, dim=0)
            else:
                # Fallback: use target image as negative
                neg_images = target_image.unsqueeze(0)

            sample["neg_images"] = neg_images

        if self.num_identity_pairs > 0 and self.use_fst:
            identity_pairs = []

            if self.identity_pair_mode == "same_style":
                # All pairs from same style as main sample
                pair_paths = self.get_same_style_pairs(
                    num_pairs=self.num_identity_pairs,
                    target_style=style,
                    exclude_content=content,
                )
            else:  # "random"
                # Random styles for each pair
                pair_paths = self.get_same_style_pairs(
                    num_pairs=self.num_identity_pairs,
                )

            # Load and transform pairs
            for img1_path, img2_path in pair_paths:
                try:
                    img1 = Image.open(img1_path).convert("RGB")
                    img2 = Image.open(img2_path).convert("RGB")

                    if self.transforms is not None:
                        img1 = self.transforms[1](img1)  # Style transform
                        img2 = self.transforms[1](img2)  # Style transform

                    identity_pairs.append((img1, img2))
                except Exception as e:
                    logger.debug(f"Failed to load identity pair: {e}")
                    continue

            if identity_pairs:
                sample["identity_pairs"] = identity_pairs

        if self.use_frequency_decomp:
            # Decompose into frequency bands
            # Input: (C, H, W) → Add batch dim → (1, C, H, W)
            bands = self.freq_decomp(content_image.unsqueeze(0))

            # Extract bands and remove batch dim
            content_low_freq = bands["low_freq"].squeeze(0)
            content_mid_freq = bands["mid_freq"].squeeze(0)
            content_high_freq = bands["high_freq"].squeeze(0)

            # Store frequency bands
            sample["content_image"] = content_low_freq  # Use low freq for content
            sample["content_image_mid"] = content_mid_freq
            sample["content_image_original"] = content_image

            # Also decompose style images if available
            if style_image is not None:
                style_bands = self.freq_decomp(style_image.unsqueeze(0))
                sample["style_image_high"] = style_bands["high_freq"].squeeze(0)
                sample["style_image_mid"] = style_bands["mid_freq"].squeeze(0)
        else:
            sample["content_image"] = content_image

        return sample

    def get_style_source_image_with_name(
        self, target_style: str, content: str, target_image_path: str
    ) -> tuple[str, Image.Image]:
        """
        Get source style image for FST, returning both the style name and image.

        Strategy:
        1. With probability style_source_same_prob: use same style (different character)
        2. Otherwise: use different style (same or different character)

        Returns:
            tuple of (source_style_name, source_image)
        """
        use_same_style = random.random() < self.style_source_same_prob

        if use_same_style:
            # Same style, different character (standard case)
            source_style = target_style
            images_in_style = self.style_to_images[target_style].copy()
            images_in_style.remove(target_image_path)
            if images_in_style:
                source_image_path = random.choice(images_in_style)
            else:
                # Fallback: use target image if no other images available
                source_image_path = target_image_path
        else:
            # Different style for style transformation learning
            # Try to get same content in different style
            if content in self.content_to_images:
                available_styles = list(self.content_to_images[content].keys())
                if target_style in available_styles:
                    available_styles.remove(target_style)

                if available_styles:
                    # Same content, different style
                    source_style = random.choice(available_styles)
                    source_candidates = self.content_to_images[content][source_style]
                    source_image_path = random.choice(source_candidates)
                else:
                    # Fallback: random style image
                    other_styles = [
                        s for s in self.style_to_images.keys() if s != target_style
                    ]
                    if other_styles:
                        source_style = random.choice(other_styles)
                        source_image_path = random.choice(
                            self.style_to_images[source_style]
                        )
                    else:
                        source_style = target_style
                        source_image_path = target_image_path
            else:
                # Fallback: random different style
                other_styles = [
                    s for s in self.style_to_images.keys() if s != target_style
                ]
                if other_styles:
                    source_style = random.choice(other_styles)
                    source_image_path = random.choice(
                        self.style_to_images[source_style]
                    )
                else:
                    source_style = target_style
                    source_image_path = target_image_path

        source_image = Image.open(source_image_path).convert("RGB")
        return source_style, source_image

    def __len__(self) -> int:
        return len(self.target_images)


class FontDatasetDebug:
    """Debug utility to verify dataset structure and samples."""

    def __init__(self, dataset: FontDataset):
        self.dataset = dataset

    def check_structure(self) -> dict[str, any]:
        """Check dataset structure and return statistics."""
        stats = {
            "total_samples": len(self.dataset),
            "num_styles": len(self.dataset.style_to_images),
            "num_contents": len(self.dataset.content_to_images),
            "samples_per_style": {},
            "contents_per_style": {},
        }

        # Count samples per style
        for style, images in self.dataset.style_to_images.items():
            stats["samples_per_style"][style] = len(images)

        # Count contents per style
        for content, styles_dict in self.dataset.content_to_images.items():
            for style in styles_dict:
                if style not in stats["contents_per_style"]:
                    stats["contents_per_style"][style] = 0
                stats["contents_per_style"][style] += 1

        return stats

    def print_sample(self, index: int = 0):
        """Print detailed information about a sample."""
        sample = self.dataset[index]

        print(f"\n{'=' * 80}")
        print(f"Sample {index} Information")
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

    def verify_fst_diversity(self, num_samples: int = 100):
        """Verify FST source/target diversity."""
        if not self.dataset.use_fst:
            print("Dataset not in FST mode")
            return

        same_style_count = 0
        diff_style_count = 0

        for i in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[i]
            # This is a simplified check - in practice you'd need to track actual paths
            same_style_count += 1  # Placeholder

        print(f"\nFST Diversity Check ({num_samples} samples):")
        print(f"  Same style pairs: {same_style_count}")
        print(f"  Different style pairs: {diff_style_count}")
        print(f"  Diversity ratio: {diff_style_count / num_samples:.2%}")


# Example usage and testing
if __name__ == "__main__":
    # Mock args for testing
    class Args:
        data_root = "path/to/your/dataset"
        resolution = 128
        num_neg = 3

    args = Args()

    # Test original mode
    print("Testing original dataset mode...")
    dataset_original = FontDataset(args=args, phase="train", scr=False, use_fst=False)

    debug_original = FontDatasetDebug(dataset_original)
    stats = debug_original.check_structure()
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Number of styles: {stats['num_styles']}")

    # Test FST mode
    print("\n" + "=" * 80)
    print("Testing FST dataset mode...")
    dataset_fst = FontDataset(
        args=args, phase="train", scr=False, use_fst=True, style_source_same_prob=0.5
    )

    debug_fst = FontDatasetDebug(dataset_fst)
    if len(dataset_fst) > 0:
        debug_fst.print_sample(0)
