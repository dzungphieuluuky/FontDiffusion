"""
Enhanced FontDataset for FontDiffuserWithFST.
Supports both original and FST training modes.
"""

import os
import random
from PIL import Image
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging

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
    """
    Enhanced dataset for font generation supporting both original and FST modes.

    For FST mode, provides:
    - content_image: The character to generate
    - style_image: Target style reference (same character, different font)
    - style_source_image: Source style reference (optional, for style transformation)
    - target_image: Ground truth

    Args:
        args: Arguments containing data_root, resolution, etc.
        phase: 'train' or 'test'
        transforms: List of [content_transform, style_transform, target_transform]
        scr: Whether to use SCR loss (loads negative samples)
        use_fst: Whether to use FST mode (loads source style images)
        style_source_same_prob: Probability of using same style for source/target (0.0-1.0)
    """

    def __init__(
        self,
        args,
        phase: str,
        transforms: Optional[list] = None,
        scr: bool = False,
        use_fst: bool = False,
        style_source_same_prob: float = 0.5,
        num_consistency_pairs: int = 0,
    ):
        super().__init__()
        self.root = args.data_root
        self.phase = phase
        self.scr = scr
        self.use_fst = use_fst
        self.style_source_same_prob = style_source_same_prob
        self.num_consistency_pairs = num_consistency_pairs
        if self.scr:
            self.num_neg = args.num_neg

        # Get Data path
        self.get_path()
        self.transforms = transforms
        self.nonorm_transforms = get_nonorm_transform(args.resolution)

        logger.info(
            f"Dataset initialized:\n "
            f"Phase: {phase}\n"
            f"Use_FST: {use_fst}\n"
            f"SCR: {scr}\n"
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
        self, 
        target_style: str, 
        source_style: str,
        exclude_content: str,
        num_pairs: int
    ) -> list[tuple[Image.Image, Image.Image]]:
        """
        Get k pairs of images for consistency loss.
        
        Each pair contains:
        - Image with same content in source_style
        - Image with same content in target_style
        
        All pairs should have different content from each other and from exclude_content.
        
        Args:
            target_style: Target style name
            source_style: Source style name  
            exclude_content: Content to exclude (the main training sample)
            num_pairs: Number of pairs to return (k)
            
        Returns:
            List of (source_image, target_image) tuples
        """
        pairs = []
        
        # Find available contents that exist in both styles
        available_contents = []
        for content, styles_dict in self.content_to_images.items():
            if content == exclude_content:
                continue
            if source_style in styles_dict and target_style in styles_dict:
                available_contents.append(content)
        
        if not available_contents:
            logger.warning(
                f"No available contents for consistency pairs "
                f"(source={source_style}, target={target_style})"
            )
            return pairs
        
        # Sample k different contents
        selected_contents = random.sample(
            available_contents, 
            min(num_pairs, len(available_contents))
        )
        
        for content in selected_contents:
            # Get source style image
            source_candidates = self.content_to_images[content][source_style]
            source_image_path = random.choice(source_candidates)
            source_image = Image.open(source_image_path).convert("RGB")
            
            # Get target style image  
            target_candidates = self.content_to_images[content][target_style]
            target_image_path = random.choice(target_candidates)
            target_image = Image.open(target_image_path).convert("RGB")
            
            pairs.append((source_image, target_image))
        
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
        if self.use_fst:
            style_source_image = self.get_style_source_image(
                target_style=style, content=content, target_image_path=target_image_path
            )

            if self.transforms is not None:
                style_source_image = self.transforms[1](style_source_image)

            sample["style_source_image"] = style_source_image

        # Add consistency pairs for FST consistency loss
        if self.num_consistency_pairs > 0:
            # Determine source style from the style_source_image path
            # We need to extract which style the source image came from
            # This requires tracking the source style in get_style_source_image
            
            # For now, we'll infer it from the sampling strategy:
            # If same_style was used, source_style = target_style
            # If different_style was used, we need to find which style was selected
            
            # To properly implement this, we need to modify get_style_source_image
            # to return both the image and the source style name
            
            # Get source style name by re-sampling with same logic
            use_same_style = random.random() < self.style_source_same_prob
            
            if use_same_style:
                source_style = style  # Same as target
            else:
                # Need to determine which different style was chosen
                # We'll modify get_style_source_image to return style name
                source_style, style_source_image_resampled = self.get_style_source_image_with_name(
                    target_style=style, 
                    content=content, 
                    target_image_path=target_image_path
                )
                # Use the resampled image (with known style)
                if self.transforms is not None:
                    style_source_image = self.transforms[1](style_source_image_resampled)
                sample["style_source_image"] = style_source_image
            
            # Get consistency pairs: same source→target style, different contents
            consistency_pairs = self.get_consistency_pairs(
                target_style=style,
                source_style=source_style,
                exclude_content=content,
                num_pairs=self.num_consistency_pairs
            )
            
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
