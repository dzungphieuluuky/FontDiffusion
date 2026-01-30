"""
Enhanced collate function for FontDiffuserWithFST.
Handles both original and FST dataset modes.
"""

import torch
import logging
from typing import List, Dict, Any


class CollateFN(object):
    """
    Enhanced collate function supporting both original and FST modes.

    Handles:
    - Standard tensors (content_image, style_image, target_image)
    - FST tensors (style_source_image)
    - SCR tensors (neg_images with variable dimensions)
    - Variable-sized tensors with automatic resizing
    """

    def __init__(
        self,
        return_tensors: str = "pt",
    ):
        """
        Args:
            return_tensors: Format for return ("pt" for PyTorch)
            num_consistency_refs: Number of additional content refs for consistency
        """
        self.return_tensors = return_tensors

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate batch items.

        Dataset provides keys: target_image, content_image, style_image, style_source_image
        Trainer expects keys: target_img, content_img, style_img, style_source_img
        
        Each batch item should contain:
        - target_image: Generated character (from dataset)
        - content_image: Content reference (from dataset)
        - style_source_image: Source style reference (FST mode, from dataset)
        - style_image: Target style reference (from dataset)
        - consistency_pairs: List of (source_tensor, target_tensor) tuples
        """
        # Stack main tensors
        target_imgs = torch.stack([item["target_image"] for item in batch])
        content_imgs = torch.stack([item["content_image"] for item in batch])
        style_imgs = torch.stack([item["style_image"] for item in batch])

        result = {
            "target_image": target_imgs,
            "content_image": content_imgs,
            "style_image": style_imgs,
        }

        # Add nonorm_target_image if present
        if "nonorm_target_image" in batch[0]:
            nonorm_targets = torch.stack([item["nonorm_target_image"] for item in batch])
            result["nonorm_target_image"] = nonorm_targets

        # Add style_source_image for FST mode
        if "style_source_image" in batch[0]:
            style_source_imgs = torch.stack([item["style_source_image"] for item in batch])
            result["style_source_image"] = style_source_imgs

        # Add neg_images for SCR mode
        if "neg_images" in batch[0]:
            neg_imgs = self._collate_neg_images([item["neg_images"] for item in batch])
            result["neg_images"] = neg_imgs

        # Add consistency pairs if present
        if "consistency_pairs" in batch[0] and batch[0]["consistency_pairs"]:
            # Each batch item has a list of (source_tensor, target_tensor) tuples
            # Each tensor is already (C, H, W)
            
            consistency_sources = []
            consistency_targets = []
            
            for item in batch:
                pairs = item.get("consistency_pairs", [])
                if pairs:
                    # Stack all pairs for this batch item
                    # pairs: [(tensor, tensor), (tensor, tensor), ...]
                    sources = torch.stack([p[0] for p in pairs])  # (k, C, H, W)
                    targets = torch.stack([p[1] for p in pairs])  # (k, C, H, W)
                    consistency_sources.append(sources)
                    consistency_targets.append(targets)
            
            # Only add to result if we have valid pairs
            if consistency_sources:
                # Stack across batch: (B, k, C, H, W)
                result["consistency_source_images"] = torch.stack(consistency_sources)
                result["consistency_target_images"] = torch.stack(consistency_targets)

        return result

    def _collate_neg_images(
        self, neg_image_tensors: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Collate negative images from SCR.

        Each sample has shape (num_neg, C, H, W).
        Output shape: (batch_size, num_neg, C, H, W)

        Args:
            neg_image_tensors: List of negative image tensors

        Returns:
            Batched negative images
        """
        # Check if all have same num_neg
        first_shape = neg_image_tensors[0].shape

        if all(t.shape == first_shape for t in neg_image_tensors):
            # Standard case: stack directly
            return torch.stack(neg_image_tensors)

        # Variable num_neg - need to handle carefully
        logging.warning(
            f"Variable number of negative samples detected: "
            f"{[t.shape[0] for t in neg_image_tensors]}"
        )

        # Find max num_neg
        max_num_neg = max(t.shape[0] for t in neg_image_tensors)

        # Pad smaller batches by repeating last negative sample
        padded_negs = []
        for neg_tensor in neg_image_tensors:
            if neg_tensor.shape[0] < max_num_neg:
                # Pad by repeating the last sample
                num_to_pad = max_num_neg - neg_tensor.shape[0]
                padding = neg_tensor[-1:].repeat(num_to_pad, 1, 1, 1)
                neg_tensor = torch.cat([neg_tensor, padding], dim=0)
            padded_negs.append(neg_tensor)

        return torch.stack(padded_negs)


class CollateFNDebug(CollateFN):
    """Debug version of CollateFN with detailed logging."""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate with debug logging."""
        self.call_count += 1

        print(f"\n{'=' * 80}")
        print(f"CollateFN Call #{self.call_count}")
        print(f"{'=' * 80}")
        print(f"Batch size: {len(batch)}")

        if batch:
            print(f"Keys in sample: {list(batch[0].keys())}")

            # Print shapes
            for key in batch[0].keys():
                if isinstance(batch[0][key], torch.Tensor):
                    shapes = [sample[key].shape for sample in batch]
                    print(f"  {key:25s}: {shapes}")

        # Call parent collate
        result = super().__call__(batch)

        # Print result shapes
        print(f"\nBatched tensor shapes:")
        for key, value in result.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key:25s}: {tuple(value.shape)}")
            elif isinstance(value, list):
                print(f"  {key:25s}: list of {len(value)} tensors")

        print(f"{'=' * 80}\n")

        return result


def test_collate_fn():
    """Test the collate function with various scenarios."""
    print("Testing CollateFN...")

    # Test 1: Standard batch (original mode)
    print("\n" + "=" * 80)
    print("Test 1: Standard batch (original mode)")
    print("=" * 80)

    batch_original = [
        {
            "content_image": torch.randn(1, 96, 96),
            "style_image": torch.randn(1, 96, 96),
            "target_image": torch.randn(1, 128, 128),
            "nonorm_target_image": torch.randn(1, 128, 128),
            "target_image_path": "path/to/image1.jpg",
        },
        {
            "content_image": torch.randn(1, 96, 96),
            "style_image": torch.randn(1, 96, 96),
            "target_image": torch.randn(1, 128, 128),
            "nonorm_target_image": torch.randn(1, 128, 128),
            "target_image_path": "path/to/image2.jpg",
        },
    ]

    collate_fn = CollateFNDebug()
    batched = collate_fn(batch_original)

    # Test 2: FST batch
    print("\n" + "=" * 80)
    print("Test 2: FST batch with style_source_image")
    print("=" * 80)

    batch_fst = [
        {
            "content_image": torch.randn(1, 96, 96),
            "style_image": torch.randn(1, 96, 96),
            "style_source_image": torch.randn(1, 96, 96),
            "target_image": torch.randn(1, 128, 128),
            "nonorm_target_image": torch.randn(1, 128, 128),
        },
        {
            "content_image": torch.randn(1, 96, 96),
            "style_image": torch.randn(1, 96, 96),
            "style_source_image": torch.randn(1, 96, 96),
            "target_image": torch.randn(1, 128, 128),
            "nonorm_target_image": torch.randn(1, 128, 128),
        },
    ]

    batched_fst = collate_fn(batch_fst)

    # Test 3: SCR batch with neg_images
    print("\n" + "=" * 80)
    print("Test 3: SCR batch with neg_images")
    print("=" * 80)

    batch_scr = [
        {
            "content_image": torch.randn(1, 96, 96),
            "style_image": torch.randn(1, 96, 96),
            "target_image": torch.randn(1, 128, 128),
            "nonorm_target_image": torch.randn(1, 128, 128),
            "neg_images": torch.randn(3, 1, 128, 128),  # 3 negative samples
        },
        {
            "content_image": torch.randn(1, 96, 96),
            "style_image": torch.randn(1, 96, 96),
            "target_image": torch.randn(1, 128, 128),
            "nonorm_target_image": torch.randn(1, 128, 128),
            "neg_images": torch.randn(3, 1, 128, 128),
        },
    ]

    batched_scr = collate_fn(batch_scr)

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_collate_fn()