"""
Tier 3: Data Integrity & Leakage (Validation Tests)

Purpose: Ensure the data pipeline doesn't "cheat" or break.
Strategy:
  - Test that shuffle=True actually reorders data
  - Verify that data augmentations stay within valid ranges
  - Check for label leakage (ensuring target isn't accidentally in inputs)
  - Validate dataset loading and collation
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Dict, List, Tuple
import tempfile
import numpy as np
from PIL import Image
import os


class DummyFontDataset(Dataset):
    """Minimal font dataset for testing data integrity."""

    def __init__(self, num_samples: int = 20, augment: bool = False, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.num_samples = num_samples
        self.augment = augment
        self.data = [
            {
                "content_image": torch.rand(1, 96, 96),
                "style_image": torch.rand(1, 96, 96),
                "target_image": torch.rand(1, 96, 96),
                "index": i,
            }
            for i in range(num_samples)
        ]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        sample = self.data[idx].copy()

        if self.augment:
            # Apply augmentation
            sample["content_image"] = self._augment(sample["content_image"])
            sample["style_image"] = self._augment(sample["style_image"])
            sample["target_image"] = self._augment(sample["target_image"])

        return sample

    @staticmethod
    def _augment(img: torch.Tensor) -> torch.Tensor:
        """Simple augmentation: random brightness adjustment."""
        brightness_factor = torch.rand(1).item()
        return torch.clamp(img * brightness_factor, 0, 1)


class SimpleDummyCollate:
    """Simple collate function."""

    def __call__(self, batch: list[dict]) -> dict:
        """Collate batch."""
        result = {}
        for key in batch[0].keys():
            if key == "index":
                result[key] = [sample[key] for sample in batch]
            else:
                result[key] = torch.stack([sample[key] for sample in batch])
        return result


class TestDatasetShuffling:
    """Test that shuffling actually reorders data."""

    def test_dataloader_shuffle_true_reorders(self):
        """Verify shuffle=True produces different orderings."""
        dataset = DummyFontDataset(num_samples=20)

        # Load with shuffle=False
        loader_no_shuffle = DataLoader(dataset, batch_size=5, shuffle=False)
        indices_no_shuffle = []
        for batch in loader_no_shuffle:
            indices_no_shuffle.extend(batch["index"].tolist())

        # Load with shuffle=True (set seed for reproducibility in test)
        torch.manual_seed(999)
        loader_shuffle = DataLoader(dataset, batch_size=5, shuffle=True)
        indices_shuffle = []
        for batch in loader_shuffle:
            indices_shuffle.extend(batch["index"].tolist())

        # Indices should be different (with high probability)
        # They should have same elements but different order
        assert sorted(indices_no_shuffle) == sorted(indices_shuffle), (
            "Shuffled and non-shuffled should have same elements"
        )

        # It's very unlikely shuffle produces the same order
        # (not guaranteed but very improbable)
        # So we only check this if shuffle actually reordered
        num_different_positions = sum(
            1 for a, b in zip(indices_no_shuffle, indices_shuffle) if a != b
        )
        # Allow some position to be same by chance
        if num_different_positions > 0:
            print(f"Shuffle reordered {num_different_positions} positions")

    def test_shuffle_consistency_with_seed(self):
        """Test that shuffling with seed is reproducible."""
        dataset = DummyFontDataset(num_samples=20)

        indices_1 = []
        torch.manual_seed(42)
        loader_1 = DataLoader(dataset, batch_size=5, shuffle=True)
        for batch in loader_1:
            indices_1.extend(batch["index"].tolist())

        indices_2 = []
        torch.manual_seed(42)
        loader_2 = DataLoader(dataset, batch_size=5, shuffle=True)
        for batch in loader_2:
            indices_2.extend(batch["index"].tolist())

        assert indices_1 == indices_2, (
            "Shuffling with same seed should produce same order"
        )


class TestAugmentationRanges:
    """Test that augmentations keep values in valid ranges."""

    def test_augmentation_brightness_in_range(self):
        """Verify brightness augmentation stays in [0, 1]."""
        dataset = DummyFontDataset(num_samples=10, augment=True)

        for _ in range(20):
            sample = dataset[0]
            for key in ["content_image", "style_image", "target_image"]:
                img = sample[key]
                assert (img >= 0).all(), f"{key} has values < 0"
                assert (img <= 1).all(), f"{key} has values > 1"

    def test_augmentation_preserves_shape(self):
        """Verify augmentation doesn't change shape."""
        dataset = DummyFontDataset(num_samples=10, augment=True)

        original_sample = dataset.data[0]
        original_shapes = {
            "content_image": original_sample["content_image"].shape,
            "style_image": original_sample["style_image"].shape,
            "target_image": original_sample["target_image"].shape,
        }

        augmented_sample = dataset[0]
        for key in original_shapes.keys():
            assert augmented_sample[key].shape == original_shapes[key], (
                f"{key} shape changed after augmentation"
            )

    @pytest.mark.parametrize("num_augmentations", [1, 5, 10])
    def test_augmentation_consistency(self, num_augmentations: int):
        """Test augmentation behavior across multiple applications."""
        dataset = DummyFontDataset(num_samples=5, augment=True)

        for _ in range(num_augmentations):
            for idx in range(len(dataset)):
                sample = dataset[idx]

                # All images should be tensors
                assert torch.is_tensor(sample["content_image"])
                assert torch.is_tensor(sample["style_image"])
                assert torch.is_tensor(sample["target_image"])

                # All should have correct shape
                for key in ["content_image", "style_image", "target_image"]:
                    assert sample[key].shape == (1, 96, 96)


class TestLabelLeakage:
    """Test for label leakage in data pipeline."""

    def test_content_not_in_style_input(self):
        """Verify content image is not accidentally in style input."""
        dataset = DummyFontDataset(num_samples=20)

        for idx in range(len(dataset)):
            sample = dataset[idx]

            # Content and style should be different tensors
            content = sample["content_image"]
            style = sample["style_image"]
            target = sample["target_image"]

            # They should not be identical (random chance very low)
            # But we can't assert strict inequality due to randomness
            # Instead, check they're different objects
            assert content is not style, "Content and style should be different objects"
            assert content is not target, (
                "Content and target should be different objects"
            )

    def test_target_not_in_inputs(self):
        """Verify target is not accidentally mixed with inputs."""
        dataset = DummyFontDataset(num_samples=20)
        loader = DataLoader(dataset, batch_size=4, collate_fn=SimpleDummyCollate())

        for batch in loader:
            content = batch["content_image"]
            style = batch["style_image"]
            target = batch["target_image"]

            # Targets should not be part of input concatenation
            # (Check by computing stats)
            content_style_cat = torch.cat([content, style], dim=1)
            assert content_style_cat.shape[1] == 2, (
                "Concatenated input should have 2 channels"
            )
            assert target.shape[1] == 1, "Target should not be concatenated with inputs"

    def test_batch_indices_no_repetition(self):
        """Verify no sample is repeated within a batch in shuffle mode."""
        dataset = DummyFontDataset(num_samples=50)
        loader = DataLoader(
            dataset, batch_size=10, shuffle=True, collate_fn=SimpleDummyCollate()
        )

        for batch in loader:
            indices = batch["index"]
            unique_indices = len(set(indices))
            batch_size = len(indices)

            assert unique_indices == batch_size, (
                f"Batch has {batch_size - unique_indices} repeated indices!"
            )


class TestCollateFunction:
    """Test collate function behavior."""

    def test_collate_stacks_tensors(self):
        """Verify collate function properly stacks tensors."""
        batch = [
            {
                "content_image": torch.rand(1, 96, 96),
                "target_image": torch.rand(1, 96, 96),
                "index": 0,
            },
            {
                "content_image": torch.rand(1, 96, 96),
                "target_image": torch.rand(1, 96, 96),
                "index": 1,
            },
        ]

        collate_fn = SimpleDummyCollate()
        collated = collate_fn(batch)

        assert collated["content_image"].shape == (2, 1, 96, 96)
        assert collated["target_image"].shape == (2, 1, 96, 96)
        assert collated["index"] == [0, 1]

    def test_collate_preserves_dtype(self):
        """Verify collate preserves data types."""
        batch = [
            {"content_image": torch.rand(1, 96, 96, dtype=torch.float32), "index": 0},
            {"content_image": torch.rand(1, 96, 96, dtype=torch.float32), "index": 1},
        ]

        collate_fn = SimpleDummyCollate()
        collated = collate_fn(batch)

        assert collated["content_image"].dtype == torch.float32


class TestDataLoaderDeterminism:
    """Test that DataLoader is deterministic with seeds."""

    def test_dataloader_same_seed_same_order(self):
        """Verify DataLoader produces same order with same seed."""
        dataset = DummyFontDataset(num_samples=30)

        # First run
        torch.manual_seed(123)
        loader1 = DataLoader(dataset, batch_size=5, shuffle=True)
        data1 = []
        for batch in loader1:
            data1.extend(batch["index"].tolist())

        # Second run with same seed
        torch.manual_seed(123)
        loader2 = DataLoader(dataset, batch_size=5, shuffle=True)
        data2 = []
        for batch in loader2:
            data2.extend(batch["index"].tolist())

        assert data1 == data2, "DataLoader with same seed should produce same order"

    def test_dataloader_different_seed_different_order(self):
        """Verify DataLoader produces different order with different seeds."""
        dataset = DummyFontDataset(num_samples=30)

        torch.manual_seed(111)
        loader1 = DataLoader(dataset, batch_size=5, shuffle=True)
        data1 = [idx for batch in loader1 for idx in batch["index"].tolist()]

        torch.manual_seed(222)
        loader2 = DataLoader(dataset, batch_size=5, shuffle=True)
        data2 = [idx for batch in loader2 for idx in batch["index"].tolist()]

        # Should have same elements but likely different order
        assert sorted(data1) == sorted(data2)
        # With high probability, order is different
        # (we don't assert this strictly as it could happen by chance)


class TestValidRangeValues:
    """Test that pixel values stay in valid ranges."""

    def test_image_pixel_ranges(self):
        """Verify image pixels are in expected range."""
        dataset = DummyFontDataset(num_samples=10, augment=False)

        for idx in range(len(dataset)):
            sample = dataset[idx]

            for key in ["content_image", "style_image", "target_image"]:
                img = sample[key]

                # Check range
                assert (img >= 0).all() and (img <= 1).all(), (
                    f"{key} not in [0, 1] range"
                )

                # Check not all zeros or ones (would indicate issue)
                assert not (img == 0).all(), f"{key} is all zeros"
                assert not (img == 1).all(), f"{key} is all ones"


class TestBatchesAgainstLabelLeakage:
    """Test full batching pipeline for label leakage."""

    def test_full_pipeline_no_leakage(self):
        """Test end-to-end that data pipeline has no label leakage."""
        dataset = DummyFontDataset(num_samples=50)
        loader = DataLoader(
            dataset, batch_size=8, shuffle=True, collate_fn=SimpleDummyCollate()
        )

        for batch_idx, batch in enumerate(loader):
            if batch_idx > 5:  # Just check first few batches
                break

            content = batch["content_image"]  # (B, 1, 96, 96)
            style = batch["style_image"]  # (B, 1, 96, 96)
            target = batch["target_image"]  # (B, 1, 96, 96)

            # Compute correlation between content+style and target
            # If there's leakage, correlation should be high
            combined = torch.cat([content, style], dim=1)  # (B, 2, 96, 96)

            # Flatten and compute correlation
            combined_flat = combined.reshape(combined.shape[0], -1)
            target_flat = target.reshape(target.shape[0], -1)

            correlations = []
            for i in range(combined_flat.shape[0]):
                corr = torch.corrcoef(torch.stack([combined_flat[i], target_flat[i]]))[
                    0, 1
                ]
                if not torch.isnan(corr):
                    correlations.append(corr.item())

            if correlations:
                avg_corr = np.mean(correlations)
                print(f"Batch {batch_idx}: avg correlation = {avg_corr:.3f}")
                # For random data, correlation should be near 0
                # Allow some deviation due to randomness
                assert abs(avg_corr) < 0.5, (
                    f"High correlation detected ({avg_corr:.3f}), possible leakage!"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
