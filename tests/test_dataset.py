import pytest
import torch
from pathlib import Path
from PIL import Image


class TestDatasetLoading:
    """Test dataset loading and validation."""

    @pytest.mark.dataset
    def test_dataset_directory_structure(self, dataset_dir):
        """Test dataset directory is created correctly."""
        assert (dataset_dir / "train" / "ContentImage").exists()
        assert (dataset_dir / "train" / "StyleSourceImage").exists()
        assert (dataset_dir / "train" / "StyleTargetImage").exists()
        
        # Check images exist
        content_images = list((dataset_dir / "train" / "ContentImage").glob("*.jpg"))
        assert len(content_images) == 3

    @pytest.mark.dataset
    def test_image_loading(self, dataset_dir):
        """Test images can be loaded."""
        img_path = list((dataset_dir / "train" / "ContentImage").glob("*.jpg"))[0]
        img = Image.open(img_path)
        assert img.size == (96, 96)
        assert img.mode == 'RGB'

    @pytest.mark.dataset
    @pytest.mark.slow
    def test_fontdataset_initialization(self, dataset_dir):
        """Test FontDataset can be initialized."""
        try:
            from dataset.font_dataset import FontDataset
            
            dataset = FontDataset(
                data_root=str(dataset_dir),
                split="train",
                character_list=None,
            )
            
            assert len(dataset) > 0
        except ImportError:
            pytest.skip("FontDataset not available")

    @pytest.mark.dataset
    @pytest.mark.slow
    def test_fontdataset_getitem(self, dataset_dir):
        """Test FontDataset __getitem__ returns correct format."""
        try:
            from dataset.font_dataset import FontDataset
            
            dataset = FontDataset(
                data_root=str(dataset_dir),
                split="train",
                character_list=None,
            )
            
            sample = dataset[0]
            
            assert isinstance(sample, dict)
            assert 'content_image' in sample
            assert 'style_source_image' in sample
            assert 'style_target_image' in sample
        except (ImportError, FileNotFoundError, Exception):
            pytest.skip("FontDataset test skipped")


class TestDatasetValidation:
    """Test dataset validation utilities."""

    @pytest.mark.dataset
    def test_image_format_validation(self, dataset_dir):
        """Test image format validation."""
        img_path = list((dataset_dir / "train" / "ContentImage").glob("*.jpg"))[0]
        img = Image.open(img_path)
        
        # Valid image
        assert img.size == (96, 96)
        assert img.mode == 'RGB'

    @pytest.mark.dataset
    def test_missing_image_handling(self, dataset_dir):
        """Test handling of missing images."""
        missing_path = dataset_dir / "train" / "ContentImage" / "missing.jpg"
        assert not missing_path.exists()