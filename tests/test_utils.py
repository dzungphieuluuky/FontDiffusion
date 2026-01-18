import pytest
import torch
from pathlib import Path


class TestTensorOperations:
    """Test tensor operations."""

    @pytest.mark.unit
    def test_tensor_reshaping(self):
        """Test tensor reshaping."""
        x = torch.randn(2, 1024, 6, 6)
        B, C, H, W = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        
        assert x_reshaped.shape == (2, 36, 1024)

    @pytest.mark.unit
    def test_tensor_concatenation(self):
        """Test tensor concatenation."""
        x1 = torch.randn(2, 10, 768)
        x2 = torch.randn(2, 5, 768)
        
        concatenated = torch.cat([x1, x2], dim=1)
        
        assert concatenated.shape == (2, 15, 768)

    @pytest.mark.unit
    def test_tensor_device_movement(self):
        """Test moving tensors between devices."""
        x = torch.randn(2, 4, 12, 12)
        device = torch.device("cpu")
        
        x_moved = x.to(device)
        
        assert x_moved.device.type == "cpu"

    @pytest.mark.unit
    def test_gradient_operations(self):
        """Test gradient operations."""
        x = torch.randn(2, 10, requires_grad=True)
        y = (x ** 2).sum()
        y.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestPathOperations:
    """Test path and file operations."""

    @pytest.mark.unit
    def test_path_exists(self, tmp_path):
        """Test path existence check."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        assert test_file.exists()

    @pytest.mark.unit
    def test_glob_pattern(self, dataset_dir):
        """Test glob pattern matching."""
        images = list((dataset_dir / "train" / "ContentImage").glob("*.jpg"))
        
        assert len(images) > 0
        assert all(img.suffix == ".jpg" for img in images)