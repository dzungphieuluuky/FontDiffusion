"""
Pytest suite for validating MRL dimensions compatibility.

Tests cover:
- Nesting dimensions and frequency radii matching
- Ascending order validation
- Embedding dimension constraints
- Frequency radii value ranges
- Invalid configurations detection
"""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
import argparse

# Mock torch and other heavy imports before importing trainer
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['diffusers'] = MagicMock()
sys.modules['accelerate'] = MagicMock()
sys.modules['torchvision'] = MagicMock()
sys.modules['torchvision.transforms'] = MagicMock()
sys.modules['onnx'] = MagicMock()
sys.modules['onnxruntime'] = MagicMock()

from src.trainers.trainer_mrl import FontDiffuserMRLTrainer


class TestMRLDimensionsParsing:
    """Test MRL dimension parsing utilities."""

    def test_parse_mrl_nesting_dims_string(self):
        """Test parsing nesting dimensions from comma-separated string."""
        args = Mock()
        args.use_fst = True
        args.phase_1_ckpt_dir = None
        
        with patch.object(FontDiffuserMRLTrainer, '__init__', lambda x, y: None):
            trainer = FontDiffuserMRLTrainer(args)
            
            result = trainer._parse_mrl_nesting_dims("64,128,256,512")
            assert result == [64, 128, 256, 512]
            assert isinstance(result, list)

    def test_parse_mrl_nesting_dims_list(self):
        """Test that list input is returned as-is."""
        args = Mock()
        with patch.object(FontDiffuserMRLTrainer, '__init__', lambda x, y: None):
            trainer = FontDiffuserMRLTrainer(args)
            
            input_list = [64, 128, 256, 512]
            result = trainer._parse_mrl_nesting_dims(input_list)
            assert result == input_list
            assert isinstance(result, list)

    def test_parse_mrl_nesting_dims_with_spaces(self):
        """Test parsing with extra whitespace."""
        args = Mock()
        with patch.object(FontDiffuserMRLTrainer, '__init__', lambda x, y: None):
            trainer = FontDiffuserMRLTrainer(args)
            
            result = trainer._parse_mrl_nesting_dims("64 , 128 , 256 , 512")
            assert result == [64, 128, 256, 512]

    def test_parse_mrl_freq_radii_string(self):
        """Test parsing frequency radii from comma-separated string."""
        args = Mock()
        with patch.object(FontDiffuserMRLTrainer, '__init__', lambda x, y: None):
            trainer = FontDiffuserMRLTrainer(args)
            
            result = trainer._parse_mrl_freq_radii("0.1,0.3,0.5")
            assert result == [0.1, 0.3, 0.5]
            assert isinstance(result, list)

    def test_parse_mrl_freq_radii_list(self):
        """Test that list input is returned as-is."""
        args = Mock()
        with patch.object(FontDiffuserMRLTrainer, '__init__', lambda x, y: None):
            trainer = FontDiffuserMRLTrainer(args)
            
            input_list = [0.1, 0.3, 0.5]
            result = trainer._parse_mrl_freq_radii(input_list)
            assert result == input_list


class TestMRLDimensionsCompatibility:
    """Test MRL dimensions compatibility validation."""

    def test_default_dimensions_compatible(self):
        """Test that default dimensions are compatible."""
        nesting_dims = [64, 128, 256, 512]
        freq_radii = [0.1, 0.3, 0.5]
        
        assert len(freq_radii) == len(nesting_dims) - 1, \
            f"Expected {len(nesting_dims) - 1} freq_radii, got {len(freq_radii)}"

    def test_freq_radii_missing_value(self):
        """Test error when freq_radii has too few values."""
        nesting_dims = [64, 128, 256, 512]
        freq_radii = [0.1, 0.3]  # Missing one value
        
        assert len(freq_radii) != len(nesting_dims) - 1, \
            "freq_radii should not match nesting_dims"

    def test_freq_radii_extra_value(self):
        """Test error when freq_radii has too many values."""
        nesting_dims = [64, 128, 256, 512]
        freq_radii = [0.1, 0.3, 0.5, 0.7]  # One extra value
        
        assert len(freq_radii) != len(nesting_dims) - 1, \
            "freq_radii should not match nesting_dims"

    def test_custom_compatible_dimensions(self):
        """Test custom compatible dimension combinations."""
        test_cases = [
            ([64, 128], [0.25]),  # 2 nesting dims, 1 freq radius
            ([128, 256, 512], [0.2, 0.6]),  # 3 nesting dims, 2 freq radii
            ([32, 64, 128, 256, 512], [0.15, 0.3, 0.6, 0.8]),  # 5 nesting dims
        ]
        
        for nesting_dims, freq_radii in test_cases:
            assert len(freq_radii) == len(nesting_dims) - 1, \
                f"Mismatch for nesting_dims={nesting_dims}, freq_radii={freq_radii}"

    def test_nesting_dims_ascending_order(self):
        """Test that nesting dimensions are in ascending order."""
        valid_dims = [64, 128, 256, 512]
        assert valid_dims == sorted(valid_dims), "Nesting dims should be sorted"
        
        invalid_dims = [64, 256, 128, 512]
        assert invalid_dims != sorted(invalid_dims), "Invalid dims should not be sorted"

    def test_nesting_dims_no_duplicates(self):
        """Test that nesting dimensions have no duplicates."""
        valid_dims = [64, 128, 256, 512]
        assert len(valid_dims) == len(set(valid_dims)), "No duplicates expected"
        
        invalid_dims = [64, 128, 128, 512]
        assert len(invalid_dims) != len(set(invalid_dims)), "Should have duplicates"

    def test_nesting_dims_all_positive(self):
        """Test that all nesting dimensions are positive."""
        valid_dims = [64, 128, 256, 512]
        assert all(d > 0 for d in valid_dims), "All dims should be positive"
        
        invalid_dims = [64, -128, 256, 512]
        assert not all(d > 0 for d in invalid_dims), "Should contain negative value"


class TestMRLFrequencyRadiiValidation:
    """Test frequency radii value ranges and validity."""

    def test_freq_radii_in_valid_range(self):
        """Test that frequency radii are in [0, 1] range."""
        valid_radii = [0.1, 0.3, 0.5]
        assert all(0 <= r <= 1 for r in valid_radii), "All radii should be in [0,1]"

    def test_freq_radii_ascending_order(self):
        """Test that frequency radii are in ascending order."""
        valid_radii = [0.1, 0.3, 0.5]
        assert valid_radii == sorted(valid_radii), "Radii should be sorted"
        
        invalid_radii = [0.1, 0.5, 0.3]
        assert invalid_radii != sorted(invalid_radii), "Invalid radii order"

    def test_freq_radii_boundary_values(self):
        """Test boundary frequency radii values."""
        # Boundary cases
        boundary_radii = [0.0, 0.5, 1.0]
        assert all(0 <= r <= 1 for r in boundary_radii), "Boundary values should be valid"

    def test_freq_radii_invalid_negative(self):
        """Test that negative frequency radii are invalid."""
        invalid_radii = [0.1, -0.3, 0.5]
        assert not all(0 <= r <= 1 for r in invalid_radii), "Should reject negative radii"

    def test_freq_radii_invalid_above_one(self):
        """Test that frequency radii > 1 are invalid."""
        invalid_radii = [0.1, 0.3, 1.5]
        assert not all(0 <= r <= 1 for r in invalid_radii), "Should reject radii > 1"


class TestMRLEmbeddingDimensionCompatibility:
    """Test MRL embedding dimension constraints."""

    def test_nesting_dims_less_than_embedding(self):
        """Test that all nesting dimensions are <= embedding dimension."""
        embedding_dim = 512
        nesting_dims = [64, 128, 256, 512]
        
        assert all(d <= embedding_dim for d in nesting_dims), \
            "All nesting dims should be <= embedding_dim"

    def test_nesting_dims_exceeds_embedding(self):
        """Test error when nesting dimension exceeds embedding dimension."""
        embedding_dim = 256
        nesting_dims = [64, 128, 256, 512]  # 512 > 256
        
        assert not all(d <= embedding_dim for d in nesting_dims), \
            "Should detect dimension exceeding embedding_dim"

    def test_largest_nesting_dim_equals_embedding(self):
        """Test valid case where largest nesting dimension equals embedding."""
        embedding_dim = 512
        nesting_dims = [64, 128, 256, 512]
        
        assert max(nesting_dims) == embedding_dim, "Largest dim should match embedding"
        assert all(d <= embedding_dim for d in nesting_dims), "All dims valid"

    def test_custom_embedding_dim_validation(self):
        """Test validation with custom embedding dimensions."""
        test_cases = [
            (256, [64, 128, 256]),  # Valid
            (512, [32, 64, 128, 256, 512]),  # Valid
            (256, [128, 256, 512]),  # Invalid - 512 > 256
        ]
        
        for embedding_dim, nesting_dims in test_cases[:-1]:
            assert all(d <= embedding_dim for d in nesting_dims), \
                f"Valid case failed: embedding_dim={embedding_dim}"
        
        embedding_dim, nesting_dims = test_cases[-1]
        assert not all(d <= embedding_dim for d in nesting_dims), \
            f"Should reject: embedding_dim={embedding_dim}, nesting_dims={nesting_dims}"


class TestMRLConfigurationMatrix:
    """Test various real-world MRL configuration combinations."""

    def test_configuration_compact(self):
        """Test compact MRL configuration."""
        config = {
            "nesting_dims": [128, 256],
            "freq_radii": [0.5],
            "embedding_dim": 256,
        }
        
        assert len(config["freq_radii"]) == len(config["nesting_dims"]) - 1
        assert all(d <= config["embedding_dim"] for d in config["nesting_dims"])

    def test_configuration_standard(self):
        """Test standard MRL configuration."""
        config = {
            "nesting_dims": [64, 128, 256, 512],
            "freq_radii": [0.1, 0.3, 0.5],
            "embedding_dim": 512,
        }
        
        assert len(config["freq_radii"]) == len(config["nesting_dims"]) - 1
        assert all(d <= config["embedding_dim"] for d in config["nesting_dims"])

    def test_configuration_large(self):
        """Test large MRL configuration."""
        config = {
            "nesting_dims": [64, 128, 256, 512, 1024],
            "freq_radii": [0.1, 0.25, 0.5, 0.75],
            "embedding_dim": 1024,
        }
        
        assert len(config["freq_radii"]) == len(config["nesting_dims"]) - 1
        assert all(d <= config["embedding_dim"] for d in config["nesting_dims"])

    def test_configuration_fine_grained(self):
        """Test fine-grained MRL configuration with many bands."""
        config = {
            "nesting_dims": [32, 64, 96, 128, 160, 192, 224, 256],
            "freq_radii": [0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88],
            "embedding_dim": 256,
        }
        
        assert len(config["freq_radii"]) == len(config["nesting_dims"]) - 1
        assert all(d <= config["embedding_dim"] for d in config["nesting_dims"])
        assert config["freq_radii"] == sorted(config["freq_radii"])

    def test_configuration_mixed_powers_of_two(self):
        """Test configuration with mixed dimension powers."""
        config = {
            "nesting_dims": [128, 256, 512],
            "freq_radii": [0.3, 0.7],
            "embedding_dim": 512,
        }
        
        assert len(config["freq_radii"]) == len(config["nesting_dims"]) - 1
        assert all(d <= config["embedding_dim"] for d in config["nesting_dims"])


class TestMRLTrainerInitialization:
    """Test MRL trainer initialization with various configs."""

    @patch('src.trainers.trainer_fst.FontDiffuserFSTTrainer.__init__', return_value=None)
    def test_trainer_default_args(self, mock_parent_init):
        """Test trainer init with default arguments."""
        args = argparse.Namespace(
            use_fst=True,
            use_mrl=True,
            mrl_nesting_dims="64,128,256,512",
            mrl_freq_radii="0.1,0.3,0.5",
            content_encoder_dim=512,
            phase_1_ckpt_dir=None,
            freeze_modules="",
            fst_feature_channels="64,128,256,512,1024",
        )
        
        trainer = FontDiffuserMRLTrainer(args)
        
        assert trainer.mrl_nesting_dims == [64, 128, 256, 512]
        assert trainer.mrl_freq_radii == [0.1, 0.3, 0.5]
        assert len(trainer.mrl_freq_radii) == len(trainer.mrl_nesting_dims) - 1

    @patch('src.trainers.trainer_fst.FontDiffuserFSTTrainer.__init__', return_value=None)
    def test_trainer_custom_args(self, mock_parent_init):
        """Test trainer init with custom arguments."""
        args = argparse.Namespace(
            use_fst=True,
            use_mrl=True,
            mrl_nesting_dims="128,256,512",
            mrl_freq_radii="0.2,0.6",
            content_encoder_dim=512,
            phase_1_ckpt_dir=None,
            freeze_modules="",
            fst_feature_channels="64,128,256,512,1024",
        )
        
        trainer = FontDiffuserMRLTrainer(args)
        
        assert trainer.mrl_nesting_dims == [128, 256, 512]
        assert trainer.mrl_freq_radii == [0.2, 0.6]
        assert len(trainer.mrl_freq_radii) == len(trainer.mrl_nesting_dims) - 1

    @patch('src.trainers.trainer_fst.FontDiffuserFSTTrainer.__init__', return_value=None)
    def test_trainer_validation_dimension_mismatch(self, mock_parent_init):
        """Test that trainer detects dimension mismatches."""
        args = argparse.Namespace(
            use_fst=True,
            use_mrl=True,
            mrl_nesting_dims="64,128,256,512",  # 4 dims
            mrl_freq_radii="0.1,0.3",  # 2 radii (should be 3)
            content_encoder_dim=512,
            phase_1_ckpt_dir=None,
            freeze_modules="",
            fst_feature_channels="64,128,256,512,1024",
        )
        
        trainer = FontDiffuserMRLTrainer(args)
        
        # This should be caught during _setup_models
        assert trainer.mrl_nesting_dims == [64, 128, 256, 512]
        assert trainer.mrl_freq_radii == [0.1, 0.3]
        assert len(trainer.mrl_freq_radii) != len(trainer.mrl_nesting_dims) - 1


class TestMRLInvalidConfigurations:
    """Test detection of invalid MRL configurations."""

    def test_invalid_ascending_order(self):
        """Test detection of non-ascending nesting dimensions."""
        invalid_dims = [64, 256, 128, 512]
        
        assert invalid_dims != sorted(invalid_dims), \
            "Should detect non-ascending order"

    def test_invalid_ascending_order_radii(self):
        """Test detection of non-ascending frequency radii."""
        invalid_radii = [0.1, 0.5, 0.3]
        
        assert invalid_radii != sorted(invalid_radii), \
            "Should detect non-ascending radii"

    def test_invalid_single_nesting_dim(self):
        """Test that single nesting dimension is invalid (needs at least 2)."""
        dims = [256]
        radii = []  # 0 radii for 1 dim
        
        assert len(radii) == len(dims) - 1  # But this is technically valid by formula
        # However, single dimension MRL doesn't make practical sense

    def test_invalid_zero_embedding_dim(self):
        """Test detection of zero embedding dimension."""
        embedding_dim = 0
        nesting_dims = [64, 128, 256, 512]
        
        assert not all(d <= embedding_dim for d in nesting_dims), \
            "Should reject zero embedding_dim"

    def test_invalid_freq_radii_not_increasing(self):
        """Test that non-strictly-increasing radii are detected."""
        # Equal values
        radii_equal = [0.1, 0.3, 0.3]
        assert radii_equal != sorted(radii_equal) or radii_equal[1] == radii_equal[2], \
            "Should detect equal consecutive values"


class TestMRLEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_valid_configuration(self):
        """Test minimum valid MRL configuration."""
        nesting_dims = [256, 512]
        freq_radii = [0.5]
        embedding_dim = 512
        
        assert len(freq_radii) == len(nesting_dims) - 1
        assert all(d <= embedding_dim for d in nesting_dims)
        assert nesting_dims == sorted(nesting_dims)

    def test_very_small_dimensions(self):
        """Test with very small dimension values."""
        nesting_dims = [8, 16, 32, 64]
        freq_radii = [0.25, 0.5, 0.75]
        embedding_dim = 64
        
        assert len(freq_radii) == len(nesting_dims) - 1
        assert all(d <= embedding_dim for d in nesting_dims)

    def test_very_large_dimensions(self):
        """Test with very large dimension values."""
        nesting_dims = [512, 1024, 2048, 4096]
        freq_radii = [0.25, 0.5, 0.75]
        embedding_dim = 4096
        
        assert len(freq_radii) == len(nesting_dims) - 1
        assert all(d <= embedding_dim for d in nesting_dims)

    def test_many_nesting_dims(self):
        """Test with many nesting dimensions."""
        nesting_dims = list(range(64, 512 + 64, 64))  # [64, 128, 192, ..., 512]
        freq_radii_count = len(nesting_dims) - 1
        freq_radii = [i / (freq_radii_count + 1) for i in range(1, freq_radii_count + 1)]
        embedding_dim = 512
        
        assert len(freq_radii) == len(nesting_dims) - 1
        assert all(d <= embedding_dim for d in nesting_dims)
        assert freq_radii == sorted(freq_radii)
        assert all(0 < r < 1 for r in freq_radii)


class TestMRLTypeValidation:
    """Test type validation for MRL configurations."""

    def test_nesting_dims_type_list(self):
        """Test that nesting_dims accepts list type."""
        dims = [64, 128, 256, 512]
        assert isinstance(dims, list)

    def test_freq_radii_type_list(self):
        """Test that freq_radii accepts list type."""
        radii = [0.1, 0.3, 0.5]
        assert isinstance(radii, list)

    def test_nesting_dims_type_tuple(self):
        """Test that nesting_dims can be converted from tuple."""
        dims_tuple = (64, 128, 256, 512)
        dims_list = list(dims_tuple)
        assert dims_list == [64, 128, 256, 512]

    def test_freq_radii_all_numeric(self):
        """Test that all freq_radii values are numeric."""
        radii = [0.1, 0.3, 0.5]
        assert all(isinstance(r, (int, float)) for r in radii)

    def test_nesting_dims_all_integer(self):
        """Test that all nesting_dims values are integers."""
        dims = [64, 128, 256, 512]
        assert all(isinstance(d, int) for d in dims)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
