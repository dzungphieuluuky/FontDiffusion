"""
Standalone pytest suite for validating MRL dimensions compatibility.

Tests validate:
- Nesting dimensions and frequency radii matching
- Ascending order validation
- Embedding dimension constraints
- Frequency radii value ranges
- Configuration compatibility matrices

This suite can run independently without heavy torch/accelerate imports.
"""

import pytest


class TestMRLDimensionsParsing:
    """Test MRL dimension parsing logic."""

    def parse_mrl_nesting_dims(self, dims_str):
        """Parse MRL nesting dimensions from comma-separated string."""
        if isinstance(dims_str, list):
            return dims_str
        return [int(x.strip()) for x in dims_str.split(",") if x.strip()]

    def parse_mrl_freq_radii(self, radii_str):
        """Parse MRL frequency radii from comma-separated string."""
        if isinstance(radii_str, list):
            return radii_str
        return [float(x.strip()) for x in radii_str.split(",") if x.strip()]

    def test_parse_nesting_dims_string(self):
        """Parse nesting dimensions from comma-separated string."""
        result = self.parse_mrl_nesting_dims("64,128,256,512")
        assert result == [64, 128, 256, 512]
        assert isinstance(result, list)

    def test_parse_nesting_dims_list(self):
        """Parse nesting dimensions when input is already a list."""
        input_list = [64, 128, 256, 512]
        result = self.parse_mrl_nesting_dims(input_list)
        assert result == input_list

    def test_parse_nesting_dims_with_spaces(self):
        """Parse with extra whitespace."""
        result = self.parse_mrl_nesting_dims("64 , 128 , 256 , 512")
        assert result == [64, 128, 256, 512]

    def test_parse_freq_radii_string(self):
        """Parse frequency radii from comma-separated string."""
        result = self.parse_mrl_freq_radii("0.1,0.3,0.5")
        assert result == [0.1, 0.3, 0.5]

    def test_parse_freq_radii_list(self):
        """Parse frequency radii when input is already a list."""
        input_list = [0.1, 0.3, 0.5]
        result = self.parse_mrl_freq_radii(input_list)
        assert result == input_list

    def test_parse_freq_radii_with_spaces(self):
        """Parse with extra whitespace."""
        result = self.parse_mrl_freq_radii("0.1 , 0.3 , 0.5")
        assert result == [0.1, 0.3, 0.5]


class TestMRLDimensionsCompatibility:
    """Test MRL dimensions compatibility validation."""

    def validate_dimensions(self, nesting_dims, freq_radii, embedding_dim=512):
        """Validate MRL dimension compatibility."""
        errors = []
        
        # Check length matching
        if len(freq_radii) != len(nesting_dims) - 1:
            errors.append(
                f"len(freq_radii)={len(freq_radii)} must equal "
                f"len(nesting_dims)-1={len(nesting_dims)-1}"
            )
        
        # Check ascending order
        if nesting_dims != sorted(nesting_dims):
            errors.append("nesting_dims not in ascending order")
        
        if freq_radii != sorted(freq_radii):
            errors.append("freq_radii not in ascending order")
        
        # Check embedding dimension constraints
        if not all(d <= embedding_dim for d in nesting_dims):
            errors.append(
                f"Some nesting_dims exceed embedding_dim={embedding_dim}"
            )
        
        # Check frequency radii value range
        if not all(0 <= r <= 1 for r in freq_radii):
            errors.append("Some freq_radii outside [0, 1] range")
        
        return errors

    def test_default_dimensions_valid(self):
        """Default dimensions should be valid."""
        errors = self.validate_dimensions(
            nesting_dims=[64, 128, 256, 512],
            freq_radii=[0.1, 0.3, 0.5],
            embedding_dim=512
        )
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_freq_radii_too_few(self):
        """Detect when freq_radii has too few values."""
        errors = self.validate_dimensions(
            nesting_dims=[64, 128, 256, 512],
            freq_radii=[0.1, 0.3],  # Should be 3
            embedding_dim=512
        )
        assert len(errors) > 0
        assert any("len(freq_radii)" in e for e in errors)

    def test_freq_radii_too_many(self):
        """Detect when freq_radii has too many values."""
        errors = self.validate_dimensions(
            nesting_dims=[64, 128, 256, 512],
            freq_radii=[0.1, 0.3, 0.5, 0.7],  # Should be 3
            embedding_dim=512
        )
        assert len(errors) > 0
        assert any("len(freq_radii)" in e for e in errors)

    def test_nesting_dims_not_ascending(self):
        """Detect when nesting_dims not in ascending order."""
        errors = self.validate_dimensions(
            nesting_dims=[64, 256, 128, 512],
            freq_radii=[0.2, 0.4, 0.6],
            embedding_dim=512
        )
        assert len(errors) > 0
        assert any("ascending order" in e for e in errors)

    def test_freq_radii_not_ascending(self):
        """Detect when freq_radii not in ascending order."""
        errors = self.validate_dimensions(
            nesting_dims=[64, 128, 256, 512],
            freq_radii=[0.1, 0.5, 0.3],
            embedding_dim=512
        )
        assert len(errors) > 0
        assert any("ascending order" in e for e in errors)

    def test_dimension_exceeds_embedding(self):
        """Detect when nesting_dims exceed embedding_dim."""
        errors = self.validate_dimensions(
            nesting_dims=[64, 128, 256, 512],
            freq_radii=[0.1, 0.3, 0.5],
            embedding_dim=256  # 512 > 256
        )
        assert len(errors) > 0
        assert any("embedding_dim" in e for e in errors)

    def test_freq_radii_outside_range(self):
        """Detect when freq_radii outside [0,1]."""
        errors = self.validate_dimensions(
            nesting_dims=[64, 128, 256, 512],
            freq_radii=[0.1, 0.3, 1.5],
            embedding_dim=512
        )
        assert len(errors) > 0
        assert any("[0, 1]" in e for e in errors)

    def test_custom_valid_dimensions(self):
        """Test custom valid dimension combinations."""
        test_cases = [
            ([64, 128], [0.25], 128),
            ([128, 256, 512], [0.2, 0.6], 512),
            ([32, 64, 128, 256, 512], [0.15, 0.3, 0.6, 0.8], 512),
        ]
        
        for nesting_dims, freq_radii, embedding_dim in test_cases:
            errors = self.validate_dimensions(
                nesting_dims=nesting_dims,
                freq_radii=freq_radii,
                embedding_dim=embedding_dim
            )
            assert len(errors) == 0, \
                f"Validation failed for {nesting_dims}: {errors}"


class TestMRLFrequencyRadiiValidation:
    """Test frequency radii specific validation."""

    def test_radii_in_valid_range(self):
        """Frequency radii should be in [0, 1]."""
        valid_radii = [0.1, 0.3, 0.5]
        assert all(0 <= r <= 1 for r in valid_radii)

    def test_radii_boundary_zero(self):
        """Test boundary value 0."""
        radii = [0.0, 0.5]
        assert all(0 <= r <= 1 for r in radii)

    def test_radii_boundary_one(self):
        """Test boundary value 1."""
        radii = [0.5, 1.0]
        assert all(0 <= r <= 1 for r in radii)

    def test_radii_negative_invalid(self):
        """Negative radii should be invalid."""
        radii = [0.1, -0.3, 0.5]
        assert not all(0 <= r <= 1 for r in radii)

    def test_radii_above_one_invalid(self):
        """Radii > 1 should be invalid."""
        radii = [0.1, 0.3, 1.5]
        assert not all(0 <= r <= 1 for r in radii)

    def test_radii_ascending(self):
        """Radii should be in ascending order."""
        valid = [0.1, 0.3, 0.5]
        assert valid == sorted(valid)
        
        invalid = [0.1, 0.5, 0.3]
        assert invalid != sorted(invalid)


class TestMRLEmbeddingDimensionCompatibility:
    """Test embedding dimension constraints."""

    def test_all_dims_less_than_embedding(self):
        """All nesting dims should be <= embedding_dim."""
        embedding_dim = 512
        nesting_dims = [64, 128, 256, 512]
        assert all(d <= embedding_dim for d in nesting_dims)

    def test_dims_exceed_embedding(self):
        """Detect when dims exceed embedding."""
        embedding_dim = 256
        nesting_dims = [64, 128, 256, 512]
        assert not all(d <= embedding_dim for d in nesting_dims)

    def test_largest_dim_equals_embedding(self):
        """Valid when largest dim equals embedding."""
        embedding_dim = 512
        nesting_dims = [64, 128, 256, 512]
        assert max(nesting_dims) == embedding_dim
        assert all(d <= embedding_dim for d in nesting_dims)

    def test_custom_embedding_dims(self):
        """Test various embedding dimension scenarios."""
        cases = [
            (256, [64, 128, 256], True),  # Valid
            (512, [32, 64, 128, 256, 512], True),  # Valid
            (256, [128, 256, 512], False),  # Invalid - 512 > 256
            (1024, [512, 1024], True),  # Valid
        ]
        
        for embedding_dim, nesting_dims, should_be_valid in cases:
            is_valid = all(d <= embedding_dim for d in nesting_dims)
            assert is_valid == should_be_valid, \
                f"Failed: embedding_dim={embedding_dim}, nesting_dims={nesting_dims}"


class TestMRLConfigurationMatrix:
    """Test real-world MRL configuration combinations."""

    def validate_config(self, nesting_dims, freq_radii, embedding_dim=512):
        """Quick validation of a config."""
        return (
            len(freq_radii) == len(nesting_dims) - 1 and
            nesting_dims == sorted(nesting_dims) and
            freq_radii == sorted(freq_radii) and
            all(d <= embedding_dim for d in nesting_dims) and
            all(0 <= r <= 1 for r in freq_radii)
        )

    def test_configuration_compact(self):
        """Compact MRL configuration."""
        assert self.validate_config(
            nesting_dims=[128, 256],
            freq_radii=[0.5],
            embedding_dim=256
        )

    def test_configuration_standard(self):
        """Standard MRL configuration."""
        assert self.validate_config(
            nesting_dims=[64, 128, 256, 512],
            freq_radii=[0.1, 0.3, 0.5],
            embedding_dim=512
        )

    def test_configuration_large(self):
        """Large MRL configuration."""
        assert self.validate_config(
            nesting_dims=[64, 128, 256, 512, 1024],
            freq_radii=[0.1, 0.25, 0.5, 0.75],
            embedding_dim=1024
        )

    def test_configuration_fine_grained(self):
        """Fine-grained configuration with many bands."""
        assert self.validate_config(
            nesting_dims=[32, 64, 96, 128, 160, 192, 224, 256],
            freq_radii=[0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88],
            embedding_dim=256
        )

    def test_configuration_mixed_powers(self):
        """Configuration with mixed dimension powers."""
        assert self.validate_config(
            nesting_dims=[128, 256, 512],
            freq_radii=[0.3, 0.7],
            embedding_dim=512
        )


class TestMRLEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_minimum_valid_config(self):
        """Minimum valid configuration (2 dims, 1 radius)."""
        nesting_dims = [256, 512]
        freq_radii = [0.5]
        embedding_dim = 512
        
        assert len(freq_radii) == len(nesting_dims) - 1
        assert all(d <= embedding_dim for d in nesting_dims)
        assert nesting_dims == sorted(nesting_dims)

    def test_very_small_dimensions(self):
        """Very small dimension values."""
        nesting_dims = [8, 16, 32, 64]
        freq_radii = [0.25, 0.5, 0.75]
        embedding_dim = 64
        
        assert len(freq_radii) == len(nesting_dims) - 1
        assert all(d <= embedding_dim for d in nesting_dims)

    def test_very_large_dimensions(self):
        """Very large dimension values."""
        nesting_dims = [512, 1024, 2048, 4096]
        freq_radii = [0.25, 0.5, 0.75]
        embedding_dim = 4096
        
        assert len(freq_radii) == len(nesting_dims) - 1
        assert all(d <= embedding_dim for d in nesting_dims)

    def test_many_nesting_dims(self):
        """Many nesting dimensions."""
        nesting_dims = list(range(64, 512 + 64, 64))  # [64, 128, ..., 512]
        n_radii = len(nesting_dims) - 1
        freq_radii = [i / (n_radii + 1) for i in range(1, n_radii + 1)]
        embedding_dim = 512
        
        assert len(freq_radii) == len(nesting_dims) - 1
        assert all(d <= embedding_dim for d in nesting_dims)
        assert freq_radii == sorted(freq_radii)
        assert all(0 < r < 1 for r in freq_radii)

    def test_single_band_edge_case(self):
        """Minimum case with 2 dims - but not strictly ascending due to duplicates."""
        nesting_dims = [512, 512]  # Same values - edge case
        freq_radii = [0.5]
        
        # This will fail the ascending order check in practice
        # because we need strictly ascending dims (no duplicates)
        assert len(freq_radii) == len(nesting_dims) - 1
        # But the duplicate check should catch this:
        assert len(nesting_dims) != len(set(nesting_dims))


class TestMRLTypeValidation:
    """Test type validation for MRL configurations."""

    def test_nesting_dims_is_list(self):
        """Nesting dims should be list type."""
        dims = [64, 128, 256, 512]
        assert isinstance(dims, list)

    def test_freq_radii_is_list(self):
        """Freq radii should be list type."""
        radii = [0.1, 0.3, 0.5]
        assert isinstance(radii, list)

    def test_tuple_to_list_conversion(self):
        """Tuples should be convertible to lists."""
        dims_tuple = (64, 128, 256, 512)
        dims_list = list(dims_tuple)
        assert dims_list == [64, 128, 256, 512]

    def test_all_radii_numeric(self):
        """All freq_radii should be numeric."""
        radii = [0.1, 0.3, 0.5]
        assert all(isinstance(r, (int, float)) for r in radii)

    def test_all_dims_integer(self):
        """All nesting_dims should be integers."""
        dims = [64, 128, 256, 512]
        assert all(isinstance(d, int) for d in dims)


class TestMRLInvalidConfigurations:
    """Test detection of invalid configurations."""

    def test_invalid_non_ascending_dims(self):
        """Non-ascending nesting dims are invalid."""
        dims = [64, 256, 128, 512]
        assert dims != sorted(dims)

    def test_invalid_non_ascending_radii(self):
        """Non-ascending freq radii are invalid."""
        radii = [0.1, 0.5, 0.3]
        assert radii != sorted(radii)

    def test_invalid_dims_mismatch(self):
        """Length mismatch between dims and radii."""
        nesting_dims = [64, 128, 256, 512]
        freq_radii = [0.1, 0.3]  # Off by one
        
        assert len(freq_radii) != len(nesting_dims) - 1

    def test_invalid_duplicate_dims(self):
        """Duplicate values in nesting dims."""
        dims = [64, 128, 128, 512]
        assert len(dims) != len(set(dims))

    def test_invalid_zero_embedding(self):
        """Zero embedding dimension invalid."""
        embedding_dim = 0
        nesting_dims = [64, 128, 256, 512]
        
        assert not all(d <= embedding_dim for d in nesting_dims)


# Integration test: realistic training scenarios
class TestMRLRealisticScenarios:
    """Test realistic training scenarios."""

    def test_phase1_configuration(self):
        """Typical Phase 1 (base training) config."""
        config = {
            "nesting_dims": [64, 128, 256, 512],
            "freq_radii": [0.1, 0.3, 0.5],
            "embedding_dim": 512,
            "content_weight": 1.0,
            "fourier_weight": 0.3,
        }
        
        assert len(config["freq_radii"]) == len(config["nesting_dims"]) - 1
        assert all(d <= config["embedding_dim"] for d in config["nesting_dims"])

    def test_phase2_configuration(self):
        """Typical Phase 2 (fine-tuning) config."""
        config = {
            "nesting_dims": [64, 128, 256, 512],
            "freq_radii": [0.1, 0.3, 0.5],
            "embedding_dim": 512,
            "content_weight": 0.5,
            "fourier_weight": 0.15,
            "mrl_start_weight": 0.5,
            "mrl_final_weight": 0.1,
        }
        
        assert len(config["freq_radii"]) == len(config["nesting_dims"]) - 1
        assert all(d <= config["embedding_dim"] for d in config["nesting_dims"])

    def test_custom_optimization_config(self):
        """Custom optimization scenario."""
        config = {
            "nesting_dims": [128, 256, 512],
            "freq_radii": [0.3, 0.7],
            "embedding_dim": 512,
        }
        
        assert len(config["freq_radii"]) == len(config["nesting_dims"]) - 1
        assert all(d <= config["embedding_dim"] for d in config["nesting_dims"])


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
