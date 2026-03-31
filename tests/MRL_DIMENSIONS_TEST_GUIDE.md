# MRL Dimensions Pytest Suite

## Overview

Comprehensive pytest suite for validating all MRL (Matryoshka Representation Learning) dimension compatibility in FontDiffuserWithMRL training.

**Files Created:**
- `tests/test_mrl_dimensions_standalone.py` - ✅ **RECOMMENDED** (standalone, no heavy imports)
- `tests/test_mrl_dimensions.py` - Alternative (requires torch environment)

**Test Results:** ✅ **47/47 tests PASSED**

---

## Test Coverage

### 1. **Dimension Parsing** (6 tests)
Validates parsing of dimension strings and configurations:
- `test_parse_nesting_dims_string` - Parse "64,128,256,512" format
- `test_parse_nesting_dims_list` - Accept list input as-is
- `test_parse_nesting_dims_with_spaces` - Handle whitespace in strings
- `test_parse_freq_radii_string` - Parse "0.1,0.3,0.5" format
- `test_parse_freq_radii_list` - Accept list input as-is
- `test_parse_freq_radii_with_spaces` - Handle whitespace

### 2. **Dimensions Compatibility** (8 tests)
Core compatibility validation:
- ✅ `test_default_dimensions_valid` - Default config passes validation
- ❌ `test_freq_radii_too_few` - Detects missing radii values
- ❌ `test_freq_radii_too_many` - Detects extra radii values
- ❌ `test_nesting_dims_not_ascending` - Detects non-sorted order
- ❌ `test_freq_radii_not_ascending` - Detects radii out of order
- ❌ `test_dimension_exceeds_embedding` - Detects dims > embedding_dim
- ❌ `test_freq_radii_outside_range` - Detects radii outside [0,1]
- ✅ `test_custom_valid_dimensions` - Multiple valid combinations

### 3. **Frequency Radii Validation** (6 tests)
Validates frequency bands:
- ✅ `test_radii_in_valid_range` - All in [0, 1]
- ✅ `test_radii_boundary_zero` - Boundary value 0
- ✅ `test_radii_boundary_one` - Boundary value 1
- ❌ `test_radii_negative_invalid` - Rejects negative values
- ❌ `test_radii_above_one_invalid` - Rejects values > 1
- ✅ `test_radii_ascending` - Checks sort order

### 4. **Embedding Dimension Constraints** (4 tests)
Validates against embedding dimension:
- ✅ `test_all_dims_less_than_embedding` - All dims ≤ embedding_dim
- ❌ `test_dims_exceed_embedding` - Detects violations
- ✅ `test_largest_dim_equals_embedding` - Valid when max equals embedding
- ✅ `test_custom_embedding_dims` - Various embedding scenarios

### 5. **Configuration Matrix** (5 tests)
Real-world MRL configurations:
- `test_configuration_compact` - 2 dims, 1 radius
- `test_configuration_standard` - 4 dims, 3 radii (default)
- `test_configuration_large` - 5 dims, 4 radii
- `test_configuration_fine_grained` - 8 dims, 7 radii
- `test_configuration_mixed_powers` - Custom mix

### 6. **Edge Cases** (5 tests)
Boundary conditions:
- `test_minimum_valid_config` - 2 dims, 1 radius
- `test_very_small_dimensions` - Small values [8, 16, 32, 64]
- `test_very_large_dimensions` - Large values [512, 1024, 2048, 4096]
- `test_many_nesting_dims` - Many bands (8+ dimensions)
- `test_single_band_edge_case` - Duplicate dims handling

### 7. **Type Validation** (5 tests)
Type safety checks:
- `test_nesting_dims_is_list` - List type
- `test_freq_radii_is_list` - List type
- `test_tuple_to_list_conversion` - Convert from tuple
- `test_all_radii_numeric` - Numeric values
- `test_all_dims_integer` - Integer values

### 8. **Invalid Configurations** (5 tests)
Detection of errors:
- ❌ `test_invalid_non_ascending_dims` - RejectsUnsorted dimensions
- ❌ `test_invalid_non_ascending_radii` - Rejects unsorted radii
- ❌ `test_invalid_dims_mismatch` - Wrong count
- ❌ `test_invalid_duplicate_dims` - Duplicate values
- ❌ `test_invalid_zero_embedding` - Zero embedding_dim

### 9. **Realistic Scenarios** (3 tests)
Real training configurations:
- `test_phase1_configuration` - Base training setup
- `test_phase2_configuration` - Fine-tuning with SCR
- `test_custom_optimization_config` - Custom optimization

---

## Key Validation Rules

All valid MRL configurations must satisfy:

```python
# 1. Length matching
len(freq_radii) == len(nesting_dims) - 1

# 2. Ascending order
nesting_dims == sorted(nesting_dims)
freq_radii == sorted(freq_radii)

# 3. Unique values in nesting_dims
len(nesting_dims) == len(set(nesting_dims))

# 4. Embedding dimension constraint
all(d <= embedding_dim for d in nesting_dims)

# 5. Frequency radii range
all(0 <= r <= 1 for r in freq_radii)
```

---

## Running the Tests

### Option 1: Run Standalone Suite (RECOMMENDED)
```bash
cd d:\School\FontDiffusion
python -m pytest tests/test_mrl_dimensions_standalone.py -v
```

### Option 2: Run with Full Environment
```bash
python -m pytest tests/test_mrl_dimensions.py -v
```

### Option 3: Run Specific Test Class
```bash
python -m pytest tests/test_mrl_dimensions_standalone.py::TestMRLDimensionsCompatibility -v
```

### Option 4: Run Single Test
```bash
python -m pytest tests/test_mrl_dimensions_standalone.py::TestMRLConfigurationMatrix::test_configuration_standard -v
```

---

## Default Configurations Tested

### ✅ Valid Configurations

| Name | Nesting Dims | Freq Radii | Embedding Dim | Status |
|------|---|---|---|---|
| **Compact** | [128, 256] | [0.5] | 256 | ✅ |
| **Standard (Default)** | [64, 128, 256, 512] | [0.1, 0.3, 0.5] | 512 | ✅ |
| **Large** | [64, 128, 256, 512, 1024] | [0.1, 0.25, 0.5, 0.75] | 1024 | ✅ |
| **Fine-Grained** | [32, 64, 96, 128, 160, 192, 224, 256] | [0.12, 0.25, 0.38, 0.5, 0.62, 0.75, 0.88] | 256 | ✅ |

### ❌ Invalid Configurations (Detected)

| Issue | Example | Error |
|-------|---------|-------|
| Length Mismatch | dims=[64,128,256,512], radii=[0.1,0.3] | `len(freq_radii) != len(nesting_dims) - 1` |
| Wrong Order | dims=[64,256,128,512] | `nesting_dims not in ascending order` |
| Exceeds Embedding | dims=[64,128,256,512], embedding=256 | `512 > 256` |
| Outside Range | radii=[0.1, 0.3, 1.5] | `1.5 not in [0, 1]` |
| Duplicates | dims=[64, 128, 128, 512] | `len != len(set())` |

---

## Integration with Training

These tests validate configurations BEFORE they reach the trainer:

```python
# trainer_mrl.py validates in _setup_models()
if len(self.mrl_freq_radii) != len(self.mrl_nesting_dims) - 1:
    raise ValueError(
        f"MRL dimension mismatch! "
        f"nesting_dims={self.mrl_nesting_dims}, "
        f"freq_radii={self.mrl_freq_radii}"
    )
```

---

## Usage Examples

### Training with Default Config
```bash
python train_mrl.py \
    --use_fst \
    --use_mrl \
    --mrl_nesting_dims="64,128,256,512" \
    --mrl_freq_radii="0.1,0.3,0.5"
```

### Training with Custom Config
```bash
python train_mrl.py \
    --use_fst \
    --use_mrl \
    --mrl_nesting_dims="128,256,512" \
    --mrl_freq_radii="0.2,0.6"
```

### Validation Before Training
The pytest suite validates:
```python
assert len([0.2, 0.6]) == len([128, 256, 512]) - 1  # ✅ 2 == 2
assert [0.2, 0.6] == sorted([0.2, 0.6])              # ✅ True
assert all(d <= 512 for d in [128, 256, 512])        # ✅ True
```

---

## Maintenance

When adding new MRL configurations, add tests to:
1. `TestMRLConfigurationMatrix` - for new valid combos
2. `TestMRLRealisticScenarios` - for training scenarios
3. `TestMRLEdgeCases` - for boundary conditions

Example:
```python
def test_new_configuration(self):
    """Test new XYZ configuration."""
    assert self.validate_config(
        nesting_dims=[YOUR_DIMS],
        freq_radii=[YOUR_RADII],
        embedding_dim=YOUR_EMBEDDING
    )
```

---

## Summary

✅ **47 comprehensive tests** ensure MRL dimensions are always compatible.
✅ **No external dependencies** (standalone version uses pure Python).
✅ **Covers edge cases** from minimum (2 dims) to large (8+ dims).
✅ **Integration-ready** catches errors before training starts.

Run tests now:
```bash
pytest tests/test_mrl_dimensions_standalone.py -v
```
