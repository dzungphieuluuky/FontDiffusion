# MRL Dimension Error - Complete Solution

## Problem Summary

**Error:** `AssertionError: len(freq_radii) must equal len(nesting_dims) - 1`

**Cause:** `--mrl_freq_radii` argument is not being passed in your training command, and the remote environment running Linux (`/usr/local/bin/accelerate`) likely has **old code without updated defaults**.

---

## Solutions

### Solution 1: Quick Fix (Immediate) ⚡

Add `--mrl_freq_radii="0.1,0.3,0.5"` to your training command.

**Your current command:**
```bash
accelerate launch train_mrl.py \
    --mrl_nesting_dims=64,128,256,512 \
    ...other args...
```

**Add this line (or similar based on your nesting_dims):**
```bash
accelerate launch train_mrl.py \
    --mrl_nesting_dims=64,128,256,512 \
    --mrl_freq_radii=0.1,0.3,0.5 \
    ...other args...
```

**Required ratio:** Always ensure `len(mrl_freq_radii) == len(mrl_nesting_dims) - 1`

| nesting_dims | Required freq_radii | Example |
|---|---|---|
| 2 dims | 1 radius | `--mrl_nesting_dims="256,512" --mrl_freq_radii="0.5"` |
| 3 dims | 2 radii | `--mrl_nesting_dims="128,256,512" --mrl_freq_radii="0.3,0.7"` |
| 4 dims | 3 radii | `--mrl_nesting_dims="64,128,256,512" --mrl_freq_radii="0.1,0.3,0.5"` |
| 5 dims | 4 radii | `--mrl_nesting_dims="64,128,256,512,1024" --mrl_freq_radii="0.1,0.25,0.5,0.75"` |

---

### Solution 2: Pre-Validation (Recommended) ✅

**Before training**, run the validation script to catch errors early:

```bash
python validate_mrl_config.py \
    --mrl_nesting_dims="64,128,256,512" \
    --mrl_freq_radii="0.1,0.3,0.5"
```

**Output if valid:**
```
✅ Length matching: 3 == 3
✅ nesting_dims in ascending order
✅ freq_radii in ascending order
✅ All nesting_dims <= embedding_dim (512)
✅ All freq_radii in range [0, 1]

✅ VALIDATION PASSED - Configuration is valid!
```

---

### Solution 3: Deploy Updated Code (Permanent) 🔄

Push your local fixes to the remote environment:

```bash
# Files that have been updated locally:
# - train_mrl.py (line 44: default="0.1,0.3,0.5")
# - src/trainers/trainer_mrl.py (line 86 + validation)

git status  # Check changes
git add train_mrl.py src/trainers/trainer_mrl.py
git commit -m "fix: correct MRL freq_radii defaults and add validation"
git push origin main

# On remote server:
cd /path/to/FontDiffusion
git pull origin main
```

---

## Validation Rules

All MRL configurations must satisfy:

```python
# 1. Length: exactly one more nesting_dim than freq_radii
len(freq_radii) == len(nesting_dims) - 1

# 2. Order: must be sorted (ascending)
nesting_dims == sorted(nesting_dims)
freq_radii == sorted(freq_radii)

# 3. Embedding: all dims must fit in embedding space
all(d <= embedding_dim for d in nesting_dims)

# 4. Range: all radii must be between 0 and 1
all(0 <= r <= 1 for r in freq_radii)
```

---

## Examples

### ✅ Valid Configurations

```bash
# Compact (2 dimensions)
--mrl_nesting_dims="256,512" --mrl_freq_radii="0.5"

# Standard (4 dimensions - DEFAULT)
--mrl_nesting_dims="64,128,256,512" --mrl_freq_radii="0.1,0.3,0.5"

# Large (5 dimensions)
--mrl_nesting_dims="64,128,256,512,1024" --mrl_freq_radii="0.1,0.25,0.5,0.75"

# Fine-grained (8 dimensions)
--mrl_nesting_dims="32,64,96,128,160,192,224,256" \
--mrl_freq_radii="0.12,0.25,0.38,0.5,0.62,0.75,0.88"
```

### ❌ Invalid Configurations

```bash
# WRONG: Too few radii (would fail)
--mrl_nesting_dims="64,128,256,512" --mrl_freq_radii="0.1,0.3"
# Error: len(freq_radii)=2 must equal len(nesting_dims)-1=3

# WRONG: Not ascending (would fail)
--mrl_nesting_dims="64,256,128,512" --mrl_freq_radii="0.1,0.3,0.5"
# Error: nesting_dims not in ascending order

# WRONG: Radii out of range (would fail)
--mrl_nesting_dims="64,128,256,512" --mrl_freq_radii="0.1,0.3,1.5"
# Error: Some freq_radii outside [0,1]
```

---

## Testing

Run the comprehensive pytest suite locally to validate all scenarios:

```bash
# 47 tests covering all dimension compatibility cases
pytest tests/test_mrl_dimensions_standalone.py -v

# Or specific scenarios:
pytest tests/test_mrl_dimensions_standalone.py::TestMRLConfigurationMatrix -v
```

---

## Files Updated

| File | Changes | Line |
|------|---------|------|
| `train_mrl.py` | Default freq_radii changed to "0.1,0.3,0.5" | 44 |
| `src/trainers/trainer_mrl.py` | Default freq_radii changed to "0.1,0.3,0.5" | 86 |
| `src/trainers/trainer_mrl.py` | Added dimension validation with clear error messages | 187-192 |
| `validate_mrl_config.py` | **NEW** - Pre-training validation script | - |
| `tests/test_mrl_dimensions_standalone.py` | **NEW** - 47 comprehensive tests | - |

---

## Next Steps

1. **Immediate:** Add `--mrl_freq_radii="0.1,0.3,0.5"` to your training command
2. **Before training:** Run `python validate_mrl_config.py --mrl_nesting_dims="64,128,256,512" --mrl_freq_radii="0.1,0.3,0.5"`
3. **Permanent:** `git push` updates to remote environment and `git pull` on remote server

---

## Support

If issues persist:

1. Check that `--mrl_freq_radii` matches your `--mrl_nesting_dims` count
2. Run validation script: `python validate_mrl_config.py ...`
3. Verify remote environment has latest code: `git pull && git log --oneline -5`
4. Check pytest results: `pytest tests/test_mrl_dimensions_standalone.py -v`
