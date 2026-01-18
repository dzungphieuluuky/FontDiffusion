# FontDiffuser Deep Learning Test Suite

Professional 4-tier pytest suite for FontDiffuser, focusing on math, memory, and convergence verification rather than just logic testing.

## Overview

This test suite follows the 2026 best practices for deep learning testing:

1. **Tier 1: Shape & Tensor Invariants (Unit Tests)**
   - Verify output tensor shapes match expectations
   - Detect NaNs and gradient explosions early
   - Parametrized tests for different batch sizes

2. **Tier 2: Overfit-a-Batch (Integration Tests)**
   - Verify model + loss + optimizer "plumbing" works
   - Run 20-50 training iterations on a single batch
   - Success metric: loss decreases significantly (>30%)

3. **Tier 3: Data Integrity & Leakage (Validation Tests)**
   - Test shuffle behavior actually reorders data
   - Verify augmentations stay in valid ranges (0-1)
   - Check for label leakage (target not in inputs)
   - Validate dataset loading and collation

4. **Tier 4: Gradient Flow Checks (Backpropagation Validation)**
   - Ensure all parameters receive gradients after `loss.backward()`
   - Identify "dead" or unused layers
   - Test gradient accumulation and stability

## Directory Structure

```
tests/
├── conftest.py                      # Shared fixtures and configuration
├── test_tier1_shapes.py            # Shape & Tensor Invariants tests
├── test_tier2_overfit.py           # Overfit-a-Batch tests
├── test_tier3_data_integrity.py    # Data Integrity & Leakage tests
├── test_tier4_gradients.py         # Gradient Flow tests
└── test_fst_integration.py         # FST module integration tests
```

## Key Fixtures (from conftest.py)

### Device Handling
- `device`: Automatically detects GPU/CPU
- `use_small_model`: Returns True for CPU testing

### Determinism
- `seed_everything`: Auto-used fixture ensuring reproducible tests
- Thread-safe seeding: `torch.manual_seed()` + `np.random.seed()`

### Test Data
- `sample_noisy_latents`: Diffusion model latents (B, 4, H//8, W//8)
- `sample_content_image`: Font content (B, 1, 96, 96)
- `sample_style_image`: Font style (B, 1, 96, 96)
- `sample_target_image`: Ground truth (B, 1, 96, 96)
- `dummy_style_features`: Multi-scale style features from MSSE

### Model Components
- `simple_encoder`: Minimal encoder for testing
- `simple_unet`: Minimal UNet for testing
- `optimizer_factory`: Create Adam optimizers with custom LR
- `scheduler_factory`: Create learning rate schedulers

### Assertion Helpers
- `assert_no_nan()`: Check tensor has no NaNs
- `assert_no_inf()`: Check tensor has no Infs
- `assert_shape()`: Verify tensor shape
- `assert_gradients_exist()`: Check all parameters have gradients

## Running Tests

### Run all tests
```bash
pytest tests/ -v
```

### Run by tier
```bash
pytest tests/test_tier1_shapes.py -v        # Shape tests
pytest tests/test_tier2_overfit.py -v       # Overfit tests
pytest tests/test_tier3_data_integrity.py -v  # Data tests
pytest tests/test_tier4_gradients.py -v     # Gradient tests
```

### Run FST-specific tests
```bash
pytest tests/test_fst_integration.py -v
```

### Run specific test class
```bash
pytest tests/test_tier1_shapes.py::TestMultiScaleStyleEncoderShapes -v
```

### Run with markers
```bash
pytest -m "tier1" -v              # All Tier 1 tests
pytest -m "fst" -v               # All FST tests
pytest -m "not slow" -v          # Skip slow tests
```

### Run with coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

### Run on GPU only
```bash
pytest tests/ -m "gpu" -v
```

### Run on CPU only
```bash
pytest tests/ -m "cpu" -v
```

## Test Configuration

Configuration is in `pytest.ini`:
- Test discovery patterns
- Custom markers for organizing tests
- Timeout settings (300 seconds default)
- Logging configuration
- Doctest options

## Key Test Patterns

### Parametrized Tests
Testing multiple configurations efficiently:
```python
@pytest.mark.parametrize("batch_size,num_scales", [
    (1, 5), (2, 5), (4, 5), (8, 3),
])
def test_msse_output_shapes(self, batch_size, num_scales, device):
    # test code
```

### NaN Detection
```python
def test_forward_no_nan(self, device, assert_no_nan):
    output = model(x)
    assert_no_nan(output, "model output")
```

### Overfit Verification
```python
losses = []
for i in range(50):
    output = model(x)
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Loss should decrease >30%
assert (losses[0] - losses[-1]) / losses[0] > 0.3
```

### Gradient Flow Checking
```python
output = model(x)
loss = loss.backward()

for name, param in model.named_parameters():
    assert param.grad is not None, f"{name} has no gradient"
    assert not torch.isnan(param.grad).any(), f"{name} has NaN"
```

### Data Leakage Prevention
```python
# Verify target not in inputs
assert torch.corrcoef(torch.stack([combined, target]))[0, 1].abs() < 0.5
```

## Common Issues & Solutions

### Tests failing with CUDA out of memory
- Use `use_small_model` fixture to reduce model sizes on CPU
- Set `CUDA_VISIBLE_DEVICES=""` to force CPU testing
- Run smaller batches with `@pytest.mark.parametrize("batch_size", [1, 2])`

### Non-deterministic test failures
- Check that `seed_everything` fixture is applied
- Verify all random sources use seeded RNG
- Use explicit device placement (`.to(device)`)

### Slow tests
- Mark with `@pytest.mark.slow` and skip with `pytest -m "not slow"`
- Use smaller models and shorter training loops for regular CI
- Use `test_tier2_overfit.py` as smoke tests

### Gradient-related failures
- Check model is in training mode: `model.train()`
- Verify all parameters have `requires_grad=True`
- Check for frozen layers preventing backprop
- Use `assert_gradients_exist()` helper

## Best Practices

1. **Always use fixtures**: Device, seeding, and data are managed centrally
2. **Test determinism**: Use seeds for reproducible failures
3. **Parametrize extensively**: Test multiple batch sizes, configurations
4. **Check NaNs early**: Catch numerical issues immediately
5. **Verify gradient flow**: Every parameter should learn
6. **Monitor loss curves**: Overfitting on small batches proves plumbing works
7. **Use small models**: Keep test runtime under 1 minute per test

## Debugging Failed Tests

### Enable verbose output
```bash
pytest tests/test_tier1_shapes.py::TestMultiScaleStyleEncoderShapes::test_msse_output_shapes -vv
```

### Print intermediate values
```python
# Add to test
print(f"\nOutput shape: {output.shape}")
print(f"Max gradient: {max(p.grad.abs().max() for p in model.parameters())}")
```

### Drop into pdb on failure
```bash
pytest tests/ --pdb
```

### Show local variables on failure
```bash
pytest tests/ -l
```

## Integration with CI/CD

Recommended workflow:
1. Run Tier 1 & 3 tests on every commit (fast, ~30 seconds)
2. Run Tier 2 & 4 tests on PRs (slower, ~2 minutes)
3. Run full suite nightly with coverage report

Example GitHub Actions workflow:
```yaml
- name: Run Tier 1 & 3 tests
  run: pytest tests/ -m "tier1 or tier3" -v

- name: Run Tier 2 & 4 tests  
  run: pytest tests/ -m "tier2 or tier4" -v

- name: Coverage report
  run: pytest tests/ --cov=src --cov-report=xml
```

## Contributing New Tests

When adding new tests:
1. Place in appropriate tier file
2. Use existing fixtures for consistency
3. Add docstring explaining test purpose
4. Use parametrization for multiple cases
5. Add assertions for all failure modes
6. Mark with relevant markers
7. Ensure deterministic with seeds
8. Run locally before committing

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [PyTorch Testing Best Practices](https://pytorch.org/docs/stable/testing.html)
- [Deep Learning Testing 2024 Survey](https://arxiv.org/abs/2402.10018)

## License

Same as FontDiffuser project
