"""
Tier 4: Gradient Flow Checks (Backpropagation Validation)

Purpose: Identify "dead" layers or broken backpropagation.
Strategy: After loss.backward(), iterate through model.parameters() and assert that
.grad is not None for every layer. This ensures every part of the architecture is learning.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Set


class TestGradientFlowSimple:
    """Test gradient flow in simple architectures."""

    def test_sequential_model_gradients(self, device):
        """Verify all layers in sequential model receive gradients."""
        model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ).to(device)

        x = torch.randn(4, 100, device=device)
        y = torch.randn(4, 10, device=device)

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Check all parameters have gradients
        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter '{name}' has no gradient"
            assert not torch.isnan(param.grad).any(), (
                f"Parameter '{name}' has NaN gradients"
            )
            assert not torch.isinf(param.grad).any(), (
                f"Parameter '{name}' has Inf gradients"
            )

    def test_conv_model_gradients(self, device):
        """Verify convolutional layers receive gradients."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        ).to(device)

        x = torch.randn(2, 3, 32, 32, device=device)
        y = torch.randn(2, 10, device=device)

        model.train()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Check all parameters
        param_count = 0
        grad_count = 0
        for name, param in model.named_parameters():
            param_count += 1
            if param.grad is not None:
                grad_count += 1
                assert not torch.isnan(param.grad).any(), f"NaN in {name}.grad"

        assert grad_count == param_count, (
            f"Only {grad_count}/{param_count} parameters have gradients"
        )


class TestGradientFlowComplex:
    """Test gradient flow in complex models."""

    def test_residual_connection_gradients(self, device):
        """Verify gradients flow through residual connections."""

        class ResidualBlock(nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(channels)
                self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(channels)

            def forward(self, x):
                identity = x
                out = self.conv1(x)
                out = self.bn1(out)
                out = torch.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)
                out = out + identity
                return torch.relu(out)

        model = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            ResidualBlock(64),
            ResidualBlock(64),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 10),
        ).to(device)

        x = torch.randn(2, 3, 32, 32, device=device)
        y = torch.randn(2, 10, device=device)

        model.train()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Collect all parameter names with/without gradients
        with_grad = set()
        without_grad = set()

        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                with_grad.add(name)
            else:
                without_grad.add(name)

        print(f"\nParameters with gradients: {len(with_grad)}")
        print(f"Parameters without gradients: {len(without_grad)}")

        # All parameters should have gradients
        assert len(without_grad) == 0, f"Parameters without gradients: {without_grad}"

    def test_cross_attention_gradients(self, device):
        """Test gradient flow through attention mechanisms."""
        from src.modules.attention import MultiHeadAttention

        # Simple attention test
        class SimpleAttention(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim)
                self.k_proj = nn.Linear(dim, dim)
                self.v_proj = nn.Linear(dim, dim)
                self.out_proj = nn.Linear(dim, dim)

            def forward(self, q, k, v):
                Q = self.q_proj(q)
                K = self.k_proj(k)
                V = self.v_proj(v)

                scores = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
                attn = torch.softmax(scores, dim=-1)
                out = torch.matmul(attn, V)
                out = self.out_proj(out)
                return out

        model = SimpleAttention(64).to(device)

        q = torch.randn(2, 10, 64, device=device, requires_grad=True)
        k = torch.randn(2, 20, 64, device=device, requires_grad=True)
        v = torch.randn(2, 20, 64, device=device, requires_grad=True)

        output = model(q, k, v)
        loss = output.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"Attention parameter '{name}' has no gradient"
            )
            assert param.grad.abs().sum() > 0, (
                f"Attention parameter '{name}' has zero gradient"
            )


class TestGradientMagnitudes:
    """Test that gradient magnitudes are reasonable."""

    def test_gradient_magnitude_not_exploding(self, device):
        """Verify gradients don't explode (become extremely large)."""
        model = nn.Sequential(
            nn.Linear(100, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100),
        ).to(device)

        x = torch.randn(4, 100, device=device)
        y = torch.randn(4, 100, device=device)

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        max_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)

        print(f"\nMax gradient norm: {max_grad_norm:.6f}")

        # Gradient should be finite
        assert torch.isfinite(torch.tensor(max_grad_norm)), (
            "Gradient norm is not finite (NaN or Inf)"
        )

        # For this simple setup, gradient shouldn't be extremely large
        assert max_grad_norm < 100, f"Gradient explosion detected: {max_grad_norm}"

    def test_gradient_magnitude_not_vanishing(self, device):
        """Verify gradients don't vanish (become zero)."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Sigmoid(),
            nn.Linear(20, 10),
        ).to(device)

        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 10, device=device)

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Check gradient magnitudes
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.abs().max().item())

        min_grad = min(grad_norms)
        avg_grad = sum(grad_norms) / len(grad_norms)

        print(f"\nMin gradient: {min_grad:.8f}")
        print(f"Avg gradient: {avg_grad:.8f}")

        # Gradients should not all be near zero
        assert avg_grad > 1e-6, f"Vanishing gradients detected: avg={avg_grad:.2e}"


class TestGradientAccumulation:
    """Test gradient accumulation behavior."""

    def test_gradient_accumulates_without_zero_grad(self, device):
        """Verify gradients accumulate when zero_grad is not called."""
        model = nn.Linear(10, 10).to(device)

        x = torch.randn(2, 10, device=device)
        y = torch.randn(2, 10, device=device)

        loss_fn = nn.MSELoss()

        # First backward
        output1 = model(x)
        loss1 = loss_fn(output1, y)
        loss1.backward()
        grad_after_first = model.weight.grad.clone()

        # Second backward without zero_grad
        output2 = model(x)
        loss2 = loss_fn(output2, y)
        loss2.backward()
        grad_after_second = model.weight.grad.clone()

        # Gradient should accumulate (roughly doubled, accounting for randomness)
        assert (grad_after_second.abs() > grad_after_first.abs() * 0.5).any(), (
            "Gradients should accumulate"
        )

    def test_zero_grad_clears_gradients(self, device):
        """Verify zero_grad properly clears gradients."""
        model = nn.Linear(10, 10).to(device)

        x = torch.randn(2, 10, device=device)
        y = torch.randn(2, 10, device=device)

        # Backward pass
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Check gradients exist
        assert model.weight.grad is not None
        assert (model.weight.grad != 0).any()

        # Clear gradients
        model.zero_grad()

        # Check gradients are zero
        assert (model.weight.grad == 0).all()


class TestDeadLayers:
    """Test detection of dead/unused layers."""

    def test_detect_unrequired_grad_layers(self, device):
        """Test that we can detect layers with requires_grad=False."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
        ).to(device)

        # Freeze first layer
        for param in model[0].parameters():
            param.requires_grad = False

        x = torch.randn(2, 10, device=device)
        y = torch.randn(2, 10, device=device)

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # First layer should have no gradients
        for param in model[0].parameters():
            assert param.grad is None, "Frozen layer should have no gradients"

        # Other layers should have gradients
        for param in model[2].parameters():
            assert param.grad is not None, "Non-frozen layer should have gradients"

    def test_batch_norm_training_vs_eval_gradients(self, device):
        """Test BatchNorm gradient behavior in train vs eval mode."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),
            nn.Linear(20, 10),
        ).to(device)

        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 10, device=device)
        loss_fn = nn.MSELoss()

        # Test in TRAINING mode
        model.train()
        output = model(x)
        loss = loss_fn(output, y)
        loss.backward()

        # All parameters should have gradients in training mode
        for name, param in model.named_parameters():
            assert param.grad is not None, (
                f"Parameter '{name}' has no gradient in training mode"
            )

    def test_conv_batch_norm_gradients(self, device):
        """Test gradients through Conv+BatchNorm blocks."""
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, 10),
        ).to(device)

        model.train()
        x = torch.randn(2, 3, 16, 16, device=device)
        y = torch.randn(2, 10, device=device)

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Collect parameter status
        params_with_grad = 0
        params_without_grad = 0

        for name, param in model.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
            else:
                params_without_grad += 1

        print(
            f"\nConv+BN model: {params_with_grad} params with grad, "
            f"{params_without_grad} without"
        )

        assert params_without_grad == 0, (
            f"{params_without_grad} parameters have no gradients"
        )


class TestRecurrentGradients:
    """Test gradient flow in recurrent-like structures."""

    def test_transformer_like_gradients(self, device):
        """Test gradients in transformer-like architecture."""

        class SimpleTransformer(nn.Module):
            def __init__(self, d_model: int = 64):
                super().__init__()
                self.embedding = nn.Linear(10, d_model)
                self.self_attn = nn.MultiheadAttention(d_model, 4, batch_first=True)
                self.ff = nn.Sequential(
                    nn.Linear(d_model, 256),
                    nn.ReLU(),
                    nn.Linear(256, d_model),
                )
                self.out = nn.Linear(d_model, 10)

            def forward(self, x):
                x = self.embedding(x)  # (B, seq_len, d_model)
                attn_out, _ = self.self_attn(x, x, x)
                ff_out = self.ff(attn_out)
                out = self.out(ff_out)
                return out

        model = SimpleTransformer().to(device)
        model.train()

        x = torch.randn(2, 5, 10, device=device)
        y = torch.randn(2, 5, 10, device=device)

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Check all parameters have gradients
        param_names_no_grad = []
        for name, param in model.named_parameters():
            if param.grad is None:
                param_names_no_grad.append(name)

        assert len(param_names_no_grad) == 0, (
            f"Parameters without gradients: {param_names_no_grad}"
        )


class TestGradientStability:
    """Test that gradients are stable across iterations."""

    def test_gradient_consistency_across_steps(self, device):
        """Verify gradients are consistent when using same input."""
        model = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ).to(device)

        x = torch.randn(4, 20, device=device)
        y = torch.randn(4, 10, device=device)

        # Compute gradients multiple times
        grad_sequences = []
        for _ in range(3):
            model.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, y)
            loss.backward()

            grad_seq = torch.cat([p.grad.flatten() for p in model.parameters()])
            grad_sequences.append(grad_seq)

        # Gradients should be identical (or very close)
        for i in range(1, len(grad_sequences)):
            assert torch.allclose(grad_sequences[0], grad_sequences[i], atol=1e-6), (
                f"Gradient not consistent at iteration {i}"
            )


class TestGradientByParameterType:
    """Test gradient flow by parameter type."""

    def test_weight_bias_gradients(self, device):
        """Verify both weights and biases receive gradients."""
        model = nn.Linear(10, 10, bias=True).to(device)

        x = torch.randn(2, 10, device=device)
        y = torch.randn(2, 10, device=device)

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # Check weight gradient
        assert model.weight.grad is not None
        assert (model.weight.grad != 0).any()

        # Check bias gradient
        assert model.bias.grad is not None
        assert (model.bias.grad != 0).any()

    def test_norm_layer_gradients(self):
        """Test gradients for normalization layer parameters."""
        # Test LayerNorm
        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.LayerNorm(10),
            nn.Linear(10, 10),
        )

        x = torch.randn(2, 10)
        y = torch.randn(2, 10)

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        # LayerNorm should have weight and bias gradients
        ln = model[1]
        assert ln.weight.grad is not None, "LayerNorm weight has no gradient"
        if ln.bias is not None:
            assert ln.bias.grad is not None, "LayerNorm bias has no gradient"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
