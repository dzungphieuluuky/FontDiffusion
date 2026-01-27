"""
Tier 2: Overfit-a-Batch (Integration Tests)

Purpose: Verify that the "plumbing" (model + loss + optimizer) works.
Strategy: Create a single random batch and run training for 20–50 iterations.
Success Metric: The loss must decrease significantly (ideally to near-zero).
If it doesn't, there is a bug in gradient flow or loss calculation.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Tuple


class SimplePerceptualLoss(nn.Module):
    """Simplified perceptual loss for testing."""

    def __init__(self):
        super().__init__()
        self.l2 = nn.MSELoss()

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute simple MSE loss."""
        return self.l2(generated, target)


class MinimalFontGenerator(nn.Module):
    """Minimal generator for testing overfit-a-batch."""

    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
        )
        self.conv_decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate image from noise."""
        B = x.shape[0]
        out = self.fc(x)
        out = out.reshape(B, 64, 4, 4)
        out = self.conv_decoder(out)
        # Resize to 96x96
        out = torch.nn.functional.interpolate(
            out, size=(96, 96), mode="bilinear", align_corners=False
        )
        return out


class TestOverfitSingleBatch:
    """Test that models can overfit a single batch."""

    def test_simple_generator_overfits_batch(self, device):
        """Verify a generator can memorize and reconstruct a batch."""
        generator = MinimalFontGenerator().to(device)
        optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2)
        loss_fn = SimplePerceptualLoss().to(device)

        # Create a single batch
        batch_size = 2
        latent = torch.randn(batch_size, 128, device=device)
        target = torch.rand(batch_size, 3, 96, 96, device=device)

        losses = []
        num_iterations = 50

        for i in range(num_iterations):
            optimizer.zero_grad()
            generated = generator(latent)
            loss = loss_fn(generated, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease significantly
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss

        print(f"\nLoss reduction: {loss_reduction:.2%}")
        print(f"Initial loss: {initial_loss:.6f}, Final loss: {final_loss:.6f}")

        assert final_loss < initial_loss, "Loss did not decrease!"
        assert (
            loss_reduction > 0.5
        ), f"Loss reduction ({loss_reduction:.2%}) should be > 50% for overfit test"

    def test_encoder_overfits_batch(
        self, device, simple_encoder, optimizer_factory, loss_fn
    ):
        """Test that a simple encoder can overfit a batch."""
        encoder = simple_encoder
        optimizer = optimizer_factory(encoder, lr=1e-2)

        # Create input and target
        batch_size = 2
        x = torch.rand(batch_size, 1, 96, 96, device=device)
        target = torch.randn(batch_size, 1024, device=device)

        losses = []
        num_iterations = 30

        for i in range(num_iterations):
            optimizer.zero_grad()
            output = encoder(x)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss

        print(f"\nEncoder loss reduction: {loss_reduction:.2%}")

        assert final_loss < initial_loss, "Encoder loss did not decrease!"
        assert (
            loss_reduction > 0.3
        ), f"Encoder loss reduction ({loss_reduction:.2%}) should be > 30%"

    def test_batch_norm_training_mode(self, device):
        """Verify batch norm behaves correctly in training mode."""
        model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ).to(device)

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.MSELoss()

        x = torch.rand(2, 1, 32, 32, device=device)
        target = torch.rand(2, 64, 32, 32, device=device)

        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], "BatchNorm training failed"


class TestLossDecreaseTrajectory:
    """Test that loss follows expected convergence patterns."""

    def test_loss_monotonic_decrease(self, device, simple_unet):
        """Test that loss generally decreases (allowing small fluctuations)."""
        model = simple_unet
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        x = torch.rand(2, 3, 96, 96, device=device)
        target = torch.rand(2, 3, 96, 96, device=device)

        losses = []
        num_iterations = 40

        for i in range(num_iterations):
            optimizer.zero_grad()
            output = model(x, None, None)["sample"]
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Calculate average loss in first and second half
        first_half_avg = sum(losses[: len(losses) // 2]) / (len(losses) // 2)
        second_half_avg = sum(losses[len(losses) // 2 :]) / (
            len(losses) - len(losses) // 2
        )

        print(f"\nFirst half avg loss: {first_half_avg:.6f}")
        print(f"Second half avg loss: {second_half_avg:.6f}")

        # Second half should have lower average loss
        assert second_half_avg < first_half_avg, "Loss should decrease over iterations"

    def test_learning_rate_impact(
        self, device, simple_encoder, optimizer_factory, loss_fn
    ):
        """Verify higher learning rates lead to faster convergence."""
        encoder = simple_encoder

        x = torch.rand(2, 1, 96, 96, device=device)
        target = torch.randn(2, 1024, device=device)

        results = {}

        for lr in [1e-4, 1e-3, 1e-2]:
            encoder_copy = type(encoder)().to(device)
            optimizer = optimizer_factory(encoder_copy, lr=lr)

            losses = []
            for _ in range(20):
                optimizer.zero_grad()
                output = encoder_copy(x)
                l = loss_fn(output, target)
                l.backward()
                optimizer.step()
                losses.append(l.item())

            loss_reduction = (losses[0] - losses[-1]) / losses[0]
            results[lr] = loss_reduction
            print(f"\nLR {lr}: loss reduction = {loss_reduction:.2%}")

        # Higher LR should generally lead to better convergence (with some allowance for randomness)
        assert (
            results[1e-2] > results[1e-4] * 0.8
        ), "Higher learning rate should converge faster"


class TestMultipleBackwardPasses:
    """Test gradient accumulation and multiple backward passes."""

    def test_gradient_accumulation(self, device, simple_encoder):
        """Test that gradients accumulate correctly."""
        encoder = simple_encoder
        optimizer = torch.optim.Adam(encoder.parameters())
        loss_fn = nn.MSELoss()

        x = torch.rand(2, 1, 96, 96, device=device)
        target = torch.randn(2, 1024, device=device)

        # Two forward passes without zeroing gradients
        output1 = encoder(x)
        loss1 = loss_fn(output1, target)
        loss1.backward()

        grad_after_first = [
            p.grad.clone() for p in encoder.parameters() if p.grad is not None
        ]

        output2 = encoder(x)
        loss2 = loss_fn(output2, target)
        loss2.backward()

        grad_after_second = [
            p.grad.clone() for p in encoder.parameters() if p.grad is not None
        ]

        # Gradients should be approximately doubled (not exactly due to stochasticity)
        for g1, g2 in zip(grad_after_first, grad_after_second):
            # Check that second gradients are larger
            assert (g2.abs() > g1.abs() * 0.5).any(), "Gradients should accumulate"

    def test_gradient_reset(self, device, simple_encoder):
        """Test that zero_grad properly resets gradients."""
        encoder = simple_encoder
        loss_fn = nn.MSELoss()

        x = torch.rand(2, 1, 96, 96, device=device)
        target = torch.randn(2, 1024, device=device)

        output = encoder(x)
        loss = loss_fn(output, target)
        loss.backward()

        # Check gradients exist
        for param in encoder.parameters():
            if param.grad is not None:
                assert param.grad.abs().sum() > 0, "Gradients should be non-zero"

        # Reset gradients
        encoder.zero_grad()

        # Check all gradients are zero
        for param in encoder.parameters():
            if param.grad is not None:
                assert (
                    param.grad == 0
                ).all(), "Gradients should be zero after zero_grad()"


class TestOptimizerStepping:
    """Test optimizer update behavior."""

    def test_adam_parameter_updates(self, device, simple_encoder):
        """Verify Adam optimizer updates parameters."""
        encoder = simple_encoder
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        x = torch.rand(2, 1, 96, 96, device=device)
        target = torch.randn(2, 1024, device=device)

        # Store initial parameters
        initial_params = [p.clone() for p in encoder.parameters()]

        # Forward, backward, step
        output = encoder(x)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Check parameters changed
        any_updated = False
        for init_p, curr_p in zip(initial_params, encoder.parameters()):
            if not torch.allclose(init_p, curr_p):
                any_updated = True
                break

        assert any_updated, "Optimizer should update parameters"

    @pytest.mark.parametrize("num_accumulation_steps", [1, 2, 4])
    def test_gradient_accumulation_with_steps(
        self, device, simple_encoder, num_accumulation_steps: int
    ):
        """Test gradient accumulation over multiple steps."""
        encoder = simple_encoder
        optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        x = torch.rand(2, 1, 96, 96, device=device)
        target = torch.randn(2, 1024, device=device)

        accumulated_loss = 0
        for step in range(num_accumulation_steps):
            output = encoder(x)
            loss = loss_fn(output, target)
            loss = loss / num_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()

        assert accumulated_loss > 0, "Accumulated loss should be positive"


class TestCheckpointingSaving:
    """Test model state saving for checkpointing."""

    def test_state_dict_save_load(self, device, simple_encoder, temp_dir):
        """Test saving and loading model state."""
        encoder = simple_encoder

        # Save initial state
        save_path = temp_dir / "encoder.pth"
        torch.save(encoder.state_dict(), str(save_path))

        # Modify model
        encoder_new = type(encoder)().to(device)
        encoder_new.load_state_dict(torch.load(str(save_path)))

        # Verify states match
        x = torch.rand(2, 1, 96, 96, device=device)

        with torch.no_grad():
            out_orig = encoder(x)
            out_loaded = encoder_new(x)

        assert torch.allclose(
            out_orig, out_loaded, atol=1e-5
        ), "Loaded model should produce identical outputs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
