"""Test consistency loss with Schulman's k3 estimator."""
<<<<<<< HEAD
import torch
import torch.distributions as dist

=======

import torch
import torch.distributions as dist


>>>>>>> main
def test_kl_estimator_accuracy():
    """Verify k3 estimator matches true KL divergence."""
    # Ground truth distributions
    p = dist.Normal(loc=0.0, scale=1.0)
    q = dist.Normal(loc=0.1, scale=1.2)
<<<<<<< HEAD
    
    # True KL divergence
    true_kl = dist.kl_divergence(p, q)
    print(f"True KL: {true_kl:.6f}")
    
    # Sample from Q
    x = q.sample((10_000_000,))
    
    # Compute log ratio
    logr = p.log_prob(x) - q.log_prob(x)
    
    # k3 estimator
    k3 = ((logr.exp() - 1.0) - logr).mean()
    
=======

    # True KL divergence
    true_kl = dist.kl_divergence(p, q)
    print(f"True KL: {true_kl:.6f}")

    # Sample from Q
    x = q.sample((10_000_000,))

    # Compute log ratio
    logr = p.log_prob(x) - q.log_prob(x)

    # k3 estimator
    k3 = ((logr.exp() - 1.0) - logr).mean()

>>>>>>> main
    # Relative error
    rel_error = ((k3 - true_kl) / true_kl).abs()
    print(f"k3 estimate: {k3:.6f}")
    print(f"Relative error: {rel_error:.4%}")
<<<<<<< HEAD
    
    assert rel_error < 0.01, "k3 estimator should be within 1% of true KL"

=======

    assert rel_error < 0.01, "k3 estimator should be within 1% of true KL"


>>>>>>> main
def test_consistency_loss_behavior():
    """Test consistency loss decreases with similar transformations."""
    from src.model import FontDiffuserWithFST
    from src.modules.msse import MultiScaleStyleEncoder
    from src.modules.fst import FontStyleTransformationModule
<<<<<<< HEAD
    
    # Create minimal model
    mss_encoder = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5)
    fst_module = FontStyleTransformationModule(
        msse_output_channels=[64, 128, 256, 512, 1024],
        num_queries=220,
        query_dim=128
    )
    
=======

    # Create minimal model
    mss_encoder = MultiScaleStyleEncoder(in_channels=1, base_channels=64, num_scales=5)
    fst_module = FontStyleTransformationModule(
        msse_output_channels=[64, 128, 256, 512, 1024], num_queries=220, query_dim=128
    )

>>>>>>> main
    # Test case 1: Identical pairs → low loss
    batch_size, num_pairs = 2, 4
    identical_source = torch.randn(batch_size, num_pairs, 1, 96, 96)
    identical_target = identical_source.clone()
<<<<<<< HEAD
    
=======

>>>>>>> main
    # Mock model
    class MockModel:
        def __init__(self):
            self.mss_encoder = mss_encoder
            self.fst_module = fst_module
<<<<<<< HEAD
    
    model = MockModel()
    # Compute loss (need to extract method)
    # loss_identical = model.compute_consistency_loss(identical_source, identical_target)
    
    # Test case 2: Different pairs → higher loss
    varied_target = torch.randn(batch_size, num_pairs, 1, 96, 96)
    # loss_varied = model.compute_consistency_loss(identical_source, varied_target)
    
    # Assert: varied loss > identical loss
    # assert loss_varied > loss_identical

if __name__ == "__main__":
    test_kl_estimator_accuracy()
    print("✅ k3 estimator validated")
=======

    model = MockModel()
    # Compute loss (need to extract method)
    # loss_identical = model.compute_consistency_loss(identical_source, identical_target)

    # Test case 2: Different pairs → higher loss
    varied_target = torch.randn(batch_size, num_pairs, 1, 96, 96)
    # loss_varied = model.compute_consistency_loss(identical_source, varied_target)

    # Assert: varied loss > identical loss
    # assert loss_varied > loss_identical


if __name__ == "__main__":
    test_kl_estimator_accuracy()
    print("✅ k3 estimator validated")
>>>>>>> main
