import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch


class TestFontDiffuserWithFST:
    """Test FontDiffuserWithFST model."""

    @pytest.mark.unit
    def test_model_initialization(self):
        """Test model can be initialized."""
        # Mock original FontDiffuser
        mock_original = Mock()
        mock_original.content_encoder = Mock(spec=nn.Module)
        mock_original.unet = Mock(spec=nn.Module)
        mock_original.style_encoder = Mock(spec=nn.Module)
        
        # Import and instantiate
        from src.model import FontDiffuserWithFST
        
        model = FontDiffuserWithFST(
            original_fontdiffuser=mock_original,
            feature_channels=[64, 128, 256, 512, 1024],
            num_queries=256,
            query_dim=128,
            num_scales=5,
        )
        
        assert model is not None
        assert hasattr(model, 'content_encoder')
        assert hasattr(model, 'diffusion_unet')
        assert hasattr(model, 'style_encoder')
        assert hasattr(model, 'mss_encoder')
        assert hasattr(model, 'fst_module')

    @pytest.mark.unit
    def test_forward_pass_shapes(
        self,
        sample_latents,
        sample_timestep,
        sample_content_images,
        sample_style_images,
    ):
        """Test forward pass produces correct output shapes."""
        from src.model import FontDiffuserWithFST
        
        # Create mock original FontDiffuser
        mock_original = Mock()
        
        # Mock content encoder output
        content_feat = torch.randn(2, 128, 12, 12)
        mock_original.content_encoder = Mock(
            return_value=(content_feat, [torch.randn(2, 64, 24, 24)])
        )
        
        # Mock style encoder output
        style_feat = torch.randn(2, 1024, 6, 6)
        style_vec = torch.randn(2, 1024)
        mock_original.style_encoder = Mock(
            return_value=(style_feat, style_vec, [torch.randn(2, 512, 12, 12)])
        )
        
        # Mock U-Net output
        noise_pred = torch.randn(2, 4, 12, 12)
        offset_out = torch.tensor(0.0)
        mock_original.unet = Mock(return_value=(noise_pred, offset_out))
        mock_original.config = Mock(cross_attention_dim=1280)
        
        model = FontDiffuserWithFST(mock_original)
        
        outputs = model(
            noisy_latents=sample_latents,
            timestep=sample_timestep,
            content_img=sample_content_images,
            style_source_img=sample_style_images,
            style_target_img=sample_style_images,
            return_dict=True,
        )
        
        assert outputs['noise_pred'].shape == (2, 4, 12, 12)
        assert 'transformation_features' in outputs
        assert 'fst_condition' in outputs

    @pytest.mark.unit
    def test_get_loss_dict(self, sample_latents):
        """Test loss computation."""
        from src.model import FontDiffuserWithFST
        
        mock_original = Mock()
        model = FontDiffuserWithFST(mock_original)
        
        outputs = {
            'noise_pred': torch.randn(2, 4, 12, 12),
            'offset_out_sum': torch.tensor(0.5),
        }
        target_noise = torch.randn(2, 4, 12, 12)
        
        losses = model.get_loss_dict(outputs, target_noise, reduction='mean')
        
        assert 'noise_loss' in losses
        assert 'offset_loss' in losses
        assert 'total_loss' in losses
        assert losses['noise_loss'].item() >= 0
        assert losses['offset_loss'].item() >= 0


class TestFontDiffuserModel:
    """Test original FontDiffuserModel."""

    @pytest.mark.unit
    def test_fontdiffuser_model_forward(self):
        """Test FontDiffuserModel forward pass."""
        from src.model import FontDiffuserModel
        
        # Create mocks
        mock_unet = Mock(return_value=(torch.randn(2, 4, 12, 12), torch.tensor(0.0)))
        mock_style_encoder = Mock(return_value=(torch.randn(2, 1024, 6, 6), None, None))
        mock_content_encoder = Mock(
            return_value=(torch.randn(2, 128, 12, 12), [torch.randn(2, 64, 24, 24)])
        )
        
        model = FontDiffuserModel(
            unet=mock_unet,
            style_encoder=mock_style_encoder,
            content_encoder=mock_content_encoder,
        )
        
        x_t = torch.randn(2, 4, 12, 12)
        timesteps = torch.tensor([100, 200])
        style_imgs = torch.randn(2, 1, 96, 96)
        content_imgs = torch.randn(2, 1, 96, 96)
        
        noise_pred, offset_out = model(
            x_t=x_t,
            timesteps=timesteps,
            style_images=style_imgs,
            content_images=content_imgs,
            content_encoder_downsample_size=4,
        )
        
        assert noise_pred.shape == (2, 4, 12, 12)