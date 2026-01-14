from dataclasses import dataclass
import logging

logger = logging.getLogger("TrainingConfig")
@dataclass
class TrainingConfig:
    """Configuration for training with validation."""
    learning_rate: float
    train_batch_size: int
    max_train_steps: int
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "fp16"
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Loss coefficients
    perceptual_coefficient: float = 1.0
    offset_coefficient: float = 0.1
    sc_coefficient: float = 1.0
    style_transform_coefficient: float = 0.1
    
    # Training phases
    phase_1: bool = False
    phase_2: bool = False
    
    # Classifier-free guidance
    drop_prob: float = 0.1
    
    # Style transformation
    enable_style_transform: bool = False
    
    def validate(self):
        """Validate configuration values."""
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.train_batch_size > 0, "Batch size must be positive"
        assert self.max_train_steps > 0, "Max train steps must be positive"
        assert 0 <= self.drop_prob <= 1, "Drop probability must be between 0 and 1"
        
        if self.phase_1 and self.phase_2:
            logger.warning("Both phase_1 and phase_2 are enabled")
