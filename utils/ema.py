"""
Exponential Moving Average (EMA) for model weights.

EMA maintains a running average of model parameters which typically leads to
better sample quality and more stable training for generative models.
"""

import torch
from torch import nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average of model parameters.
    
    Args:
        model: The model to track
        decay: Decay rate (typical values: 0.999, 0.9999)
        update_every: Update EMA every N steps (for efficiency)
        update_after_step: Start EMA updates after N steps
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        update_every: int = 1,
        update_after_step: int = 0,
    ):
        self.decay = decay
        self.update_every = update_every
        self.update_after_step = update_after_step
        
        # Create shadow parameters
        self.model = model
        
        # Get the actual model (unwrap if DDP/FSDP wrapped)
        actual_model = model
        if hasattr(model, 'module'):
            actual_model = model.module
        
        # Deep copy and ensure it's on the same device
        self.shadow = deepcopy(actual_model)
        
        # Move shadow to same device as model
        device = next(model.parameters()).device
        self.shadow = self.shadow.to(device)
        
        self.shadow.eval()
        self.shadow.requires_grad_(False)
        
        self.step = 0
        
    @torch.no_grad()
    def update(self):
        """Update EMA parameters."""
        self.step += 1
        
        # Don't update before update_after_step
        if self.step < self.update_after_step:
            return
        
        # Only update every N steps
        if self.step % self.update_every != 0:
            return
        
        # Calculate effective decay
        # For the first few updates, use lower decay for faster convergence
        decay = self.decay
        if self.step < 1000:
            decay = min(self.decay, (1 + self.step) / (10 + self.step))
        
        # Get the actual model (unwrap if DDP/FSDP wrapped)
        actual_model = self.model
        if hasattr(self.model, 'module'):
            actual_model = self.model.module
        
        # Update shadow parameters
        for shadow_param, model_param in zip(
            self.shadow.parameters(), 
            actual_model.parameters()
        ):
            shadow_param.data.lerp_(model_param.data, 1 - decay)
    
    def get_model(self):
        """Get the EMA model."""
        return self.shadow
    
    def state_dict(self):
        """Get state dict for checkpointing."""
        return {
            'shadow': self.shadow.state_dict(),
            'step': self.step,
            'decay': self.decay,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.shadow.load_state_dict(state_dict['shadow'])
        self.step = state_dict['step']
        self.decay = state_dict['decay']

