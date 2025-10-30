import torch
from torch import nn
from tqdm import tqdm
import os
import sys
from typing import Callable

sys.path.append(os.path.abspath("."))


class FlowMatchingSampler(nn.Module):
    """
    Algorithm 1: Sampling from a Flow Model with Euler method.
    """
    
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 100,
        scheduler: Callable = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.T = timesteps
        # scheduler is not used in flow matching but kept for API compatibility
        
    @torch.no_grad()
    def forward(self, x_t, steps: int = None):
        """
        Sample by integrating the ODE from t=0 to t=1 using Euler method.
        
        Args:
            x_t: Initial noise sample X_0 ~ p_init (at t=0)
            steps: Number of integration steps n (defaults to self.T)
        
        Returns:
            Sample X_1 at t=1
        """
        if steps is None:
            steps = self.T
            
        B, *_ = x_t.shape
        
        # Set t = 0
        t = 0.0
        
        # Set step size h = 1/n
        h = 1.0 / steps
        
        # X_0 is already drawn from p_init (passed as x_t)
        X_t = x_t
        
        progress_bar = tqdm(range(steps))
        
        # for i = 1, ..., n do
        for i in range(steps):
            # Create time tensor
            t_tensor = torch.full((B,), t, device=X_t.device, dtype=torch.float32)
            
            # X_{t+h} = X_t + h * u_θ_t(X_t)
            u_theta_t = self.model(t_tensor, X_t)
            X_t = X_t + h * u_theta_t
            
            # Update t ← t + h
            t = t + h
            
            progress_bar.update(1)
        
        # return X_1
        return X_t

