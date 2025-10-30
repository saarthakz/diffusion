import torch
from torch import nn
import math


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal time embedding."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConvBlock(nn.Module):
    """Convolutional block with time conditioning."""
    
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        
        # Add time conditioning
        t_emb = self.time_mlp(t_emb)
        h = h + t_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h + self.residual_conv(x)


class FlowNet(nn.Module):
    """Simple ConvNet for Flow Matching with time conditioning."""
    
    def __init__(
        self,
        num_channels: int = 1,
        hidden_dims: list = [32, 64, 128, 64, 32],
        time_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.num_channels = num_channels
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        
        # Initial projection
        self.input_conv = nn.Conv2d(num_channels, hidden_dims[0], 3, padding=1)
        
        # Convolutional blocks with time conditioning
        self.blocks = nn.ModuleList([
            ConvBlock(hidden_dims[i], hidden_dims[i+1], time_dim)
            for i in range(len(hidden_dims) - 1)
        ])
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, hidden_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(hidden_dims[-1], num_channels, 3, padding=1),
        )
        
    def forward(self, t: torch.Tensor, x_t: torch.Tensor):
        """
        Forward pass with time conditioning.
        
        Args:
            t: Time tensor of shape (batch_size,)
            x_t: Image tensor at time t of shape (batch_size, channels, height, width)
        
        Returns:
            Predicted velocity field (same shape as x_t)
        """
        # Get time embedding
        t_emb = self.time_mlp(t)
        
        # Initial projection
        h = self.input_conv(x_t)
        
        # Apply blocks with time conditioning
        for block in self.blocks:
            h = block(h, t_emb)
        
        # Output
        out = self.output_conv(h)
        
        return out

