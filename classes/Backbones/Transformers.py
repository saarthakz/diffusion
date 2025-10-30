import torch
from torch import nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Use PyTorch's optimized scaled_dot_product_attention
        # Automatically uses Flash Attention or Memory-Efficient Attention when available
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False
        )
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_features = hidden_features or in_features * 4
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with self-attention and MLP."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, dropout=dropout)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding with MLP."""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # MLP to process the sinusoidal embedding
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, t: torch.Tensor):
        """
        Args:
            t: (batch_size,) timesteps in [0, 1]
        Returns:
            (batch_size, dim) timestep embeddings
        """
        half_dim = self.dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        emb = self.mlp(emb)
        return emb


class AdaLNBlock(nn.Module):
    """Transformer block with Adaptive Layer Normalization (AdaLN) for time conditioning.
    
    This is the key innovation in DiT - instead of adding time embeddings,
    we modulate the layer norm parameters based on timestep.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.mlp = MLP(embed_dim, dropout=dropout)
        
        # AdaLN modulation - predicts scale and shift for each norm layer
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, 6 * embed_dim, bias=True)
        )
        
    def forward(self, x, t_emb):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
            t_emb: (batch_size, embed_dim) timestep embeddings
        """
        # Get modulation parameters: scale and shift for each of the two norms
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        
        # Attention block with AdaLN
        x = x + gate_msa.unsqueeze(1) * self.attn(
            self.modulate(self.norm1(x), shift_msa, scale_msa)
        )
        
        # MLP block with AdaLN
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        
        return x
    
    @staticmethod
    def modulate(x, shift, scale):
        """Apply affine transformation: x * (1 + scale) + shift"""
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiffusionTransformer(nn.Module):
    """
    Diffusion Transformer (DiT) for pixel-space or patch-based generation.
    
    Supports two modes:
    - Pixel mode: Each pixel is a token (good for small images like MNIST)
    - Patch mode: Image is divided into patches (better for larger images)
    
    Args:
        img_size: Input image size (assumed square)
        patch_size: Patch size (if 1, operates in pixel mode)
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        dropout: Dropout rate
        use_adaln: Whether to use Adaptive Layer Norm (recommended for diffusion)
    """
    
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        dropout: float = 0.0,
        use_adaln: bool = True,
        num_channels: int = None,  # Alias for in_channels
        input_res: list = None,  # Extract img_size from input_res
        **kwargs,
    ):
        super().__init__()
        # Handle parameter aliases from config
        if num_channels is not None:
            in_channels = num_channels
        if input_res is not None:
            img_size = input_res[0]
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.use_adaln = use_adaln
        
        # Number of patches
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Patch embedding
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        
        # Positional embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Timestep embedding
        self.time_embed = TimestepEmbedding(embed_dim)
        
        # Transformer blocks
        block_class = AdaLNBlock if use_adaln else Block
        if use_adaln:
            self.blocks = nn.ModuleList([
                AdaLNBlock(embed_dim, num_heads, dropout)
                for _ in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, dropout)
                for _ in range(depth)
            ])
        
        # Output layers
        self.norm_final = nn.LayerNorm(embed_dim, elementwise_affine=not use_adaln)
        self.output_proj = nn.Linear(embed_dim, self.patch_dim)
        
        # Final modulation for AdaLN
        if use_adaln:
            self.final_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(embed_dim, 2 * embed_dim, bias=True)
            )
    
    def patchify(self, x):
        """
        Convert image to patches.
        Args:
            x: (B, C, H, W)
        Returns:
            patches: (B, num_patches, patch_dim)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Reshape to patches
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H//p, W//p, C, p, p)
        x = x.reshape(B, -1, C * p * p)  # (B, num_patches, patch_dim)
        
        return x
    
    def unpatchify(self, x):
        """
        Convert patches back to image.
        Args:
            x: (B, num_patches, patch_dim)
        Returns:
            image: (B, C, H, W)
        """
        B = x.shape[0]
        p = self.patch_size
        h = w = int(self.num_patches ** 0.5)
        
        x = x.reshape(B, h, w, self.in_channels, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, h, p, w, p)
        x = x.reshape(B, self.in_channels, h * p, w * p)
        
        return x
    
    def forward(self, t: torch.Tensor, x_t: torch.Tensor):
        """
        Forward pass.
        
        Args:
            t: (batch_size,) timesteps in [0, 1]
            x_t: (batch_size, in_channels, img_size, img_size) noisy images
        
        Returns:
            (batch_size, in_channels, img_size, img_size) predicted velocity/noise
        """
        # Patchify input
        x = self.patchify(x_t)  # (B, num_patches, patch_dim)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Get timestep embedding
        t_emb = self.time_embed(t)  # (B, embed_dim)
        
        # Apply transformer blocks
        if self.use_adaln:
            for block in self.blocks:
                x = block(x, t_emb)
            
            # Final norm with modulation
            shift, scale = self.final_modulation(t_emb).chunk(2, dim=-1)
            x = self.norm_final(x)
            x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        else:
            # Add time embedding to input (simpler approach)
            x = x + t_emb.unsqueeze(1)
            for block in self.blocks:
                x = block(x)
            x = self.norm_final(x)
        
        # Project to output
        x = self.output_proj(x)  # (B, num_patches, patch_dim)
        
        # Unpatchify to image
        x = self.unpatchify(x)  # (B, C, H, W)
        
        return x


