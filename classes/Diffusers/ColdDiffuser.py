import torch
import torch.nn as nn
import os
import sys

sys.path.append(os.path.abspath("."))
from utils.extract import extract


class ColdDiffuser(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        scheduler: function,
        betas=None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.T = timesteps

        # generate T steps of beta
        beta_t = scheduler(timesteps, betas)
        self.register_buffer("beta_t", beta_t)

        # calculate the cumulative product of $\alpha$ , named $\bar{\alpha_t}$ in paper
        alpha_t = 1.0 - self.beta_t
        alpha_t_bar = torch.cumprod(alpha_t, dim=0)

        # calculate and store two coefficient of $q(x_t | x_0)$
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_t_bar))
        self.register_buffer(
            "sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_t_bar)
        )

    @torch.no_grad
    def forward(self, x: torch.Tensor, *args, **kwargs):

        # Noise for a random number of timesteps
        timesteps = torch.randint(low=0, high=self.T, size=(x.shape,), device=x.device)

        epsilon = torch.randn_like(x)

        x_t = (
            extract(self.sqrt_alpha_cumprod, timesteps, x.shape) * x  #
            + extract(self.sqrt_one_minus_alpha_cumprod, timesteps, x.shape) * epsilon
        )

        x_recon = self.model(x_t, timesteps)

        loss = torch.nn.functional.mse_loss(x, x_recon)

        return loss
