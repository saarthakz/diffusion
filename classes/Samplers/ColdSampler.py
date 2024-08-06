import torch
import torch.nn as nn
import os
import sys
from typing import Callable

sys.path.append(os.path.abspath("."))
from utils.extract import extract


class ColdGaussianDiffuserSampler(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        scheduler: Callable,
        betas=None,
        **kwargs,
    ):
        super().__init__()
        self.model = model

        self.num_timesteps = int(timesteps)

        betas = scheduler(timesteps, betas)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    @torch.no_grad()
    def forward(self, noise, t=1):

        batch_size, *_ = noise.shape
        self.model.eval()
        if t == None:
            t = self.num_timesteps

        curr = noise
        direct_recons = None

        while t:
            step = torch.full(
                (batch_size,),
                t - 1,
                dtype=torch.long,
                device=noise.device,
            )

            x_0_bar = self.model(curr, step)  # Get 1 Shot recon

            predicted_noise = (
                curr - extract(self.sqrt_alphas_cumprod, step, x_0_bar.shape) * x_0_bar
            ) / extract(self.sqrt_one_minus_alphas_cumprod, step, x_0_bar.shape)

            # 1 Shot Reconstruction
            if direct_recons is None:
                direct_recons = x_0_bar

            x_t_bar = self.add_noise(
                x_start=x_0_bar, noise=predicted_noise, timesteps=step
            )

            x_t_minus_one_bar = x_0_bar

            if t - 1 != 0:
                step_minus_one = torch.full(
                    (batch_size,),
                    t - 2,
                    dtype=torch.long,
                    device=noise.device,
                )

                x_t_minus_one_bar = self.add_noise(
                    x_start=x_0_bar,
                    noise=predicted_noise,
                    timesteps=step_minus_one,
                )

            curr = curr - x_t_bar + x_t_minus_one_bar
            t = t - 1

        self.model.train()

        return noise, direct_recons, curr
