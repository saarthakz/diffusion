import torch
import torch.nn as nn

import os
import sys

sys.path.append(os.path.abspath("."))
from utils.extract import extract
from typing import Callable


class StandardDiffuser(nn.Module):

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

    def forward(self, x):
        # get a random training step $t \sim Uniform({1, ..., T})$
        t = torch.randint(low=0, high=self.T, size=(x.shape[0],), device=x.device)

        # generate $\epsilon \sim N(0, 1)$
        epsilon = torch.randn_like(x)

        # predict the noise added from $x_{t-1}$ to $x_t$
        x_t = (
            extract(self.sqrt_alpha_cumprod, t, x.shape) * x
            + extract(self.sqrt_one_minus_alpha_cumprod, t, x.shape) * epsilon
        )

        epsilon_theta = self.model(x_t, t)

        # get the gradient
        loss = torch.nn.functional.mse_loss(epsilon_theta, epsilon)

        return loss
