from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import sys
from typing import Callable

sys.path.append(os.path.abspath("."))

from utils.extract import extract


class DDIM(nn.Module):
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

    @torch.no_grad()
    def forward(
        self,
        x_t,
        steps: int = 1,
    ):
        B, *_ = x_t.shape

        time_steps = np.asarray(
            list(range(0, self.T, self.T // steps))
        )  # This ensures that the number of time steps is equal to the arg: steps

        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        # previous sequence
        time_steps_prev = np.concatenate([[0], time_steps[:-1]])

        progress_bar = tqdm(range(len(time_steps)))

        for idx, (time_step, prev_time_step) in enumerate(
            zip(reversed(time_steps), reversed(time_steps_prev))
        ):

            t = torch.full((B,), time_step, device=x_t.device, dtype=torch.long)
            t_minus_one = torch.full(
                (B,), prev_time_step, device=x_t.device, dtype=torch.long
            )

            # # predict noise using model
            epsilon_theta_t = self.model(x_t, t)

            x_zero_hat = (
                x_t
                - epsilon_theta_t
                * extract(self.sqrt_one_minus_alpha_cumprod, t, x_t.shape)
            ) / extract(self.sqrt_alpha_cumprod, t, x_t.shape)

            x_t = (
                extract(self.sqrt_alpha_cumprod, t_minus_one, x_t.shape) * x_zero_hat
                + extract(self.sqrt_one_minus_alpha_cumprod, t_minus_one, x_t.shape)
                * epsilon_theta_t
            )

            progress_bar.update(1)

        return x_t
