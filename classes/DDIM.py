from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np


def extract(v, i, shape):
    """
    Get the i-th number in v, and the shape of v is mostly (T, ), the shape of i is mostly (batch_size, ).
    equal to [v[index] for index in i]
    """
    out = torch.gather(v, index=i, dim=0)
    out = out.to(device=i.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([i.shape[0]] + [1] * (len(shape) - 1))
    return out


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def linear_beta_schedule(timesteps, betas=None):
    if betas == None:
        betas = [1e-4, 0.02]
    return torch.linspace(*betas, timesteps)


class GaussianDiffuser(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        timesteps: int,
        betas=None,
        scheduler="linear",
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.T = timesteps

        # generate T steps of beta
        beta_t = (
            linear_beta_schedule(timesteps, betas)
            if scheduler == "linear"
            else cosine_beta_schedule(timesteps)
        )
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
        loss = F.mse_loss(epsilon_theta, epsilon)
        # loss = torch.sum(loss)
        return loss

    @torch.no_grad()
    def sample(
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
