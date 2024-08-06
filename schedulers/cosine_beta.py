import torch


def scheduler(timesteps, scale=0.008, **kwargs):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Returns all the betas for the scheduler
    """
    steps = timesteps + 1
    x = torch.linspace(0, steps, steps)
    alphas_cumprod = (
        torch.cos(((x / steps) + scale) / (1 + scale) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)
