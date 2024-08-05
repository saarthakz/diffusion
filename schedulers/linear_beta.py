import torch


def linear_beta_schedule(timesteps, betas=None, **kwargs):
    if betas == None:
        betas = [1e-4, 0.02]
    return torch.linspace(*betas, timesteps)
