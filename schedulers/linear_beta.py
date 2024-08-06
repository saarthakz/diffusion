import torch


def scheduler(timesteps, betas=None, **kwargs):
    if betas == None:
        betas = [1e-4, 0.02]
    return torch.linspace(*betas, timesteps)
