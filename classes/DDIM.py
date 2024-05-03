import torch
import torch.nn as nn


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


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


class DDIM(nn.Module):
    def __init__(
        self,
        noise_pred_fn,
        *,
        input_res,
        num_channels=3,
        timesteps=1000,
        **kwargs,
    ):
        super().__init__()
        self.channels = num_channels
        self.image_size = input_res
        self.noise_pred_fn = noise_pred_fn

        self.num_timesteps = int(timesteps)

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    @torch.no_grad()
    def sample(self, noise: torch.Tensor, t=None, batch_size=16):

        self.noise_pred_fn.eval()
        if t == None:
            t = self.num_timesteps

        curr = noise
        direct_recons = None

        for curr_t in range(t, 1, -1):
            step = torch.full(
                (batch_size,),
                curr_t - 1,
                dtype=torch.long,
                device=curr.device,
            )

            predicted_noise = self.noise_pred_fn(curr, step)
            x_0_bar = (
                curr
                - extract(self.sqrt_one_minus_alphas_cumprod, step, curr.shape)
                * predicted_noise
            ) / extract(self.sqrt_alphas_cumprod, step, curr.shape)

            # 1 Shot Reconstruction
            if direct_recons is None:
                direct_recons = x_0_bar

            step_minus_one = torch.full(
                (batch_size,),
                curr_t - 2,
                dtype=torch.long,
                device=curr.device,
            )

            curr = self.add_noise(x_0_bar, predicted_noise, step_minus_one)

        self.noise_pred_fn.train()

        return noise, direct_recons, curr

    def add_noise(self, x_start, noise, timesteps):
        # simply use the alphas to interpolate
        return (
            extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape) * x_start  #
            + extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)
            * noise
        )

    def get_loss(self, x_start, noise, timesteps):
        noised = self.add_noise(x_start=x_start, noise=noise, timesteps=timesteps)
        predicted_noise = self.noise_pred_fn(noised, timesteps)

        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        return loss

    def forward(self, x: torch.Tensor, *args, **kwargs):
        (b, c, h, w, img_h, img_w) = (*x.shape, *self.image_size)

        assert (
            h == img_h and w == img_w
        ), f"height and width of image must be {img_h} {img_w}"

        # Noise for a random number of timesteps
        timesteps = torch.randint(
            low=0, high=self.num_timesteps, size=(b,), device=x.device
        ).long()

        noise = torch.randn_like(x, device=x.device)
        return self.get_loss(x, noise, timesteps, *args, **kwargs)
