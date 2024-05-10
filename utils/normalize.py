import torch


def normalize(tensor: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    """
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean: Mean of size (C, )
        std: Standard deviation of size (C, )
    Returns:
        Tensor: Normalized image.
    """
    for tensor_slice, channel_mean, channel_std in zip(tensor, mean, std):
        tensor_slice.sub_(channel_std).div_(channel_mean)
    return tensor
