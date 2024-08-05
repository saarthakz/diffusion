import torch


def extract(vec: torch.Tensor, indices: torch.Tensor, shape: torch.Size):
    """
    Get the indices-th number in vec, and the shape of vec is mostly (T, ), the shape of indices is mostly (batch_size, ).
    equal to [vec[index] for index in indices]
    """
    out = torch.gather(vec, index=indices, dim=0)
    out = out.to(device=indices.device, dtype=torch.float32)

    # reshape to (batch_size, 1, 1, 1, 1, ...) for broadcasting purposes.
    out = out.view([indices.shape[0]] + [1] * (len(shape) - 1))
    return out
