import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Callable

sys.path.append(os.path.abspath("."))
from utils.import_file import import_file


def get_backbone_arch(arch: str) -> nn.Module:

    base_dir = os.path.join(os.getcwd(), "backbones")
    all_modules = import_file(
        "",
        path=os.path.join(base_dir, f"{arch}.py"),
    )
    return all_modules.Backbone


def get_sampler_arch(arch: str) -> nn.Module:

    base_dir = os.path.join(os.getcwd(), "samplers")
    all_modules = import_file(
        "",
        path=os.path.join(base_dir, f"{arch}.py"),
    )
    return all_modules.Sampler


def get_diffuser_arch(arch: str) -> nn.Module:

    base_dir = os.path.join(os.getcwd(), "diffusers")
    all_modules = import_file(
        "",
        path=os.path.join(base_dir, f"{arch}.py"),
    )
    return all_modules.Diffuser


def get_scheduler(scheduler: str) -> Callable:

    base_dir = os.path.join(os.getcwd(), "schedulers")
    all_modules = import_file(
        "",
        path=os.path.join(base_dir, f"{scheduler}.py"),
    )
    return all_modules.scheduler


def get_optimizer(optimizer: str) -> torch.optim.Optimizer:

    base_dir = os.path.join(os.getcwd(), "optimizers")
    all_modules = import_file(
        "",
        path=os.path.join(base_dir, f"{optimizer}.py"),
    )
    return all_modules.get_optim()


def get_dataset(
    dataset_name: str, input_res: list[int] = [32, 32], mean=[0.5], std=[0.5]
):

    base_dataset_dir = os.path.join(os.getcwd(), "datasets")
    all_modules = import_file(
        "",
        path=os.path.join(base_dataset_dir, f"{dataset_name}.py"),
    )
    return all_modules.get_dataset(input_res, mean, std)


def get_mean(
    dataloader: DataLoader,
    device: torch.device = "cpu",
):
    mean = 0.0
    for images, _ in dataloader:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataloader.dataset)
    return mean


def get_std(
    input_res: list[int],
    mean: torch.Tensor,
    dataloader: DataLoader,
    device: torch.device = "cpu",
):
    H, W = input_res
    var = 0.0
    for images, _ in dataloader:
        images = images.to(device)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(dataloader.dataset) * H * W))
    return std


def get_batch_mean(x: torch.Tensor):
    mean = 0
    batch_samples = x.size(0)
    images = x.view(batch_samples, x.size(1), -1)
    mean += images.mean(2).sum(0) / batch_samples
    return mean


def get_batch_std(x: torch.Tensor, mean: torch.Tensor, input_res: list[int]):
    H, W = input_res
    var = 0
    batch_samples = x.size(0)
    images = x.view(batch_samples, x.size(1), -1)
    var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (H * W * batch_samples))
    return std


def get_flow_backbone(config: dict):
    """
    Get flow matching backbone class based on config.
    
    Supports:
    - FlowNet: Convolutional backbone
    - DiT: Diffusion Transformer
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Backbone class (not instantiated)
    """
    model_type = config.get("model_type", "flownet")
    
    if model_type == "dit":
        from classes.Backbones.Transformers import DiffusionTransformer
        return DiffusionTransformer
    else:  # flownet or default
        from classes.Backbones.FlowNet import FlowNet
        return FlowNet
