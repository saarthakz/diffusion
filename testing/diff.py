import os
import sys

sys.path.append(os.path.abspath("."))
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from classes.UNet import UNet
from classes.UNet_two import UNet as UNet_two
from classes.ColdGaussianDiffuser import ColdGaussianDiffuser
from accelerate import Accelerator, DistributedDataParallelKwargs
import argparse
import json
from utils.get_model_arch import get_model_arch
from utils.unnormalize import unnormalize
from random import random
import math


def main(config: dict):
    model_name = config["model_name"]
    model_dir = os.path.join(os.getcwd(), "models", model_name)

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=config["accelerate_find_unused_params"]
    )
    accelerator = Accelerator(
        project_dir=model_dir,
        kwargs_handlers=[ddp_kwargs],
    )

    # Print the config file
    accelerator.print(config)

    # Model
    unet = UNet_two(**config)

    model = get_model_arch(config["model_arch"])(unet, **config)

    accelerator.print(model)

    # Print # of model parameters
    accelerator.print(
        sum(param.numel() for param in model.parameters()) / 1e6, "M parameters"
    )

    # Load a model checkpoint
    if config["model_from_checkpoint"]:
        model.load_state_dict(torch.load(f=config["model_checkpoint_path"]))
        accelerator.print(
            "Model loaded from checkpoint: ", config["model_checkpoint_path"]
        )

    # Acceleration :P
    model = accelerator.prepare_model(model=model)

    starting_noise = torch.randn(
        size=(
            config["sample_batch_size"],
            config["num_channels"],
            *config["input_res"],
        ),
        device=accelerator.device,
    )

    curr = model.module.sample(starting_noise, config["sampling_steps"])
    curr = unnormalize(curr, config["dataset_mean"], config["dataset_std"])

    base_path = os.path.join(model_dir, "images")
    os.makedirs(base_path, exist_ok=True)

    rand = str(int((random() * 100)))

    save_image(
        make_grid(curr, nrow=int(math.sqrt(config["sample_batch_size"]))),
        fp=os.path.join(
            base_path,
            f"{rand}_samples.png",
        ),
    )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to the config JSON file",
    )
    args = vars(arg_parser.parse_args())
    config_file = open(file=args["config_file"], mode="r")
    config = json.load(config_file)
    main(config)
