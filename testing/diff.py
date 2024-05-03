import os
import sys

sys.path.append(os.path.abspath("."))
import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from classes.Unet import Unet
from classes.ColdGaussianDiffuser import ColdGaussianDiffuser
from accelerate import Accelerator, DistributedDataParallelKwargs
import argparse
import json
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
    unet = Unet(
        dim=config["dim"],
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=config["with_time_emb"],
        residual=config["residual"],
    )

    model = ColdGaussianDiffuser(
        unet,
        image_size=config["input_res"],
        channels=config["num_channels"],
        timesteps=config["timesteps"],  # number of steps
    )

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

    noise, direct_recons, curr = model.module.sample(
        noise=starting_noise, t=config["timesteps"]
    )

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
    save_image(
        make_grid(direct_recons, nrow=int(math.sqrt(config["sample_batch_size"]))),
        fp=os.path.join(
            base_path,
            f"{rand}_oneshots.png",
        ),
    )
    accelerator.print(curr.shape)


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
