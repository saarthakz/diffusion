import os
import sys

sys.path.append(os.path.abspath("."))
import torch
import torch.nn as nn
from classes.Unet import Unet
from classes.ColdGaussianDiffuser import ColdGaussianDiffuser
from torchvision.utils import save_image, make_grid
from utils.get_dataset import get_dataset
from torch.utils.data import DataLoader
import argparse
import json
from random import random
import math


def main(config: dict):
    model_name = config["model_name"]
    model_dir = os.path.join(os.getcwd(), "models", model_name)

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

    dataset = get_dataset(config["dataset"], config["input_res"])

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
    )

    x, y = next(iter(data_loader))

    step_counts = 10
    step_diff = config["timesteps"] // step_counts

    steps = [
        torch.full(size=(1,), fill_value=(idx + 1) * step_diff - 1).long()
        for idx in range(step_counts)
    ]

    noise = torch.randn_like(x)

    for idx in range(step_counts):
        save_image(
            model.add_noise(x, noise, steps[idx]),
            fp=f"temp/{idx}.png",
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
