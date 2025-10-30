import os
import sys

sys.path.append(os.path.abspath("."))
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.auto import tqdm
from datetime import datetime, UTC
import argparse
import json
from utils.logger import Logger
from utils.get import get_dataset, get_optimizer


def main(config: dict):
    model_name = f'{config["model_name"]} {datetime.now(UTC).strftime("%d-%m-%Y %H:%M:%S")} UTC'
    model_dir = os.path.join(os.getcwd(), "models", model_name)

    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=config.get("accelerate_find_unused_params", False)
    )
    accelerator = Accelerator(
        project_dir=model_dir,
        log_with="wandb" if config["tracking"] else None,
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        kwargs_handlers=[ddp_kwargs],
    )
    
    accelerator.print(config)

    if config["tracking"]:
        accelerator.init_trackers(
            project_name="Flow-Matching",
            config=config,
            init_kwargs={
                "wandb": {
                    "name": model_name,
                    "entity": config.get("wandb_entity", "default"),
                },
            },
        )

    if accelerator.is_main_process:
        os.makedirs(model_dir, exist_ok=True)
        if config.get("logging", True):
            logger = Logger(os.path.join(model_dir, "log.txt"))
            with open(os.path.join(model_dir, "config.json"), "w") as f:
                json.dump(config, f, indent=2)

    epochs = config["epochs"]
    batch_size = config["batch_size"]

    # Dataset and Dataloaders
    train_dataset = get_dataset(
        config["dataset"],
        config["input_res"],
        config["dataset_mean"],
        config["dataset_std"],
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.get("num_workers", 4),
    )

    # Model (Flow Matching backbone)
    from classes.Backbones.FlowNet import FlowNet

    model = FlowNet(
        num_channels=config["num_channels"],
        hidden_dims=config.get("hidden_dims", [64, 128, 256, 128, 64]),
        time_dim=config.get("time_dim", 128),
    )
    
    # Sampler (for checkpointing)
    from classes.Samplers.FlowMatching import FlowMatchingSampler
    sampler = FlowMatchingSampler(
        model=model,
        timesteps=config.get("sampling_steps", 100),
    )
    
    accelerator.print(model)
    accelerator.print(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M"
    )

    # Load model checkpoint if specified
    if config.get("model_from_checkpoint", False):
        model.load_state_dict(torch.load(config["model_checkpoint_path"]))
        accelerator.print(f"Model loaded from checkpoint: {config['model_checkpoint_path']}")

    # Optimizer
    optim = get_optimizer(config["optimizer"])(
        model.parameters(), 
        lr=config["lr"],
    )
    loss_fn = nn.MSELoss()

    # Prepare with accelerate
    train_loader = accelerator.prepare_data_loader(data_loader=train_loader)
    model = accelerator.prepare_model(model=model)
    optim = accelerator.prepare_optimizer(optimizer=optim)

    # Load state from checkpoint if specified
    if config.get("state_from_checkpoint", False):
        accelerator.load_state(input_dir=config["state_checkpoint_path"])
        accelerator.print(f"State loaded from checkpoint: {config['state_checkpoint_path']}")

    total_steps = epochs * len(train_loader)
    checkpoint_step = max(1, total_steps // config["num_checkpoints"])
    accelerator.print(
        f"Total steps: {total_steps} and checkpoint every {checkpoint_step} steps"
    )

    if accelerator.is_main_process:
        progress_bar = tqdm(range(total_steps))

    total_steps_done = 0
    total_loss = 0

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader):
            with accelerator.accumulate(model):
                x_1, _ = batch
                
                # Flow matching: x_t = (1 - t) * x_0 + t * x_1
                x_0 = torch.randn_like(x_1)
                t = torch.rand(len(x_1), device=x_1.device)
                t_expanded = t[:, None, None, None]
                x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
                
                # Target velocity
                dx_t = x_1 - x_0

                pred = model(t, x_t)
                loss = loss_fn(pred, dx_t)

                epoch_loss += loss.detach().item()
                optim.zero_grad()
                accelerator.backward(loss)
                optim.step()
                total_steps_done += 1

                accelerator.log({"train_loss": loss.item()}, step=total_steps_done)

                if total_steps_done % checkpoint_step == 0:
                    ckpt_dir = os.path.join(model_dir, "checkpoints", f"{total_steps_done}")
                    
                    # Sampling during checkpointing
                    if accelerator.is_main_process:
                        model.eval()
                        with torch.no_grad():
                            # Generate samples (use x_1.device for consistency)
                            sample_batch_size = config.get("sample_batch_size", 16)
                            noise = torch.randn(
                                sample_batch_size,
                                config["num_channels"],
                                config["input_res"][0],
                                config["input_res"][1],
                                device=x_1.device
                            )
                            
                            # Use sampler to generate images
                            samples = sampler(noise, steps=config.get("sampling_steps", 50))
                            
                            # Denormalize
                            mean = torch.tensor(config["dataset_mean"]).view(1, -1, 1, 1).to(x_1.device)
                            std = torch.tensor(config["dataset_std"]).view(1, -1, 1, 1).to(x_1.device)
                            samples = samples * std + mean
                            samples = torch.clamp(samples, 0, 1)
                            
                            # Save samples
                            os.makedirs(ckpt_dir, exist_ok=True)
                            grid = make_grid(samples, nrow=4, padding=2)
                            save_image(grid, os.path.join(ckpt_dir, "samples.png"))
                            
                            # Log to wandb if tracking
                            if config["tracking"]:
                                import wandb
                                accelerator.log({"samples": wandb.Image(grid)}, step=total_steps_done)
                        
                        model.train()
                    
                    accelerator.save_state(ckpt_dir, safe_serialization=False)

                if accelerator.is_main_process:
                    progress_bar.set_postfix_str(f"Loss: {loss.item():.4f}")
                    progress_bar.update(1)

        total_loss += epoch_loss
        if accelerator.is_main_process:
            epoch_loss_log = f"Epoch: {epoch}, Avg Epoch Loss {epoch_loss / (step + 1)}, Net Avg Loss: {total_loss / (total_steps_done + 1)}"
            if config.get("logging", True):
                logger.log(epoch_loss_log)
            accelerator.print(epoch_loss_log)

    accelerator.wait_for_everyone()
    accelerator.end_training()
    accelerator.save_model(model, os.path.join(model_dir), safe_serialization=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()
    config = json.load(open(args.config_file))
    main(config)

