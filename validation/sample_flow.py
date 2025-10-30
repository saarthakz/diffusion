#!/usr/bin/env python3
"""
Generate samples from a trained flow matching model.
"""

import os
import sys
import argparse
import torch
from torchvision.utils import save_image, make_grid
import json
from datetime import datetime

sys.path.append(os.path.abspath("."))

from utils.get import get_flow_backbone


def main():
    parser = argparse.ArgumentParser(description="Sample from trained flow matching model")
    parser.add_argument(
        "--model_dir", 
        type=str, 
        required=True, 
        help="Path to model directory (e.g., models/flow_matching_mnist ...)"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=64, 
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--sampling_steps", 
        type=int, 
        default=250, 
        help="Number of ODE integration steps"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Output image path (default: model_dir/generated_samples.png)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA model weights if available (pytorch_model_ema.bin)"
    )
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load config
    config_path = os.path.join(args.model_dir, "config.json")
    print(f"Loading config from {config_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load model
    print(f"Loading model from {args.model_dir}...")
    Backbone = get_flow_backbone(config)
    model = Backbone(**config)
    
    # Load model weights
    # Try to use EMA model if requested and available
    if args.use_ema:
        model_path = os.path.join(args.model_dir, "pytorch_model_ema.bin")
        if not os.path.exists(model_path):
            print("EMA model not found, falling back to regular model...")
            model_path = os.path.join(args.model_dir, "pytorch_model.bin")
        else:
            print("Using EMA model for sampling...")
    else:
        model_path = os.path.join(args.model_dir, "pytorch_model.bin")
    
    if not os.path.exists(model_path):
        # Try alternative path
        model_path = os.path.join(args.model_dir, "model.safetensors")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found in {args.model_dir}")
    
    print(f"Loading weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=args.device)
    
    # Handle different checkpoint formats
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    
    # Handle DDP wrapped models
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model = model.to(args.device)
    model.eval()
    
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create sampler
    from classes.Samplers.FlowMatching import FlowMatchingSampler
    sampler = FlowMatchingSampler(
        model=model,
        timesteps=args.sampling_steps,
    )
    
    # Generate samples
    print(f"Generating {args.num_samples} samples with {args.sampling_steps} steps...")
    
    shape = (
        args.num_samples,
        config["num_channels"],
        config["input_res"][0],
        config["input_res"][1]
    )
    
    noise = torch.randn(shape, device=args.device)
    samples = sampler(noise, steps=args.sampling_steps)
    
    # Denormalize
    mean = torch.tensor(config["dataset_mean"]).view(1, -1, 1, 1).to(args.device)
    std = torch.tensor(config["dataset_std"]).view(1, -1, 1, 1).to(args.device)
    samples = samples * std + mean
    
    # Clip to valid range
    samples = torch.clamp(samples, 0, 1)
    
    # Save grid
    if args.output is None:
        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"samples_grid_{timestamp}_steps{args.sampling_steps}.png"
        args.output = os.path.join(args.model_dir, filename)
    
    print(f"Saving grid...")
    
    # Create and save grid
    nrow = int(args.num_samples ** 0.5)
    if nrow * nrow < args.num_samples:
        nrow += 1  # Ensure all samples fit in the grid
    
    grid = make_grid(samples, nrow=nrow, padding=4, normalize=False, pad_value=1.0)
    save_image(grid, args.output)
    
    print(f"\n{'='*60}")
    print(f"✅ Grid saved to: {args.output}")
    print(f"   Samples: {args.num_samples} | Steps: {args.sampling_steps}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

