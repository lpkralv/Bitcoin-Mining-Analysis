#!/usr/bin/env python3
"""
SHA-256 ML Redux Phase 3.2 & 3.3: Conditional Diffusion + GAN
Final two architectures to test for completeness after MLP/VAE/CLIP null results.

Phase 3.2: Conditional Diffusion Model
- Learn to denoise random bits into valid nonce conditioned on header
- DDPM framework with linear noise schedule

Phase 3.3: GAN (Wasserstein with GP)
- Generator: header+noise -> nonce
- Discriminator: header+nonce -> real/fake
- Straight-through estimator for discrete outputs
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# Global shutdown signal
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print("\nShutdown requested, finishing current epoch...")

signal.signal(signal.SIGINT, signal_handler)


class DiffusionDenoiser(nn.Module):
    """Denoiser network for conditional diffusion model."""

    def __init__(self, depth: int = 4, width: int = 1024):
        super().__init__()
        # Input: 32 nonce + 992 header + 1 timestep = 1025
        # Output: 32 (predicted noise for nonce)

        layers = []
        input_dim = 1025

        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, 32))  # predict noise for nonce

        self.net = nn.Sequential(*layers)

    def forward(self, noisy_nonce: torch.Tensor, header: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        # noisy_nonce: (batch, 32), header: (batch, 992), timestep: (batch, 1)
        x = torch.cat([noisy_nonce, header, timestep], dim=1)
        return self.net(x)


class DiffusionSchedule:
    """Linear noise schedule for DDPM."""

    def __init__(self, num_timesteps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_timesteps = num_timesteps

        # Linear schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean data according to forward process."""
        noise = torch.randn_like(x_0)
        alpha_cumprod_t = self.alphas_cumprod[t]

        # Reshape for broadcasting
        alpha_cumprod_t = alpha_cumprod_t.view(-1, 1)

        # Forward process: q(x_t | x_0) = N(sqrt(alpha_cumprod) * x_0, (1 - alpha_cumprod) * I)
        noisy_x = torch.sqrt(alpha_cumprod_t) * x_0 + torch.sqrt(1 - alpha_cumprod_t) * noise
        return noisy_x, noise

    def denoise_step(self, x_t: torch.Tensor, predicted_noise: torch.Tensor, t: int) -> torch.Tensor:
        """Single denoising step in reverse process."""
        if t == 0:
            return x_t - predicted_noise / torch.sqrt(1 - self.alphas_cumprod[t])

        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t-1]

        # Mean of reverse process
        coeff1 = torch.sqrt(alpha_cumprod_prev) * self.betas[t] / (1 - alpha_cumprod_t)
        coeff2 = torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)

        mean = coeff1 * (x_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise / torch.sqrt(alpha_cumprod_t)) + coeff2 * x_t

        if t > 0:
            # Add noise for non-final steps
            variance = self.betas[t] * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean


class Generator(nn.Module):
    """GAN Generator: noise + header -> nonce."""

    def __init__(self, noise_dim: int = 64, depth: int = 4, width: int = 1024):
        super().__init__()
        # Input: noise_dim + 992 header = noise_dim + 992
        # Output: 32 nonce logits

        input_dim = noise_dim + 992
        layers = []

        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, 32))  # nonce logits

        self.net = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor, header: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, header], dim=1)
        logits = self.net(x)

        # Straight-through estimator: forward = sigmoid -> round, backward = pass through
        probs = torch.sigmoid(logits)
        bits_hard = (probs > 0.5).float()
        bits_soft = probs + (bits_hard - probs).detach()  # STE

        return bits_soft


class Discriminator(nn.Module):
    """GAN Discriminator: nonce + header -> real/fake probability."""

    def __init__(self, depth: int = 4, width: int = 1024):
        super().__init__()
        # Input: 32 nonce + 992 header = 1024
        # Output: 1 (real/fake logit)

        layers = []
        layers.append(nn.Linear(1024, width))
        layers.append(nn.LeakyReLU(0.2))

        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.LeakyReLU(0.2))

        layers.append(nn.Linear(width, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, nonce: torch.Tensor, header: torch.Tensor) -> torch.Tensor:
        x = torch.cat([nonce, header], dim=1)
        return self.net(x)


def sha256d(data_bits: np.ndarray) -> np.ndarray:
    """Double SHA-256 hash of bit array."""
    import hashlib

    # Convert bits to bytes
    if len(data_bits) % 8 != 0:
        raise ValueError("Data length must be multiple of 8 bits")

    bytes_data = np.packbits(data_bits.astype(np.uint8))

    # Double SHA-256
    hash1 = hashlib.sha256(bytes_data.tobytes()).digest()
    hash2 = hashlib.sha256(hash1).digest()

    # Convert back to bits
    hash_bits = np.unpackbits(np.frombuffer(hash2, dtype=np.uint8))
    return hash_bits


def count_leading_zeros(hash_bits: np.ndarray) -> int:
    """Count leading zeros in hash."""
    for i, bit in enumerate(hash_bits):
        if bit == 1:
            return i
    return len(hash_bits)


def validate_nonce(header_bits: np.ndarray, nonce_bits: np.ndarray, target_zeros: int = 1) -> bool:
    """Check if nonce produces valid hash with target leading zeros."""
    full_data = np.concatenate([header_bits, nonce_bits])
    hash_bits = sha256d(full_data)
    leading_zeros = count_leading_zeros(hash_bits)
    return leading_zeros >= target_zeros


def run_diffusion_experiment(sandbox_path: str) -> Dict:
    """Run Phase 3.2: Conditional Diffusion experiments."""
    print("\n" + "="*80)
    print("PHASE 3.2: CONDITIONAL DIFFUSION")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = Path(sandbox_path) / "data" / "dataset_reduced_r64.npy"
    print(f"Loading data from {data_path}")

    data = np.load(data_path)
    print(f"Loaded {len(data)} samples, shape: {data.shape}")

    # Split header and nonce
    headers = data[:, :992].astype(np.float32)
    nonces = data[:, 992:].astype(np.float32)

    # Train/test split
    split_idx = int(0.8 * len(data))
    train_headers = torch.tensor(headers[:split_idx])
    train_nonces = torch.tensor(nonces[:split_idx])
    test_headers = torch.tensor(headers[split_idx:])
    test_nonces = torch.tensor(nonces[split_idx:])

    print(f"Train: {len(train_headers)}, Test: {len(test_headers)}")

    # AutoResearch configurations
    configs = [
        {"depth": 2, "width": 512, "lr": 1e-4, "batch_size": 256},
        {"depth": 2, "width": 512, "lr": 3e-4, "batch_size": 256},
        {"depth": 2, "width": 1024, "lr": 1e-4, "batch_size": 256},
        {"depth": 2, "width": 1024, "lr": 3e-4, "batch_size": 256},
        {"depth": 2, "width": 2048, "lr": 1e-4, "batch_size": 256},
        {"depth": 4, "width": 512, "lr": 1e-4, "batch_size": 256},
        {"depth": 4, "width": 512, "lr": 3e-4, "batch_size": 256},
        {"depth": 4, "width": 1024, "lr": 1e-4, "batch_size": 256},
        {"depth": 4, "width": 1024, "lr": 3e-4, "batch_size": 256},
        {"depth": 4, "width": 2048, "lr": 1e-4, "batch_size": 256},
        {"depth": 4, "width": 2048, "lr": 3e-4, "batch_size": 256},
        {"depth": 6, "width": 512, "lr": 1e-4, "batch_size": 256},
        {"depth": 6, "width": 512, "lr": 3e-4, "batch_size": 256},
        {"depth": 6, "width": 1024, "lr": 1e-4, "batch_size": 256},
        {"depth": 6, "width": 1024, "lr": 3e-4, "batch_size": 256},
        {"depth": 6, "width": 2048, "lr": 1e-4, "batch_size": 256},
        {"depth": 6, "width": 2048, "lr": 3e-4, "batch_size": 256},
    ]

    results = []
    scaler = torch.amp.GradScaler("cuda")

    for config_idx, config in enumerate(configs):
        if shutdown_requested:
            break

        print(f"\n--- Diffusion Config {config_idx + 1}/{len(configs)} ---")
        print(f"Config: {config}")

        # Model and optimizer
        model = DiffusionDenoiser(depth=config["depth"], width=config["width"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        schedule = DiffusionSchedule().to(device)

        # Data loader
        dataset = TensorDataset(train_headers, train_nonces)
        dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

        # Training
        model.train()
        train_losses = []

        for epoch in range(20):
            if shutdown_requested:
                break

            epoch_losses = []

            for batch_idx, (batch_headers, batch_nonces) in enumerate(dataloader):
                if shutdown_requested:
                    break

                batch_headers = batch_headers.to(device)
                batch_nonces = batch_nonces.to(device)
                batch_size = batch_headers.size(0)

                # Sample timesteps
                t = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device)

                # Add noise to nonces
                noisy_nonces, noise = schedule.add_noise(batch_nonces, t)

                # Predict noise
                with torch.amp.autocast(device_type="cuda"):
                    timestep_embed = t.float().unsqueeze(1) / schedule.num_timesteps
                    predicted_noise = model(noisy_nonces, batch_headers, timestep_embed)
                    loss = F.mse_loss(predicted_noise, noise)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_losses.append(loss.item())

            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                train_losses.append(avg_loss)
                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: loss = {avg_loss:.6f}")

        if shutdown_requested:
            break

        # Evaluation
        print("Evaluating...")
        model.eval()

        # Generate 1000 nonces
        num_test = min(1000, len(test_headers))
        test_batch = test_headers[:num_test].to(device)

        with torch.no_grad():
            # Start from pure noise
            generated_nonces = torch.randn((num_test, 32), device=device)

            # Denoise over all timesteps
            for t in reversed(range(schedule.num_timesteps)):
                timestep_embed = torch.full((num_test, 1), t / schedule.num_timesteps, device=device)
                predicted_noise = model(generated_nonces, test_batch, timestep_embed)
                generated_nonces = schedule.denoise_step(generated_nonces, predicted_noise, t)

        # Convert to binary and validate
        generated_bits = (generated_nonces.cpu().numpy() > 0.5).astype(np.uint8)
        test_headers_np = test_headers[:num_test].cpu().numpy().astype(np.uint8)

        valid_count = 0
        for i in range(num_test):
            if validate_nonce(test_headers_np[i], generated_bits[i], target_zeros=1):
                valid_count += 1

        validity_rate = valid_count / num_test
        random_baseline = 0.5  # 50% chance for 1 leading zero

        result = {
            "config": config,
            "final_loss": train_losses[-1] if train_losses else float('inf'),
            "validity_rate": validity_rate,
            "valid_count": int(valid_count),
            "total_tested": int(num_test),
            "random_baseline": random_baseline,
            "improvement_over_random": validity_rate - random_baseline,
            "train_losses": [float(x) for x in train_losses]
        }
        results.append(result)

        print(f"Validity rate: {validity_rate:.4f} (vs random {random_baseline:.4f})")
        print(f"Valid nonces: {valid_count}/{num_test}")

        # Update status
        status = {
            "phase": "3.2-diffusion",
            "config": config_idx + 1,
            "total_configs": len(configs),
            "current_validity": validity_rate,
            "best_validity": max(r["validity_rate"] for r in results),
            "timestamp": time.time()
        }

        status_path = Path(sandbox_path) / "data" / "status.json"
        with open(status_path, 'w') as f:
            json.dump(status, f)

    # Find best result
    if results:
        best_result = max(results, key=lambda x: x["validity_rate"])
        print(f"\nBest diffusion result: {best_result['validity_rate']:.4f} validity rate")
        print(f"Config: {best_result['config']}")

    return {
        "method": "conditional_diffusion",
        "results": results,
        "best_result": best_result if results else None,
        "summary": {
            "total_configs": len(results),
            "best_validity_rate": best_result["validity_rate"] if results else 0.0,
            "random_baseline": 0.5,
            "null_result": (best_result["validity_rate"] if results else 0.0) <= 0.52  # Within noise of random
        }
    }


def run_gan_experiment(sandbox_path: str) -> Dict:
    """Run Phase 3.3: GAN experiments."""
    print("\n" + "="*80)
    print("PHASE 3.3: WASSERSTEIN GAN")
    print("="*80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    data_path = Path(sandbox_path) / "data" / "dataset_reduced_r64.npy"
    print(f"Loading data from {data_path}")

    data = np.load(data_path)
    print(f"Loaded {len(data)} samples, shape: {data.shape}")

    # Split header and nonce
    headers = data[:, :992].astype(np.float32)
    nonces = data[:, 992:].astype(np.float32)

    # Train/test split
    split_idx = int(0.8 * len(data))
    train_headers = torch.tensor(headers[:split_idx])
    train_nonces = torch.tensor(nonces[:split_idx])
    test_headers = torch.tensor(headers[split_idx:])
    test_nonces = torch.tensor(nonces[split_idx:])

    print(f"Train: {len(train_headers)}, Test: {len(test_headers)}")

    # AutoResearch configurations
    configs = [
        {"gen_depth": 2, "gen_width": 512, "disc_depth": 2, "disc_width": 512, "noise_dim": 32, "disc_steps": 3, "lr": 1e-4},
        {"gen_depth": 2, "gen_width": 512, "disc_depth": 2, "disc_width": 512, "noise_dim": 32, "disc_steps": 3, "lr": 3e-4},
        {"gen_depth": 2, "gen_width": 512, "disc_depth": 2, "disc_width": 512, "noise_dim": 32, "disc_steps": 5, "lr": 1e-4},
        {"gen_depth": 2, "gen_width": 512, "disc_depth": 2, "disc_width": 512, "noise_dim": 64, "disc_steps": 3, "lr": 1e-4},
        {"gen_depth": 2, "gen_width": 512, "disc_depth": 4, "disc_width": 512, "lr": 1e-4, "noise_dim": 32, "disc_steps": 3},
        {"gen_depth": 2, "gen_width": 1024, "disc_depth": 2, "disc_width": 1024, "noise_dim": 32, "disc_steps": 3, "lr": 1e-4},
        {"gen_depth": 2, "gen_width": 1024, "disc_depth": 2, "disc_width": 1024, "noise_dim": 32, "disc_steps": 3, "lr": 3e-4},
        {"gen_depth": 2, "gen_width": 1024, "disc_depth": 2, "disc_width": 1024, "noise_dim": 64, "disc_steps": 3, "lr": 1e-4},
        {"gen_depth": 2, "gen_width": 1024, "disc_depth": 4, "disc_width": 1024, "noise_dim": 32, "disc_steps": 5, "lr": 1e-4},
        {"gen_depth": 4, "gen_width": 512, "disc_depth": 2, "disc_width": 512, "noise_dim": 32, "disc_steps": 3, "lr": 1e-4},
        {"gen_depth": 4, "gen_width": 512, "disc_depth": 4, "disc_width": 512, "noise_dim": 32, "disc_steps": 3, "lr": 1e-4},
        {"gen_depth": 4, "gen_width": 512, "disc_depth": 4, "disc_width": 512, "noise_dim": 64, "disc_steps": 3, "lr": 3e-4},
        {"gen_depth": 4, "gen_width": 1024, "disc_depth": 2, "disc_width": 1024, "noise_dim": 32, "disc_steps": 3, "lr": 1e-4},
        {"gen_depth": 4, "gen_width": 1024, "disc_depth": 4, "disc_width": 1024, "noise_dim": 32, "disc_steps": 3, "lr": 1e-4},
        {"gen_depth": 4, "gen_width": 1024, "disc_depth": 4, "disc_width": 1024, "noise_dim": 32, "disc_steps": 5, "lr": 1e-4},
        {"gen_depth": 4, "gen_width": 1024, "disc_depth": 4, "disc_width": 1024, "noise_dim": 64, "disc_steps": 3, "lr": 1e-4},
        {"gen_depth": 4, "gen_width": 1024, "disc_depth": 4, "disc_width": 1024, "noise_dim": 64, "disc_steps": 5, "lr": 3e-4},
    ]

    results = []
    scaler = torch.amp.GradScaler("cuda")

    def gradient_penalty(discriminator, real_data, fake_data, headers, device):
        """WGAN-GP gradient penalty."""
        batch_size = real_data.size(0)
        alpha = torch.rand(batch_size, 1, device=device)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        disc_interpolated = discriminator(interpolated, headers)

        gradients = torch.autograd.grad(
            outputs=disc_interpolated, inputs=interpolated,
            grad_outputs=torch.ones_like(disc_interpolated),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradient_norm = gradients.norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        return penalty

    for config_idx, config in enumerate(configs):
        if shutdown_requested:
            break

        print(f"\n--- GAN Config {config_idx + 1}/{len(configs)} ---")
        print(f"Config: {config}")

        # Models
        generator = Generator(
            noise_dim=config["noise_dim"],
            depth=config["gen_depth"],
            width=config["gen_width"]
        ).to(device)

        discriminator = Discriminator(
            depth=config["disc_depth"],
            width=config["disc_width"]
        ).to(device)

        # Optimizers
        gen_optimizer = torch.optim.Adam(generator.parameters(), lr=config["lr"], betas=(0.0, 0.9))
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config["lr"], betas=(0.0, 0.9))

        # Data loader
        dataset = TensorDataset(train_headers, train_nonces)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

        # Training
        gen_losses = []
        disc_losses = []

        for epoch in range(20):
            if shutdown_requested:
                break

            epoch_gen_losses = []
            epoch_disc_losses = []

            for batch_idx, (batch_headers, batch_nonces) in enumerate(dataloader):
                if shutdown_requested:
                    break

                batch_headers = batch_headers.to(device)
                batch_nonces = batch_nonces.to(device)
                batch_size = batch_headers.size(0)

                # Train discriminator
                for _ in range(config["disc_steps"]):
                    disc_optimizer.zero_grad()

                    with torch.amp.autocast(device_type="cuda"):
                        # Real samples
                        real_validity = discriminator(batch_nonces, batch_headers)

                        # Fake samples
                        noise = torch.randn(batch_size, config["noise_dim"], device=device)
                        fake_nonces = generator(noise, batch_headers)
                        fake_validity = discriminator(fake_nonces.detach(), batch_headers)

                        # WGAN loss + gradient penalty
                        gp = gradient_penalty(discriminator, batch_nonces, fake_nonces, batch_headers, device)
                        disc_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + 10 * gp

                    scaler.scale(disc_loss).backward()
                    scaler.step(disc_optimizer)
                    scaler.update()

                    epoch_disc_losses.append(disc_loss.item())

                # Train generator
                gen_optimizer.zero_grad()

                with torch.amp.autocast(device_type="cuda"):
                    noise = torch.randn(batch_size, config["noise_dim"], device=device)
                    fake_nonces = generator(noise, batch_headers)
                    fake_validity = discriminator(fake_nonces, batch_headers)
                    gen_loss = -torch.mean(fake_validity)

                scaler.scale(gen_loss).backward()
                scaler.step(gen_optimizer)
                scaler.update()

                epoch_gen_losses.append(gen_loss.item())

            if epoch_gen_losses and epoch_disc_losses:
                avg_gen_loss = np.mean(epoch_gen_losses)
                avg_disc_loss = np.mean(epoch_disc_losses)
                gen_losses.append(avg_gen_loss)
                disc_losses.append(avg_disc_loss)

                if epoch % 5 == 0:
                    print(f"Epoch {epoch}: Gen loss = {avg_gen_loss:.6f}, Disc loss = {avg_disc_loss:.6f}")

        if shutdown_requested:
            break

        # Evaluation
        print("Evaluating...")
        generator.eval()

        # Generate 1000 nonces
        num_test = min(1000, len(test_headers))
        test_batch = test_headers[:num_test].to(device)

        with torch.no_grad():
            noise = torch.randn(num_test, config["noise_dim"], device=device)
            generated_nonces = generator(noise, test_batch)

        # Convert to binary and validate
        generated_bits = (generated_nonces.cpu().numpy() > 0.5).astype(np.uint8)
        test_headers_np = test_headers[:num_test].cpu().numpy().astype(np.uint8)

        valid_count = 0
        for i in range(num_test):
            if validate_nonce(test_headers_np[i], generated_bits[i], target_zeros=1):
                valid_count += 1

        validity_rate = valid_count / num_test
        random_baseline = 0.5  # 50% chance for 1 leading zero

        result = {
            "config": config,
            "final_gen_loss": gen_losses[-1] if gen_losses else float('inf'),
            "final_disc_loss": disc_losses[-1] if disc_losses else float('inf'),
            "validity_rate": validity_rate,
            "valid_count": int(valid_count),
            "total_tested": int(num_test),
            "random_baseline": random_baseline,
            "improvement_over_random": validity_rate - random_baseline,
            "gen_losses": [float(x) for x in gen_losses],
            "disc_losses": [float(x) for x in disc_losses]
        }
        results.append(result)

        print(f"Validity rate: {validity_rate:.4f} (vs random {random_baseline:.4f})")
        print(f"Valid nonces: {valid_count}/{num_test}")

        # Update status
        status = {
            "phase": "3.3-gan",
            "config": config_idx + 1,
            "total_configs": len(configs),
            "current_validity": validity_rate,
            "best_validity": max(r["validity_rate"] for r in results),
            "timestamp": time.time()
        }

        status_path = Path(sandbox_path) / "data" / "status.json"
        with open(status_path, 'w') as f:
            json.dump(status, f)

    # Find best result
    if results:
        best_result = max(results, key=lambda x: x["validity_rate"])
        print(f"\nBest GAN result: {best_result['validity_rate']:.4f} validity rate")
        print(f"Config: {best_result['config']}")

    return {
        "method": "wasserstein_gan",
        "results": results,
        "best_result": best_result if results else None,
        "summary": {
            "total_configs": len(results),
            "best_validity_rate": best_result["validity_rate"] if results else 0.0,
            "random_baseline": 0.5,
            "null_result": (best_result["validity_rate"] if results else 0.0) <= 0.52  # Within noise of random
        }
    }


def main():
    parser = argparse.ArgumentParser(description="SHA-256 ML Redux Phase 3.2 & 3.3")
    parser.add_argument("--sandbox", required=True, help="Sandbox directory path")
    parser.add_argument("--diffusion-only", action="store_true", help="Run only diffusion experiment")
    parser.add_argument("--gan-only", action="store_true", help="Run only GAN experiment")
    args = parser.parse_args()

    sandbox_path = Path(args.sandbox)
    data_dir = sandbox_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Setup logging
    log_file = data_dir / "phase3_diffusion_gan.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    print("SHA-256 ML Redux Phase 3.2 & 3.3: Conditional Diffusion + GAN")
    print(f"Sandbox: {sandbox_path}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check for shutdown signal
    shutdown_signal = sandbox_path / "shutdown.signal"
    if shutdown_signal.exists():
        print("Shutdown signal detected, exiting")
        return

    all_results = {}

    try:
        # Run experiments
        if not args.gan_only:
            diffusion_results = run_diffusion_experiment(str(sandbox_path))
            all_results["diffusion"] = diffusion_results

            # Save intermediate results
            diffusion_file = data_dir / "phase3_diffusion_results.json"
            with open(diffusion_file, 'w') as f:
                json.dump(diffusion_results, f, indent=2)

            print(f"Diffusion results saved to {diffusion_file}")

            # Check for shutdown
            if shutdown_signal.exists() or shutdown_requested:
                print("Shutdown requested after diffusion")
                return

        if not args.diffusion_only:
            gan_results = run_gan_experiment(str(sandbox_path))
            all_results["gan"] = gan_results

            # Save results
            gan_file = data_dir / "phase3_gan_results.json"
            with open(gan_file, 'w') as f:
                json.dump(gan_results, f, indent=2)

            print(f"GAN results saved to {gan_file}")

        # Final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        for method, results in all_results.items():
            best_validity = results["summary"]["best_validity_rate"]
            is_null = results["summary"]["null_result"]
            print(f"{method.upper()}: Best validity = {best_validity:.4f}, Null result = {is_null}")

        print("\nPhase 3.2 & 3.3 Complete!")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final status update
        status = {
            "phase": "complete",
            "experiments_completed": list(all_results.keys()),
            "timestamp": time.time()
        }

        status_path = data_dir / "status.json"
        with open(status_path, 'w') as f:
            json.dump(status, f)


if __name__ == "__main__":
    main()