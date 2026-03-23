#!/usr/bin/env python3
"""
SHA-256 ML Redux: Deep Investigation (Phases D2, D3, D4)
Comprehensive PyTorch implementation for RTX 4070 Ti with CUDA
"""

import argparse
import json
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import hashlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# ============================================================================
# Global Configuration and Utilities
# ============================================================================

class Config:
    """Global configuration"""
    RANDOM_SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # File paths (relative to sandbox)
    DATA_DIR = "data"
    DATASET_REAL_BITCOIN = "dataset_real_bitcoin.npy"
    DATASET_REDUCED_R64 = "dataset_reduced_r64.npy"
    SHUTDOWN_SIGNAL = "shutdown.signal"
    STATUS_JSON = "status.json"
    LOG_FILE = "deep_investigation.log"

    # Results files
    D2_RESULTS = "d2_results.json"
    D3_RESULTS = "d3_results.json"
    D4_RESULTS = "d4_results.json"


def setup_logging(log_path: Path) -> logging.Logger:
    """Setup logging to file and console"""
    logger = logging.getLogger("deep_investigation")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def save_status(status_path: Path, phase: str, experiment: str, progress: str):
    """Save current status to JSON"""
    status = {
        "timestamp": time.time(),
        "phase": phase,
        "experiment": experiment,
        "progress": progress
    }
    with open(status_path, 'w') as f:
        json.dump(status, f)


def should_shutdown(signal_path: Path) -> bool:
    """Check if shutdown signal exists"""
    return signal_path.exists()


def convert_for_json(obj: Any) -> Any:
    """Convert numpy/torch types to JSON-serializable types"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    else:
        return obj


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> List[bool]:
    """Apply Bonferroni correction to p-values"""
    corrected_alpha = alpha / len(p_values)
    return [p < corrected_alpha for p in p_values]


def z_test_p_value(accuracy: float, n_samples: int, null_accuracy: float = 0.5) -> float:
    """Compute p-value for binary classification using z-test"""
    if n_samples == 0:
        return 1.0

    # Z-test for binomial proportion
    p_hat = accuracy
    p_0 = null_accuracy

    # Standard error under null hypothesis
    se = math.sqrt(p_0 * (1 - p_0) / n_samples)

    if se == 0:
        return 1.0

    # Z-score
    z = (p_hat - p_0) / se

    # Two-tailed p-value using complementary error function
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))

    return min(1.0, max(0.0, p_value))


# ============================================================================
# Neural Network Architectures
# ============================================================================

class Autoencoder(nn.Module):
    """Vanilla Autoencoder for SHA-256 data"""

    def __init__(self, input_dim: int, latent_dim: int, encoder_depth: int,
                 encoder_width: int, decoder_depth: int, decoder_width: int):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        in_dim = input_dim

        for i in range(encoder_depth):
            out_dim = encoder_width if i < encoder_depth - 1 else latent_dim
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < encoder_depth - 1 else nn.Identity()
            ])
            in_dim = out_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim

        for i in range(decoder_depth):
            out_dim = decoder_width if i < decoder_depth - 1 else input_dim
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < decoder_depth - 1 else nn.Identity()
            ])
            in_dim = out_dim

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)  # Returns logits for BCEWithLogitsLoss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)  # Returns logits


class DiffusionModel(nn.Module):
    """Simple diffusion model for nonce generation"""

    def __init__(self, nonce_dim: int = 32, header_dim: int = 992,
                 depth: int = 2, width: int = 1024, timesteps: int = 100):
        super().__init__()

        self.nonce_dim = nonce_dim
        self.header_dim = header_dim
        self.timesteps = timesteps

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, width // 4),
            nn.ReLU(),
            nn.Linear(width // 4, width // 4)
        )

        # Main network
        layers = []
        input_dim = nonce_dim + header_dim + width // 4

        for i in range(depth):
            output_dim = width if i < depth - 1 else nonce_dim
            layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.ReLU() if i < depth - 1 else nn.Identity()
            ])
            input_dim = output_dim

        self.network = nn.Sequential(*layers)

        # Noise schedule
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean data"""
        noise = torch.randn_like(x0)
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        noisy_x = torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise
        return noisy_x, noise

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor, header: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy data, time, and header condition"""
        # Time embedding
        t_embed = self.time_embed(t.float().view(-1, 1))

        # Concatenate inputs
        combined = torch.cat([x_t, header, t_embed], dim=1)

        return self.network(combined)

    def forward(self, nonce: torch.Tensor, header: torch.Tensor) -> torch.Tensor:
        """Forward pass for training"""
        batch_size = nonce.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=nonce.device)

        noisy_nonce, noise = self.add_noise(nonce, t)
        predicted_noise = self.predict_noise(noisy_nonce, t, header)

        return F.mse_loss(predicted_noise, noise)

    def sample(self, header: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """Generate nonces given headers"""
        device = header.device

        if header.dim() == 1:
            header = header.unsqueeze(0)

        batch_size = header.shape[0] * num_samples
        header = header.repeat(num_samples, 1)

        # Start from pure noise
        x = torch.randn(batch_size, self.nonce_dim, device=device)

        # Reverse diffusion process
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device)

            with torch.no_grad():
                predicted_noise = self.predict_noise(x, t_tensor, header)

                alpha_t = self.alphas[t]
                alpha_cumprod_t = self.alphas_cumprod[t]

                # Compute mean of reverse process
                mean = (x - self.betas[t] * predicted_noise / torch.sqrt(1 - alpha_cumprod_t)) / torch.sqrt(alpha_t)

                if t > 0:
                    noise = torch.randn_like(x)
                    x = mean + torch.sqrt(self.betas[t]) * noise
                else:
                    x = mean

        return torch.sigmoid(x)  # Ensure [0,1] range


class Generator(nn.Module):
    """GAN Generator"""

    def __init__(self, noise_dim: int = 100, header_dim: int = 992,
                 nonce_dim: int = 32, depth: int = 3, width: int = 512):
        super().__init__()

        layers = []
        input_dim = noise_dim + header_dim

        for i in range(depth):
            output_dim = width if i < depth - 1 else nonce_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < depth - 1:
                layers.append(nn.ReLU())
            # No sigmoid on last layer — use BCEWithLogitsLoss
            input_dim = output_dim

        self.network = nn.Sequential(*layers)
        self.noise_dim = noise_dim

    def forward(self, noise: torch.Tensor, header: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([noise, header], dim=1)
        return self.network(combined)


class Discriminator(nn.Module):
    """GAN Discriminator"""

    def __init__(self, header_dim: int = 992, nonce_dim: int = 32,
                 depth: int = 3, width: int = 512):
        super().__init__()

        layers = []
        input_dim = header_dim + nonce_dim

        for i in range(depth):
            output_dim = width if i < depth - 1 else 1
            layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.LeakyReLU(0.2) if i < depth - 1 else nn.Identity()
            ])
            input_dim = output_dim

        self.network = nn.Sequential(*layers)

    def forward(self, header: torch.Tensor, nonce: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([header, nonce], dim=1)
        return self.network(combined)


class MLPClassifier(nn.Module):
    """MLP for nonce bit prediction"""

    def __init__(self, input_dim: int, output_dim: int = 32, depth: int = 2,
                 width: int = 1024, dropout: float = 0.1):
        super().__init__()

        layers = []
        in_dim = input_dim

        for i in range(depth):
            out_dim = width if i < depth - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))

            if i < depth - 1:
                layers.extend([
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])

            in_dim = out_dim

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================================
# Training Functions
# ============================================================================

def train_autoencoder(model: Autoencoder, train_loader: DataLoader,
                     val_loader: DataLoader, num_epochs: int = 100,
                     lr: float = 1e-4, patience: int = 20,
                     min_epochs: int = 50, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Train autoencoder with early stopping"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                output = model(data)
                loss = criterion(output, data)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, in val_loader:
                data = data.to(Config.DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type="cuda"):
                    output = model(data)
                    loss = criterion(output, data)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if logger and epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch >= min_epochs:
            if logger:
                logger.info(f"Early stopping at epoch {epoch}")
            break

    return {
        "final_epoch": epoch,
        "best_val_loss": float(best_val_loss),
        "train_losses": train_losses,
        "val_losses": val_losses
    }


def train_diffusion(model: DiffusionModel, train_loader: DataLoader,
                   val_loader: DataLoader, num_epochs: int = 100,
                   lr: float = 1e-4, logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Train diffusion model"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda")

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(Config.DEVICE, non_blocking=True)
            header = data[:, :992]  # First 992 bits
            nonce = data[:, 992:1024]  # Last 32 bits

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                loss = model(nonce, header)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, in val_loader:
                data = data.to(Config.DEVICE, non_blocking=True)
                header = data[:, :992]
                nonce = data[:, 992:1024]

                with torch.amp.autocast(device_type="cuda"):
                    loss = model(nonce, header)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if logger and epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses
    }


def train_gan(generator: Generator, discriminator: Discriminator,
              train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int = 100, lr: float = 2e-4,
              logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Train GAN"""
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()
    scaler_g = torch.amp.GradScaler("cuda")
    scaler_d = torch.amp.GradScaler("cuda")

    g_losses = []
    d_losses = []

    for epoch in range(num_epochs):
        g_loss_epoch = 0
        d_loss_epoch = 0

        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(Config.DEVICE, non_blocking=True)
            batch_size = data.shape[0]
            header = data[:, :992]
            real_nonce = data[:, 992:1024]

            # Train discriminator
            discriminator.train()
            generator.eval()

            d_optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                # Real samples
                real_labels = torch.ones(batch_size, 1, device=Config.DEVICE)
                real_output = discriminator(header, real_nonce)
                d_loss_real = criterion(real_output, real_labels)

                # Fake samples
                noise = torch.randn(batch_size, generator.noise_dim, device=Config.DEVICE)
                fake_nonce = generator(noise, header)
                fake_labels = torch.zeros(batch_size, 1, device=Config.DEVICE)
                fake_output = discriminator(header, fake_nonce.detach())
                d_loss_fake = criterion(fake_output, fake_labels)

                d_loss = (d_loss_real + d_loss_fake) / 2

            scaler_d.scale(d_loss).backward()
            scaler_d.step(d_optimizer)
            scaler_d.update()

            # Train generator
            generator.train()
            discriminator.eval()

            g_optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                noise = torch.randn(batch_size, generator.noise_dim, device=Config.DEVICE)
                fake_nonce = generator(noise, header)
                fake_output = discriminator(header, fake_nonce)
                g_loss = criterion(fake_output, real_labels)  # Want discriminator to think it's real

            scaler_g.scale(g_loss).backward()
            scaler_g.step(g_optimizer)
            scaler_g.update()

            g_loss_epoch += g_loss.item()
            d_loss_epoch += d_loss.item()

        g_loss_epoch /= len(train_loader)
        d_loss_epoch /= len(train_loader)

        g_losses.append(g_loss_epoch)
        d_losses.append(d_loss_epoch)

        if logger and epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: g_loss={g_loss_epoch:.6f}, d_loss={d_loss_epoch:.6f}")

    return {
        "g_losses": g_losses,
        "d_losses": d_losses
    }


def train_mlp_classifier(model: MLPClassifier, train_loader: DataLoader,
                        val_loader: DataLoader, num_epochs: int = 100,
                        lr: float = 3e-4, patience: int = 50, min_epochs: int = 100,
                        logger: Optional[logging.Logger] = None) -> Dict[str, Any]:
    """Train MLP classifier with early stopping"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(Config.DEVICE, non_blocking=True)
            targets = targets.to(Config.DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type="cuda"):
                outputs = model(data)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(Config.DEVICE, non_blocking=True)
                targets = targets.to(Config.DEVICE, non_blocking=True)

                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if logger and epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch >= min_epochs:
            if logger:
                logger.info(f"Early stopping at epoch {epoch}")
            break

    return {
        "final_epoch": epoch,
        "best_val_loss": float(best_val_loss),
        "train_losses": train_losses,
        "val_losses": val_losses
    }


# ============================================================================
# Data Processing Functions
# ============================================================================

def load_dataset(path: Path, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Load dataset from numpy file"""
    if logger:
        logger.info(f"Loading dataset from {path}")

    data = np.load(path)

    if logger:
        logger.info(f"Loaded {data.shape[0]} samples with shape {data.shape}")

    return data


def create_shuffled_data(data: np.ndarray, logger: Optional[logging.Logger] = None) -> np.ndarray:
    """Create shuffled version: same headers, randomly permuted nonces"""
    if logger:
        logger.info("Creating shuffled data control")

    shuffled_data = data.copy()
    headers = data[:, :992]  # Keep headers in place
    nonces = data[:, 992:1024]  # Extract nonces

    # Randomly permute nonce assignments
    np.random.shuffle(nonces)
    shuffled_data[:, 992:1024] = nonces

    if logger:
        logger.info(f"Created shuffled dataset with {shuffled_data.shape[0]} samples")

    return shuffled_data


def create_data_splits(data: np.ndarray, train_ratio: float = 0.8,
                      val_ratio: float = 0.1, test_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/val/test"""
    n = len(data)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def bits_to_header(bits: np.ndarray) -> bytes:
    """Convert bit array to 80-byte header for SHA-256 verification"""
    # bits[0:608] = header bytes 0-75 (76 bytes, no nonce)
    # bits[992:1024] = nonce bits (4 bytes)

    header_bits = bits[:608]
    nonce_bits = bits[992:1024]

    # Pad header_bits to multiple of 8 if needed
    if len(header_bits) % 8 != 0:
        padding = 8 - (len(header_bits) % 8)
        header_bits = np.concatenate([header_bits, np.zeros(padding, dtype=np.uint8)])

    # Pack bits to bytes
    header_bytes = np.packbits(header_bits.astype(np.uint8)).tobytes()[:76]  # Ensure 76 bytes
    nonce_bytes = np.packbits(nonce_bits.astype(np.uint8)).tobytes()  # 4 bytes

    return header_bytes + nonce_bytes


def verify_sha256_nonce(header_bits: np.ndarray, nonce_bits: np.ndarray) -> int:
    """Verify nonce using SHA-256d and return number of leading zeros"""
    try:
        full_header = bits_to_header(np.concatenate([header_bits, np.zeros(384), nonce_bits]))

        # SHA-256d (double SHA-256)
        hash1 = hashlib.sha256(full_header).digest()
        hash2 = hashlib.sha256(hash1).digest()

        # Count leading zero bits
        leading_zeros = 0
        for byte in hash2:
            if byte == 0:
                leading_zeros += 8
            else:
                # Count leading zeros in this byte
                for i in range(8):
                    if (byte >> (7 - i)) & 1 == 0:
                        leading_zeros += 1
                    else:
                        break
                break

        return leading_zeros

    except Exception:
        return 0


def evaluate_nonce_reconstruction(model: Autoencoder, test_loader: DataLoader,
                                logger: Optional[logging.Logger] = None) -> float:
    """Evaluate nonce reconstruction accuracy"""
    model.eval()
    total_correct = 0
    total_bits = 0

    with torch.no_grad():
        for data, in test_loader:
            data = data.to(Config.DEVICE, non_blocking=True)
            output = model(data)

            # Extract nonce predictions and targets
            nonce_pred = output[:, 992:1024]  # Last 32 bits
            nonce_target = data[:, 992:1024]

            # Convert to binary predictions
            nonce_pred_binary = (nonce_pred > 0.0).float()  # logits threshold

            # Count correct bits
            correct_bits = (nonce_pred_binary == nonce_target).sum().item()
            total_correct += correct_bits
            total_bits += nonce_pred_binary.numel()

    accuracy = total_correct / total_bits if total_bits > 0 else 0.0

    if logger:
        logger.info(f"Nonce reconstruction accuracy: {accuracy:.6f}")

    return accuracy


def evaluate_per_bit_accuracy(model: MLPClassifier, test_loader: DataLoader,
                             logger: Optional[logging.Logger] = None) -> Tuple[List[float], List[float]]:
    """Evaluate per-bit accuracies and p-values"""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(Config.DEVICE, non_blocking=True)
            targets = targets.to(Config.DEVICE, non_blocking=True)

            outputs = model(data)
            predictions = torch.sigmoid(outputs)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Compute per-bit accuracies
    accuracies = []
    p_values = []

    for bit in range(32):
        pred_binary = (predictions[:, bit] > 0.0).astype(int)  # logits threshold
        target_binary = targets[:, bit].astype(int)

        accuracy = np.mean(pred_binary == target_binary)
        p_value = z_test_p_value(accuracy, len(pred_binary))

        accuracies.append(float(accuracy))
        p_values.append(float(p_value))

    if logger:
        significant_bits = bonferroni_correction(p_values)
        n_significant = sum(significant_bits)
        logger.info(f"Found {n_significant}/32 bits with significant accuracy (Bonferroni corrected)")

    return accuracies, p_values


# ============================================================================
# Phase D2: VAE Controls
# ============================================================================

def run_phase_d2(sandbox_dir: Path, logger: logging.Logger, status_path: Path) -> Dict[str, Any]:
    """Run Phase D2: VAE Controls"""
    logger.info("=" * 60)
    logger.info("PHASE D2: VAE CONTROLS")
    logger.info("=" * 60)

    results = {}

    # Set random seed
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    data_dir = sandbox_dir / Config.DATA_DIR

    # D2.1: Shuffled-Data Control
    save_status(status_path, "D2", "D2.1", "Loading real Bitcoin data")
    logger.info("-" * 40)
    logger.info("D2.1: Shuffled-Data Control")
    logger.info("-" * 40)

    # Load real Bitcoin data
    bitcoin_data = load_dataset(data_dir / Config.DATASET_REAL_BITCOIN, logger)

    # Create shuffled version
    shuffled_data = create_shuffled_data(bitcoin_data, logger)

    # Create data splits for both datasets
    train_real, val_real, test_real = create_data_splits(bitcoin_data)
    train_shuffled, val_shuffled, test_shuffled = create_data_splits(shuffled_data)

    logger.info(f"Real data splits: {len(train_real)}/{len(val_real)}/{len(test_real)}")
    logger.info(f"Shuffled data splits: {len(train_shuffled)}/{len(val_shuffled)}/{len(test_shuffled)}")

    # Best VAE config from Phase 3.4
    ae_config = {
        "input_dim": 1024,
        "latent_dim": 256,
        "encoder_depth": 4,
        "encoder_width": 1024,
        "decoder_depth": 2,
        "decoder_width": 1024
    }

    # Train on real data
    save_status(status_path, "D2", "D2.1", "Training AE on real data")
    logger.info("Training autoencoder on real (paired) data...")

    model_real = Autoencoder(**ae_config).to(Config.DEVICE)

    train_loader_real = DataLoader(
        TensorDataset(torch.FloatTensor(train_real)),
        batch_size=256, shuffle=True, pin_memory=True
    )
    val_loader_real = DataLoader(
        TensorDataset(torch.FloatTensor(val_real)),
        batch_size=256, shuffle=False, pin_memory=True
    )
    test_loader_real = DataLoader(
        TensorDataset(torch.FloatTensor(test_real)),
        batch_size=256, shuffle=False, pin_memory=True
    )

    train_results_real = train_autoencoder(
        model_real, train_loader_real, val_loader_real,
        num_epochs=200, lr=1e-4, patience=20, min_epochs=50, logger=logger
    )

    accuracy_real = evaluate_nonce_reconstruction(model_real, test_loader_real, logger)

    # Train on shuffled data
    save_status(status_path, "D2", "D2.1", "Training AE on shuffled data")
    logger.info("Training autoencoder on shuffled data...")

    model_shuffled = Autoencoder(**ae_config).to(Config.DEVICE)

    train_loader_shuffled = DataLoader(
        TensorDataset(torch.FloatTensor(train_shuffled)),
        batch_size=256, shuffle=True, pin_memory=True
    )
    val_loader_shuffled = DataLoader(
        TensorDataset(torch.FloatTensor(val_shuffled)),
        batch_size=256, shuffle=False, pin_memory=True
    )
    test_loader_shuffled = DataLoader(
        TensorDataset(torch.FloatTensor(test_shuffled)),
        batch_size=256, shuffle=False, pin_memory=True
    )

    train_results_shuffled = train_autoencoder(
        model_shuffled, train_loader_shuffled, val_loader_shuffled,
        num_epochs=200, lr=1e-4, patience=20, min_epochs=50, logger=logger
    )

    accuracy_shuffled = evaluate_nonce_reconstruction(model_shuffled, test_loader_shuffled, logger)

    logger.info(f"Real data nonce accuracy: {accuracy_real:.6f}")
    logger.info(f"Shuffled data nonce accuracy: {accuracy_shuffled:.6f}")
    logger.info(f"Difference: {accuracy_real - accuracy_shuffled:.6f}")

    results["d2_1_shuffled_control"] = {
        "accuracy_real": float(accuracy_real),
        "accuracy_shuffled": float(accuracy_shuffled),
        "difference": float(accuracy_real - accuracy_shuffled),
        "train_results_real": convert_for_json(train_results_real),
        "train_results_shuffled": convert_for_json(train_results_shuffled)
    }

    if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
        return results

    # D2.2: Capacity Control
    save_status(status_path, "D2", "D2.2", "Starting capacity control")
    logger.info("-" * 40)
    logger.info("D2.2: Capacity Control")
    logger.info("-" * 40)

    # Use 50K subset for speed
    subset_size = 50000
    bitcoin_subset = bitcoin_data[:subset_size]
    train_subset, val_subset, test_subset = create_data_splits(bitcoin_subset)

    latent_dims = [8, 16, 32, 64, 128, 256]
    capacity_results = {}

    for latent_dim in latent_dims:
        save_status(status_path, "D2", "D2.2", f"Training latent_dim={latent_dim}")
        logger.info(f"Training with latent_dim={latent_dim}")

        config = ae_config.copy()
        config["latent_dim"] = latent_dim

        model = Autoencoder(**config).to(Config.DEVICE)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_subset)),
            batch_size=256, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(val_subset)),
            batch_size=256, shuffle=False, pin_memory=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(test_subset)),
            batch_size=256, shuffle=False, pin_memory=True
        )

        train_results = train_autoencoder(
            model, train_loader, val_loader,
            num_epochs=50, lr=1e-4, patience=10, min_epochs=20, logger=logger
        )

        accuracy = evaluate_nonce_reconstruction(model, test_loader, logger)

        capacity_results[str(latent_dim)] = {
            "accuracy": float(accuracy),
            "train_results": convert_for_json(train_results)
        }

        logger.info(f"Latent dim {latent_dim}: nonce accuracy = {accuracy:.6f}")

        if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
            break

    results["d2_2_capacity_control"] = capacity_results

    if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
        return results

    # D2.3: Nonce-Only Autoencoder
    save_status(status_path, "D2", "D2.3", "Training nonce-only AE")
    logger.info("-" * 40)
    logger.info("D2.3: Nonce-Only Autoencoder")
    logger.info("-" * 40)

    # Extract only nonce bits for training
    nonce_train = train_subset[:, 992:1024]  # Only the 32 nonce bits
    nonce_val = val_subset[:, 992:1024]
    nonce_test = test_subset[:, 992:1024]

    nonce_only_results = {}

    for latent_dim in [8, 16, 32]:
        save_status(status_path, "D2", "D2.3", f"Nonce-only latent_dim={latent_dim}")
        logger.info(f"Training nonce-only AE with latent_dim={latent_dim}")

        model = Autoencoder(
            input_dim=32,
            latent_dim=latent_dim,
            encoder_depth=2,
            encoder_width=128,
            decoder_depth=2,
            decoder_width=128
        ).to(Config.DEVICE)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(nonce_train)),
            batch_size=256, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(nonce_val)),
            batch_size=256, shuffle=False, pin_memory=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(nonce_test)),
            batch_size=256, shuffle=False, pin_memory=True
        )

        train_results = train_autoencoder(
            model, train_loader, val_loader,
            num_epochs=50, lr=1e-4, patience=10, min_epochs=20, logger=logger
        )

        # Evaluate reconstruction accuracy on nonces
        model.eval()
        total_correct = 0
        total_bits = 0

        with torch.no_grad():
            for data, in test_loader:
                data = data.to(Config.DEVICE, non_blocking=True)
                output = model(data)

                pred_binary = (output > 0.0).float()  # logits threshold
                correct_bits = (pred_binary == data).sum().item()
                total_correct += correct_bits
                total_bits += pred_binary.numel()

        accuracy = total_correct / total_bits if total_bits > 0 else 0.0

        nonce_only_results[str(latent_dim)] = {
            "accuracy": float(accuracy),
            "train_results": convert_for_json(train_results)
        }

        logger.info(f"Nonce-only latent dim {latent_dim}: accuracy = {accuracy:.6f}")

        if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
            break

    results["d2_3_nonce_only"] = nonce_only_results

    logger.info("Phase D2 completed!")
    return results


# ============================================================================
# Phase D3: Generative Model Power Tests
# ============================================================================

def run_phase_d3(sandbox_dir: Path, logger: logging.Logger, status_path: Path) -> Dict[str, Any]:
    """Run Phase D3: Generative Model Power Tests"""
    logger.info("=" * 60)
    logger.info("PHASE D3: GENERATIVE MODEL POWER TESTS")
    logger.info("=" * 60)

    results = {}

    # Set random seed
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    data_dir = sandbox_dir / Config.DATA_DIR

    # Load re-mined data
    save_status(status_path, "D3", "setup", "Loading re-mined data")
    reduced_data = load_dataset(data_dir / Config.DATASET_REDUCED_R64, logger)

    # Create data splits
    train_data, val_data, test_data = create_data_splits(reduced_data)
    logger.info(f"Data splits: {len(train_data)}/{len(val_data)}/{len(test_data)}")

    # Create shuffled version for controls
    shuffled_reduced = create_shuffled_data(reduced_data, logger)
    train_shuffled, val_shuffled, test_shuffled = create_data_splits(shuffled_reduced)

    if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
        return results

    # D3.1: Diffusion High-Power
    save_status(status_path, "D3", "D3.1", "Training diffusion model")
    logger.info("-" * 40)
    logger.info("D3.1: Diffusion High-Power")
    logger.info("-" * 40)

    # Train diffusion model on real data
    logger.info("Training diffusion model on re-mined data...")

    diffusion_model = DiffusionModel(
        nonce_dim=32,
        header_dim=992,
        depth=2,
        width=1024,
        timesteps=100
    ).to(Config.DEVICE)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(train_data)),
        batch_size=32, shuffle=True, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(val_data)),
        batch_size=32, shuffle=False, pin_memory=True
    )

    diffusion_train_results = train_diffusion(
        diffusion_model, train_loader, val_loader,
        num_epochs=100, lr=1e-4, logger=logger
    )

    # Generate and validate nonces
    save_status(status_path, "D3", "D3.1", "Generating and validating nonces")
    logger.info("Generating 50,000 nonces and validating...")

    test_headers = torch.FloatTensor(test_data[:, :992]).to(Config.DEVICE)
    n_test_samples = len(test_headers)
    n_generations = 50000

    valid_nonces = 0
    leading_zeros_list = []

    with torch.no_grad():
        for i in range(0, n_generations, 1000):  # Process in batches
            batch_size = min(1000, n_generations - i)

            # Sample random headers for this batch
            header_indices = np.random.choice(n_test_samples, batch_size)
            batch_headers = test_headers[header_indices]

            # Generate nonces
            generated_nonces = diffusion_model.sample(batch_headers, num_samples=1)
            generated_nonces_binary = (generated_nonces > 0.5).cpu().numpy()

            # Validate each nonce
            for j in range(batch_size):
                header_bits = test_data[header_indices[j], :992]
                nonce_bits = generated_nonces_binary[j]

                leading_zeros = verify_sha256_nonce(header_bits, nonce_bits)
                leading_zeros_list.append(leading_zeros)

                if leading_zeros > 0:  # Any leading zeros indicate some validity
                    valid_nonces += 1

            if (i + batch_size) % 10000 == 0:
                logger.info(f"Processed {i + batch_size}/{n_generations} generations")

            if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
                break

    validity_rate = valid_nonces / len(leading_zeros_list) if leading_zeros_list else 0.0

    # Calculate 95% confidence interval
    n = len(leading_zeros_list)
    if n > 0:
        p = validity_rate
        ci_margin = 1.96 * math.sqrt(p * (1 - p) / n)
        ci_lower = max(0, p - ci_margin)
        ci_upper = min(1, p + ci_margin)
    else:
        ci_lower = ci_upper = 0

    logger.info(f"Diffusion validity rate: {validity_rate:.6f} [{ci_lower:.6f}, {ci_upper:.6f}]")
    logger.info(f"Average leading zeros: {np.mean(leading_zeros_list):.3f}")

    results["d3_1_diffusion"] = {
        "validity_rate": float(validity_rate),
        "confidence_interval": [float(ci_lower), float(ci_upper)],
        "n_generated": len(leading_zeros_list),
        "average_leading_zeros": float(np.mean(leading_zeros_list)),
        "leading_zeros_distribution": convert_for_json(leading_zeros_list),
        "train_results": convert_for_json(diffusion_train_results)
    }

    if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
        return results

    # Train diffusion model on shuffled data
    save_status(status_path, "D3", "D3.1", "Training diffusion on shuffled data")
    logger.info("Training diffusion model on shuffled data...")

    diffusion_shuffled = DiffusionModel(
        nonce_dim=32,
        header_dim=992,
        depth=2,
        width=1024,
        timesteps=100
    ).to(Config.DEVICE)

    train_loader_shuffled = DataLoader(
        TensorDataset(torch.FloatTensor(train_shuffled)),
        batch_size=32, shuffle=True, pin_memory=True
    )
    val_loader_shuffled = DataLoader(
        TensorDataset(torch.FloatTensor(val_shuffled)),
        batch_size=32, shuffle=False, pin_memory=True
    )

    diffusion_shuffled_results = train_diffusion(
        diffusion_shuffled, train_loader_shuffled, val_loader_shuffled,
        num_epochs=100, lr=1e-4, logger=logger
    )

    # Generate and validate on shuffled
    save_status(status_path, "D3", "D3.1", "Validating shuffled diffusion")
    logger.info("Generating from shuffled diffusion model...")

    valid_nonces_shuffled = 0
    leading_zeros_shuffled = []

    with torch.no_grad():
        for i in range(0, 10000, 1000):  # Smaller sample for control
            batch_size = min(1000, 10000 - i)

            header_indices = np.random.choice(len(test_shuffled), batch_size)
            batch_headers = torch.FloatTensor(test_shuffled[header_indices, :992]).to(Config.DEVICE)

            generated_nonces = diffusion_shuffled.sample(batch_headers, num_samples=1)
            generated_nonces_binary = (generated_nonces > 0.5).cpu().numpy()

            for j in range(batch_size):
                header_bits = test_shuffled[header_indices[j], :992]
                nonce_bits = generated_nonces_binary[j]

                leading_zeros = verify_sha256_nonce(header_bits, nonce_bits)
                leading_zeros_shuffled.append(leading_zeros)

                if leading_zeros > 0:
                    valid_nonces_shuffled += 1

            if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
                break

    validity_rate_shuffled = valid_nonces_shuffled / len(leading_zeros_shuffled) if leading_zeros_shuffled else 0.0

    logger.info(f"Diffusion shuffled validity rate: {validity_rate_shuffled:.6f}")

    results["d3_1_diffusion"]["shuffled_control"] = {
        "validity_rate": float(validity_rate_shuffled),
        "n_generated": len(leading_zeros_shuffled),
        "average_leading_zeros": float(np.mean(leading_zeros_shuffled)),
        "train_results": convert_for_json(diffusion_shuffled_results)
    }

    if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
        return results

    # D3.2: GAN High-Power
    save_status(status_path, "D3", "D3.2", "Training GAN")
    logger.info("-" * 40)
    logger.info("D3.2: GAN High-Power")
    logger.info("-" * 40)

    # Train GAN on real data
    logger.info("Training GAN on re-mined data...")

    generator = Generator(
        noise_dim=100,
        header_dim=992,
        nonce_dim=32,
        depth=3,
        width=512
    ).to(Config.DEVICE)

    discriminator = Discriminator(
        header_dim=992,
        nonce_dim=32,
        depth=3,
        width=512
    ).to(Config.DEVICE)

    gan_train_results = train_gan(
        generator, discriminator, train_loader, val_loader,
        num_epochs=100, lr=2e-4, logger=logger
    )

    # Generate and validate nonces
    save_status(status_path, "D3", "D3.2", "Generating and validating GAN nonces")
    logger.info("Generating 50,000 nonces from GAN...")

    valid_nonces_gan = 0
    leading_zeros_gan = []

    with torch.no_grad():
        for i in range(0, n_generations, 1000):
            batch_size = min(1000, n_generations - i)

            header_indices = np.random.choice(n_test_samples, batch_size)
            batch_headers = test_headers[header_indices]

            # Generate nonces
            noise = torch.randn(batch_size, generator.noise_dim, device=Config.DEVICE)
            generated_nonces = generator(noise, batch_headers)
            generated_nonces_binary = (generated_nonces > 0.5).cpu().numpy()

            for j in range(batch_size):
                header_bits = test_data[header_indices[j], :992]
                nonce_bits = generated_nonces_binary[j]

                leading_zeros = verify_sha256_nonce(header_bits, nonce_bits)
                leading_zeros_gan.append(leading_zeros)

                if leading_zeros > 0:
                    valid_nonces_gan += 1

            if (i + batch_size) % 10000 == 0:
                logger.info(f"Processed {i + batch_size}/{n_generations} GAN generations")

            if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
                break

    validity_rate_gan = valid_nonces_gan / len(leading_zeros_gan) if leading_zeros_gan else 0.0

    # Calculate confidence interval
    n = len(leading_zeros_gan)
    if n > 0:
        p = validity_rate_gan
        ci_margin = 1.96 * math.sqrt(p * (1 - p) / n)
        ci_lower_gan = max(0, p - ci_margin)
        ci_upper_gan = min(1, p + ci_margin)
    else:
        ci_lower_gan = ci_upper_gan = 0

    logger.info(f"GAN validity rate: {validity_rate_gan:.6f} [{ci_lower_gan:.6f}, {ci_upper_gan:.6f}]")

    results["d3_2_gan"] = {
        "validity_rate": float(validity_rate_gan),
        "confidence_interval": [float(ci_lower_gan), float(ci_upper_gan)],
        "n_generated": len(leading_zeros_gan),
        "average_leading_zeros": float(np.mean(leading_zeros_gan)),
        "leading_zeros_distribution": convert_for_json(leading_zeros_gan),
        "train_results": convert_for_json(gan_train_results)
    }

    if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
        return results

    # D3.3: Three-Way Comparison Summary
    save_status(status_path, "D3", "D3.3", "Creating three-way comparison")
    logger.info("-" * 40)
    logger.info("D3.3: Three-Way Comparison Summary")
    logger.info("-" * 40)

    # For demonstration, we'll use the same re-mined data for "random nonces"
    # and compare with shuffled data as "broken association"

    comparison_summary = {
        "diffusion": {
            "remined_data": float(validity_rate),
            "shuffled_data": float(validity_rate_shuffled),
            "difference": float(validity_rate - validity_rate_shuffled)
        },
        "gan": {
            "remined_data": float(validity_rate_gan),
            "shuffled_data": "not_computed",  # Would need to train GAN on shuffled
            "difference": "not_computed"
        }
    }

    results["d3_3_comparison"] = comparison_summary

    logger.info("Phase D3 completed!")
    logger.info(f"Diffusion: remined={validity_rate:.6f}, shuffled={validity_rate_shuffled:.6f}")
    logger.info(f"GAN: remined={validity_rate_gan:.6f}")

    return results


# ============================================================================
# Phase D4: High-Power Phase 2C Replication
# ============================================================================

def run_phase_d4(sandbox_dir: Path, logger: logging.Logger, status_path: Path) -> Dict[str, Any]:
    """Run Phase D4: High-Power Phase 2C Replication"""
    logger.info("=" * 60)
    logger.info("PHASE D4: HIGH-POWER PHASE 2C REPLICATION")
    logger.info("=" * 60)

    results = {}

    # Set random seed
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

    data_dir = sandbox_dir / Config.DATA_DIR

    # D4.1: High-Power MLP at Each Round Count
    save_status(status_path, "D4", "D4.1", "Starting high-power MLP training")
    logger.info("-" * 40)
    logger.info("D4.1: High-Power MLP at Each Round Count")
    logger.info("-" * 40)

    round_counts = [4, 5, 8, 10, 15, 20, 32, 64]
    round_results = {}

    # Best MLP config from Phase 2A
    mlp_config = {
        "input_dim": 992,  # Header bits only
        "output_dim": 32,   # Nonce bits
        "depth": 2,
        "width": 1024,
        "dropout": 0.1
    }

    for round_count in round_counts:
        save_status(status_path, "D4", "D4.1", f"Training round {round_count}")
        logger.info(f"Training MLP for {round_count} rounds...")

        # Load dataset for this round count
        dataset_file = f"dataset_reduced_r{round_count}.npy"
        dataset_path = data_dir / dataset_file

        if not dataset_path.exists():
            logger.warning(f"Dataset {dataset_file} not found, skipping round {round_count}")
            continue

        data = load_dataset(dataset_path, logger)

        # Split data: 40K train, 5K val, 5K test
        train_data = data[:40000]
        val_data = data[40000:45000]
        test_data = data[45000:50000]

        # Separate headers and nonces
        train_headers = train_data[:, :992]
        train_nonces = train_data[:, 992:1024]
        val_headers = val_data[:, :992]
        val_nonces = val_data[:, 992:1024]
        test_headers = test_data[:, :992]
        test_nonces = test_data[:, 992:1024]

        logger.info(f"Round {round_count}: {len(train_data)}/{len(val_data)}/{len(test_data)} samples")

        # Create data loaders
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(train_headers), torch.FloatTensor(train_nonces)),
            batch_size=128, shuffle=True, pin_memory=True
        )
        val_loader = DataLoader(
            TensorDataset(torch.FloatTensor(val_headers), torch.FloatTensor(val_nonces)),
            batch_size=128, shuffle=False, pin_memory=True
        )
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(test_headers), torch.FloatTensor(test_nonces)),
            batch_size=128, shuffle=False, pin_memory=True
        )

        # Train model
        model = MLPClassifier(**mlp_config).to(Config.DEVICE)

        train_results = train_mlp_classifier(
            model, train_loader, val_loader,
            num_epochs=500, lr=3e-4, patience=50, min_epochs=100, logger=logger
        )

        # Evaluate per-bit accuracies
        per_bit_accuracies, per_bit_p_values = evaluate_per_bit_accuracy(model, test_loader, logger)

        # Apply Bonferroni correction
        significant_bits = bonferroni_correction(per_bit_p_values)
        n_significant = sum(significant_bits)

        # Overall accuracy
        overall_accuracy = np.mean(per_bit_accuracies)

        logger.info(f"Round {round_count}: overall accuracy = {overall_accuracy:.6f}")
        logger.info(f"Round {round_count}: {n_significant}/32 bits significant after Bonferroni correction")

        round_results[str(round_count)] = {
            "overall_accuracy": float(overall_accuracy),
            "per_bit_accuracies": convert_for_json(per_bit_accuracies),
            "per_bit_p_values": convert_for_json(per_bit_p_values),
            "significant_bits": convert_for_json(significant_bits),
            "n_significant_bits": int(n_significant),
            "train_results": convert_for_json(train_results)
        }

        if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
            break

    results["d4_1_round_results"] = round_results

    if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
        return results

    # D4.2: Per-Bit Signal Map
    save_status(status_path, "D4", "D4.2", "Creating per-bit signal map")
    logger.info("-" * 40)
    logger.info("D4.2: Per-Bit Signal Map")
    logger.info("-" * 40)

    # Create signal map: round_count x bit_position matrix of accuracies and significance
    signal_map = {
        "round_counts": [],
        "bit_accuracies": [],  # [round_count][bit_position]
        "bit_significance": [],  # [round_count][bit_position] - boolean
        "bit_p_values": []  # [round_count][bit_position]
    }

    for round_count in sorted([int(r) for r in round_results.keys()]):
        round_key = str(round_count)
        if round_key in round_results:
            signal_map["round_counts"].append(round_count)
            signal_map["bit_accuracies"].append(round_results[round_key]["per_bit_accuracies"])
            signal_map["bit_significance"].append(round_results[round_key]["significant_bits"])
            signal_map["bit_p_values"].append(round_results[round_key]["per_bit_p_values"])

    # Find bits that show signal at any round count
    any_significant_bits = [False] * 32
    for round_significance in signal_map["bit_significance"]:
        for bit in range(32):
            if round_significance[bit]:
                any_significant_bits[bit] = True

    n_any_significant = sum(any_significant_bits)
    logger.info(f"Bits showing significance at any round count: {n_any_significant}/32")

    # Find earliest round where each bit becomes significant
    earliest_significant_round = {}
    for bit in range(32):
        for i, round_count in enumerate(signal_map["round_counts"]):
            if signal_map["bit_significance"][i][bit]:
                earliest_significant_round[bit] = round_count
                break

    if earliest_significant_round:
        logger.info(f"Earliest significant rounds: {earliest_significant_round}")

    signal_map["summary"] = {
        "any_significant_bits": convert_for_json(any_significant_bits),
        "n_any_significant": int(n_any_significant),
        "earliest_significant_round": convert_for_json(earliest_significant_round)
    }

    results["d4_2_signal_map"] = signal_map

    logger.info("Phase D4 completed!")
    return results


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="SHA-256 ML Redux Deep Investigation")
    parser.add_argument("--sandbox", required=True, type=Path, help="Sandbox directory")
    parser.add_argument("--d2-only", action="store_true", help="Run only Phase D2")
    parser.add_argument("--d3-only", action="store_true", help="Run only Phase D3")
    parser.add_argument("--d4-only", action="store_true", help="Run only Phase D4")

    args = parser.parse_args()

    sandbox_dir = args.sandbox
    if not sandbox_dir.exists():
        print(f"Error: Sandbox directory {sandbox_dir} does not exist")
        sys.exit(1)

    data_dir = sandbox_dir / Config.DATA_DIR
    data_dir.mkdir(exist_ok=True)

    # Setup logging
    log_path = data_dir / Config.LOG_FILE
    logger = setup_logging(log_path)

    logger.info("Starting SHA-256 ML Redux Deep Investigation")
    logger.info(f"Sandbox: {sandbox_dir}")
    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")

    # Paths
    status_path = data_dir / Config.STATUS_JSON

    # Setup signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, creating shutdown.signal file")
        (sandbox_dir / Config.SHUTDOWN_SIGNAL).touch()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Determine which phases to run
        run_d2 = not (args.d3_only or args.d4_only)
        run_d3 = not (args.d2_only or args.d4_only)
        run_d4 = not (args.d2_only or args.d3_only)

        if args.d2_only:
            run_d2 = True
        if args.d3_only:
            run_d3 = True
        if args.d4_only:
            run_d4 = True

        logger.info(f"Phases to run: D2={run_d2}, D3={run_d3}, D4={run_d4}")

        # Phase D2: VAE Controls
        if run_d2:
            logger.info("Starting Phase D2...")
            d2_results = run_phase_d2(sandbox_dir, logger, status_path)

            # Save D2 results
            d2_results_path = data_dir / Config.D2_RESULTS
            with open(d2_results_path, 'w') as f:
                json.dump(convert_for_json(d2_results), f, indent=2)

            logger.info(f"Phase D2 results saved to {d2_results_path}")

            if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
                logger.info("Shutdown signal detected, stopping after D2")
                return

        # Phase D3: Generative Model Power Tests
        if run_d3:
            logger.info("Starting Phase D3...")
            d3_results = run_phase_d3(sandbox_dir, logger, status_path)

            # Save D3 results
            d3_results_path = data_dir / Config.D3_RESULTS
            with open(d3_results_path, 'w') as f:
                json.dump(convert_for_json(d3_results), f, indent=2)

            logger.info(f"Phase D3 results saved to {d3_results_path}")

            if should_shutdown(sandbox_dir / Config.SHUTDOWN_SIGNAL):
                logger.info("Shutdown signal detected, stopping after D3")
                return

        # Phase D4: High-Power Phase 2C Replication
        if run_d4:
            logger.info("Starting Phase D4...")
            d4_results = run_phase_d4(sandbox_dir, logger, status_path)

            # Save D4 results
            d4_results_path = data_dir / Config.D4_RESULTS
            with open(d4_results_path, 'w') as f:
                json.dump(convert_for_json(d4_results), f, indent=2)

            logger.info(f"Phase D4 results saved to {d4_results_path}")

        logger.info("Deep investigation completed successfully!")

    except Exception as e:
        logger.error(f"Error during investigation: {e}", exc_info=True)
        sys.exit(1)

    finally:
        # Clean up
        if (sandbox_dir / Config.SHUTDOWN_SIGNAL).exists():
            (sandbox_dir / Config.SHUTDOWN_SIGNAL).unlink()

        save_status(status_path, "completed", "finished", "All phases completed")


if __name__ == "__main__":
    main()