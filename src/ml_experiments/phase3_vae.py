#!/usr/bin/env python3
"""
Phase 3.4: VAE Analysis for SHA-256 ML Redux
Test whether (header, nonce) pairs have learnable latent structure using autoencoders.

This asks a different question from Phase 2: not "can we predict nonce from header?"
but "is there non-trivial joint structure in the (header, nonce) representation?"

Author: Claude Code (Math Musings)
Date: 2026-03-21
"""

import argparse
import json
import logging
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class VariationalAutoEncoder(nn.Module):
    """Variational Autoencoder for binary sequences."""

    def __init__(self, input_dim: int = 1024, latent_dim: int = 64,
                 encoder_depth: int = 4, encoder_width: int = 512,
                 decoder_depth: int = 4, decoder_width: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for i in range(encoder_depth):
            current_dim = encoder_width if i < encoder_depth - 1 else latent_dim * 2  # mu and log_var
            encoder_layers.extend([
                nn.Linear(prev_dim, current_dim),
                nn.ReLU() if i < encoder_depth - 1 else nn.Identity()
            ])
            prev_dim = current_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for i in range(decoder_depth):
            current_dim = decoder_width if i < decoder_depth - 1 else input_dim
            decoder_layers.extend([
                nn.Linear(prev_dim, current_dim),
                nn.ReLU() if i < decoder_depth - 1 else nn.Identity()
            ])
            prev_dim = current_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        encoded = self.encoder(x)
        mu, log_var = encoded[:, :self.latent_dim], encoded[:, self.latent_dim:]
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return recon, mu, log_var


class PlainAutoEncoder(nn.Module):
    """Plain Autoencoder (VAE with KL weight = 0)."""

    def __init__(self, input_dim: int = 1024, latent_dim: int = 64,
                 encoder_depth: int = 4, encoder_width: int = 512,
                 decoder_depth: int = 4, decoder_width: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for i in range(encoder_depth):
            current_dim = encoder_width if i < encoder_depth - 1 else latent_dim
            encoder_layers.extend([
                nn.Linear(prev_dim, current_dim),
                nn.ReLU() if i < encoder_depth - 1 else nn.Identity()
            ])
            prev_dim = current_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for i in range(decoder_depth):
            current_dim = decoder_width if i < decoder_depth - 1 else input_dim
            decoder_layers.extend([
                nn.Linear(prev_dim, current_dim),
                nn.ReLU() if i < decoder_depth - 1 else nn.Identity()
            ])
            prev_dim = current_dim
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass (returns dummy mu, log_var for compatibility)."""
        z = self.encode(x)
        recon = self.decode(z)
        # Return dummy values for mu, log_var for compatibility
        dummy_mu = torch.zeros_like(z)
        dummy_log_var = torch.zeros_like(z)
        return recon, dummy_mu, dummy_log_var


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             log_var: torch.Tensor, kl_weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """VAE loss function."""
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')

    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss


def compute_reconstruction_accuracy(recon_logits: torch.Tensor, target: torch.Tensor) -> float:
    """Compute binary reconstruction accuracy."""
    recon_probs = torch.sigmoid(recon_logits)
    predictions = (recon_probs > 0.5).float()
    accuracy = (predictions == target).float().mean().item()
    return accuracy


def compute_per_bit_accuracy(recon_logits: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute reconstruction accuracy for header vs nonce bits."""
    recon_probs = torch.sigmoid(recon_logits)
    predictions = (recon_probs > 0.5).float()

    # Header bits: 0:992, Nonce bits: 992:1024
    header_acc = (predictions[:, :992] == target[:, :992]).float().mean().item()
    nonce_acc = (predictions[:, 992:] == target[:, 992:]).float().mean().item()

    return {
        'header_accuracy': header_acc,
        'nonce_accuracy': nonce_acc,
        'overall_accuracy': (predictions == target).float().mean().item()
    }


def train_nonce_decoder(latent_representations: torch.Tensor, nonce_bits: torch.Tensor,
                       device: torch.device) -> float:
    """Train a small MLP to decode nonce bits from latent representations."""
    latent_dim = latent_representations.shape[1]
    nonce_dim = nonce_bits.shape[1]  # Should be 32

    # Simple MLP: latent_dim -> 128 -> nonce_dim
    nonce_decoder = nn.Sequential(
        nn.Linear(latent_dim, 128),
        nn.ReLU(),
        nn.Linear(128, nonce_dim)
    ).to(device)

    optimizer = torch.optim.Adam(nonce_decoder.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Train for 50 epochs
    dataset = TensorDataset(latent_representations, nonce_bits)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    nonce_decoder.train()
    for epoch in range(50):
        for latent_batch, nonce_batch in dataloader:
            optimizer.zero_grad()
            pred_logits = nonce_decoder(latent_batch)
            loss = criterion(pred_logits, nonce_batch)
            loss.backward()
            optimizer.step()

    # Evaluate
    nonce_decoder.eval()
    with torch.no_grad():
        all_pred_logits = nonce_decoder(latent_representations)
        accuracy = compute_reconstruction_accuracy(all_pred_logits, nonce_bits)

    return accuracy


def analyze_latent_space(model: nn.Module, test_loader: DataLoader, device: torch.device,
                        is_vae: bool = True) -> Dict:
    """Analyze latent space properties."""
    model.eval()
    latent_representations = []
    nonce_bits_list = []
    header_bits_list = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch[0].to(device)
            if is_vae:
                mu, log_var = model.encode(x)
                z = mu  # Use mean for analysis
            else:
                z = model.encode(x)

            latent_representations.append(z.cpu())
            nonce_bits_list.append(x[:, 992:].cpu())  # Nonce bits
            header_bits_list.append(x[:, :992].cpu())  # Header bits

    latent_representations = torch.cat(latent_representations, dim=0)
    nonce_bits = torch.cat(nonce_bits_list, dim=0)
    header_bits = torch.cat(header_bits_list, dim=0)

    # Move back to device for analysis
    latent_representations = latent_representations.to(device)
    nonce_bits = nonce_bits.to(device)
    header_bits = header_bits.to(device)

    # Analyze nonce decodability
    nonce_decodability = train_nonce_decoder(latent_representations, nonce_bits, device)

    # Compute latent space statistics
    latent_mean = latent_representations.mean(dim=0)
    latent_std = latent_representations.std(dim=0)

    # Compute correlations between latent dimensions and input bits
    latent_np = latent_representations.cpu().numpy()
    nonce_np = nonce_bits.cpu().numpy()
    header_np = header_bits.cpu().numpy()

    # Average correlation with nonce vs header bits
    nonce_correlations = []
    header_correlations = []

    for i in range(latent_np.shape[1]):
        # Correlation with average nonce value
        nonce_avg = nonce_np.mean(axis=1)
        nonce_corr = abs(np.corrcoef(latent_np[:, i], nonce_avg)[0, 1])
        if not np.isnan(nonce_corr):
            nonce_correlations.append(nonce_corr)

        # Correlation with average header value
        header_avg = header_np.mean(axis=1)
        header_corr = abs(np.corrcoef(latent_np[:, i], header_avg)[0, 1])
        if not np.isnan(header_corr):
            header_correlations.append(header_corr)

    return {
        'nonce_decodability_accuracy': float(nonce_decodability),
        'latent_mean_norm': float(torch.norm(latent_mean).item()),
        'latent_std_mean': float(latent_std.mean().item()),
        'avg_nonce_correlation': float(np.mean(nonce_correlations)) if nonce_correlations else 0.0,
        'avg_header_correlation': float(np.mean(header_correlations)) if header_correlations else 0.0,
        'latent_dimensions': int(latent_representations.shape[1])
    }


def load_data(data_path: Path, dataset_name: str, max_samples: Optional[int] = None) -> torch.Tensor:
    """Load and prepare data."""
    file_path = data_path / f"{dataset_name}.npy"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = np.load(file_path)
    if max_samples:
        data = data[:max_samples]

    # Convert to float32 tensor
    return torch.from_numpy(data).float()


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   scaler: torch.amp.GradScaler, epoch: int,
                   checkpoint_path: Path, config: Dict) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'epoch': epoch,
        'config': config
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                   scaler: torch.amp.GradScaler) -> Tuple[int, Dict]:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    return checkpoint['epoch'], checkpoint['config']


def update_status(status_path: Path, status: Dict) -> None:
    """Update status JSON file."""
    with open(status_path, 'w') as f:
        json.dump(status, f, indent=2)


def train_config(config: Dict, data_tensor: torch.Tensor, device: torch.device,
                sandbox_path: Path, logger: logging.Logger) -> Dict:
    """Train a single configuration."""
    # Create model
    is_vae = config['kl_weight'] > 0.0
    if is_vae:
        model = VariationalAutoEncoder(
            input_dim=1024,
            latent_dim=config['latent_dim'],
            encoder_depth=config['encoder_depth'],
            encoder_width=config['encoder_width'],
            decoder_depth=config['decoder_depth'],
            decoder_width=config['decoder_width']
        ).to(device)
    else:
        model = PlainAutoEncoder(
            input_dim=1024,
            latent_dim=config['latent_dim'],
            encoder_depth=config['encoder_depth'],
            encoder_width=config['encoder_width'],
            decoder_depth=config['decoder_depth'],
            decoder_width=config['decoder_width']
        ).to(device)

    # Split data
    train_size = int(0.8 * len(data_tensor))
    val_size = int(0.1 * len(data_tensor))
    test_size = len(data_tensor) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        TensorDataset(data_tensor), [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = torch.amp.GradScaler(device)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_total_loss = 0.0
        train_recon_loss = 0.0
        train_kl_loss = 0.0

        for batch_idx, (batch,) in enumerate(train_loader):
            x = batch.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                recon_x, mu, log_var = model(x)
                total_loss, recon_loss, kl_loss = vae_loss(
                    recon_x, x, mu, log_var, config['kl_weight']
                )
                # Normalize by batch size
                total_loss = total_loss / x.size(0)
                recon_loss = recon_loss / x.size(0)
                kl_loss = kl_loss / x.size(0)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_total_loss += total_loss.item()
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()

        # Validation
        model.eval()
        val_total_loss = 0.0
        val_recon_loss = 0.0
        val_kl_loss = 0.0

        with torch.no_grad():
            for batch, in val_loader:
                x = batch.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    recon_x, mu, log_var = model(x)
                    total_loss, recon_loss, kl_loss = vae_loss(
                        recon_x, x, mu, log_var, config['kl_weight']
                    )
                    # Normalize by batch size
                    total_loss = total_loss / x.size(0)
                    recon_loss = recon_loss / x.size(0)
                    kl_loss = kl_loss / x.size(0)

                val_total_loss += total_loss.item()
                val_recon_loss += recon_loss.item()
                val_kl_loss += kl_loss.item()

        # Compute averages
        train_total_loss /= len(train_loader)
        train_recon_loss /= len(train_loader)
        train_kl_loss /= len(train_loader)
        val_total_loss /= len(val_loader)
        val_recon_loss /= len(val_loader)
        val_kl_loss /= len(val_loader)

        logger.info(f"Epoch {epoch+1}: train_recon={train_recon_loss:.4f}, "
                   f"val_recon={val_recon_loss:.4f}, kl={val_kl_loss:.4f}")

        # Early stopping
        if val_recon_loss < best_val_loss:
            best_val_loss = val_recon_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Final evaluation
    model.eval()
    test_results = {}

    with torch.no_grad():
        test_total_loss = 0.0
        test_recon_loss = 0.0
        test_kl_loss = 0.0
        all_accuracies = []

        for batch, in test_loader:
            x = batch.to(device)
            with torch.amp.autocast(device_type="cuda"):
                recon_x, mu, log_var = model(x)
                total_loss, recon_loss, kl_loss = vae_loss(
                    recon_x, x, mu, log_var, config['kl_weight']
                )
                # Normalize by batch size
                total_loss = total_loss / x.size(0)
                recon_loss = recon_loss / x.size(0)
                kl_loss = kl_loss / x.size(0)

            test_total_loss += total_loss.item()
            test_recon_loss += recon_loss.item()
            test_kl_loss += kl_loss.item()

            # Compute per-bit accuracies
            per_bit_acc = compute_per_bit_accuracy(recon_x, x)
            all_accuracies.append(per_bit_acc)

    # Average test metrics
    test_total_loss /= len(test_loader)
    test_recon_loss /= len(test_loader)
    test_kl_loss /= len(test_loader)

    # Average accuracies
    avg_header_acc = sum(acc['header_accuracy'] for acc in all_accuracies) / len(all_accuracies)
    avg_nonce_acc = sum(acc['nonce_accuracy'] for acc in all_accuracies) / len(all_accuracies)
    avg_overall_acc = sum(acc['overall_accuracy'] for acc in all_accuracies) / len(all_accuracies)

    # Analyze latent space
    latent_analysis = analyze_latent_space(model, test_loader, device, is_vae)

    return {
        'config': config,
        'test_reconstruction_loss': float(test_recon_loss),
        'test_kl_loss': float(test_kl_loss),
        'test_total_loss': float(test_total_loss),
        'header_reconstruction_accuracy': float(avg_header_acc),
        'nonce_reconstruction_accuracy': float(avg_nonce_acc),
        'overall_reconstruction_accuracy': float(avg_overall_acc),
        'latent_analysis': latent_analysis,
        'epochs_trained': epoch + 1
    }


def generate_search_configs(num_configs: int = 40) -> List[Dict]:
    """Generate random search configurations."""
    configs = []

    latent_dims = [32, 64, 128, 256]
    depths = [2, 4, 6]
    widths = [256, 512, 1024]
    kl_weights = [0.0, 0.01, 0.1, 1.0]  # 0.0 = plain AE
    learning_rates = [1e-4, 3e-4, 1e-3]
    batch_sizes = [128, 256]

    for _ in range(num_configs):
        config = {
            'latent_dim': random.choice(latent_dims),
            'encoder_depth': random.choice(depths),
            'encoder_width': random.choice(widths),
            'decoder_depth': random.choice(depths),
            'decoder_width': random.choice(widths),
            'kl_weight': random.choice(kl_weights),
            'learning_rate': random.choice(learning_rates),
            'batch_size': random.choice(batch_sizes),
            'epochs': 30  # Fixed for search
        }
        configs.append(config)

    return configs


def run_architecture_search(sandbox_path: Path, logger: logging.Logger) -> Dict:
    """Run architecture search on 50K subset of real Bitcoin data."""
    data_path = sandbox_path / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load 50K subset of real Bitcoin data
    logger.info("Loading data for architecture search...")
    data_tensor = load_data(data_path, "dataset_real_bitcoin", max_samples=50000)
    logger.info(f"Loaded {len(data_tensor)} samples for search")

    # Generate search configurations
    search_configs = generate_search_configs(40)
    logger.info(f"Generated {len(search_configs)} search configurations")

    # Results storage
    results = []
    best_config = None
    best_val_loss = float('inf')

    # Status tracking
    status_path = data_path / "status.json"

    for i, config in enumerate(search_configs):
        try:
            logger.info(f"Training config {i+1}/{len(search_configs)}: {config}")

            # Update status
            update_status(status_path, {
                'phase': 'phase3_vae_search',
                'progress': f"{i+1}/{len(search_configs)}",
                'current_config': config,
                'timestamp': time.time()
            })

            # Train configuration
            result = train_config(config, data_tensor, device, sandbox_path, logger)
            results.append(result)

            # Track best configuration
            val_loss = result['test_reconstruction_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_config = config
                logger.info(f"New best config: val_loss={val_loss:.4f}")

            # Save checkpoint every 5 configs
            if (i + 1) % 5 == 0:
                checkpoint_data = {
                    'results': results,
                    'best_config': best_config,
                    'best_val_loss': best_val_loss,
                    'completed_configs': i + 1
                }
                checkpoint_path = data_path / "phase3_vae_search_checkpoint.json"
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
                logger.info(f"Saved search checkpoint at config {i+1}")

        except Exception as e:
            logger.error(f"Error training config {i+1}: {e}")
            results.append({
                'config': config,
                'error': str(e)
            })

    # Final results
    search_results = {
        'best_config': best_config,
        'best_validation_loss': float(best_val_loss),
        'all_results': results,
        'search_completed': True,
        'total_configs': len(search_configs),
        'timestamp': time.time()
    }

    # Save search results
    results_path = data_path / "phase3_vae_search_results.json"
    with open(results_path, 'w') as f:
        json.dump(search_results, f, indent=2)

    logger.info(f"Architecture search completed. Best config: {best_config}")
    return search_results


def run_full_training(sandbox_path: Path, logger: logging.Logger,
                     best_config: Optional[Dict] = None) -> Dict:
    """Run full training and analysis on both datasets."""
    data_path = sandbox_path / "data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best config if not provided
    if best_config is None:
        search_results_path = data_path / "phase3_vae_search_results.json"
        if search_results_path.exists():
            with open(search_results_path, 'r') as f:
                search_results = json.load(f)
                best_config = search_results['best_config']
        else:
            logger.warning("No search results found, using default config")
            best_config = {
                'latent_dim': 64,
                'encoder_depth': 4,
                'encoder_width': 512,
                'decoder_depth': 4,
                'decoder_width': 512,
                'kl_weight': 0.1,
                'learning_rate': 3e-4,
                'batch_size': 256
            }

    # Update config for full training
    best_config = best_config.copy()
    best_config['epochs'] = 300  # Increased for full training

    logger.info(f"Starting full training with config: {best_config}")

    # Results storage
    final_results = {}

    # Experiment 1: Real Bitcoin data
    logger.info("Experiment 1: Training on real Bitcoin data...")
    try:
        real_data = load_data(data_path, "dataset_real_bitcoin")
        logger.info(f"Loaded {len(real_data)} real Bitcoin samples")

        real_results = train_config(best_config, real_data, device, sandbox_path, logger)
        final_results['real_bitcoin_experiment'] = real_results

    except Exception as e:
        logger.error(f"Error in real Bitcoin experiment: {e}")
        final_results['real_bitcoin_experiment'] = {'error': str(e)}

    # Experiment 2: Re-mined data
    logger.info("Experiment 2: Training on re-mined data...")
    try:
        remined_data = load_data(data_path, "dataset_reduced_r64")
        logger.info(f"Loaded {len(remined_data)} re-mined samples")

        remined_results = train_config(best_config, remined_data, device, sandbox_path, logger)
        final_results['remined_experiment'] = remined_results

    except Exception as e:
        logger.error(f"Error in re-mined experiment: {e}")
        final_results['remined_experiment'] = {'error': str(e)}

    # Compare experiments
    comparison = {}
    if 'real_bitcoin_experiment' in final_results and 'remined_experiment' in final_results:
        real_exp = final_results['real_bitcoin_experiment']
        remined_exp = final_results['remined_experiment']

        if 'error' not in real_exp and 'error' not in remined_exp:
            comparison = {
                'nonce_accuracy_difference': (
                    real_exp['nonce_reconstruction_accuracy'] -
                    remined_exp['nonce_reconstruction_accuracy']
                ),
                'nonce_decodability_difference': (
                    real_exp['latent_analysis']['nonce_decodability_accuracy'] -
                    remined_exp['latent_analysis']['nonce_decodability_accuracy']
                ),
                'reconstruction_loss_difference': (
                    real_exp['test_reconstruction_loss'] -
                    remined_exp['test_reconstruction_loss']
                )
            }

    # Final results summary
    final_results.update({
        'best_config_used': best_config,
        'comparison_analysis': comparison,
        'timestamp': time.time(),
        'training_completed': True
    })

    # Save final results
    results_path = data_path / "phase3_vae_final_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info("Full training and analysis completed")
    return final_results


def setup_logging(log_path: Path) -> logging.Logger:
    """Set up logging."""
    logger = logging.getLogger('phase3_vae')
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


def setup_signal_handler():
    """Set up graceful shutdown signal handler."""
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        # Create shutdown signal file
        shutdown_path = Path.cwd() / "shutdown.signal"
        shutdown_path.touch()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    parser = argparse.ArgumentParser(description="Phase 3.4: VAE Analysis for SHA-256 ML Redux")
    parser.add_argument("--sandbox", type=str, required=True,
                       help="Path to sandbox directory")
    parser.add_argument("--search-only", action="store_true",
                       help="Run only architecture search")
    parser.add_argument("--full-only", action="store_true",
                       help="Run only full training (requires search results)")

    args = parser.parse_args()

    # Setup paths
    sandbox_path = Path(args.sandbox)
    data_path = sandbox_path / "data"
    data_path.mkdir(exist_ok=True)

    # Setup logging
    log_path = data_path / "phase3_vae.log"
    logger = setup_logging(log_path)

    # Setup signal handler
    setup_signal_handler()

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    logger.info("Starting Phase 3.4: VAE Analysis")
    logger.info(f"Sandbox path: {sandbox_path}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    try:
        # Run architecture search
        if not args.full_only:
            logger.info("Starting architecture search...")
            search_results = run_architecture_search(sandbox_path, logger)
            logger.info("Architecture search completed successfully")

        # Run full training
        if not args.search_only:
            logger.info("Starting full training and analysis...")
            final_results = run_full_training(sandbox_path, logger)
            logger.info("Full training completed successfully")

        logger.info("Phase 3.4 VAE analysis completed successfully")

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()