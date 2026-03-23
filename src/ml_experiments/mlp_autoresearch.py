#!/usr/bin/env python3
"""
SHA-256 ML Redux Phase 2A: AutoResearch MLP Hyperparameter Search

AutoResearch methodology: Random search over ~80 MLP configurations on Bitcoin
block header → nonce prediction task, followed by full training of best config.

Usage:
    python mlp_autoresearch.py --sandbox /mnt/d/sha256-ml-redux [--search-only] [--full-only] [--resume]
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.amp import GradScaler, autocast
    AMP_DEVICE = "cuda"
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_DEVICE = None
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset, random_split
import math


class SHA256Dataset(Dataset):
    """Dataset for Bitcoin block header → nonce prediction."""

    def __init__(self, data: np.ndarray):
        """
        Args:
            data: uint8 array of shape (N, 1024) with values in {0, 1}
                  bits[0:992] = header (input), bits[992:1024] = nonce (target)
        """
        self.data = data.astype(np.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        header = sample[:992]
        nonce = sample[992:1024]
        return header, nonce


class MLPBlock(nn.Module):
    """MLP block with optional skip connection."""

    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu",
                 dropout: float = 0.0, use_skip: bool = False):
        super().__init__()
        self.use_skip = use_skip and input_dim == output_dim

        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU() if activation.lower() == "relu" else nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x if self.use_skip else None
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.use_skip and residual is not None:
            x = x + residual

        return x


class SHA256MLP(nn.Module):
    """MLP for SHA-256 nonce prediction with optional skip connections."""

    def __init__(self, depth: int = 4, width: int = 1024, activation: str = "relu",
                 dropout: float = 0.0, use_skip: bool = False):
        super().__init__()

        layers = []
        input_dim = 992  # header bits

        # Create hidden layers
        for i in range(depth):
            output_dim = width
            use_skip_this_layer = use_skip and i >= 1 and i % 2 == 1  # every 2nd layer starting from layer 1

            layers.append(MLPBlock(input_dim, output_dim, activation, dropout, use_skip_this_layer))
            input_dim = output_dim

        # Output layer (no sigmoid — use BCEWithLogitsLoss for amp compatibility)
        layers.append(nn.Linear(input_dim, 32))  # 32 nonce bits

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)  # returns logits


class AutoResearchConfig:
    """Configuration for AutoResearch hyperparameter search."""

    @staticmethod
    def generate_random_configs(num_configs: int = 80) -> List[Dict[str, Any]]:
        """Generate random configurations for search."""
        configs = []

        # Architecture options
        depths = [2, 4, 6, 8]
        widths = [512, 1024, 2048, 4096]
        activations = ["relu", "gelu"]

        # Training options
        learning_rates = [1e-4, 3e-4, 1e-3, 3e-3]
        batch_sizes = [64, 128, 256]
        dropouts = [0.0, 0.1]

        for i in range(num_configs):
            depth = random.choice(depths)
            config = {
                "config_id": i,
                "depth": depth,
                "width": random.choice(widths),
                "activation": random.choice(activations),
                "use_skip": random.choice([True, False]) if depth >= 4 else False,
                "learning_rate": random.choice(learning_rates),
                "batch_size": random.choice(batch_sizes),
                "dropout": random.choice(dropouts),
            }
            configs.append(config)

        return configs


class AutoResearchTrainer:
    """Main trainer for AutoResearch methodology."""

    def __init__(self, sandbox_dir: str):
        self.sandbox_dir = Path(sandbox_dir)
        self.data_dir = self.sandbox_dir / "data"
        self.checkpoint_path = self.sandbox_dir / "checkpoint_phase2a.pt"
        self.status_path = self.sandbox_dir / "status.json"
        self.log_path = self.data_dir / "phase2a_search_log.jsonl"

        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)

        # Set random seeds for reproducibility
        self.set_random_seeds(42)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Mixed precision scaler
        self.scaler = GradScaler()

        # Graceful shutdown handling
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)

        # Search state
        self.search_results = []
        self.best_config = None
        self.best_accuracy = 0.0

    def _signal_handler(self, signum, frame):
        """Handle SIGINT for graceful shutdown."""
        print("\nShutdown signal received. Will save checkpoint and exit gracefully...")
        self.shutdown_requested = True

    def set_random_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def check_shutdown_signal(self) -> bool:
        """Check for shutdown signal file or SIGINT."""
        shutdown_file = self.sandbox_dir / "shutdown.signal"
        if shutdown_file.exists() or self.shutdown_requested:
            return True
        return False

    def update_status(self, phase: str, message: str, progress: str = ""):
        """Update status JSON file."""
        status = {
            "phase": phase,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "progress": progress,
            "current_config": getattr(self, 'current_config', {}),
            "best_config": self.best_config,
            "best_accuracy": self.best_accuracy
        }

        with open(self.status_path, 'w') as f:
            json.dump(status, f, indent=2)

    def save_checkpoint(self, **additional_state):
        """Save full training state checkpoint."""
        state = {
            "search_results": self.search_results,
            "best_config": self.best_config,
            "best_accuracy": self.best_accuracy,
            "torch_rng_state": torch.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            **additional_state
        }

        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()

        torch.save(state, self.checkpoint_path)
        print(f"Checkpoint saved to {self.checkpoint_path}")

    def load_checkpoint(self) -> Optional[Dict]:
        """Load checkpoint if it exists."""
        if not self.checkpoint_path.exists():
            return None

        print(f"Loading checkpoint from {self.checkpoint_path}")
        state = torch.load(self.checkpoint_path, map_location=self.device)

        self.search_results = state.get("search_results", [])
        self.best_config = state.get("best_config")
        self.best_accuracy = state.get("best_accuracy", 0.0)

        # Restore RNG states (must be CPU ByteTensors)
        torch.set_rng_state(state["torch_rng_state"].cpu().byte())
        np.random.set_state(state["numpy_rng_state"])
        random.setstate(state["python_rng_state"])

        if torch.cuda.is_available() and "cuda_rng_state" in state:
            cuda_state = state["cuda_rng_state"]
            if hasattr(cuda_state, 'cpu'):
                cuda_state = cuda_state.cpu().byte()
            torch.cuda.set_rng_state(cuda_state)

        return state

    def load_dataset(self, dataset_path: Path) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load dataset and create train/val/test splits."""
        data = np.load(dataset_path)
        dataset = SHA256Dataset(data)

        # 80/10/10 split
        total_size = len(dataset)
        train_size = int(0.8 * total_size)
        val_size = int(0.1 * total_size)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )

        return train_dataset, val_dataset, test_dataset

    def create_data_loaders(self, train_dataset, val_dataset, test_dataset, batch_size: int):
        """Create data loaders."""
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, val_loader, test_loader

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: torch.optim.Optimizer, criterion: nn.Module) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0.0

        for batch_idx, (headers, nonces) in enumerate(train_loader):
            headers, nonces = headers.to(self.device), nonces.to(self.device)

            optimizer.zero_grad()

            with autocast(device_type="cuda"):
                outputs = model(headers)
                loss = criterion(outputs, nonces)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def evaluate(self, model: nn.Module, data_loader: DataLoader,
                criterion: nn.Module) -> Tuple[float, float, np.ndarray]:
        """Evaluate model and return loss, accuracy, and per-bit accuracies."""
        model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for headers, nonces in data_loader:
                headers, nonces = headers.to(self.device), nonces.to(self.device)

                with autocast(device_type="cuda"):
                    outputs = model(headers)
                    loss = criterion(outputs, nonces)

                total_loss += loss.item()

                # Convert logits to binary predictions (threshold at 0 for logits)
                predictions = (outputs > 0.0).float()
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(nonces.cpu().numpy())

        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate per-bit accuracies
        per_bit_accuracies = np.mean(all_predictions == all_targets, axis=0)
        overall_accuracy = np.mean(per_bit_accuracies)

        return total_loss / len(data_loader), overall_accuracy, per_bit_accuracies

    def train_config_on_difficulty(self, config: Dict, difficulty: int) -> float:
        """Train a config on a specific difficulty level for 20 epochs."""
        dataset_path = self.data_dir / f"dataset_n{difficulty:02d}_real.npy"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        # Load data and create splits
        train_dataset, val_dataset, test_dataset = self.load_dataset(dataset_path)
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_dataset, val_dataset, test_dataset, config["batch_size"]
        )

        # Create model
        model = SHA256MLP(
            depth=config["depth"],
            width=config["width"],
            activation=config["activation"],
            dropout=config["dropout"],
            use_skip=config["use_skip"]
        ).to(self.device)

        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

        best_val_accuracy = 0.0

        for epoch in range(20):
            if self.check_shutdown_signal():
                return best_val_accuracy

            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_accuracy, val_per_bit_acc = self.evaluate(model, val_loader, criterion)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

        return best_val_accuracy

    def hyperparameter_search(self) -> Dict:
        """Run AutoResearch hyperparameter search."""
        print("Starting AutoResearch hyperparameter search...")

        # Load checkpoint if resuming
        checkpoint = self.load_checkpoint()
        start_config_idx = 0

        if checkpoint and "search_phase_config_idx" in checkpoint:
            start_config_idx = checkpoint["search_phase_config_idx"]
            print(f"Resuming search from config {start_config_idx}")

        # Generate or load configs
        if checkpoint and "search_configs" in checkpoint:
            configs = checkpoint["search_configs"]
        else:
            configs = AutoResearchConfig.generate_random_configs(80)

        difficulty_levels = [1, 2, 3, 4]

        for config_idx in range(start_config_idx, len(configs)):
            if self.check_shutdown_signal():
                self.save_checkpoint(
                    search_phase_config_idx=config_idx,
                    search_configs=configs
                )
                return self.best_config

            config = configs[config_idx]
            self.current_config = config

            print(f"\nTesting config {config_idx + 1}/{len(configs)}")
            print(f"Config: {config}")

            # Test on each difficulty level
            difficulty_accuracies = []
            for difficulty in difficulty_levels:
                self.update_status(
                    "phase2a_search",
                    f"Config {config_idx + 1}/{len(configs)}, difficulty n={difficulty}",
                    f"{config_idx + 1}/{len(configs)} configs, best_acc={self.best_accuracy:.4f}"
                )

                print(f"  Testing on difficulty n={difficulty}...")
                accuracy = self.train_config_on_difficulty(config, difficulty)
                difficulty_accuracies.append(accuracy)
                print(f"    Validation accuracy: {accuracy:.4f}")

            # Average accuracy across all difficulties
            avg_accuracy = np.mean(difficulty_accuracies)

            # Record results
            result = {
                **config,
                "difficulty_accuracies": difficulty_accuracies,
                "avg_accuracy": avg_accuracy,
                "timestamp": datetime.now().isoformat()
            }

            self.search_results.append(result)

            # Update best config
            if avg_accuracy > self.best_accuracy:
                self.best_accuracy = avg_accuracy
                self.best_config = config.copy()
                print(f"  *** NEW BEST CONFIG! Average accuracy: {avg_accuracy:.4f} ***")

            # Log result
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(result) + '\n')

            print(f"  Average accuracy: {avg_accuracy:.4f} (best so far: {self.best_accuracy:.4f})")

            # Save checkpoint every 5 configs
            if (config_idx + 1) % 5 == 0:
                self.save_checkpoint(
                    search_phase_config_idx=config_idx + 1,
                    search_configs=configs
                )

        # Final checkpoint
        self.save_checkpoint(
            search_phase_config_idx=len(configs),
            search_configs=configs,
            search_complete=True
        )

        print(f"\nHyperparameter search complete!")
        print(f"Best configuration (avg accuracy {self.best_accuracy:.4f}):")
        print(json.dumps(self.best_config, indent=2))

        return self.best_config

    def load_full_bitcoin_dataset(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load full Bitcoin dataset with chronological split."""
        dataset_path = self.data_dir / "dataset_real_bitcoin.npy"

        if not dataset_path.exists():
            raise FileNotFoundError(f"Full Bitcoin dataset not found: {dataset_path}")

        data = np.load(dataset_path)

        # Data is already in block-height order (from Electrum download)
        # Use chronological split: first 80% train, next 10% val, last 10% test
        # This is more realistic than random split (no future leakage)
        n = len(data)
        train_end = int(0.8 * n)
        val_end = int(0.9 * n)

        train_dataset = SHA256Dataset(data[:train_end])
        val_dataset = SHA256Dataset(data[train_end:val_end])
        test_dataset = SHA256Dataset(data[val_end:])

        return train_dataset, val_dataset, test_dataset

    def full_training(self, config: Dict) -> Dict:
        """Train best config to convergence on full dataset."""
        print(f"\nStarting full training with best config:")
        print(json.dumps(config, indent=2))

        # Load full dataset
        train_dataset, val_dataset, test_dataset = self.load_full_bitcoin_dataset()
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_dataset, val_dataset, test_dataset, config["batch_size"]
        )

        print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

        # Create model
        model = SHA256MLP(
            depth=config["depth"],
            width=config["width"],
            activation=config["activation"],
            dropout=config["dropout"],
            use_skip=config["use_skip"]
        ).to(self.device)

        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20)

        # Load checkpoint if resuming full training
        checkpoint = self.load_checkpoint()
        start_epoch = 0
        best_val_loss = float('inf')
        epochs_without_improvement = 0

        if checkpoint and "full_training_epoch" in checkpoint:
            start_epoch = checkpoint["full_training_epoch"]
            best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print(f"Resuming full training from epoch {start_epoch}")

        # Training loop
        max_epochs = 500
        min_epochs = 50
        patience = 20

        for epoch in range(start_epoch, max_epochs):
            if self.check_shutdown_signal():
                self.save_checkpoint(
                    full_training_epoch=epoch,
                    best_val_loss=best_val_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict()
                )
                break

            self.update_status(
                "phase2a_full_training",
                f"Epoch {epoch + 1}/{max_epochs}",
                f"best_val_loss={best_val_loss:.6f}, patience={epochs_without_improvement}/{patience}"
            )

            # Training
            train_loss = self.train_epoch(model, train_loader, optimizer, criterion)
            val_loss, val_accuracy, val_per_bit_acc = self.evaluate(model, val_loader, criterion)

            scheduler.step()

            print(f"Epoch {epoch + 1:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, val_acc={val_accuracy:.4f}")

            # Early stopping logic
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                # Save best model
                best_model_path = self.sandbox_dir / "best_model_phase2a.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_per_bit_acc": val_per_bit_acc,
                    "epoch": epoch
                }, best_model_path)
            else:
                epochs_without_improvement += 1

            # Check early stopping
            if epoch >= min_epochs and epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(
                    full_training_epoch=epoch + 1,
                    best_val_loss=best_val_loss,
                    epochs_without_improvement=epochs_without_improvement,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scheduler_state_dict=scheduler.state_dict()
                )

        # Final evaluation on test set
        print("\nEvaluating on test set...")
        best_model_path = self.sandbox_dir / "best_model_phase2a.pt"
        best_checkpoint = torch.load(best_model_path)
        model.load_state_dict(best_checkpoint["model_state_dict"])

        test_loss, test_accuracy, test_per_bit_acc = self.evaluate(model, test_loader, criterion)

        # Statistical significance testing
        n_samples = len(test_dataset)
        p_values = []

        for bit_idx in range(32):
            # Two-tailed test: H0: accuracy = 0.5, H1: accuracy ≠ 0.5
            bit_accuracy = test_per_bit_acc[bit_idx]
            z_score = (bit_accuracy - 0.5) / np.sqrt(0.25 / n_samples)
            # Approximate normal CDF without scipy
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z_score) / math.sqrt(2))))
            p_values.append(p_value)

        # Bonferroni correction
        bonferroni_alpha = 0.01 / 32
        significant_bits = np.array(p_values) < bonferroni_alpha
        signal_detected = np.any(significant_bits & (test_per_bit_acc > 0.505))

        results = {
            "config": config,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
            "test_per_bit_accuracies": test_per_bit_acc.tolist(),
            "p_values": p_values,
            "significant_bits": significant_bits.tolist(),
            "signal_detected": bool(signal_detected),
            "bonferroni_alpha": float(bonferroni_alpha),
            "n_test_samples": int(n_samples),
            "timestamp": datetime.now().isoformat()
        }

        # Save final results
        results_path = self.sandbox_dir / "phase2a_final_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nFinal Results:")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Signal detected: {signal_detected}")
        print(f"Significant nonce bits (p < {bonferroni_alpha:.6f}): {np.sum(significant_bits)}/32")

        if signal_detected:
            sig_bit_indices = np.where(significant_bits & (test_per_bit_acc > 0.505))[0]
            print(f"Signal bits: {sig_bit_indices} with accuracies: {test_per_bit_acc[sig_bit_indices]}")

        return results


def main():
    parser = argparse.ArgumentParser(description="SHA-256 ML Redux Phase 2A: AutoResearch MLP")
    parser.add_argument("--sandbox", required=True, help="Sandbox directory path")
    parser.add_argument("--search-only", action="store_true", help="Only run hyperparameter search")
    parser.add_argument("--full-only", action="store_true", help="Only run full training (skip search)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint (auto-detected anyway)")

    args = parser.parse_args()

    trainer = AutoResearchTrainer(args.sandbox)

    try:
        if not args.full_only:
            # Run hyperparameter search
            search_results_path = trainer.sandbox_dir / "phase2a_search_results.json"

            if search_results_path.exists() and not args.resume:
                print("Search results already exist. Loading best config...")
                with open(search_results_path) as f:
                    search_data = json.load(f)
                    trainer.best_config = search_data["best_config"]
                    trainer.best_accuracy = search_data["best_accuracy"]
            else:
                best_config = trainer.hyperparameter_search()

                # Save search results
                search_summary = {
                    "best_config": trainer.best_config,
                    "best_accuracy": trainer.best_accuracy,
                    "total_configs": len(trainer.search_results),
                    "search_results": trainer.search_results,
                    "timestamp": datetime.now().isoformat()
                }

                with open(search_results_path, 'w') as f:
                    json.dump(search_summary, f, indent=2)

        if not args.search_only:
            # Run full training
            final_results_path = trainer.sandbox_dir / "phase2a_final_results.json"

            if final_results_path.exists():
                print("Full training results already exist.")
                with open(final_results_path) as f:
                    results = json.load(f)
                    print(f"Test accuracy: {results['test_accuracy']:.4f}")
                    print(f"Signal detected: {results['signal_detected']}")
            else:
                if trainer.best_config is None:
                    # Load from search results
                    search_results_path = trainer.sandbox_dir / "phase2a_search_results.json"
                    with open(search_results_path) as f:
                        search_data = json.load(f)
                        trainer.best_config = search_data["best_config"]

                trainer.full_training(trainer.best_config)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        # Final cleanup
        trainer.update_status("phase2a_complete", "Phase 2A training finished")
        print("Phase 2A complete.")


if __name__ == "__main__":
    main()