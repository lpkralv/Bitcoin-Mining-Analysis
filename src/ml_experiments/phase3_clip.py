#!/usr/bin/env python3
"""
Phase 3.1: CLIP-style Contrastive Learning for SHA-256 Headers and Nonces

Treats headers as "text" and nonces as "images" in a CLIP-style framework.
Learns a shared embedding space where matching (header, nonce) pairs are close
and non-matching pairs are far apart.

Data format: uint8 (1024,) where bits[0:992]=header, bits[992:1024]=nonce
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


# ============================================================================
# Configuration and Global State
# ============================================================================

class Config:
    """Configuration for Phase 3.1 CLIP training"""

    # Data paths
    REAL_BITCOIN_DATA = "data/dataset_real_bitcoin.npy"
    REMINED_DATA = "data/dataset_reduced_r64.npy"

    # Logging and checkpointing
    LOG_FILE = "data/phase3_clip.log"
    RESULTS_FILE = "data/phase3_clip_results.jsonl"
    STATUS_FILE = "data/status.json"
    CHECKPOINT_DIR = "data/checkpoints/phase3_clip"

    # Search results
    SEARCH_RESULTS_FILE = "data/phase3_clip_search_results.json"
    FINAL_RESULTS_FILE = "data/phase3_clip_final_results.json"

    # AutoResearch grid
    SEARCH_CONFIGS = {
        'header_depth': [2, 4, 6],
        'header_width': [128, 256, 512],
        'nonce_depth': [1, 2, 3],
        'nonce_width': [64, 128, 256],
        'embedding_dim': [64, 128, 256],
        'learning_rate': [1e-4, 3e-4, 1e-3],
        'batch_size': [128, 256, 512],
    }

    # Training parameters
    SEARCH_EPOCHS = 30
    SEARCH_SUBSET_SIZE = 50000

    FULL_MIN_EPOCHS = 50
    FULL_MAX_EPOCHS = 200
    FULL_PATIENCE = 20

    TEMPERATURE_INIT = 0.07

    # Evaluation
    EVAL_BATCH_SIZE = 1024
    RECALL_THRESHOLDS = [0.01, 0.10]  # 1%, 10%

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Global state for graceful shutdown
SHUTDOWN_REQUESTED = False

def signal_handler(signum, frame):
    global SHUTDOWN_REQUESTED
    print(f"\nReceived signal {signum}. Setting shutdown flag...")
    SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def check_shutdown_signal(sandbox_dir: str) -> bool:
    """Check for shutdown.signal file"""
    return os.path.exists(os.path.join(sandbox_dir, "shutdown.signal"))


def update_status(sandbox_dir: str, status_data: Dict):
    """Update status.json with current progress"""
    status_path = os.path.join(sandbox_dir, Config.STATUS_FILE)
    try:
        with open(status_path, 'w') as f:
            json.dump(status_data, f, indent=2)
    except Exception as e:
        logging.warning(f"Failed to update status: {e}")


# ============================================================================
# Dataset and Data Loading
# ============================================================================

class SHA256Dataset(Dataset):
    """Dataset for SHA-256 headers and nonces"""

    def __init__(self, data: np.ndarray):
        """
        Args:
            data: numpy array of shape (N, 1024) with uint8 values
        """
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        header = sample[:992]  # First 992 bits
        nonce = sample[992:]   # Last 32 bits
        return header, nonce


def load_data(sandbox_dir: str, subset_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load both real Bitcoin and re-mined datasets"""

    real_path = os.path.join(sandbox_dir, Config.REAL_BITCOIN_DATA)
    remined_path = os.path.join(sandbox_dir, Config.REMINED_DATA)

    logging.info(f"Loading real Bitcoin data from {real_path}")
    real_data = np.load(real_path)

    logging.info(f"Loading re-mined data from {remined_path}")
    remined_data = np.load(remined_path)

    if subset_size:
        # Use subset for search
        real_data = real_data[:min(subset_size, len(real_data))]
        remined_data = remined_data[:min(subset_size, len(remined_data))]

    logging.info(f"Real Bitcoin data: {real_data.shape}")
    logging.info(f"Re-mined data: {remined_data.shape}")

    return real_data, remined_data


def create_dataloaders(data: np.ndarray, batch_size: int, train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders"""

    dataset = SHA256Dataset(data)

    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader


# ============================================================================
# Model Architecture
# ============================================================================

class MLPEncoder(nn.Module):
    """MLP encoder for either header or nonce"""

    def __init__(self, input_dim: int, hidden_dim: int, depth: int, output_dim: int):
        super().__init__()

        layers = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))

        # Hidden layers
        for _ in range(depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CLIPModel(nn.Module):
    """CLIP-style model for SHA-256 headers and nonces"""

    def __init__(self,
                 header_depth: int = 4,
                 header_width: int = 256,
                 nonce_depth: int = 2,
                 nonce_width: int = 128,
                 embedding_dim: int = 128,
                 temperature_init: float = Config.TEMPERATURE_INIT):
        super().__init__()

        # Encoders
        self.header_encoder = MLPEncoder(992, header_width, header_depth, embedding_dim)
        self.nonce_encoder = MLPEncoder(32, nonce_width, nonce_depth, embedding_dim)

        # Learnable temperature parameter
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature_init)))

    def encode_header(self, headers):
        """Encode headers to normalized embeddings"""
        embeddings = self.header_encoder(headers)
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_nonce(self, nonces):
        """Encode nonces to normalized embeddings"""
        embeddings = self.nonce_encoder(nonces)
        return F.normalize(embeddings, p=2, dim=-1)

    def forward(self, headers, nonces):
        """
        Forward pass for contrastive learning

        Returns:
            logits: (batch_size, batch_size) similarity matrix
            temperature: current temperature value
        """
        header_embeddings = self.encode_header(headers)
        nonce_embeddings = self.encode_nonce(nonces)

        # Compute similarity matrix
        temperature = torch.exp(self.log_temperature)
        logits = torch.matmul(header_embeddings, nonce_embeddings.T) / temperature

        return logits, temperature


def contrastive_loss(logits):
    """InfoNCE loss (same as CLIP)"""
    batch_size = logits.size(0)
    labels = torch.arange(batch_size, device=logits.device)

    # Symmetric loss: header->nonce and nonce->header
    loss_h2n = F.cross_entropy(logits, labels)
    loss_n2h = F.cross_entropy(logits.T, labels)

    return (loss_h2n + loss_n2h) / 2


# ============================================================================
# Evaluation Metrics
# ============================================================================

def evaluate_retrieval(model, val_loader, device):
    """
    Evaluate retrieval performance

    For each header, rank all nonces in batch by similarity.
    Compute MRR and Recall@K metrics.
    """
    model.eval()

    all_ranks = []
    total_samples = 0

    with torch.no_grad():
        for headers, nonces in val_loader:
            headers = headers.to(device)
            nonces = nonces.to(device)

            batch_size = headers.size(0)

            # Get embeddings
            header_embeddings = model.encode_header(headers)
            nonce_embeddings = model.encode_nonce(nonces)

            # Compute similarity matrix
            similarity = torch.matmul(header_embeddings, nonce_embeddings.T)

            # For each header, find rank of correct nonce
            for i in range(batch_size):
                similarities = similarity[i]  # Similarities to all nonces
                sorted_indices = torch.argsort(similarities, descending=True)
                rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1  # 1-indexed
                all_ranks.append(rank)
                total_samples += 1

    # Compute metrics
    ranks = np.array(all_ranks)

    mrr = np.mean(1.0 / ranks)
    mean_rank = np.mean(ranks)

    recall_metrics = {}
    for threshold in Config.RECALL_THRESHOLDS:
        recall_k = np.mean(ranks <= threshold * len(ranks))
        recall_metrics[f'recall@{int(threshold*100)}%'] = recall_k

    return {
        'mrr': mrr,
        'mean_rank': mean_rank,
        **recall_metrics,
        'total_samples': total_samples
    }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, train_loader, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_batches = 0

    for headers, nonces in train_loader:
        headers = headers.to(device, non_blocking=True)
        nonces = nonces.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            logits, temperature = model(headers, nonces)
            loss = contrastive_loss(logits)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_batches += 1

        # Check shutdown
        if SHUTDOWN_REQUESTED:
            break

    return total_loss / total_batches if total_batches > 0 else float('inf')


def validate_epoch(model, val_loader, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    total_batches = 0

    with torch.no_grad():
        for headers, nonces in val_loader:
            headers = headers.to(device, non_blocking=True)
            nonces = nonces.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda"):
                logits, _ = model(headers, nonces)
                loss = contrastive_loss(logits)

            total_loss += loss.item()
            total_batches += 1

    return total_loss / total_batches if total_batches > 0 else float('inf')


def save_checkpoint(model, optimizer, scaler, epoch, loss, checkpoint_dir, suffix=""):
    """Save training checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'timestamp': time.time()
    }

    filename = f"checkpoint_epoch_{epoch}{suffix}.pt"
    path = os.path.join(checkpoint_dir, filename)
    torch.save(checkpoint, path)

    return path


def load_checkpoint(path, model, optimizer, scaler):
    """Load training checkpoint"""
    checkpoint = torch.load(path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])

    return checkpoint['epoch'], checkpoint['loss']


# ============================================================================
# AutoResearch Grid Search
# ============================================================================

def generate_search_configs():
    """Generate all combinations for grid search"""
    import itertools

    keys = list(Config.SEARCH_CONFIGS.keys())
    values = [Config.SEARCH_CONFIGS[key] for key in keys]

    configs = []
    for combination in itertools.product(*values):
        config = dict(zip(keys, combination))
        configs.append(config)

    return configs


def run_search_config(config, data, sandbox_dir, config_idx, total_configs):
    """Run a single search configuration"""

    logging.info(f"Config {config_idx+1}/{total_configs}: {config}")

    # Setup
    device = torch.device(Config.DEVICE)
    torch.manual_seed(42 + config_idx)
    np.random.seed(42 + config_idx)

    # Data
    train_loader, val_loader = create_dataloaders(data, config['batch_size'])

    # Model
    model = CLIPModel(
        header_depth=config['header_depth'],
        header_width=config['header_width'],
        nonce_depth=config['nonce_depth'],
        nonce_width=config['nonce_width'],
        embedding_dim=config['embedding_dim']
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    best_val_loss = float('inf')
    best_mrr = 0

    for epoch in range(Config.SEARCH_EPOCHS):
        if SHUTDOWN_REQUESTED or check_shutdown_signal(sandbox_dir):
            logging.info("Shutdown requested during search")
            break

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = validate_epoch(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Evaluate retrieval on small subset
            eval_metrics = evaluate_retrieval(model, val_loader, device)
            best_mrr = eval_metrics['mrr']

        # Log progress
        if epoch % 10 == 0:
            logging.info(f"Config {config_idx+1}, Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Checkpoint every few configs
        if config_idx % 5 == 0 and epoch % 10 == 0:
            checkpoint_dir = os.path.join(sandbox_dir, Config.CHECKPOINT_DIR, f"search_config_{config_idx}")
            save_checkpoint(model, optimizer, scaler, epoch, val_loss, checkpoint_dir)

    result = {
        'config': config,
        'config_idx': config_idx,
        'best_val_loss': best_val_loss,
        'best_mrr': best_mrr,
        'completed_epochs': epoch + 1
    }

    return result


def run_autoresearch_search(sandbox_dir: str, data: np.ndarray):
    """Run AutoResearch grid search"""

    logging.info("Starting AutoResearch grid search")

    configs = generate_search_configs()
    logging.info(f"Generated {len(configs)} search configurations")

    results = []

    for i, config in enumerate(configs):
        if SHUTDOWN_REQUESTED or check_shutdown_signal(sandbox_dir):
            logging.info("Shutdown requested, stopping search")
            break

        try:
            result = run_search_config(config, data, sandbox_dir, i, len(configs))
            results.append(result)

            # Log result
            logging.info(f"Config {i+1} result: val_loss={result['best_val_loss']:.6f}, mrr={result['best_mrr']:.6f}")

            # Update status
            update_status(sandbox_dir, {
                'phase': 'search',
                'config': i + 1,
                'total_configs': len(configs),
                'best_mrr_so_far': max([r['best_mrr'] for r in results]),
                'timestamp': time.time()
            })

        except Exception as e:
            logging.error(f"Config {i+1} failed: {e}")
            results.append({
                'config': config,
                'config_idx': i,
                'error': str(e),
                'best_val_loss': float('inf'),
                'best_mrr': 0
            })

    # Find best configuration
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        best_result = max(valid_results, key=lambda x: x['best_mrr'])
        logging.info(f"Best configuration: {best_result['config']}")
        logging.info(f"Best MRR: {best_result['best_mrr']:.6f}")
    else:
        best_result = None
        logging.error("No valid configurations found")

    # Save results
    search_results = {
        'total_configs': len(configs),
        'completed_configs': len([r for r in results if 'error' not in r]),
        'best_config': best_result['config'] if best_result else None,
        'best_mrr': best_result['best_mrr'] if best_result else 0,
        'all_results': results,
        'timestamp': time.time()
    }

    results_path = os.path.join(sandbox_dir, Config.SEARCH_RESULTS_FILE)
    with open(results_path, 'w') as f:
        json.dump(search_results, f, indent=2, default=float)

    logging.info(f"Search results saved to {results_path}")

    return best_result


# ============================================================================
# Full Training
# ============================================================================

def run_full_training(sandbox_dir: str, real_data: np.ndarray, remined_data: np.ndarray, config: Dict):
    """Run full training on both datasets with best configuration"""

    logging.info("Starting full training with best configuration")
    logging.info(f"Configuration: {config}")

    results = {}

    for dataset_name, data in [('real_bitcoin', real_data), ('remined', remined_data)]:
        logging.info(f"\nTraining on {dataset_name} dataset ({len(data)} samples)")

        result = train_full_dataset(sandbox_dir, data, config, dataset_name)
        results[dataset_name] = result

        if SHUTDOWN_REQUESTED or check_shutdown_signal(sandbox_dir):
            logging.info("Shutdown requested during full training")
            break

    # Save final results
    final_results = {
        'config': config,
        'results': results,
        'timestamp': time.time()
    }

    results_path = os.path.join(sandbox_dir, Config.FINAL_RESULTS_FILE)
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=float)

    logging.info(f"Final results saved to {results_path}")

    return final_results


def train_full_dataset(sandbox_dir: str, data: np.ndarray, config: Dict, dataset_name: str):
    """Train on a single dataset to convergence"""

    device = torch.device(Config.DEVICE)
    torch.manual_seed(42)
    np.random.seed(42)

    # Data
    train_loader, val_loader = create_dataloaders(data, config['batch_size'])

    # Model
    model = CLIPModel(
        header_depth=config['header_depth'],
        header_width=config['header_width'],
        nonce_depth=config['nonce_depth'],
        nonce_width=config['nonce_width'],
        embedding_dim=config['embedding_dim']
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scaler = torch.amp.GradScaler("cuda")

    # Training state
    best_val_loss = float('inf')
    best_mrr = 0
    patience_counter = 0
    epoch = 0

    checkpoint_dir = os.path.join(sandbox_dir, Config.CHECKPOINT_DIR, f"full_{dataset_name}")

    # Training loop
    while epoch < Config.FULL_MAX_EPOCHS:
        if SHUTDOWN_REQUESTED or check_shutdown_signal(sandbox_dir):
            logging.info("Shutdown requested during training")
            break

        epoch += 1

        train_loss = train_epoch(model, train_loader, optimizer, scaler, device)
        val_loss = validate_epoch(model, val_loader, device)

        # Evaluate retrieval
        eval_metrics = evaluate_retrieval(model, val_loader, device)
        current_mrr = eval_metrics['mrr']

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_mrr = current_mrr
            patience_counter = 0

            # Save best model
            save_checkpoint(model, optimizer, scaler, epoch, val_loss, checkpoint_dir, "_best")
        else:
            patience_counter += 1

        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, scaler, epoch, val_loss, checkpoint_dir)
            logging.info(f"{dataset_name} Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, mrr={current_mrr:.6f}")

        # Update status
        update_status(sandbox_dir, {
            'phase': 'full_training',
            'dataset': dataset_name,
            'epoch': epoch,
            'max_epochs': Config.FULL_MAX_EPOCHS,
            'best_mrr': best_mrr,
            'patience': patience_counter,
            'timestamp': time.time()
        })

        # Early stopping
        if epoch >= Config.FULL_MIN_EPOCHS and patience_counter >= Config.FULL_PATIENCE:
            logging.info(f"Early stopping at epoch {epoch} (patience {Config.FULL_PATIENCE})")
            break

    # Final evaluation
    logging.info(f"Final evaluation for {dataset_name}")
    final_metrics = evaluate_retrieval(model, val_loader, device)

    result = {
        'dataset_name': dataset_name,
        'dataset_size': len(data),
        'total_epochs': epoch,
        'best_val_loss': best_val_loss,
        'final_metrics': final_metrics,
        'config': config
    }

    logging.info(f"{dataset_name} results: MRR={final_metrics['mrr']:.6f}, Mean Rank={final_metrics['mean_rank']:.1f}")
    for threshold in Config.RECALL_THRESHOLDS:
        key = f'recall@{int(threshold*100)}%'
        logging.info(f"{dataset_name} {key}: {final_metrics[key]:.4f}")

    return result


# ============================================================================
# Main Function
# ============================================================================

def setup_logging(sandbox_dir: str):
    """Setup logging configuration"""
    log_path = os.path.join(sandbox_dir, Config.LOG_FILE)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Phase 3.1: CLIP-style Contrastive Learning")
    parser.add_argument("--sandbox", required=True, help="Sandbox directory path")
    parser.add_argument("--search-only", action="store_true", help="Run only AutoResearch search")
    parser.add_argument("--full-only", action="store_true", help="Run only full training (requires search results)")

    args = parser.parse_args()

    sandbox_dir = os.path.abspath(args.sandbox)
    setup_logging(sandbox_dir)

    logging.info("Starting Phase 3.1: CLIP-style Contrastive Learning")
    logging.info(f"Sandbox directory: {sandbox_dir}")
    logging.info(f"Device: {Config.DEVICE}")
    logging.info(f"PyTorch version: {torch.__version__}")

    # Load data
    if args.full_only:
        # Load full datasets
        real_data, remined_data = load_data(sandbox_dir)
    else:
        # Load subset for search
        real_data, remined_data = load_data(sandbox_dir, Config.SEARCH_SUBSET_SIZE)

    # Use re-mined data for search (smaller, more controlled)
    search_data = remined_data

    try:
        if not args.full_only:
            # Run AutoResearch search
            update_status(sandbox_dir, {'phase': 'search', 'status': 'starting', 'timestamp': time.time()})
            best_config = run_autoresearch_search(sandbox_dir, search_data)

            if not best_config:
                logging.error("Search failed to find a valid configuration")
                return 1

        if not args.search_only:
            # Load best config if doing full training only
            if args.full_only:
                search_results_path = os.path.join(sandbox_dir, Config.SEARCH_RESULTS_FILE)
                if not os.path.exists(search_results_path):
                    logging.error("Search results not found. Run search first or use --search-only.")
                    return 1

                with open(search_results_path, 'r') as f:
                    search_results = json.load(f)

                best_config = {'config': search_results['best_config']}

                # Reload full data if not loaded
                if 'real_data' not in locals():
                    real_data, remined_data = load_data(sandbox_dir)

            # Run full training
            update_status(sandbox_dir, {'phase': 'full_training', 'status': 'starting', 'timestamp': time.time()})
            final_results = run_full_training(sandbox_dir, real_data, remined_data, best_config['config'])

        update_status(sandbox_dir, {'phase': 'complete', 'status': 'success', 'timestamp': time.time()})
        logging.info("Phase 3.1 completed successfully")
        return 0

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        update_status(sandbox_dir, {'phase': 'interrupted', 'timestamp': time.time()})
        return 1
    except Exception as e:
        logging.error(f"Training failed: {e}")
        update_status(sandbox_dir, {'phase': 'error', 'error': str(e), 'timestamp': time.time()})
        return 1


if __name__ == "__main__":
    sys.exit(main())