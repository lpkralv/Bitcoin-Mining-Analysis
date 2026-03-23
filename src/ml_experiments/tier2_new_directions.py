#!/usr/bin/env python3
"""
Tier 2 New Directions: Three Novel SHA-256 Mining Experiments
============================================================

Experiment 1: Learned Hash Approximation (MOST IMPORTANT)
- Train a neural network to approximate SHA-256d
- Use gradient descent through the approximation to find nonces

Experiment 2: Word-Level Attention Transformer
- SHA-256 operates on 32-bit words, use transformer attending over words

Experiment 3: Timestamp Micro-Optimization
- Test whether timestamp choice affects mining efficiency within ±2h window
"""

import argparse
import json
import logging
import hashlib
import numpy as np
import os
import random
import signal
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Global shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    shutdown_requested = True
    print(f"\nShutdown signal {signum} received. Will stop after current operation...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"tier2_new_directions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def save_status(status_file: Path, status: Dict):
    """Save current status to JSON file"""
    try:
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        logging.error(f"Failed to save status: {e}")

def check_shutdown_signal(shutdown_file: Path) -> bool:
    """Check if shutdown signal file exists"""
    return shutdown_file.exists() or shutdown_requested

def sha256d(data: bytes) -> bytes:
    """Double SHA-256 (Bitcoin's hash function)"""
    return hashlib.sha256(hashlib.sha256(data).digest()).digest()

def count_leading_zeros(hash_bytes: bytes) -> int:
    """Count leading zero bits in hash"""
    count = 0
    for byte in hash_bytes:
        if byte == 0:
            count += 8
        else:
            count += (7 - byte.bit_length() + 1)
            break
    return count

def create_bitcoin_header_stub(block_height: int = 800000) -> bytes:
    """Create a realistic Bitcoin header stub (76 bytes before nonce)"""
    # Version (4 bytes)
    version = struct.pack('<I', 0x20000000)

    # Previous block hash (32 bytes) - use a reasonable hash
    prev_hash = hashlib.sha256(f"block_{block_height-1}".encode()).digest()

    # Merkle root (32 bytes) - use a reasonable hash
    merkle_root = hashlib.sha256(f"transactions_{block_height}".encode()).digest()

    # Timestamp (4 bytes) - recent timestamp
    timestamp = struct.pack('<I', int(time.time()))

    # Bits (4 bytes) - difficulty target
    bits = struct.pack('<I', 0x1703a30c)  # Realistic difficulty

    return version + prev_hash + merkle_root + timestamp + bits

def generate_training_data(num_samples: int, logger) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data for neural SHA-256 approximation"""
    logger.info(f"Generating {num_samples} training samples...")

    inputs = []
    outputs = []

    for i in range(num_samples):
        if i % 10000 == 0:
            logger.info(f"Generated {i}/{num_samples} samples")
            if check_shutdown_signal(Path("shutdown.signal")):
                logger.info("Shutdown requested during data generation")
                break

        # Create header stub
        header_stub = create_bitcoin_header_stub(800000 + i % 1000)

        # Add random nonce
        nonce = struct.pack('<I', random.randint(0, 2**32-1))

        # Create full header (80 bytes)
        full_header = header_stub + nonce

        # Pad to 128 bytes (1024 bits) as required for SHA-256 block processing
        padded = full_header + b'\x80' + b'\x00' * 39 + struct.pack('>Q', 80 * 8)

        # Convert to bits
        input_bits = np.unpackbits(np.frombuffer(padded, dtype=np.uint8)).astype(np.float32)

        # Compute SHA-256d hash
        hash_result = sha256d(full_header)
        output_bits = np.unpackbits(np.frombuffer(hash_result, dtype=np.uint8)).astype(np.float32)

        inputs.append(input_bits)
        outputs.append(output_bits)

    return np.array(inputs), np.array(outputs)

class NeuralSHA256(nn.Module):
    """Neural network to approximate SHA-256d"""

    def __init__(self, input_size=1024, hidden_size=4096, num_layers=8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Deep MLP with residual connections
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        # Output projection
        self.output_proj = nn.Linear(hidden_size, 256)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        x = self.dropout(x)

        # Deep MLP with residual connections every 2 layers
        for i, layer in enumerate(self.layers):
            residual = x
            x = F.relu(layer(x))
            x = self.dropout(x)

            # Add residual connection every 2 layers
            if i % 2 == 1 and i > 0:
                x = x + residual

        # Output logits (no activation - will use BCEWithLogitsLoss)
        x = self.output_proj(x)
        return x

class WordLevelTransformer(nn.Module):
    """Transformer operating on 32-bit words for SHA-256 prediction"""

    def __init__(self, num_words=31, word_dim=32, embed_dim=128, num_heads=8, num_layers=4):
        super().__init__()
        self.num_words = num_words
        self.word_dim = word_dim
        self.embed_dim = embed_dim

        # Word embedding (linear projection of 32 bits to embed_dim)
        self.word_embedding = nn.Linear(word_dim, embed_dim)

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_words + 1, embed_dim))

        # Add CLS token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection (predict 32 nonce bits)
        self.output_proj = nn.Linear(embed_dim, 32)

    def forward(self, x):
        # x shape: (batch_size, num_words, word_dim)
        batch_size = x.shape[0]

        # Embed words
        word_embeds = self.word_embedding(x)  # (batch_size, num_words, embed_dim)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, word_embeds], dim=1)  # (batch_size, num_words+1, embed_dim)

        # Add positional embeddings
        x = x + self.pos_embedding

        # Apply transformer
        x = self.transformer(x)

        # Use CLS token for prediction
        cls_output = x[:, 0, :]  # (batch_size, embed_dim)

        # Project to nonce bits
        nonce_logits = self.output_proj(cls_output)  # (batch_size, 32)

        return nonce_logits

def prepare_word_level_data(data_file: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare data for word-level transformer"""
    if not data_file.exists():
        raise FileNotFoundError(f"Data file {data_file} not found")

    data = np.load(data_file)
    headers = data[:, :992]  # First 992 bits (31 words of 32 bits)
    nonces = data[:, 992:1024]  # Last 32 bits (nonce)

    # Reshape headers to words
    headers_words = headers.reshape(-1, 31, 32)

    return torch.FloatTensor(headers_words), torch.FloatTensor(nonces)

def experiment1_learned_hash_approximation(sandbox_dir: Path, logger):
    """Experiment 1: Train neural network to approximate SHA-256d"""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 1: Learned Hash Approximation")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Generate training data
    logger.info("Generating training data...")
    train_inputs, train_outputs = generate_training_data(500000, logger)
    if check_shutdown_signal(Path("shutdown.signal")):
        return {"status": "interrupted", "experiment": "exp1"}

    val_inputs, val_outputs = generate_training_data(50000, logger)
    if check_shutdown_signal(Path("shutdown.signal")):
        return {"status": "interrupted", "experiment": "exp1"}

    # Convert to tensors
    train_dataset = TensorDataset(
        torch.FloatTensor(train_inputs),
        torch.FloatTensor(train_outputs)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_inputs),
        torch.FloatTensor(val_outputs)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=False)

    # Create model
    model = NeuralSHA256().to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 10

    logger.info("Starting training...")
    for epoch in range(100):
        if check_shutdown_signal(Path("shutdown.signal")):
            logger.info("Shutdown requested during training")
            break

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate per-bit accuracy
            predictions = torch.sigmoid(outputs) > 0.5
            train_correct += (predictions == targets).float().sum().item()
            train_total += targets.numel()

            if batch_idx % 1000 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item()
                predictions = torch.sigmoid(outputs) > 0.5
                val_correct += (predictions == targets).float().sum().item()
                val_total += targets.numel()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, "
                   f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), sandbox_dir / "neural_sha256_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping after {epoch} epochs")
                break

    # Load best model for evaluation
    model.load_state_dict(torch.load(sandbox_dir / "neural_sha256_best.pth"))
    model.eval()

    # Evaluation: gradient descent through approximation
    logger.info("Evaluating gradient descent through approximation...")

    test_results = []
    for test_idx in range(100):  # Test on 100 headers
        if check_shutdown_signal(Path("shutdown.signal")):
            break

        # Create test header
        header_stub = create_bitcoin_header_stub(900000 + test_idx)

        # Initialize random nonce and make it require gradient
        nonce_tensor = torch.randn(1, 32, device=device, requires_grad=True)

        # Optimizer for nonce
        nonce_optimizer = optim.Adam([nonce_tensor], lr=0.01)

        best_predicted_zeros = 0
        best_nonce_bits = None

        # Gradient descent to find nonce with many leading zeros
        for step in range(100):
            nonce_optimizer.zero_grad()

            # Convert nonce to binary
            nonce_binary = torch.sigmoid(nonce_tensor)

            # Create full input (header + nonce + padding)
            header_bits = torch.from_numpy(
                np.unpackbits(np.frombuffer(header_stub, dtype=np.uint8)).astype(np.float32)
            ).to(device)

            # Combine header and nonce
            full_input = torch.cat([header_bits, nonce_binary.flatten()])

            # Pad to 1024 bits (simplified padding for gradient flow)
            padding_bits = torch.zeros(1024 - len(full_input), device=device)
            padded_input = torch.cat([full_input, padding_bits]).unsqueeze(0)

            # Predict hash
            with torch.amp.autocast(device_type="cuda"):
                predicted_hash = model(padded_input)

            # Maximize leading zeros (minimize sum of first N bits)
            leading_bits = predicted_hash[0, :20]  # Focus on first 20 bits
            loss = leading_bits.mean()  # Minimize to get zeros

            loss.backward()
            nonce_optimizer.step()

            # Check predicted leading zeros
            with torch.no_grad():
                hash_sigmoid = torch.sigmoid(predicted_hash[0])
                predicted_zeros = 0
                for bit in hash_sigmoid:
                    if bit < 0.5:
                        predicted_zeros += 1
                    else:
                        break

                if predicted_zeros > best_predicted_zeros:
                    best_predicted_zeros = predicted_zeros
                    best_nonce_bits = (nonce_binary > 0.5).cpu().numpy().flatten()

        # Test the best found nonce with real SHA-256d
        if best_nonce_bits is not None:
            # Convert binary nonce to integer
            nonce_int = 0
            for i, bit in enumerate(best_nonce_bits):
                if bit:
                    nonce_int |= (1 << (31 - i))

            # Create real header and hash it
            real_nonce = struct.pack('<I', nonce_int)
            real_header = header_stub + real_nonce
            real_hash = sha256d(real_header)
            real_zeros = count_leading_zeros(real_hash)

            test_results.append({
                "predicted_zeros": best_predicted_zeros,
                "real_zeros": real_zeros,
                "nonce": nonce_int
            })

        if test_idx % 10 == 0:
            logger.info(f"Tested {test_idx}/100 headers")

    # Compare to random baseline
    logger.info("Testing random baseline...")
    random_results = []
    for test_idx in range(100):
        if check_shutdown_signal(Path("shutdown.signal")):
            break

        header_stub = create_bitcoin_header_stub(950000 + test_idx)
        best_zeros = 0

        for _ in range(100):  # Try 100 random nonces
            nonce = struct.pack('<I', random.randint(0, 2**32-1))
            hash_result = sha256d(header_stub + nonce)
            zeros = count_leading_zeros(hash_result)
            best_zeros = max(best_zeros, zeros)

        random_results.append(best_zeros)

    # Calculate statistics
    if test_results and random_results:
        gradient_zeros = [r["real_zeros"] for r in test_results]
        gradient_mean = np.mean(gradient_zeros)
        random_mean = np.mean(random_results)

        results = {
            "experiment": "learned_hash_approximation",
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "final_val_accuracy": val_acc,
            "gradient_descent_results": {
                "mean_leading_zeros": gradient_mean,
                "max_leading_zeros": max(gradient_zeros),
                "samples": len(test_results)
            },
            "random_baseline": {
                "mean_leading_zeros": random_mean,
                "max_leading_zeros": max(random_results),
                "samples": len(random_results)
            },
            "improvement": gradient_mean - random_mean
        }

        logger.info(f"Gradient descent mean zeros: {gradient_mean:.2f}")
        logger.info(f"Random baseline mean zeros: {random_mean:.2f}")
        logger.info(f"Improvement: {gradient_mean - random_mean:.2f}")

        return results

    return {"status": "completed_with_errors", "experiment": "exp1"}

def experiment2_word_level_transformer(sandbox_dir: Path, logger):
    """Experiment 2: Word-level attention transformer"""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 2: Word-Level Attention Transformer")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Try to load reduced round data
    data_files = [
        sandbox_dir / "data" / "dataset_reduced_r64.npy",
        sandbox_dir / "data" / "dataset_reduced_r04.npy"
    ]

    results = {}

    for data_file in data_files:
        if not data_file.exists():
            logger.warning(f"Data file {data_file} not found, skipping")
            continue

        if check_shutdown_signal(Path("shutdown.signal")):
            break

        round_name = "R64" if "r64" in data_file.name else "R04"
        logger.info(f"Training on {round_name} data...")

        try:
            # Load and prepare data
            X, y = prepare_word_level_data(data_file)

            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # Create data loaders
            train_dataset = TensorDataset(X_train.to(device), y_train.to(device))
            val_dataset = TensorDataset(X_val.to(device), y_val.to(device))

            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

            # Create model
            model = WordLevelTransformer().to(device)

            # Training setup
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
            criterion = nn.BCEWithLogitsLoss()

            # Training loop
            best_val_acc = 0.0
            patience_counter = 0

            for epoch in range(200):
                if check_shutdown_signal(Path("shutdown.signal")):
                    break

                # Training
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for inputs, targets in train_loader:
                    with torch.amp.autocast(device_type="cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    predictions = torch.sigmoid(outputs) > 0.5
                    train_correct += (predictions == targets).float().sum().item()
                    train_total += targets.numel()

                # Validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        with torch.amp.autocast(device_type="cuda"):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)

                        val_loss += loss.item()
                        predictions = torch.sigmoid(outputs) > 0.5
                        val_correct += (predictions == targets).float().sum().item()
                        val_total += targets.numel()

                train_acc = train_correct / train_total
                val_acc = val_correct / val_total

                scheduler.step(val_loss)

                if epoch % 10 == 0:
                    logger.info(f"{round_name} Epoch {epoch}: Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")

                # Early stopping
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 30:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

            results[round_name] = {
                "best_validation_accuracy": best_val_acc,
                "epochs_trained": epoch + 1,
                "model_parameters": sum(p.numel() for p in model.parameters())
            }

            logger.info(f"{round_name} best validation accuracy: {best_val_acc:.4f}")

        except Exception as e:
            logger.error(f"Error processing {data_file}: {e}")
            results[round_name] = {"error": str(e)}

    return {
        "experiment": "word_level_transformer",
        "results": results
    }

def experiment3_timestamp_optimization(sandbox_dir: Path, logger):
    """Experiment 3: Test timestamp micro-optimization"""
    logger.info("=" * 60)
    logger.info("EXPERIMENT 3: Timestamp Micro-Optimization")
    logger.info("=" * 60)

    # Parameters
    num_headers = 10
    num_timestamps = 100  # Reduced from 1000 for performance
    num_nonces_per_test = 100000
    difficulty = 8  # Require 8 leading zero bits

    target_mask = (1 << (256 - difficulty)) - 1  # For difficulty checking

    results = []

    for header_idx in range(num_headers):
        if check_shutdown_signal(Path("shutdown.signal")):
            break

        logger.info(f"Processing header {header_idx + 1}/{num_headers}")

        # Create base header stub
        base_stub = create_bitcoin_header_stub(1000000 + header_idx)

        # Extract timestamp position (bytes 68-72 in header)
        base_timestamp = struct.unpack('<I', base_stub[68:72])[0]

        counts_for_header = []
        timestamps_tested = []

        # Test different timestamps within ±2 hour window
        for ts_offset in range(-7200, 7200, int(14400 / num_timestamps)):
            if check_shutdown_signal(Path("shutdown.signal")):
                break

            new_timestamp = base_timestamp + ts_offset

            # Create modified header stub with new timestamp
            modified_stub = (base_stub[:68] +
                           struct.pack('<I', new_timestamp) +
                           base_stub[72:])

            # Count valid nonces
            valid_count = 0
            for _ in range(num_nonces_per_test):
                nonce = struct.pack('<I', random.randint(0, 2**32-1))
                full_header = modified_stub + nonce
                hash_result = sha256d(full_header)

                # Check if hash meets difficulty requirement
                hash_int = int.from_bytes(hash_result, 'big')
                if hash_int <= target_mask:
                    valid_count += 1

            counts_for_header.append(valid_count)
            timestamps_tested.append(new_timestamp)

        if counts_for_header:
            # Calculate statistics for this header
            counts_array = np.array(counts_for_header)
            mean_count = np.mean(counts_array)
            var_count = np.var(counts_array)

            # Expected Poisson variance should equal the mean
            expected_var = mean_count

            header_result = {
                "header_index": header_idx,
                "num_timestamps_tested": len(counts_for_header),
                "mean_valid_count": mean_count,
                "observed_variance": var_count,
                "expected_poisson_variance": expected_var,
                "variance_ratio": var_count / expected_var if expected_var > 0 else 0,
                "min_count": int(np.min(counts_array)),
                "max_count": int(np.max(counts_array)),
                "range": int(np.max(counts_array) - np.min(counts_array))
            }

            results.append(header_result)

            logger.info(f"Header {header_idx}: Mean={mean_count:.2f}, "
                       f"Var={var_count:.2f}, Expected Var={expected_var:.2f}, "
                       f"Ratio={var_count/expected_var:.3f}")

    # Overall analysis
    if results:
        variance_ratios = [r["variance_ratio"] for r in results]
        mean_variance_ratio = np.mean(variance_ratios)

        overall_results = {
            "experiment": "timestamp_micro_optimization",
            "parameters": {
                "num_headers": num_headers,
                "num_timestamps_per_header": num_timestamps,
                "num_nonces_per_test": num_nonces_per_test,
                "difficulty": difficulty
            },
            "individual_headers": results,
            "summary": {
                "mean_variance_ratio": mean_variance_ratio,
                "variance_ratio_std": np.std(variance_ratios),
                "min_variance_ratio": min(variance_ratios),
                "max_variance_ratio": max(variance_ratios)
            }
        }

        logger.info(f"Overall mean variance ratio: {mean_variance_ratio:.3f}")
        logger.info("(Ratio = 1.0 indicates perfect Poisson behavior)")

        return overall_results

    return {"status": "no_results", "experiment": "exp3"}

def main():
    parser = argparse.ArgumentParser(description="Tier 2 New Directions: Novel SHA-256 Mining Experiments")
    parser.add_argument("--sandbox", required=True, help="Sandbox directory path")
    args = parser.parse_args()

    sandbox_dir = Path(args.sandbox)
    if not sandbox_dir.exists():
        print(f"Error: Sandbox directory {sandbox_dir} does not exist")
        return 1

    # Setup
    logger = setup_logging(sandbox_dir)
    logger.info("Starting Tier 2 New Directions experiments")
    logger.info(f"Sandbox directory: {sandbox_dir}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    status_file = sandbox_dir / "status.json"
    shutdown_file = sandbox_dir / "shutdown.signal"
    results_file = sandbox_dir / "data" / "tier2_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {},
        "system_info": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
        }
    }

    # Remove shutdown signal if it exists
    if shutdown_file.exists():
        shutdown_file.unlink()

    experiments = [
        ("experiment1", experiment1_learned_hash_approximation),
        ("experiment2", experiment2_word_level_transformer),
        ("experiment3", experiment3_timestamp_optimization)
    ]

    for exp_name, exp_func in experiments:
        if check_shutdown_signal(shutdown_file):
            logger.info("Shutdown requested, stopping experiments")
            break

        save_status(status_file, {
            "current_experiment": exp_name,
            "timestamp": datetime.now().isoformat(),
            "status": "running"
        })

        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting {exp_name}")
            logger.info(f"{'='*60}")

            start_time = time.time()
            result = exp_func(sandbox_dir, logger)
            end_time = time.time()

            result["duration_seconds"] = end_time - start_time
            all_results["experiments"][exp_name] = result

            logger.info(f"Completed {exp_name} in {end_time - start_time:.2f} seconds")

            # Save intermediate results
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)

        except Exception as e:
            logger.error(f"Error in {exp_name}: {e}", exc_info=True)
            all_results["experiments"][exp_name] = {
                "status": "error",
                "error": str(e)
            }

    # Final status
    save_status(status_file, {
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
        "experiments_completed": len(all_results["experiments"])
    })

    # Save final results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info("\nAll experiments completed!")
    logger.info(f"Results saved to: {results_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())