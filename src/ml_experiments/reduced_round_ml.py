#!/usr/bin/env python3
"""
Phase 2C: Reduced-Round SHA-256 ML Investigation

Test the MLP from Phase 2A on reduced-round SHA-256 to find the exact threshold
where learnable structure vanishes. Uses RE-MINED nonces (random start) to
eliminate miner behavioral patterns.

Usage:
    python reduced_round_ml.py --sandbox /path/to/sandbox [--generate-only] [--train-only]
"""

import argparse
import json
import math
import os
import signal
import hashlib
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.amp as amp


# Global shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\nShutdown signal {signum} received. Finishing current work...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# Complete Inline SHA-256 Implementation with Configurable Round Counts
# ============================================================================

# SHA-256 constants
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

# Initial hash values
H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
]


def rotr(x: int, n: int) -> int:
    """Right rotate a 32-bit integer."""
    return ((x >> n) | (x << (32 - n))) & 0xffffffff


def sha256_compress(block_words: List[int], initial_state: List[int], num_rounds: int = 64) -> List[int]:
    """
    SHA-256 compression function with configurable round count.

    Args:
        block_words: 16 32-bit words from the message block
        initial_state: 8 32-bit words of the current hash state
        num_rounds: Number of rounds to execute (1-64)

    Returns:
        List of 8 32-bit words representing the new hash state
    """
    # Extend the sixteen 32-bit words into sixty-four 32-bit words
    w = block_words[:]
    for i in range(16, min(64, num_rounds + 16)):  # Only generate what we need
        s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
        w.append((w[i-16] + s0 + w[i-7] + s1) & 0xffffffff)

    # Initialize working variables
    a, b, c, d, e, f, g, h = initial_state

    # Main loop (reduced rounds)
    for i in range(min(num_rounds, 64)):
        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        temp1 = (h + S1 + ch + K[i] + w[i]) & 0xffffffff
        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (S0 + maj) & 0xffffffff

        h = g
        g = f
        f = e
        e = (d + temp1) & 0xffffffff
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xffffffff

    # Add the compressed chunk to the current hash value
    return [
        (initial_state[0] + a) & 0xffffffff,
        (initial_state[1] + b) & 0xffffffff,
        (initial_state[2] + c) & 0xffffffff,
        (initial_state[3] + d) & 0xffffffff,
        (initial_state[4] + e) & 0xffffffff,
        (initial_state[5] + f) & 0xffffffff,
        (initial_state[6] + g) & 0xffffffff,
        (initial_state[7] + h) & 0xffffffff,
    ]


def sha256_reduced(data: bytes, num_rounds: int = 64) -> bytes:
    """
    Reduced-round SHA-256 hash function.

    Args:
        data: Input bytes to hash
        num_rounds: Number of rounds per 512-bit block (1-64)

    Returns:
        32-byte hash digest
    """
    # Pre-processing: adding a single 1 bit
    msg = bytearray(data)
    msg.append(0x80)

    # Pre-processing: padding with zeros
    while len(msg) % 64 != 56:
        msg.append(0)

    # Append original length in bits mod 2^64 as 64-bit big-endian integer
    msg.extend(struct.pack('>Q', len(data) * 8))

    # Process the message in successive 512-bit chunks
    hash_state = H0[:]
    for chunk_start in range(0, len(msg), 64):
        chunk = msg[chunk_start:chunk_start + 64]
        block_words = list(struct.unpack('>16I', chunk))
        hash_state = sha256_compress(block_words, hash_state, num_rounds)

    # Produce the final hash value as a 256-bit number
    return struct.pack('>8I', *hash_state)


def sha256d_reduced(header_80bytes: bytes, inner_rounds: int = 64, outer_rounds: int = 64) -> bytes:
    """
    Reduced-round double SHA-256 (SHA-256d) like Bitcoin uses.

    Args:
        header_80bytes: 80-byte Bitcoin header
        inner_rounds: Number of rounds for first SHA-256
        outer_rounds: Number of rounds for second SHA-256

    Returns:
        32-byte double hash
    """
    first_hash = sha256_reduced(header_80bytes, inner_rounds)
    return sha256_reduced(first_hash, outer_rounds)


def sha256_pad_80bytes(header: bytes) -> bytes:
    """SHA-256 pad 80-byte input to 128 bytes."""
    assert len(header) == 80
    padded = bytearray(header)
    padded.append(0x80)
    padded.extend(b'\x00' * 39)
    padded += struct.pack('>Q', 640)
    return bytes(padded)


def count_leading_zeros(hash_bytes: bytes) -> int:
    """Count leading zero bits in a hash."""
    count = 0
    for byte in hash_bytes:
        if byte == 0:
            count += 8
        else:
            # Count leading zeros in this byte
            for i in range(7, -1, -1):
                if (byte >> i) & 1 == 0:
                    count += 1
                else:
                    return count
            break
    return count


def find_nonce_reduced(stub_76bytes: bytes, num_rounds: int, target_zeros: int = 1, max_attempts: int = 10000000) -> Optional[int]:
    """
    Find a nonce such that reduced-round SHA-256d has at least target_zeros leading zero bits.
    Uses RANDOM starting position to avoid miner bias.

    Args:
        stub_76bytes: First 76 bytes of Bitcoin header (without nonce)
        num_rounds: Number of rounds for both inner and outer SHA-256
        target_zeros: Minimum number of leading zero bits required
        max_attempts: Maximum nonce attempts

    Returns:
        Nonce value if found, None if not found within max_attempts
    """
    # Random starting nonce to avoid miner bias
    import random as _rng
    start_nonce = _rng.randint(0, 2**32 - 1)

    for attempt in range(max_attempts):
        nonce = (start_nonce + attempt) % (2**32)
        header = stub_76bytes + struct.pack('<I', nonce)
        hash_result = sha256d_reduced(header, num_rounds, num_rounds)

        if count_leading_zeros(hash_result) >= target_zeros:
            return nonce

    return None


# ============================================================================
# MLP Architecture (Exact copy from Phase 2A best config)
# ============================================================================

class BitPredictionMLP(nn.Module):
    """MLP for predicting nonce bits from header bits."""

    def __init__(self, input_dim: int = 992, hidden_dim: int = 1024, output_dim: int = 32, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================================
# Data Generation and Management
# ============================================================================

def header_to_bits(header_80bytes: bytes) -> np.ndarray:
    """Convert 80-byte header to 640 bits."""
    bits = []
    for byte in header_80bytes:
        for i in range(8):
            bits.append((byte >> (7-i)) & 1)
    return np.array(bits, dtype=np.float32)


def nonce_to_bits(nonce: int) -> np.ndarray:
    """Convert 32-bit nonce to 32 bits (little-endian)."""
    bits = []
    for i in range(32):
        bits.append((nonce >> i) & 1)
    return np.array(bits, dtype=np.float32)


def _make_training_vector(stub_76bytes, nonce):
    """Convert (stub, nonce) to 1024-bit training vector."""
    header_80 = stub_76bytes + struct.pack('<I', nonce)
    padded = sha256_pad_80bytes(header_80)
    all_bits = np.unpackbits(np.frombuffer(padded, dtype=np.uint8))
    # Training format: move nonce bits (608-639) to end (992-1023)
    non_nonce = np.concatenate([all_bits[:608], all_bits[640:]])  # 992 bits
    nonce_bits = all_bits[608:640]  # 32 bits
    return np.concatenate([non_nonce, nonce_bits]).astype(np.uint8)


def _find_nonce_hashlib(stub_76bytes, target_zeros=1):
    """Fast nonce finding using hashlib (R=64 only)."""
    import hashlib, random
    start = random.randint(0, 2**32 - 1)
    for i in range(2**32):
        nonce = (start + i) & 0xFFFFFFFF
        header = stub_76bytes + struct.pack('<I', nonce)
        h = hashlib.sha256(hashlib.sha256(header).digest()).digest()
        if count_leading_zeros(h) >= target_zeros:
            return nonce
    return None


def generate_reduced_round_data(sandbox_path: Path, round_count: int, num_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate training data for reduced-round SHA-256."""
    data_file = sandbox_path / "data" / f"dataset_reduced_r{round_count:02d}.npy"

    if data_file.exists():
        print(f"Loading existing data for R={round_count} from {data_file}")
        data = np.load(data_file)
        features = data[:, :992].astype(np.float32)
        targets = data[:, 992:].astype(np.float32)
        return features, targets

    print(f"Generating {num_samples} samples for R={round_count} rounds...")

    bitcoin_headers_file = sandbox_path / "data" / "dataset_real_bitcoin.npy"
    if not bitcoin_headers_file.exists():
        raise FileNotFoundError(f"Real Bitcoin headers not found: {bitcoin_headers_file}")

    bitcoin_data = np.load(bitcoin_headers_file)
    n_headers = len(bitcoin_data)
    print(f"Loaded {n_headers} real Bitcoin headers")

    # Pre-extract ALL stubs as bytes (once, ~30s for 943K)
    print(f"Pre-extracting header stubs...")
    all_stubs = []
    for idx in range(min(n_headers, 100000)):  # Cap at 100K stubs
        header_bits_raw = bitcoin_data[idx][:608]
        stub = np.packbits(header_bits_raw.astype(np.uint8)).tobytes()
        all_stubs.append(stub)
    del bitcoin_data
    print(f"  {len(all_stubs)} stubs ready")

    # Use C nonce finder for all round counts (much faster than Python)
    c_finder = sandbox_path / "sha256_nonce_finder"
    use_c = c_finder.exists() and round_count >= 4

    if round_count == 64:
        print(f"  Using hashlib (native C) for R=64")
    elif use_c:
        print(f"  Using C nonce finder for R={round_count}")
    else:
        print(f"  Using pure Python SHA-256 for R={round_count}")

    results = []
    start_time = time.time()

    if use_c and round_count != 64:
        # Batch mode: write all stubs to temp file, run C program, read nonces
        import subprocess, tempfile
        stubs_needed = min(num_samples * 2, len(all_stubs))
        stub_hexes = [all_stubs[i % len(all_stubs)].hex() for i in range(stubs_needed)]

        input_data = "\n".join(stub_hexes[:num_samples]) + "\n"
        proc = subprocess.run(
            [str(c_finder), str(round_count)],
            input=input_data, capture_output=True, text=True, timeout=600
        )
        nonces = [int(line.strip()) for line in proc.stdout.strip().split("\n") if line.strip()]
        print(f"  C finder returned {len(nonces)} nonces in {time.time()-start_time:.1f}s")

        for i, nonce in enumerate(nonces):
            if len(results) >= num_samples:
                break
            stub = all_stubs[i % len(all_stubs)]
            vec = _make_training_vector(stub, nonce)
            results.append(vec)

            if (i + 1) % 10000 == 0:
                print(f"  {len(results)}/{num_samples} vectors built", flush=True)
    else:
        # Use hashlib for R=64 or fallback to Python
        find_fn = _find_nonce_hashlib if round_count == 64 else \
                  lambda stub, tz=1: find_nonce_reduced(stub, round_count, target_zeros=tz)

        for i in range(num_samples * 2):
            if len(results) >= num_samples:
                break

            stub = all_stubs[i % len(all_stubs)]
            nonce = find_fn(stub)

            if nonce is not None:
                vec = _make_training_vector(stub, nonce)
                results.append(vec)

            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = len(results) / max(elapsed, 0.1)
                eta = (num_samples - len(results)) / max(rate, 0.01)
                print(f"  {len(results)}/{num_samples} ({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)

    if len(results) < num_samples:
        print(f"Warning: only generated {len(results)}/{num_samples} samples")

    combined = np.stack(results[:num_samples]).astype(np.uint8)
    data_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(data_file, combined)

    features = combined[:, :992].astype(np.float32)
    targets = combined[:, 992:].astype(np.float32)

    elapsed = time.time() - start_time
    print(f"Generated {len(results)} samples for R={round_count} in {elapsed:.1f}s")

    return features, targets


def calculate_p_value(accuracy: float, num_samples: int) -> float:
    """Calculate p-value for binary classification accuracy using normal approximation."""
    # Under null hypothesis, accuracy ~ Normal(0.5, sqrt(0.25/n))
    expected = 0.5
    std_error = math.sqrt(0.25 / num_samples)
    z_score = (accuracy - expected) / std_error

    # Two-tailed test
    p_value = 2 * (1 - norm_cdf(abs(z_score)))
    return p_value


def norm_cdf(x: float) -> float:
    """Normal CDF approximation using error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def count_significant_bits(per_bit_accuracies: List[float], num_samples: int, alpha: float = 0.05) -> int:
    """Count bits with statistically significant prediction accuracy (Bonferroni corrected)."""
    corrected_alpha = alpha / len(per_bit_accuracies)  # Bonferroni correction
    significant_count = 0

    for accuracy in per_bit_accuracies:
        p_value = calculate_p_value(accuracy, num_samples)
        if p_value < corrected_alpha:
            significant_count += 1

    return significant_count


# ============================================================================
# Training and Evaluation
# ============================================================================

def train_model(features: np.ndarray, targets: np.ndarray, device: torch.device,
                results_log: List[Dict], round_count: int) -> Dict:
    """Train MLP model and evaluate performance."""

    print(f"\nTraining model for R={round_count}...")

    # Convert to tensors
    X = torch.from_numpy(features).float()
    y = torch.from_numpy(targets).float()

    # Random split (80/10/10) - no temporal structure in re-mined data
    n_samples = len(X)
    indices = torch.randperm(n_samples)

    train_end = int(0.8 * n_samples)
    val_end = int(0.9 * n_samples)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    # Create datasets
    train_dataset = TensorDataset(X[train_indices], y[train_indices])
    val_dataset = TensorDataset(X[val_indices], y[val_indices])
    test_dataset = TensorDataset(X[test_indices], y[test_indices])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Model
    model = BitPredictionMLP(input_dim=992, hidden_dim=1024, output_dim=32, dropout=0.1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scaler = amp.GradScaler()

    # Training loop
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    min_epochs = 50
    max_epochs = 200

    training_start = time.time()

    for epoch in range(max_epochs):
        if shutdown_requested:
            print("Shutdown requested during training")
            break

        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()

            with amp.autocast(device_type="cuda"):
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                with amp.autocast(device_type="cuda"):
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)

                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'/tmp/best_model_r{round_count:02d}.pth')
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping check
        if patience_counter >= patience and epoch >= min_epochs:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model for evaluation
    model.load_state_dict(torch.load(f'/tmp/best_model_r{round_count:02d}.pth'))

    # Final evaluation on test set
    model.eval()
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            with amp.autocast(device_type="cuda"):
                outputs = model(batch_x)
                predictions = torch.sigmoid(outputs)

            all_predictions.append(predictions.cpu())
            all_targets.append(batch_y.cpu())

    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Convert to binary predictions
    binary_predictions = (predictions > 0.5).float()

    # Calculate metrics
    num_test_samples = len(targets)

    # Overall accuracy
    correct_predictions = (binary_predictions == targets).float()
    overall_accuracy = correct_predictions.mean().item()

    # Per-bit accuracy
    per_bit_accuracies = []
    for bit_idx in range(32):
        bit_correct = (binary_predictions[:, bit_idx] == targets[:, bit_idx]).float()
        bit_accuracy = bit_correct.mean().item()
        per_bit_accuracies.append(bit_accuracy)

    # Statistical significance
    overall_p_value = calculate_p_value(overall_accuracy, num_test_samples * 32)  # Total predictions
    significant_bits = count_significant_bits(per_bit_accuracies, num_test_samples)

    training_time = time.time() - training_start

    # Results
    result = {
        'round_count': int(round_count),
        'overall_accuracy': float(overall_accuracy),
        'per_bit_accuracies': [float(acc) for acc in per_bit_accuracies],
        'overall_p_value': float(overall_p_value),
        'significant_bits': int(significant_bits),
        'signal_detected': bool(overall_p_value < 0.05 or significant_bits > 0),
        'num_train_samples': int(len(train_indices)),
        'num_test_samples': int(num_test_samples),
        'training_time_seconds': float(training_time),
        'best_epoch': int(epoch - patience_counter) if patience_counter >= patience else int(epoch)
    }

    print(f"R={round_count} Results:")
    print(f"  Overall accuracy: {overall_accuracy:.4f} (p={overall_p_value:.2e})")
    print(f"  Significant bits: {significant_bits}/32")
    print(f"  Signal detected: {result['signal_detected']}")
    print(f"  Training time: {training_time:.1f}s")

    # Clean up model file
    os.unlink(f'/tmp/best_model_r{round_count:02d}.pth')

    return result


# ============================================================================
# Main Experiment Orchestration
# ============================================================================

def save_status(sandbox_path: Path, status: Dict):
    """Save current status to JSON file."""
    status_file = sandbox_path / "data" / "status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    status = convert_types(status)

    with open(status_file, 'w') as f:
        json.dump(status, f, indent=2)


def load_checkpoint(sandbox_path: Path) -> Dict:
    """Load existing checkpoint if available."""
    checkpoint_file = sandbox_path / "data" / "phase2c_checkpoint.json"
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f)
    return {'completed_rounds': [], 'results': []}


def save_checkpoint(sandbox_path: Path, checkpoint: Dict):
    """Save checkpoint data."""
    checkpoint_file = sandbox_path / "data" / "phase2c_checkpoint.json"
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    checkpoint = convert_types(checkpoint)

    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def setup_logging(sandbox_path: Path):
    """Set up logging files."""
    log_dir = sandbox_path / "data"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create log files if they don't exist
    (log_dir / "phase2c.log").touch()
    (log_dir / "phase2c_results.jsonl").touch()


def log_message(sandbox_path: Path, message: str):
    """Log message to phase2c.log."""
    log_file = sandbox_path / "data" / "phase2c.log"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")


def log_result(sandbox_path: Path, result: Dict):
    """Log result to phase2c_results.jsonl."""
    results_file = sandbox_path / "data" / "phase2c_results.jsonl"

    # Convert numpy types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        return obj

    result = convert_types(result)

    with open(results_file, 'a') as f:
        f.write(json.dumps(result) + '\n')


def check_shutdown_signal(sandbox_path: Path) -> bool:
    """Check for shutdown signal file."""
    return (sandbox_path / "shutdown.signal").exists() or shutdown_requested


def run_phase2c_experiment(sandbox_path: Path, generate_only: bool = False, train_only: bool = False):
    """Run the complete Phase 2C experiment."""

    print("=== Phase 2C: Reduced-Round SHA-256 ML Investigation ===")
    print(f"Sandbox: {sandbox_path}")
    print(f"Generate only: {generate_only}")
    print(f"Train only: {train_only}")

    # Setup
    setup_logging(sandbox_path)
    log_message(sandbox_path, "Phase 2C experiment started")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    log_message(sandbox_path, f"Using device: {device}")

    # Round counts to test
    # R=64 first (uses hashlib, fast), then ascending reduced rounds
    # NOTE: nonce is W[3] in block 2, first used in round 3.
    # R=1,2,3 are meaningless (nonce doesn't affect output).
    # R=4 is the minimum where nonce has any effect.
    round_counts = [64, 4, 5, 8, 10, 15, 20, 32]

    # Load checkpoint
    checkpoint = load_checkpoint(sandbox_path)
    completed_rounds = set(checkpoint['completed_rounds'])
    results = checkpoint['results']

    print(f"Resuming from checkpoint. Completed rounds: {sorted(completed_rounds)}")

    # Data generation phase
    if not train_only:
        print("\n=== Data Generation Phase ===")
        for round_count in round_counts:
            if check_shutdown_signal(sandbox_path):
                print("Shutdown signal detected during data generation")
                break

            if round_count in completed_rounds:
                print(f"Skipping R={round_count} (already completed)")
                continue

            log_message(sandbox_path, f"Starting data generation for R={round_count}")

            try:
                features, targets = generate_reduced_round_data(sandbox_path, round_count)
                print(f"Generated data for R={round_count}: {features.shape[0]} samples")
                log_message(sandbox_path, f"Data generation complete for R={round_count}: {features.shape[0]} samples")
            except Exception as e:
                print(f"ERROR: Data generation failed for R={round_count}: {e}")
                log_message(sandbox_path, f"ERROR: Data generation failed for R={round_count}: {e}")
                break

            # Update status
            save_status(sandbox_path, {
                'phase': 'data_generation',
                'current_round': round_count,
                'completed_rounds': sorted(completed_rounds),
                'total_rounds': len(round_counts),
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })

    # Exit early if generate-only mode
    if generate_only:
        print("Data generation complete (generate-only mode)")
        log_message(sandbox_path, "Data generation complete (generate-only mode)")
        return

    # Training phase
    print("\n=== Training Phase ===")
    for round_count in round_counts:
        if check_shutdown_signal(sandbox_path):
            print("Shutdown signal detected during training")
            break

        if round_count in completed_rounds:
            print(f"Skipping R={round_count} training (already completed)")
            continue

        # Check if data exists
        data_file = sandbox_path / "data" / f"dataset_reduced_r{round_count:02d}.npy"
        if not data_file.exists():
            print(f"ERROR: Data file missing for R={round_count}: {data_file}")
            log_message(sandbox_path, f"ERROR: Data file missing for R={round_count}")
            break

        log_message(sandbox_path, f"Starting training for R={round_count}")

        try:
            # Load data
            data = np.load(data_file)
            features = data[:, :992]
            targets = data[:, 992:]

            # Train and evaluate
            result = train_model(features, targets, device, results, round_count)
            results.append(result)

            # Log result
            log_result(sandbox_path, result)
            log_message(sandbox_path, f"Training complete for R={round_count}: accuracy={result['overall_accuracy']:.4f}")

            # Update checkpoint
            completed_rounds.add(round_count)
            checkpoint['completed_rounds'] = sorted(completed_rounds)
            checkpoint['results'] = results
            save_checkpoint(sandbox_path, checkpoint)

            # Update status
            save_status(sandbox_path, {
                'phase': 'training',
                'current_round': round_count,
                'completed_rounds': sorted(completed_rounds),
                'total_rounds': len(round_counts),
                'latest_accuracy': result['overall_accuracy'],
                'latest_significant_bits': result['significant_bits'],
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            })

        except Exception as e:
            print(f"ERROR: Training failed for R={round_count}: {e}")
            log_message(sandbox_path, f"ERROR: Training failed for R={round_count}: {e}")
            import traceback
            traceback.print_exc()
            break

    # Final results summary
    if results:
        print("\n=== Final Results Summary ===")

        # Create summary table
        summary_table = []
        for result in sorted(results, key=lambda x: x['round_count']):
            summary_table.append({
                'round_count': result['round_count'],
                'accuracy': result['overall_accuracy'],
                'significant_bits': result['significant_bits'],
                'signal_detected': result['signal_detected'],
                'p_value': result['overall_p_value']
            })

        # Save final results
        final_results = {
            'experiment': 'phase2c_reduced_round_sha256',
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'summary_table': summary_table,
            'detailed_results': results
        }

        final_file = sandbox_path / "data" / "phase2c_final_results.json"
        with open(final_file, 'w') as f:
            json.dump(final_results, f, indent=2)

        # Print summary table
        print("\nRound | Accuracy | Significant Bits | Signal | P-Value")
        print("------|----------|------------------|--------|----------")
        for row in summary_table:
            signal_str = "YES" if row['signal_detected'] else "NO"
            print(f"{row['round_count']:5d} | {row['accuracy']:8.4f} | {row['significant_bits']:15d} | {signal_str:6s} | {row['p_value']:8.2e}")

        print(f"\nDetailed results saved to: {final_file}")
        log_message(sandbox_path, f"Experiment complete. Results saved to {final_file}")

    else:
        print("No results generated")
        log_message(sandbox_path, "Experiment completed with no results")


def main():
    parser = argparse.ArgumentParser(description="Phase 2C: Reduced-Round SHA-256 ML Investigation")
    parser.add_argument("--sandbox", required=True, type=Path, help="Sandbox directory path")
    parser.add_argument("--generate-only", action="store_true", help="Only generate data, don't train")
    parser.add_argument("--train-only", action="store_true", help="Only train (assume data exists)")

    args = parser.parse_args()

    if not args.sandbox.exists():
        print(f"ERROR: Sandbox directory does not exist: {args.sandbox}")
        sys.exit(1)

    try:
        run_phase2c_experiment(args.sandbox, args.generate_only, args.train_only)
    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"ERROR: Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()