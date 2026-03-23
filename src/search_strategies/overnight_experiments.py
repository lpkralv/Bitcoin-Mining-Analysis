#!/usr/bin/env python3
"""
Overnight SHA-256 Nonce Finding Experiments
==========================================

Three experiments for CUDA-accelerated nonce discovery:
1. Reduced-Round RL Mining (MOST IMPORTANT) - Train policy to find valid nonces
2. Nonce Clustering Analysis - Test random oracle hypothesis
3. Full-Round RL (conditional) - Scale up if reduced rounds show signal

Usage: python overnight_experiments.py --sandbox /mnt/d/sha256-ml-redux
"""

import argparse
import hashlib
import json
import logging
import os
import signal
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


# ============================================================================
# Globals and Configuration
# ============================================================================

SHUTDOWN_SIGNAL_FILE = None
INTERRUPTED = False

def signal_handler(signum, frame):
    global INTERRUPTED
    INTERRUPTED = True
    print(f"\nReceived signal {signum}, setting interrupt flag...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================================
# Inline Reduced-Round SHA-256 Implementation
# ============================================================================

# SHA-256 constants
K_CONST = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

H0 = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]

MASK = 0xFFFFFFFF

def rotr(x: int, n: int) -> int:
    """Right rotate 32-bit integer"""
    return ((x >> n) | (x << (32 - n))) & MASK

def sha256_compress_reduced(block_words: List[int], state: List[int], num_rounds: int) -> List[int]:
    """SHA-256 compression function with configurable round count"""
    w = block_words[:]

    # Message schedule (extend to 64 words)
    for i in range(16, 64):
        s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3)
        s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10)
        w.append((w[i-16] + s0 + w[i-7] + s1) & MASK)

    # Working variables
    a, b, c, d, e, f, g, h = state

    # Main compression rounds (limited by num_rounds)
    for i in range(num_rounds):
        S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)
        ch = (e & f) ^ ((~e) & g)
        temp1 = (h + S1 + ch + K_CONST[i] + w[i]) & MASK
        S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)
        maj = (a & b) ^ (a & c) ^ (b & c)
        temp2 = (S0 + maj) & MASK

        h = g
        g = f
        f = e
        e = (d + temp1) & MASK
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & MASK

    # Add to previous state
    return [
        (state[0] + a) & MASK,
        (state[1] + b) & MASK,
        (state[2] + c) & MASK,
        (state[3] + d) & MASK,
        (state[4] + e) & MASK,
        (state[5] + f) & MASK,
        (state[6] + g) & MASK,
        (state[7] + h) & MASK
    ]

def sha256d_reduced_bytes(header_80bytes: bytes, inner_rounds: int = 64, outer_rounds: int = 64) -> bytes:
    """Double SHA-256 with configurable round counts, returns hash bytes"""
    # First SHA-256 (inner)
    # Pad to 512 bits (64 bytes)
    padded = header_80bytes + b'\x80' + b'\x00' * 43 + struct.pack('>Q', 80 * 8)

    # Process 512-bit block
    block_words = list(struct.unpack('>16I', padded))
    hash1_state = sha256_compress_reduced(block_words, H0[:], inner_rounds)

    # Convert first hash to bytes
    hash1_bytes = struct.pack('>8I', *hash1_state)

    # Second SHA-256 (outer)
    # Pad first hash to 512 bits
    padded2 = hash1_bytes + b'\x80' + b'\x00' * 31 + struct.pack('>Q', 32 * 8)

    # Process 512-bit block
    block_words2 = list(struct.unpack('>16I', padded2))
    hash2_state = sha256_compress_reduced(block_words2, H0[:], outer_rounds)

    return struct.pack('>8I', *hash2_state)

def count_leading_zero_bits(hash_bytes: bytes) -> int:
    """Count leading zero bits in hash"""
    count = 0
    for byte in hash_bytes:
        if byte == 0:
            count += 8
        else:
            # Count leading zeros in this byte
            for i in range(8):
                if byte & (0x80 >> i):
                    break
                count += 1
            break
    return count


# ============================================================================
# Policy Network for RL
# ============================================================================

class NoncePolicy(nn.Module):
    """MLP policy network for nonce generation"""

    def __init__(self, header_bits=992, hidden_dim=4096, num_layers=6):
        super().__init__()

        layers = []
        layers.append(nn.Linear(header_bits, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 32))  # 32 nonce bits
        layers.append(nn.Sigmoid())  # Bernoulli probabilities

        self.network = nn.Sequential(*layers)

    def forward(self, header_bits):
        """Return Bernoulli probabilities for each nonce bit"""
        return self.network(header_bits)

    def sample_nonces(self, header_bits, num_samples=16):
        """Sample nonces from the policy"""
        probs = self.forward(header_bits)  # [batch_size, 32]
        batch_size = probs.shape[0]

        # Expand for sampling: [batch_size, num_samples, 32]
        probs_expanded = probs.unsqueeze(1).expand(-1, num_samples, -1)

        # Sample from Bernoulli distribution
        dist = Bernoulli(probs_expanded)
        samples = dist.sample()  # [batch_size, num_samples, 32]

        # Calculate log probabilities for REINFORCE
        log_probs = dist.log_prob(samples).sum(dim=2)  # [batch_size, num_samples]

        return samples, log_probs


# ============================================================================
# Data Loading Utilities
# ============================================================================

def load_bitcoin_headers(sandbox_path: Path, num_headers: int = 10000) -> List[bytes]:
    """Load real Bitcoin headers from dataset_real_bitcoin.npy"""
    npy_file = sandbox_path / "data" / "dataset_real_bitcoin.npy"

    if not npy_file.exists():
        logging.error(f"Dataset not found at {npy_file}")
        # Fallback to synthetic
        headers = []
        for i in range(num_headers):
            header = bytearray(80)
            header[0:4] = struct.pack('<I', 1)
            header[4:36] = np.random.randint(0, 256, 32, dtype=np.uint8).tobytes()
            header[36:68] = np.random.randint(0, 256, 32, dtype=np.uint8).tobytes()
            header[68:72] = struct.pack('<I', int(time.time()) + i)
            header[72:76] = struct.pack('<I', 0x1d00ffff)
            header[76:80] = struct.pack('<I', 0)
            headers.append(bytes(header))
        return headers

    # Load from npy — training bit format
    data = np.load(npy_file)
    n = min(num_headers, len(data))
    logging.info(f"Loading {n} headers from {npy_file} ({len(data)} available)")

    headers = []
    for i in range(n):
        # Reconstruct 76-byte stub from bits[0:608], add zero nonce
        stub_bits = data[i, :608]
        stub_bytes = np.packbits(stub_bits.astype(np.uint8)).tobytes()  # 76 bytes
        header = stub_bytes + b'\x00\x00\x00\x00'  # 80 bytes with zero nonce
        headers.append(header)

    del data

    return headers

def header_to_bits(header_bytes: bytes) -> torch.Tensor:
    """Convert header bytes to bit representation, excluding nonce"""
    # Take first 76 bytes (excluding 4-byte nonce at end)
    header_no_nonce = header_bytes[:76]

    # Convert to bits
    bits = []
    for byte in header_no_nonce:
        for i in range(8):
            bits.append(float((byte >> (7-i)) & 1))

    # Pad to 992 bits if needed (76 * 8 = 608, pad to 992)
    while len(bits) < 992:
        bits.append(0.0)

    return torch.tensor(bits, dtype=torch.float32)

def nonce_bits_to_uint32(nonce_bits: torch.Tensor) -> int:
    """Convert 32-bit tensor to uint32"""
    nonce = 0
    for i, bit in enumerate(nonce_bits):
        if bit > 0.5:
            nonce |= (1 << (31 - i))
    return nonce


# ============================================================================
# Experiment 1: Reduced-Round RL Mining
# ============================================================================

def experiment_1_reduced_round_rl(sandbox: Path, logger: logging.Logger):
    """Train RL policy to find valid nonces at reduced SHA-256 rounds"""
    logger.info("Starting Experiment 1: Reduced-Round RL Mining")

    results = {
        'experiment': 'reduced_round_rl',
        'round_results': {},
        'start_time': time.time()
    }

    # Load training data
    logger.info("Loading Bitcoin headers...")
    headers = load_bitcoin_headers(sandbox, num_headers=5000)
    train_headers = headers[:4000]
    test_headers = headers[4000:]

    # Convert to tensors
    train_header_bits = torch.stack([header_to_bits(h) for h in train_headers])
    test_header_bits = torch.stack([header_to_bits(h) for h in test_headers])

    # CUDA setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    train_header_bits = train_header_bits.to(device)
    test_header_bits = test_header_bits.to(device)

    # Test round counts — R=64 first (fast with hashlib), then R=4, R=8
    round_counts = [64, 4, 8]

    for round_count in round_counts:
        if INTERRUPTED:
            logger.info("Interrupted during round count loop")
            break

        logger.info(f"Training policy for {round_count} rounds...")

        # Initialize policy
        policy = NoncePolicy().to(device)
        optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

        # Training parameters — more steps for R=64 (fast hashlib)
        batch_size = 256
        num_samples = 16
        num_steps = 20000 if round_count == 64 else 5000
        baseline = 0.0
        baseline_momentum = 0.99

        # Training loop
        step_rewards = []
        step_entropies = []

        for step in range(num_steps):
            if INTERRUPTED:
                logger.info(f"Interrupted at step {step}")
                break

            # Sample batch of headers
            batch_indices = torch.randint(0, len(train_header_bits), (batch_size,))
            header_batch = train_header_bits[batch_indices]

            # Sample nonces from policy
            nonce_samples, log_probs = policy.sample_nonces(header_batch, num_samples)

            # Calculate rewards (on CPU for hash computation)
            rewards = torch.zeros(batch_size, num_samples)

            for i, header_bits in enumerate(header_batch):
                # Reconstruct header bytes (move to CPU)
                header_no_nonce = bytearray(76)
                header_bits_cpu = header_bits.cpu()

                for byte_idx in range(76):
                    byte_val = 0
                    for bit_idx in range(8):
                        bit_pos = byte_idx * 8 + bit_idx
                        if bit_pos < len(header_bits_cpu) and header_bits_cpu[bit_pos] > 0.5:
                            byte_val |= (1 << (7 - bit_idx))
                    header_no_nonce[byte_idx] = byte_val

                for j, nonce_bits in enumerate(nonce_samples[i]):
                    # Convert nonce bits to bytes
                    nonce_val = nonce_bits_to_uint32(nonce_bits.cpu())
                    nonce_bytes = struct.pack('<I', nonce_val)

                    # Full 80-byte header
                    full_header = bytes(header_no_nonce) + nonce_bytes

                    # Compute hash with reduced rounds
                    if round_count == 64:
                        # Use hashlib for full rounds (faster)
                        hash_result = hashlib.sha256(hashlib.sha256(full_header).digest()).digest()
                    else:
                        # Use our reduced-round implementation
                        hash_result = sha256d_reduced_bytes(full_header, round_count, round_count)

                    # Reward = number of leading zero bits
                    leading_zeros = count_leading_zero_bits(hash_result)
                    rewards[i, j] = leading_zeros

            rewards = rewards.to(device)

            # REINFORCE with baseline
            mean_reward = rewards.mean().item()
            baseline = baseline_momentum * baseline + (1 - baseline_momentum) * mean_reward

            advantages = rewards - baseline

            # Policy loss with entropy regularization to prevent collapse
            probs_for_entropy = policy(header_batch)
            entropy = -(probs_for_entropy * torch.log(probs_for_entropy + 1e-8) +
                       (1 - probs_for_entropy) * torch.log(1 - probs_for_entropy + 1e-8)).mean()
            entropy_coeff = 0.1  # Strong entropy bonus

            policy_loss = -(advantages * log_probs).mean()
            loss = policy_loss - entropy_coeff * entropy  # Subtract because we MAXIMIZE entropy

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Calculate entropy for logging
            with torch.no_grad():
                probs = policy(header_batch)
                entropy = -(probs * torch.log(probs + 1e-8) + (1-probs) * torch.log(1-probs + 1e-8)).mean()

            step_rewards.append(mean_reward)
            step_entropies.append(entropy.item())

            if step % 100 == 0:
                logger.info(f"Round {round_count}, Step {step}: reward={mean_reward:.3f}, "
                          f"baseline={baseline:.3f}, entropy={entropy.item():.3f}")

                # Update status
                update_status(sandbox, {
                    'experiment': f'reduced_round_rl_R{round_count}',
                    'step': step,
                    'reward': mean_reward,
                    'total_steps': num_steps
                })

        # Evaluation: compare model vs random search
        logger.info(f"Evaluating policy for {round_count} rounds...")

        eval_results = evaluate_policy_vs_random(
            policy, test_headers[:100], round_count, device, logger
        )

        results['round_results'][round_count] = {
            'training_rewards': step_rewards[-100:],  # Last 100 steps
            'training_entropies': step_entropies[-100:],
            'final_baseline': baseline,
            'evaluation': eval_results
        }

        logger.info(f"Round {round_count} complete. Speedup: {eval_results.get('speedup', 0):.2f}x")

    results['end_time'] = time.time()
    results['duration'] = results['end_time'] - results['start_time']

    logger.info(f"Experiment 1 complete. Duration: {results['duration']:.1f}s")
    return results

def evaluate_policy_vs_random(policy: NoncePolicy, test_headers: List[bytes],
                            round_count: int, device: torch.device,
                            logger: logging.Logger) -> Dict:
    """Compare policy-guided vs random search for finding valid nonces"""

    # Target difficulty (8 leading zero bits = 1/256 probability)
    target_zeros = 8
    max_attempts = 10000

    policy_attempts = []
    random_attempts = []

    with torch.no_grad():
        for i, header_bytes in enumerate(test_headers[:20]):  # Test on 20 headers
            if INTERRUPTED:
                break

            header_bits = header_to_bits(header_bytes).unsqueeze(0).to(device)
            header_no_nonce = header_bytes[:76]

            # Policy-guided search
            policy_found = False
            policy_count = 0

            for attempt in range(max_attempts // 16):  # Sample in batches of 16
                nonce_samples, _ = policy.sample_nonces(header_bits, 16)

                for nonce_bits in nonce_samples[0]:
                    policy_count += 1

                    nonce_val = nonce_bits_to_uint32(nonce_bits.cpu())
                    nonce_bytes = struct.pack('<I', nonce_val)
                    full_header = header_no_nonce + nonce_bytes

                    if round_count == 64:
                        hash_result = hashlib.sha256(hashlib.sha256(full_header).digest()).digest()
                    else:
                        hash_result = sha256d_reduced_bytes(full_header, round_count, round_count)

                    if count_leading_zero_bits(hash_result) >= target_zeros:
                        policy_found = True
                        break

                if policy_found:
                    break

            if policy_found:
                policy_attempts.append(policy_count)
            else:
                policy_attempts.append(max_attempts)

            # Random search
            random_found = False
            random_count = 0

            for attempt in range(max_attempts):
                random_count += 1

                nonce_val = np.random.randint(0, 2**32, dtype=np.uint32)
                nonce_bytes = struct.pack('<I', nonce_val)
                full_header = header_no_nonce + nonce_bytes

                if round_count == 64:
                    hash_result = hashlib.sha256(hashlib.sha256(full_header).digest()).digest()
                else:
                    hash_result = sha256d_reduced_bytes(full_header, round_count, round_count)

                if count_leading_zero_bits(hash_result) >= target_zeros:
                    random_found = True
                    break

            if random_found:
                random_attempts.append(random_count)
            else:
                random_attempts.append(max_attempts)

            if i % 5 == 0:
                logger.info(f"Evaluated {i+1}/20 headers for round {round_count}")

    # Calculate statistics
    policy_mean = np.mean(policy_attempts)
    random_mean = np.mean(random_attempts)
    speedup = random_mean / policy_mean if policy_mean > 0 else 0

    return {
        'policy_mean_attempts': float(policy_mean),
        'random_mean_attempts': float(random_mean),
        'speedup': float(speedup),
        'policy_success_rate': float(np.mean([a < max_attempts for a in policy_attempts])),
        'random_success_rate': float(np.mean([a < max_attempts for a in random_attempts])),
        'num_tested': len(policy_attempts)
    }


# ============================================================================
# Experiment 2: Nonce Clustering Analysis
# ============================================================================

def experiment_2_nonce_clustering(sandbox: Path, logger: logging.Logger):
    """Analyze spatial distribution of valid nonces to test random oracle hypothesis"""
    logger.info("Starting Experiment 2: Nonce Clustering Analysis")

    results = {
        'experiment': 'nonce_clustering',
        'headers': [],
        'start_time': time.time()
    }

    # Load headers
    headers = load_bitcoin_headers(sandbox, num_headers=10)
    target_difficulty = 12  # ~1/4096 valid nonces
    sample_size = 10_000_000  # 10M random samples per header

    logger.info(f"Analyzing {len(headers)} headers with {sample_size:,} samples each")
    logger.info(f"Target difficulty: {target_difficulty} leading zero bits")

    for header_idx, header_bytes in enumerate(headers):
        if INTERRUPTED:
            break

        logger.info(f"Processing header {header_idx + 1}/{len(headers)}")

        header_no_nonce = header_bytes[:76]
        valid_nonces = []
        total_tested = 0

        # Random sampling approach (faster than full enumeration)
        batch_size = 100_000

        for batch in range(sample_size // batch_size):
            if INTERRUPTED:
                break

            # Generate batch of random nonces
            random_nonces = np.random.randint(0, 2**32, batch_size, dtype=np.uint32)

            for nonce_val in random_nonces:
                total_tested += 1

                nonce_bytes = struct.pack('<I', nonce_val)
                full_header = header_no_nonce + nonce_bytes

                # Use hashlib for speed
                hash_result = hashlib.sha256(hashlib.sha256(full_header).digest()).digest()
                leading_zeros = count_leading_zero_bits(hash_result)

                if leading_zeros >= target_difficulty:
                    valid_nonces.append(int(nonce_val))

            if batch % 10 == 0:
                logger.info(f"  Batch {batch+1}/{sample_size//batch_size}, "
                          f"found {len(valid_nonces)} valid nonces")

                update_status(sandbox, {
                    'experiment': 'nonce_clustering',
                    'header': header_idx + 1,
                    'batch': batch + 1,
                    'valid_found': len(valid_nonces)
                })

        if len(valid_nonces) == 0:
            logger.warning(f"No valid nonces found for header {header_idx}")
            continue

        # Analyze distribution
        logger.info(f"Analyzing {len(valid_nonces)} valid nonces...")

        # Sort for gap analysis
        valid_nonces.sort()

        # Gap analysis (consecutive differences)
        gaps = [valid_nonces[i+1] - valid_nonces[i] for i in range(len(valid_nonces)-1)]

        # Uniformity test - chi-squared
        num_bins = 1000
        bin_counts, _ = np.histogram(valid_nonces, bins=num_bins, range=(0, 2**32))
        expected_per_bin = len(valid_nonces) / num_bins
        chi_squared = np.sum((bin_counts - expected_per_bin)**2 / expected_per_bin)

        # Autocorrelation test (lag-1)
        if len(valid_nonces) > 1:
            nonces_array = np.array(valid_nonces, dtype=np.float64)
            nonces_normalized = (nonces_array - np.mean(nonces_array)) / np.std(nonces_array)
            autocorr = np.corrcoef(nonces_normalized[:-1], nonces_normalized[1:])[0, 1]
        else:
            autocorr = 0.0

        header_results = {
            'header_index': header_idx,
            'total_tested': total_tested,
            'valid_nonces_found': len(valid_nonces),
            'success_rate': len(valid_nonces) / total_tested,
            'expected_success_rate': 1.0 / (2 ** target_difficulty),
            'gap_statistics': {
                'mean_gap': float(np.mean(gaps)) if gaps else 0,
                'std_gap': float(np.std(gaps)) if gaps else 0,
                'min_gap': int(np.min(gaps)) if gaps else 0,
                'max_gap': int(np.max(gaps)) if gaps else 0
            },
            'uniformity_test': {
                'chi_squared': float(chi_squared),
                'degrees_freedom': num_bins - 1,
                'critical_value_99': 1172.1  # Approximate for 999 df at 99% confidence
            },
            'autocorrelation': float(autocorr) if not np.isnan(autocorr) else 0.0,
            'sample_nonces': valid_nonces[:100] if len(valid_nonces) >= 100 else valid_nonces
        }

        results['headers'].append(header_results)

        logger.info(f"Header {header_idx}: {len(valid_nonces)} valid nonces, "
                   f"chi²={chi_squared:.1f}, autocorr={autocorr:.4f}")

    results['end_time'] = time.time()
    results['duration'] = results['end_time'] - results['start_time']

    logger.info(f"Experiment 2 complete. Duration: {results['duration']:.1f}s")
    return results


# ============================================================================
# Experiment 3: Full-Round RL (Conditional)
# ============================================================================

def experiment_3_full_rl(sandbox: Path, logger: logging.Logger):
    """Scale up RL approach to full SHA-256 if reduced rounds showed signal"""
    logger.info("Starting Experiment 3: Full-Round RL")

    # Larger model and more training for full complexity
    results = {
        'experiment': 'full_round_rl',
        'start_time': time.time()
    }

    # Load more training data
    headers = load_bitcoin_headers(sandbox, num_headers=20000)
    train_headers = headers[:16000]
    test_headers = headers[16000:]

    train_header_bits = torch.stack([header_to_bits(h) for h in train_headers])
    test_header_bits = torch.stack([header_to_bits(h) for h in test_headers])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_header_bits = train_header_bits.to(device)
    test_header_bits = test_header_bits.to(device)

    # Larger policy network
    policy = NoncePolicy(header_bits=992, hidden_dim=4096, num_layers=6).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50000)

    # Extended training
    batch_size = 128  # Smaller batch for larger model
    num_samples = 32  # More samples per header
    num_steps = 50000
    baseline = 0.0
    baseline_momentum = 0.99

    logger.info("Training full-round policy...")

    for step in range(num_steps):
        if INTERRUPTED:
            logger.info(f"Interrupted at step {step}")
            break

        # Sample batch
        batch_indices = torch.randint(0, len(train_header_bits), (batch_size,))
        header_batch = train_header_bits[batch_indices]

        # Sample nonces
        nonce_samples, log_probs = policy.sample_nonces(header_batch, num_samples)

        # Calculate rewards using hashlib (faster for full rounds)
        rewards = torch.zeros(batch_size, num_samples)

        for i, header_bits in enumerate(header_batch):
            header_no_nonce = bytearray(76)
            header_bits_cpu = header_bits.cpu()

            for byte_idx in range(76):
                byte_val = 0
                for bit_idx in range(8):
                    bit_pos = byte_idx * 8 + bit_idx
                    if bit_pos < len(header_bits_cpu) and header_bits_cpu[bit_pos] > 0.5:
                        byte_val |= (1 << (7 - bit_idx))
                header_no_nonce[byte_idx] = byte_val

            for j, nonce_bits in enumerate(nonce_samples[i]):
                nonce_val = nonce_bits_to_uint32(nonce_bits.cpu())
                nonce_bytes = struct.pack('<I', nonce_val)
                full_header = bytes(header_no_nonce) + nonce_bytes

                hash_result = hashlib.sha256(hashlib.sha256(full_header).digest()).digest()
                leading_zeros = count_leading_zero_bits(hash_result)
                rewards[i, j] = leading_zeros

        rewards = rewards.to(device)

        # REINFORCE with entropy regularization
        mean_reward = rewards.mean().item()
        baseline = baseline_momentum * baseline + (1 - baseline_momentum) * mean_reward

        advantages = rewards - baseline

        probs_ent = policy(header_batch)
        entropy = -(probs_ent * torch.log(probs_ent + 1e-8) +
                    (1 - probs_ent) * torch.log(1 - probs_ent + 1e-8)).mean()

        policy_loss = -(advantages * log_probs).mean()
        loss = policy_loss - 0.1 * entropy

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        if step % 500 == 0:
            logger.info(f"Step {step}: reward={mean_reward:.3f}, baseline={baseline:.3f}")

            update_status(sandbox, {
                'experiment': 'full_round_rl',
                'step': step,
                'reward': mean_reward,
                'total_steps': num_steps
            })

    # Evaluation
    logger.info("Evaluating full-round policy...")
    eval_results = evaluate_policy_vs_random(policy, test_headers[:50], 64, device, logger)

    results['evaluation'] = eval_results
    results['end_time'] = time.time()
    results['duration'] = results['end_time'] - results['start_time']

    logger.info(f"Experiment 3 complete. Speedup: {eval_results.get('speedup', 0):.2f}x")
    return results


# ============================================================================
# Utilities and Main
# ============================================================================

def setup_logging(sandbox: Path) -> logging.Logger:
    """Setup logging to file and console"""
    log_file = sandbox / "data" / "overnight.log"
    log_file.parent.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)

def update_status(sandbox: Path, status: Dict):
    """Update status.json file"""
    status_file = sandbox / "status.json"
    status['timestamp'] = time.time()

    try:
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print(f"Failed to update status: {e}")

def save_results(sandbox: Path, results: Dict, logger: logging.Logger):
    """Save results to JSON file"""
    results_file = sandbox / "data" / "overnight_results.json"
    results_file.parent.mkdir(exist_ok=True)

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results_clean = convert_numpy(results)

    try:
        with open(results_file, 'w') as f:
            json.dump(results_clean, f, indent=2)
        logger.info(f"Results saved to {results_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def check_shutdown_signal(sandbox: Path) -> bool:
    """Check if shutdown signal file exists"""
    signal_file = sandbox / "shutdown.signal"
    return signal_file.exists()

def main():
    parser = argparse.ArgumentParser(description="Overnight SHA-256 nonce experiments")
    parser.add_argument("--sandbox", required=True, type=Path,
                      help="Path to sandbox directory")
    args = parser.parse_args()

    sandbox = Path(args.sandbox)
    if not sandbox.exists():
        print(f"Sandbox directory {sandbox} does not exist")
        sys.exit(1)

    # Setup
    global SHUTDOWN_SIGNAL_FILE
    SHUTDOWN_SIGNAL_FILE = sandbox / "shutdown.signal"

    logger = setup_logging(sandbox)
    logger.info("Starting overnight experiments")
    logger.info(f"Sandbox: {sandbox}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")

    all_results = {
        'start_time': time.time(),
        'experiments': {}
    }

    try:
        # Experiment 2: Fast clustering analysis first
        if not INTERRUPTED and not check_shutdown_signal(sandbox):
            logger.info("="*50)
            results_2 = experiment_2_nonce_clustering(sandbox, logger)
            all_results['experiments']['nonce_clustering'] = results_2
            save_results(sandbox, all_results, logger)

        # Experiment 1: Main RL experiment
        if not INTERRUPTED and not check_shutdown_signal(sandbox):
            logger.info("="*50)
            results_1 = experiment_1_reduced_round_rl(sandbox, logger)
            all_results['experiments']['reduced_round_rl'] = results_1
            save_results(sandbox, all_results, logger)

            # Check if any round showed promising results (speedup > 1.5)
            max_speedup = 0
            for round_count, round_results in results_1.get('round_results', {}).items():
                speedup = round_results.get('evaluation', {}).get('speedup', 0)
                max_speedup = max(max_speedup, speedup)

            logger.info(f"Maximum speedup achieved: {max_speedup:.2f}x")

            # Experiment 3: Full RL (always run — uses hashlib, fast)
            if not INTERRUPTED and not check_shutdown_signal(sandbox):
                logger.info("="*50)
                logger.info(f"Running full-round RL (max reduced-round speedup: {max_speedup:.2f}x)")
                results_3 = experiment_3_full_rl(sandbox, logger)
                all_results['experiments']['full_round_rl'] = results_3

    except KeyboardInterrupt:
        logger.info("Experiments interrupted by user")
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
    finally:
        all_results['end_time'] = time.time()
        all_results['duration'] = all_results['end_time'] - all_results['start_time']
        all_results['interrupted'] = INTERRUPTED

        save_results(sandbox, all_results, logger)
        update_status(sandbox, {'experiment': 'completed', 'interrupted': INTERRUPTED})

        logger.info(f"Experiments complete. Total duration: {all_results['duration']:.1f}s")

        # Summary
        if 'reduced_round_rl' in all_results['experiments']:
            logger.info("RL Experiment Results:")
            for round_count, results in all_results['experiments']['reduced_round_rl']['round_results'].items():
                speedup = results.get('evaluation', {}).get('speedup', 0)
                logger.info(f"  R={round_count}: {speedup:.2f}x speedup")

        if 'nonce_clustering' in all_results['experiments']:
            clustering_results = all_results['experiments']['nonce_clustering']
            num_headers = len(clustering_results.get('headers', []))
            logger.info(f"Clustering analysis completed for {num_headers} headers")

if __name__ == '__main__':
    main()