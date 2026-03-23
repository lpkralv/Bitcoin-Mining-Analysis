#!/usr/bin/env python3
"""Benchmark: model-guided nonce search vs random search.

Uses model probabilities to rank bits by confidence. Flips least-confident
bits first, searching the probability-ranked nonce space. This is how a
real mining application would use the model."""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import time
from itertools import combinations

class MLPBlock(nn.Module):
    def __init__(self, in_d, out_d, act="relu", drop=0.0, skip=False):
        super().__init__()
        self.linear = nn.Linear(in_d, out_d)
        self.activation = nn.ReLU() if act == "relu" else nn.GELU()
        self.dropout = nn.Dropout(drop)
        self.use_skip = skip and in_d == out_d
    def forward(self, x):
        r = x if self.use_skip else None
        x = self.dropout(self.activation(self.linear(x)))
        return x + r if self.use_skip and r is not None else x

class SHA256MLP(nn.Module):
    def __init__(self, depth, width, activation, dropout, use_skip):
        super().__init__()
        layers = []
        in_d = 992
        for i in range(depth):
            skip = use_skip and i >= 1 and i % 2 == 1
            layers.append(MLPBlock(in_d, width, activation, dropout, skip))
            in_d = width
        layers.append(nn.Linear(in_d, 32))
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)


def count_lz_btc(h):
    """Count leading zeros in Bitcoin reversed format."""
    rev = h[::-1]
    c = 0
    for b in rev:
        if b == 0:
            c += 8
        else:
            return c + (8 - b.bit_length())
    return c


def model_guided_search(stub, probs, target_lz, max_hashes):
    """Search nonce space guided by model probabilities.

    Start with most likely nonce, then flip bits from least confident
    to most confident."""

    best_bits = (probs > 0.5).astype(np.uint8)
    confidence = np.abs(probs - 0.5)
    flip_order = np.argsort(confidence)  # least confident first

    hashes_tried = 0

    # Try best guess
    nonce_bytes = np.packbits(best_bits).tobytes()
    h = hashlib.sha256(hashlib.sha256(stub + nonce_bytes).digest()).digest()
    hashes_tried += 1
    if count_lz_btc(h) >= target_lz:
        return hashes_tried

    # Flip combinations: prioritize least-confident bits
    # For efficiency, only consider the K least confident bits for flipping
    K = min(20, 32)  # search among top-20 least confident
    flip_candidates = flip_order[:K]

    for n_flips in range(1, K + 1):
        for combo in combinations(flip_candidates, n_flips):
            if hashes_tried >= max_hashes:
                return max_hashes

            candidate = best_bits.copy()
            for bit_idx in combo:
                candidate[bit_idx] = 1 - candidate[bit_idx]

            nonce_bytes = np.packbits(candidate).tobytes()
            h = hashlib.sha256(hashlib.sha256(stub + nonce_bytes).digest()).digest()
            hashes_tried += 1
            if count_lz_btc(h) >= target_lz:
                return hashes_tried

    return max_hashes


def random_search(stub, target_lz, max_hashes):
    """Random nonce search baseline."""
    for i in range(max_hashes):
        nonce = np.random.bytes(4)
        h = hashlib.sha256(hashlib.sha256(stub + nonce).digest()).digest()
        if count_lz_btc(h) >= target_lz:
            return i + 1
    return max_hashes


def main():
    # Load model
    cp = torch.load("/mnt/d/sha256-ml-redux/best_model_phase2a.pt",
                    map_location="cpu", weights_only=False)
    model = SHA256MLP(
        cp["config"]["depth"], cp["config"]["width"],
        cp["config"]["activation"], cp["config"]["dropout"],
        cp["config"]["use_skip"])
    model.load_state_dict(cp["model_state_dict"])
    model.eval()

    # Load test data
    data = np.load("/mnt/d/sha256-ml-redux/data/dataset_real_bitcoin.npy")
    n = len(data)
    test_data = data[int(0.9 * n):int(0.9 * n) + 2000]
    del data

    # Get model probabilities
    headers_t = torch.FloatTensor(test_data[:, :992])
    with torch.no_grad():
        logits = model(headers_t)
        probs = torch.sigmoid(logits).numpy()

    # Show probability distribution
    print("Model probability statistics:")
    print(f"  Mean prob: {probs.mean():.4f}")
    print(f"  Std prob:  {probs.std():.4f}")
    print(f"  Min/Max:   {probs.min():.4f} / {probs.max():.4f}")
    mean_confidence = np.abs(probs - 0.5).mean()
    print(f"  Mean confidence (|p-0.5|): {mean_confidence:.4f}")
    print(f"  Bits with mean p < 0.4 or p > 0.6: "
          f"{sum(1 for i in range(32) if probs[:,i].mean() < 0.4 or probs[:,i].mean() > 0.6)}/32")
    print()

    # Benchmark
    N_HEADERS = 200
    MAX_HASHES = 100000

    for target_lz in [1, 4, 8, 12, 16]:
        expected = 2 ** target_lz
        if expected > MAX_HASHES:
            print(f"\n=== Target {target_lz} LZ: skipped (expected {expected} > {MAX_HASHES}) ===")
            continue

        print(f"\n=== Target: {target_lz} leading zeros (Bitcoin format) ===")
        print(f"Expected random hashes: ~{expected}")

        model_hashes = []
        random_hashes = []
        t0 = time.time()

        for i in range(N_HEADERS):
            stub = np.packbits(test_data[i, :608].astype(np.uint8)).tobytes()

            mh = model_guided_search(stub, probs[i], target_lz, MAX_HASHES)
            model_hashes.append(mh)

            rh = random_search(stub, target_lz, MAX_HASHES)
            random_hashes.append(rh)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{N_HEADERS} ({elapsed:.1f}s)")

        elapsed = time.time() - t0

        m_mean = np.mean(model_hashes)
        r_mean = np.mean(random_hashes)
        m_med = np.median(model_hashes)
        r_med = np.median(random_hashes)
        speedup_mean = r_mean / m_mean if m_mean > 0 else 0
        speedup_med = r_med / m_med if m_med > 0 else 0

        m_found = sum(1 for x in model_hashes if x < MAX_HASHES)
        r_found = sum(1 for x in random_hashes if x < MAX_HASHES)

        print(f"  Model-guided: mean={m_mean:.1f}, median={m_med:.0f} hashes "
              f"({m_found}/{N_HEADERS} found)")
        print(f"  Random:        mean={r_mean:.1f}, median={r_med:.0f} hashes "
              f"({r_found}/{N_HEADERS} found)")
        print(f"  Speedup (mean): {speedup_mean:.3f}x")
        print(f"  Speedup (median): {speedup_med:.3f}x")
        print(f"  Time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
