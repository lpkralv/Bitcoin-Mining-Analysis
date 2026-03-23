#!/usr/bin/env python3
"""
Evolutionary Mining: Population-based nonce search with novelty.

Instead of gradient-based RL (which collapses), use an evolutionary
strategy that maintains a POPULATION of nonce-generating strategies
and selects based on hash quality.

Two approaches:
1. CMA-ES on the nonce space (treat 32 nonce bits as a continuous vector)
2. Header-conditioned evolution: evolve a neural network that generates
   nonces, selected by actual hash quality

Also tests: for a FIXED header, does CMA-ES find valid nonces faster
than random search? This doesn't require any header conditioning —
just asks whether evolutionary optimization can exploit structure
in the nonce→hash mapping for a single header.
"""

import hashlib
import struct
import numpy as np
import json
import time
from pathlib import Path


def count_lz_btc(h):
    rev = h[::-1]
    c = 0
    for b in rev:
        if b == 0:
            c += 8
        else:
            return c + (8 - b.bit_length())
    return c


def hash_nonce(stub_76, nonce_uint32):
    """Hash header with given nonce, return leading zeros (Bitcoin format)."""
    header = stub_76 + struct.pack('<I', nonce_uint32 & 0xFFFFFFFF)
    h = hashlib.sha256(hashlib.sha256(header).digest()).digest()
    return count_lz_btc(h)


def cmaes_single_header(stub_76, target_lz=8, max_evals=10000):
    """
    CMA-ES optimization on a single header's nonce space.

    Treats the 32-bit nonce as a continuous optimization problem in [0, 2^32).
    Fitness = leading zeros in the hash.
    """
    # Simple (1+1)-ES with self-adaptation
    dim = 1  # Searching in 1D (nonce is a single uint32)

    # Start from random point
    x = np.random.uniform(0, 2**32)
    sigma = 2**31  # Initial step size (half the range)
    best_fitness = hash_nonce(stub_76, int(x))
    best_x = x
    evals = 1

    while evals < max_evals:
        # Mutate
        x_new = x + sigma * np.random.randn()
        x_new = x_new % (2**32)  # Wrap around

        fitness = hash_nonce(stub_76, int(x_new))
        evals += 1

        if fitness >= target_lz:
            return evals, int(x_new), fitness

        if fitness > best_fitness:
            x = x_new
            best_fitness = fitness
            sigma *= 1.2  # Increase step size on success
        else:
            sigma *= 0.85  # Decrease on failure
            sigma = max(sigma, 1)  # Don't go below 1

    return max_evals, int(best_x), best_fitness


def cmaes_bitwise(stub_76, target_lz=8, max_evals=10000, pop_size=50):
    """
    CMA-ES on 32-dimensional binary space.

    Population of 32-dimensional continuous vectors in [0,1],
    thresholded to produce nonce bits. Fitness = leading zeros.
    """
    dim = 32

    # Initialize population
    mean = np.full(dim, 0.5)
    sigma = 0.25

    best_fitness = 0
    best_nonce = 0
    evals = 0

    while evals < max_evals:
        # Sample population
        population = np.random.normal(mean, sigma, (pop_size, dim))
        population = np.clip(population, 0, 1)

        # Evaluate
        fitnesses = []
        for individual in population:
            nonce_bits = (individual > 0.5).astype(np.uint8)
            nonce_val = int.from_bytes(np.packbits(nonce_bits).tobytes(), 'big')
            fitness = hash_nonce(stub_76, nonce_val)
            fitnesses.append(fitness)
            evals += 1

            if fitness >= target_lz:
                return evals, nonce_val, fitness

        fitnesses = np.array(fitnesses)

        # Select top quartile
        top_k = pop_size // 4
        top_indices = np.argsort(fitnesses)[-top_k:]
        top_individuals = population[top_indices]

        # Update mean and sigma
        new_mean = top_individuals.mean(axis=0)
        mean = 0.5 * mean + 0.5 * new_mean  # Soft update

        if fitnesses.max() > best_fitness:
            best_fitness = fitnesses.max()
            best_idx = fitnesses.argmax()
            best_bits = (population[best_idx] > 0.5).astype(np.uint8)
            best_nonce = int.from_bytes(np.packbits(best_bits).tobytes(), 'big')

    return max_evals, best_nonce, best_fitness


def random_search(stub_76, target_lz=8, max_evals=10000):
    """Baseline random search."""
    for i in range(max_evals):
        nonce = np.random.randint(0, 2**32)
        if hash_nonce(stub_76, nonce) >= target_lz:
            return i + 1, nonce
    return max_evals, 0


def main():
    sandbox = Path("/mnt/d/sha256-ml-redux")
    data = np.load(sandbox / "data" / "dataset_real_bitcoin.npy")

    # Get 20 header stubs from different eras
    indices = [50000, 150000, 250000, 350000, 450000,
               550000, 650000, 750000, 850000, 940000]
    stubs = []
    for idx in indices:
        bits = data[idx, :608]
        stub = np.packbits(bits.astype(np.uint8)).tobytes()
        stubs.append(stub)
    del data

    results = {'experiments': []}

    for target_lz in [4, 8]:
        print(f"\n{'='*60}")
        print(f"Target: {target_lz} leading zeros (expected ~{2**target_lz} random hashes)")
        print(f"{'='*60}")

        random_evals_list = []
        cmaes_1d_evals_list = []
        cmaes_32d_evals_list = []

        for i, stub in enumerate(stubs):
            # Random baseline
            re, _ = random_search(stub, target_lz, max_evals=50000)
            random_evals_list.append(re)

            # CMA-ES 1D
            ce1, _, _ = cmaes_single_header(stub, target_lz, max_evals=50000)
            cmaes_1d_evals_list.append(ce1)

            # CMA-ES 32D bitwise
            ce32, _, _ = cmaes_bitwise(stub, target_lz, max_evals=50000, pop_size=50)
            cmaes_32d_evals_list.append(ce32)

            print(f"  Header {i}: random={re:6d}, cmaes_1d={ce1:6d}, cmaes_32d={ce32:6d}")

        r_mean = np.mean(random_evals_list)
        c1_mean = np.mean(cmaes_1d_evals_list)
        c32_mean = np.mean(cmaes_32d_evals_list)

        print(f"\n  Mean evals: random={r_mean:.0f}, cmaes_1d={c1_mean:.0f}, cmaes_32d={c32_mean:.0f}")
        print(f"  Speedup 1D:  {r_mean/c1_mean:.3f}x")
        print(f"  Speedup 32D: {r_mean/c32_mean:.3f}x")

        results['experiments'].append({
            'target_lz': target_lz,
            'random_mean': float(r_mean),
            'cmaes_1d_mean': float(c1_mean),
            'cmaes_32d_mean': float(c32_mean),
            'speedup_1d': float(r_mean / c1_mean),
            'speedup_32d': float(r_mean / c32_mean),
            'random_evals': [int(x) for x in random_evals_list],
            'cmaes_1d_evals': [int(x) for x in cmaes_1d_evals_list],
            'cmaes_32d_evals': [int(x) for x in cmaes_32d_evals_list],
        })

    with open(sandbox / "data" / "evolutionary_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved.")


if __name__ == '__main__':
    main()
