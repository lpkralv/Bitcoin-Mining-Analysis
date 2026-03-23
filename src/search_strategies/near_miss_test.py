#!/usr/bin/env python3
"""
Near-Miss Structure Test

For a fixed header, do nonces producing k leading zeros tend to be
"close" (in Hamming/integer distance) to nonces producing k+1 zeros?

If so, a hill-climbing approach could work: find a k-zero nonce, then
search its neighborhood for (k+1)-zero nonces.

This tests whether the valid-nonce landscape has gradient structure
even though the individual mapping nonce→hash is pseudo-random.
"""

import hashlib
import struct
import numpy as np
import json
import time
from collections import defaultdict
from pathlib import Path

def count_lz_btc(h):
    rev = h[::-1]
    c = 0
    for b in rev:
        if b == 0: c += 8
        else: return c + (8 - b.bit_length())
    return c

def hamming_distance_uint32(a, b):
    return bin(a ^ b).count('1')

def run_near_miss_test(sandbox_path):
    sandbox = Path(sandbox_path)
    data = np.load(sandbox / "data" / "dataset_real_bitcoin.npy")

    # Use 5 different headers from different eras
    header_indices = [100000, 300000, 500000, 700000, 900000]

    results = []

    for hidx in header_indices:
        print(f"\n=== Header {hidx} ===")
        stub_bits = data[hidx, :608]
        stub = np.packbits(stub_bits.astype(np.uint8)).tobytes()

        # Sample lots of random nonces and categorize by leading zeros
        nonces_by_lz = defaultdict(list)
        N_SAMPLES = 5_000_000

        t0 = time.time()
        for i in range(N_SAMPLES):
            nonce = np.random.randint(0, 2**32)
            header = stub + struct.pack('<I', nonce)
            h = hashlib.sha256(hashlib.sha256(header).digest()).digest()
            lz = count_lz_btc(h)
            if lz >= 4:  # Only store interesting ones to save memory
                nonces_by_lz[lz].append(nonce)

            if (i+1) % 1000000 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{N_SAMPLES} ({elapsed:.0f}s)")

        # Distribution
        print(f"  Leading zero distribution:")
        for lz in sorted(nonces_by_lz.keys()):
            print(f"    {lz} zeros: {len(nonces_by_lz[lz])} nonces")

        # Near-miss analysis: distance from k-zero nonces to (k+1)-zero nonces
        header_result = {
            'header_index': hidx,
            'samples': N_SAMPLES,
            'lz_counts': {str(k): len(v) for k, v in nonces_by_lz.items()},
            'near_miss_analysis': {}
        }

        for k in range(4, 12):
            if k not in nonces_by_lz or (k+1) not in nonces_by_lz:
                continue
            if len(nonces_by_lz[k]) < 10 or len(nonces_by_lz[k+1]) < 5:
                continue

            k_nonces = np.array(nonces_by_lz[k])
            k1_nonces = np.array(nonces_by_lz[k+1])

            # For each (k+1)-zero nonce, find the nearest k-zero nonce
            min_hamming_dists = []
            min_integer_dists = []

            for n1 in k1_nonces[:100]:  # Cap at 100 for speed
                # Hamming distances to all k-zero nonces
                h_dists = [hamming_distance_uint32(n1, n0) for n0 in k_nonces[:1000]]
                min_hamming_dists.append(min(h_dists))

                # Integer distances
                i_dists = [abs(int(n1) - int(n0)) for n0 in k_nonces[:1000]]
                min_integer_dists.append(min(i_dists))

            # Random baseline: distance between random nonce pairs
            random_hamming = []
            for _ in range(100):
                a, b = np.random.randint(0, 2**32, 2)
                random_hamming.append(hamming_distance_uint32(int(a), int(b)))

            # Distance from random nonces to k-zero nonces
            random_to_k_hamming = []
            for _ in range(100):
                rn = np.random.randint(0, 2**32)
                h_dists = [hamming_distance_uint32(int(rn), n0) for n0 in k_nonces[:1000]]
                random_to_k_hamming.append(min(h_dists))

            mean_near_miss_h = np.mean(min_hamming_dists)
            mean_random_to_k_h = np.mean(random_to_k_hamming)
            mean_random_h = np.mean(random_hamming)

            print(f"  k={k}→{k+1}: "
                  f"min_hamming(k+1→k)={mean_near_miss_h:.2f}, "
                  f"min_hamming(random→k)={mean_random_to_k_h:.2f}, "
                  f"random_hamming={mean_random_h:.2f}")

            header_result['near_miss_analysis'][f'{k}_to_{k+1}'] = {
                'n_k_nonces': len(k_nonces),
                'n_k1_nonces': len(k1_nonces),
                'mean_min_hamming_k1_to_k': float(mean_near_miss_h),
                'mean_min_hamming_random_to_k': float(mean_random_to_k_h),
                'mean_random_hamming': float(mean_random_h),
                'mean_min_integer_dist': float(np.mean(min_integer_dists)),
                'closer_than_random': float(mean_near_miss_h) < float(mean_random_to_k_h)
            }

        results.append(header_result)

    # Save results
    output = {
        'experiment': 'near_miss_structure',
        'results': results,
        'timestamp': time.time()
    }

    output_file = sandbox / "data" / "near_miss_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n=== SUMMARY ===")
    all_closer = []
    for r in results:
        for k, v in r['near_miss_analysis'].items():
            all_closer.append(v['closer_than_random'])
            ratio = v['mean_min_hamming_k1_to_k'] / max(v['mean_min_hamming_random_to_k'], 0.01)
            print(f"  Header {r['header_index']}, {k}: "
                  f"near-miss/random ratio = {ratio:.3f} "
                  f"({'CLOSER' if v['closer_than_random'] else 'NOT closer'})")

    n_closer = sum(all_closer)
    print(f"\n  {n_closer}/{len(all_closer)} cases where (k+1)-zero nonces are closer to k-zero nonces than random")
    print(f"  Expected if no structure: ~50%")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sandbox', default='/mnt/d/sha256-ml-redux')
    args = parser.parse_args()
    run_near_miss_test(args.sandbox)
