#!/usr/bin/env python3
"""
Bitcoin Nonce Structure Analysis (D1)
=====================================

Analyzes 943K real Bitcoin block headers to characterize nonce distribution,
temporal evolution, conditional structure, and mutual information with header fields.

Usage: python d1_nonce_analysis.py --data <path_to_dataset_real_bitcoin.npy> [--output <output.json>]
"""

import numpy as np
import argparse
import json
import time
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def pearsonr(x, y):
    """Compute Pearson correlation coefficient and p-value."""
    import math
    n = len(x)
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    if sx == 0 or sy == 0:
        return 0.0, 1.0
    r = np.mean((x - mx) * (y - my)) / (sx * sy) * n / (n - 1)
    r = max(-1.0, min(1.0, r))
    if abs(r) == 1.0:
        return float(r), 0.0
    t = r * math.sqrt((n - 2) / (1 - r*r))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return float(r), float(p)

def pack_bits_to_uint32(bits_array, start_bit, little_endian=True):
    """
    Pack 32 bits into uint32 values.

    Args:
        bits_array: (N, total_bits) array of bits
        start_bit: starting bit position
        little_endian: if True, treat as little-endian (LSB first)

    Returns:
        (N,) array of uint32 values
    """
    if bits_array.shape[1] < start_bit + 32:
        raise ValueError(f"Not enough bits: need {start_bit + 32}, got {bits_array.shape[1]}")

    bits = bits_array[:, start_bit:start_bit+32]

    if little_endian:
        # For little-endian: bit 0 is LSB, bit 31 is MSB
        powers = 2 ** np.arange(32)
    else:
        # For big-endian: bit 0 is MSB, bit 31 is LSB
        powers = 2 ** np.arange(31, -1, -1)

    return np.sum(bits * powers[np.newaxis, :], axis=1).astype(np.uint32)

def compute_mutual_information_vectorized(header_bits, nonce_bits):
    """
    Efficiently compute mutual information for all header-nonce bit pairs.

    Args:
        header_bits: (N, 608) array of header bits
        nonce_bits: (N, 32) array of nonce bits

    Returns:
        mi_matrix: (608, 32) array of MI values
        top_pairs: list of (header_bit, nonce_bit, mi_value) tuples
    """
    N = header_bits.shape[0]
    n_header_bits = header_bits.shape[1]
    n_nonce_bits = nonce_bits.shape[1]

    print(f"Computing MI for {n_header_bits} × {n_nonce_bits} = {n_header_bits * n_nonce_bits} bit pairs...")

    # Vectorized MI computation
    mi_matrix = np.zeros((n_header_bits, n_nonce_bits))

    # Process in chunks to manage memory
    chunk_size = 50
    for i in range(0, n_header_bits, chunk_size):
        end_i = min(i + chunk_size, n_header_bits)

        for j in range(0, n_nonce_bits, chunk_size):
            end_j = min(j + chunk_size, n_nonce_bits)

            # Get chunks
            h_chunk = header_bits[:, i:end_i]  # (N, chunk_h)
            n_chunk = nonce_bits[:, j:end_j]   # (N, chunk_n)

            # Compute 2x2 contingency tables for all pairs in chunk
            for hi in range(h_chunk.shape[1]):
                for nj in range(n_chunk.shape[1]):
                    h_bit = h_chunk[:, hi]
                    n_bit = n_chunk[:, nj]

                    # 2x2 contingency table
                    n00 = np.sum((h_bit == 0) & (n_bit == 0))
                    n01 = np.sum((h_bit == 0) & (n_bit == 1))
                    n10 = np.sum((h_bit == 1) & (n_bit == 0))
                    n11 = np.sum((h_bit == 1) & (n_bit == 1))

                    # Marginals
                    n0x = n00 + n01
                    n1x = n10 + n11
                    nx0 = n00 + n10
                    nx1 = n01 + n11

                    # Compute MI
                    mi = 0.0
                    for (cell, row_marg, col_marg) in [(n00, n0x, nx0), (n01, n0x, nx1),
                                                        (n10, n1x, nx0), (n11, n1x, nx1)]:
                        if cell > 0 and row_marg > 0 and col_marg > 0:
                            pij = cell / N
                            pi = row_marg / N
                            pj = col_marg / N
                            mi += pij * np.log2(pij / (pi * pj))

                    mi_matrix[i + hi, j + nj] = mi

        if (i // chunk_size + 1) % 5 == 0:
            print(f"  Processed {i + chunk_size}/{n_header_bits} header bits...")

    # Find top pairs
    flat_indices = np.argsort(mi_matrix.ravel())[::-1]
    top_pairs = []
    for idx in flat_indices[:50]:
        header_bit, nonce_bit = np.unravel_index(idx, mi_matrix.shape)
        mi_val = mi_matrix[header_bit, nonce_bit]
        top_pairs.append((int(header_bit), int(nonce_bit), float(mi_val)))

    return mi_matrix, top_pairs

def analyze_temporal_evolution(data):
    """D1.1: Temporal Evolution Analysis"""
    print("\n" + "="*60)
    print("D1.1: TEMPORAL EVOLUTION ANALYSIS")
    print("="*60)

    N = data.shape[0]
    nonce_bits = data[:, 992:1024]

    # Define era boundaries
    era_boundaries = [0, 188697, 377394, 566091, 754788, N]
    era_names = ['Era 0 (0-189K)', 'Era 1 (189K-377K)', 'Era 2 (378K-566K)',
                 'Era 3 (567K-755K)', 'Era 4 (755K-943K)']

    results = {}

    print(f"\nSplitting {N} blocks into 5 eras:")
    for i, (start, end) in enumerate(zip(era_boundaries[:-1], era_boundaries[1:])):
        print(f"  {era_names[i]}: blocks {start:,} to {end-1:,} ({end-start:,} blocks)")

    print("\nEra Analysis:")
    print("-" * 120)
    print(f"{'Era':<20} {'Mean Nonce':<15} {'Median':<15} {'Std Dev':<15} {'<2^16':<10} {'<2^24':<10} {'Max Bit Dev':<15}")
    print("-" * 120)

    for i, (start, end) in enumerate(zip(era_boundaries[:-1], era_boundaries[1:])):
        era_nonce_bits = nonce_bits[start:end]
        era_nonces = pack_bits_to_uint32(era_nonce_bits, 0, little_endian=True)

        # Basic statistics
        mean_nonce = float(np.mean(era_nonces))
        median_nonce = float(np.median(era_nonces))
        std_nonce = float(np.std(era_nonces))

        # Fraction below thresholds
        frac_below_16 = float(np.mean(era_nonces < 2**16))
        frac_below_24 = float(np.mean(era_nonces < 2**24))

        # Bit-level analysis
        bit_means = np.mean(era_nonce_bits, axis=0)
        max_bit_deviation = float(np.max(np.abs(bit_means - 0.5)))

        # Store results
        results[f'era_{i}'] = {
            'range': (int(start), int(end-1)),
            'n_blocks': int(end - start),
            'mean_nonce': mean_nonce,
            'median_nonce': median_nonce,
            'std_nonce': std_nonce,
            'frac_below_2e16': frac_below_16,
            'frac_below_2e24': frac_below_24,
            'bit_means': bit_means.tolist(),
            'max_bit_deviation': max_bit_deviation,
            'percentiles': {
                '25': float(np.percentile(era_nonces, 25)),
                '75': float(np.percentile(era_nonces, 75)),
                '95': float(np.percentile(era_nonces, 95)),
                '99': float(np.percentile(era_nonces, 99))
            }
        }

        print(f"{era_names[i]:<20} {mean_nonce:<15,.0f} {median_nonce:<15,.0f} {std_nonce:<15,.0f} "
              f"{frac_below_16:<10.3f} {frac_below_24:<10.3f} {max_bit_deviation:<15.4f}")

    # Bit evolution across eras
    print("\nBit-wise evolution (bits with strongest deviation from 0.5):")
    all_bit_means = np.array([results[f'era_{i}']['bit_means'] for i in range(5)])
    bit_deviations = np.abs(all_bit_means - 0.5)
    max_deviations_per_bit = np.max(bit_deviations, axis=0)

    top_bits = np.argsort(max_deviations_per_bit)[::-1][:10]
    print(f"{'Bit':<5} {'Era 0':<8} {'Era 1':<8} {'Era 2':<8} {'Era 3':<8} {'Era 4':<8} {'Max Dev':<10}")
    print("-" * 65)
    for bit in top_bits:
        bit_vals = [f"{all_bit_means[era, bit]:.3f}" for era in range(5)]
        max_dev = max_deviations_per_bit[bit]
        print(f"{bit:<5} {' '.join(f'{val:<8}' for val in bit_vals)} {max_dev:<10.4f}")

    return results

def analyze_conditional_structure(data):
    """D1.2: Conditional Structure Analysis"""
    print("\n" + "="*60)
    print("D1.2: CONDITIONAL STRUCTURE ANALYSIS")
    print("="*60)

    nonce_bits = data[:, 992:1024]

    results = {}

    # 1. Difficulty stratification
    print("\n1. Difficulty Stratification:")
    print("-" * 40)

    difficulty_bits = data[:, 576:608]  # bits field
    difficulties = pack_bits_to_uint32(difficulty_bits, 0, little_endian=True)

    # Split into quartiles
    difficulty_quartiles = np.percentile(difficulties, [25, 50, 75])
    quartile_labels = ['Q1 (Easy)', 'Q2', 'Q3', 'Q4 (Hard)']

    diff_results = {}
    print(f"{'Quartile':<12} {'N Blocks':<10} {'Diff Range':<25} {'Mean Bit Deviation':<20}")
    print("-" * 70)

    for q in range(4):
        if q == 0:
            mask = difficulties <= difficulty_quartiles[0]
            range_str = f"<= {difficulty_quartiles[0]:,.0f}"
        elif q == 3:
            mask = difficulties > difficulty_quartiles[2]
            range_str = f"> {difficulty_quartiles[2]:,.0f}"
        else:
            mask = (difficulties > difficulty_quartiles[q-1]) & (difficulties <= difficulty_quartiles[q])
            range_str = f"{difficulty_quartiles[q-1]:,.0f} - {difficulty_quartiles[q]:,.0f}"

        quartile_nonce_bits = nonce_bits[mask]
        bit_means = np.mean(quartile_nonce_bits, axis=0)
        mean_deviation = float(np.mean(np.abs(bit_means - 0.5)))

        diff_results[f'quartile_{q}'] = {
            'label': quartile_labels[q],
            'n_blocks': int(np.sum(mask)),
            'difficulty_range': range_str,
            'bit_means': bit_means.tolist(),
            'mean_bit_deviation': mean_deviation
        }

        print(f"{quartile_labels[q]:<12} {np.sum(mask):<10,} {range_str:<25} {mean_deviation:<20.4f}")

    results['difficulty_stratification'] = diff_results

    # 2. Version stratification
    print("\n2. Version Stratification:")
    print("-" * 40)

    version_bits = data[:, 0:32]
    versions = pack_bits_to_uint32(version_bits, 0, little_endian=True)

    version_counts = Counter(versions)
    top_versions = version_counts.most_common(5)

    version_results = {}
    print(f"{'Version':<12} {'N Blocks':<10} {'Percentage':<12} {'Mean Bit Deviation':<20}")
    print("-" * 60)

    for version, count in top_versions:
        mask = versions == version
        version_nonce_bits = nonce_bits[mask]
        bit_means = np.mean(version_nonce_bits, axis=0)
        mean_deviation = float(np.mean(np.abs(bit_means - 0.5)))
        percentage = 100 * count / len(versions)

        version_results[f'version_{version}'] = {
            'version': int(version),
            'n_blocks': count,
            'percentage': float(percentage),
            'bit_means': bit_means.tolist(),
            'mean_bit_deviation': mean_deviation
        }

        print(f"{version:<12} {count:<10,} {percentage:<12.1f}% {mean_deviation:<20.4f}")

    results['version_stratification'] = version_results

    # 3. Correlation matrix
    print("\n3. Correlation Analysis:")
    print("-" * 40)

    # Significant MLP bits from problem description
    significant_nonce_bits = [0, 1, 2, 7, 9, 10, 14, 15, 16, 17, 24, 25, 26, 27]
    header_bits = data[:, :608]  # first 608 bits

    correlations = []

    print("Computing correlations for 14 significant nonce bits vs 608 header bits...")

    for nonce_bit in significant_nonce_bits:
        nonce_bit_values = nonce_bits[:, nonce_bit]

        for header_bit in range(608):
            header_bit_values = header_bits[:, header_bit]

            # Only compute correlation if both bits have variation
            if np.var(nonce_bit_values) > 0 and np.var(header_bit_values) > 0:
                corr, p_val = pearsonr(header_bit_values, nonce_bit_values)
                correlations.append((header_bit, nonce_bit, float(corr), float(p_val)))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    top_correlations = correlations[:20]

    print(f"\nTop 20 Header-Nonce bit correlations:")
    print(f"{'Header Bit':<12} {'Nonce Bit':<12} {'Correlation':<15} {'P-value':<12}")
    print("-" * 55)

    for header_bit, nonce_bit, corr, p_val in top_correlations:
        print(f"{header_bit:<12} {nonce_bit:<12} {corr:<15.6f} {p_val:<12.2e}")

    results['correlations'] = {
        'top_20': [(int(h), int(n), float(c), float(p)) for h, n, c, p in top_correlations],
        'total_computed': len(correlations)
    }

    return results

def analyze_mutual_information(data):
    """D1.3: Mutual Information Analysis"""
    print("\n" + "="*60)
    print("D1.3: MUTUAL INFORMATION ANALYSIS")
    print("="*60)

    header_bits = data[:, :608]
    nonce_bits = data[:, 992:1024]
    N = data.shape[0]

    results = {}

    # 1. Bit-level MI
    print("\n1. Computing bit-level mutual information...")
    start_time = time.time()

    mi_matrix, top_pairs = compute_mutual_information_vectorized(header_bits, nonce_bits)

    compute_time = time.time() - start_time
    print(f"   Completed in {compute_time:.1f} seconds")

    total_mi = float(np.sum(mi_matrix))
    mean_mi = float(np.mean(mi_matrix))

    print(f"\nMutual Information Summary:")
    print(f"  Total MI across all bit pairs: {total_mi:.6f}")
    print(f"  Mean MI per bit pair: {mean_mi:.8f}")
    print(f"  Maximum MI: {np.max(mi_matrix):.6f}")

    print(f"\nTop 20 highest-MI (header_bit, nonce_bit) pairs:")
    print(f"{'Rank':<6} {'Header Bit':<12} {'Nonce Bit':<12} {'MI Value':<15}")
    print("-" * 50)
    for rank, (header_bit, nonce_bit, mi_val) in enumerate(top_pairs[:20], 1):
        print(f"{rank:<6} {header_bit:<12} {nonce_bit:<12} {mi_val:<15.8f}")

    # 2. Shuffled control
    print(f"\n2. Computing shuffled control...")

    # Randomly permute nonce assignments
    np.random.seed(42)  # For reproducibility
    shuffled_indices = np.random.permutation(N)
    shuffled_nonce_bits = nonce_bits[shuffled_indices]

    print("   Computing MI for shuffled data...")
    shuffled_mi_matrix, shuffled_top_pairs = compute_mutual_information_vectorized(header_bits, shuffled_nonce_bits)

    shuffled_total_mi = float(np.sum(shuffled_mi_matrix))
    shuffled_mean_mi = float(np.mean(shuffled_mi_matrix))

    true_mi = total_mi - shuffled_total_mi

    print(f"\nShuffled Control Results:")
    print(f"  Shuffled total MI: {shuffled_total_mi:.6f}")
    print(f"  True information (Real - Shuffled): {true_mi:.6f}")
    print(f"  Signal-to-noise ratio: {true_mi/shuffled_total_mi:.3f}")

    # 3. Statistical significance test
    print(f"\n3. Statistical significance testing...")

    # Expected MI under independence for 2x2 table with N samples
    expected_mi_indep = 1.0 / (2 * N * np.log(2))

    # Standard deviation (approximate)
    # For large N, variance of MI under null is approximately expected_mi_indep / N
    mi_std = np.sqrt(expected_mi_indep / N)
    threshold_3sigma = expected_mi_indep + 3 * mi_std

    # Count significant pairs
    significant_mask = mi_matrix > threshold_3sigma
    n_significant = int(np.sum(significant_mask))

    print(f"\nStatistical Significance (3-sigma test):")
    print(f"  Expected MI under independence: {expected_mi_indep:.8f}")
    print(f"  Standard deviation: {mi_std:.8f}")
    print(f"  3-sigma threshold: {threshold_3sigma:.8f}")
    print(f"  Number of significant pairs: {n_significant:,} / {mi_matrix.size:,}")
    print(f"  Percentage significant: {100*n_significant/mi_matrix.size:.3f}%")

    if n_significant > 0:
        # Find significant pairs
        sig_indices = np.where(significant_mask)
        significant_pairs = []
        for i in range(len(sig_indices[0])):
            h_bit = int(sig_indices[0][i])
            n_bit = int(sig_indices[1][i])
            mi_val = float(mi_matrix[h_bit, n_bit])
            z_score = (mi_val - expected_mi_indep) / mi_std
            significant_pairs.append((h_bit, n_bit, mi_val, z_score))

        # Sort by MI value
        significant_pairs.sort(key=lambda x: x[2], reverse=True)

        print(f"\nTop 10 statistically significant pairs:")
        print(f"{'Header Bit':<12} {'Nonce Bit':<12} {'MI Value':<15} {'Z-score':<10}")
        print("-" * 55)
        for header_bit, nonce_bit, mi_val, z_score in significant_pairs[:10]:
            print(f"{header_bit:<12} {nonce_bit:<12} {mi_val:<15.8f} {z_score:<10.2f}")

        results['significant_pairs'] = significant_pairs[:50]  # Top 50

    results.update({
        'total_mi': total_mi,
        'mean_mi': mean_mi,
        'max_mi': float(np.max(mi_matrix)),
        'shuffled_total_mi': shuffled_total_mi,
        'true_information': true_mi,
        'expected_mi_indep': expected_mi_indep,
        'mi_threshold_3sigma': threshold_3sigma,
        'n_significant_pairs': n_significant,
        'top_50_pairs': top_pairs,
        'computation_time_seconds': compute_time
    })

    return results

def main():
    parser = argparse.ArgumentParser(description='Bitcoin Nonce Structure Analysis')
    parser.add_argument('--data', required=True, help='Path to dataset_real_bitcoin.npy')
    parser.add_argument('--output', default='d1_results.json', help='Output JSON file path')

    args = parser.parse_args()

    print("Bitcoin Nonce Structure Analysis (D1)")
    print("=" * 60)
    print(f"Data file: {args.data}")
    print(f"Output file: {args.output}")

    # Load data
    print(f"\nLoading data from {args.data}...")
    try:
        data = np.load(args.data)
        print(f"Loaded data shape: {data.shape}")
        print(f"Data type: {data.dtype}")

        if data.shape[1] != 1024:
            print(f"Warning: Expected 1024 bits per sample, got {data.shape[1]}")

        if data.dtype != np.uint8:
            print(f"Warning: Expected uint8 data, got {data.dtype}")
            data = data.astype(np.uint8)

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Verify data format
    print(f"\nData format verification:")
    print(f"  Total samples: {data.shape[0]:,}")
    print(f"  Bits per sample: {data.shape[1]}")
    print(f"  Header bits (0:992): {992}")
    print(f"  Nonce bits (992:1024): {32}")
    print(f"  Expected block range: 0 to {data.shape[0]-1}")

    # Run analyses
    all_results = {
        'metadata': {
            'data_file': args.data,
            'n_samples': int(data.shape[0]),
            'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_bits': int(data.shape[1])
        }
    }

    # D1.1: Temporal Evolution
    temporal_results = analyze_temporal_evolution(data)
    all_results['temporal_evolution'] = temporal_results

    # D1.2: Conditional Structure
    conditional_results = analyze_conditional_structure(data)
    all_results['conditional_structure'] = conditional_results

    # D1.3: Mutual Information
    mi_results = analyze_mutual_information(data)
    all_results['mutual_information'] = mi_results

    # Save results
    print(f"\n" + "="*60)
    print(f"ANALYSIS COMPLETE")
    print(f"="*60)
    print(f"Saving results to {args.output}...")

    try:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved successfully!")
        print(f"Total analysis time: {time.time() - start_time:.1f} seconds")
    except Exception as e:
        print(f"Error saving results: {e}")

if __name__ == '__main__':
    start_time = time.time()
    main()