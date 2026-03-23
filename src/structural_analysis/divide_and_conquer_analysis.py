#!/usr/bin/env python3
"""
Phase 2B: Divide and Conquer Analysis for SHA-256 Mining

Decomposes SHA-256d(header || nonce) into sub-computations. Classifies each as:
  (a) nonce-independent / precomputable
  (b) mechanically invertible / linear
  (c) data-dependent / hard

Four sub-analyses:
  2B.1: First-Block Precomputation Boundary
  2B.2: Nonce Bit Propagation Map
  2B.3: Carry Chain Boundary Invariants
  2B.4: Double-Hash Decomposition & Bit Importance

Bitcoin mining context:
  80-byte block header -> pad to 128 bytes (two 64-byte SHA-256 blocks)
  Block 1 (bytes 0-63): entirely nonce-independent
  Block 2 (bytes 64-127): W_3 = nonce (only free variable), rest fixed per template
  Mining: SHA-256(SHA-256(header)), check for leading zeros
"""

import sys
import os
import json
import hashlib
import struct
import time
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from reduced_round_sha256 import (
    sha256_compress, parse_block, expand_message_schedule, pad_message,
    H0, K, MASK32, rotr, shr, ch, maj, sigma0, sigma1, small_sigma0, small_sigma1,
    sha256_raw
)

from round_function_diffusion import (
    initial_state_deps, initial_w_deps_mining,
    expand_message_schedule as expand_schedule_deps,
    add_mod32_deps, add_mod32_multi_deps,
    sigma0_deps, sigma1_deps, ch_deps, maj_deps,
    STATE_BITS, WORD_SIZE, NUM_REGISTERS
)

NONCE_OFFSET = STATE_BITS  # 256
NONCE_BITS = set(range(NONCE_OFFSET, NONCE_OFFSET + 32))  # indices 256..287
ALL_STATE_BITS = set(range(STATE_BITS))  # 0..255
REG_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


def create_bitcoin_header(prev_hash=None, merkle_root=None, timestamp=None,
                          bits=None, nonce=0):
    """Create an 80-byte Bitcoin block header."""
    if prev_hash is None:
        prev_hash = b'\x00' * 32
    if merkle_root is None:
        merkle_root = b'\x01' * 32
    if timestamp is None:
        timestamp = 1640995200  # 2022-01-01
    if bits is None:
        bits = 0x207fffff  # easy target

    return (struct.pack('<I', 1) +         # version
            prev_hash +                     # prev_block_hash
            merkle_root +                   # merkle_root
            struct.pack('<I', timestamp) +  # timestamp
            struct.pack('<I', bits) +       # bits/target
            struct.pack('<I', nonce))        # nonce


def count_leading_zeros(hash_bytes):
    """Count leading zero bits in a hash digest."""
    count = 0
    for byte in hash_bytes:
        if byte == 0:
            count += 8
        else:
            for bit_pos in range(7, -1, -1):
                if byte & (1 << bit_pos):
                    return count
                count += 1
    return count


# ═══════════════════════════════════════════════════════════════
# 2B.1: FIRST-BLOCK PRECOMPUTATION BOUNDARY
# ═══════════════════════════════════════════════════════════════

def analysis_2b1():
    """
    Compute and document the midstate H₁ = compress(H₀, Block1).
    Block 1 is entirely nonce-independent → 256 bits precomputable.
    """
    print("=" * 70)
    print("2B.1: FIRST-BLOCK PRECOMPUTATION BOUNDARY")
    print("=" * 70)

    header = create_bitcoin_header(nonce=0)
    padded = pad_message(header)
    assert len(padded) == 128, f"Expected 128 bytes, got {len(padded)}"

    block1 = padded[:64]
    block2 = padded[64:128]
    w1 = parse_block(block1)
    w2 = parse_block(block2)

    # Compute midstate
    result = sha256_compress(w1, list(H0), 64)
    h1 = result.final_hash_words

    print(f"\nBlock layout (80-byte header → 128-byte padded):")
    print(f"  Block 1 (bytes 0-63):  version + prev_hash + merkle_root[:28]")
    print(f"  Block 2 (bytes 64-127): merkle_root[28:32] + time + bits + NONCE + padding")

    print(f"\nBlock 2 message words:")
    labels = {0: "merkle_root tail", 1: "timestamp", 2: "bits/target",
              3: "NONCE", 4: "0x80 pad start", 15: "length=640 bits"}
    for i in range(16):
        lbl = labels.get(i, "padding zero" if 5 <= i <= 14 else "")
        marker = " ← ONLY FREE VARIABLE" if i == 3 else ""
        print(f"    W_{i:2d} = 0x{w2[i]:08x}  ({lbl}){marker}")

    print(f"\nMidstate H₁ = compress(H₀, Block1):")
    for i in range(8):
        print(f"    H₁[{i}] = 0x{h1[i]:08x}")

    # Verify nonce independence
    header2 = create_bitcoin_header(nonce=0xDEADBEEF)
    padded2 = pad_message(header2)
    assert padded[:64] == padded2[:64], "Block 1 differs across nonces!"
    assert padded[64:128] != padded2[64:128], "Block 2 should differ!"
    print(f"\nNonce independence verified: Block 1 identical for nonce=0 and nonce=0xDEADBEEF")

    # Count precomputable operations
    # Block 1: 64 rounds × (4 additions + bitwise ops) = all precomputable
    # Block 2 rounds 0-2: W_0,W_1,W_2 are constants, but state carries from midstate
    #   So rounds 0-2 of Block 2 are also nonce-free (state is deterministic, W is constant)
    print(f"\nPrecomputation summary:")
    print(f"  Block 1: 64 rounds fully precomputable (= midstate)")
    print(f"  Block 2 rounds 0-2: nonce-free (W_0..W_2 constant, state deterministic)")
    print(f"  Block 2 round 3+: nonce-dependent (W_3 = nonce enters T₁)")
    print(f"  Total precomputable rounds: 64 + 3 = 67 of 128")
    print(f"  Precomputable state: 256 bits (midstate)")

    return {
        'block1_nonce_independent': True,
        'midstate_h1': [f"0x{x:08x}" for x in h1],
        'precomputable_rounds': 67,
        'total_rounds': 128,
        'nonce_word': 'W_3 of Block 2',
        'nonce_enters_round': 3,
    }


# ═══════════════════════════════════════════════════════════════
# 2B.2: NONCE BIT PROPAGATION MAP
# ═══════════════════════════════════════════════════════════════

def analysis_2b2():
    """
    Trace nonce bit propagation through Block 2's compression, round by round.
    Uses the symbolic dependency tracker from round_function_diffusion.py.

    Data structure: state[reg] = list of 32 sets, where state[reg][bit] is
    the set of input-bit indices that output bit depends on.
    Nonce bits are indices 256..287.
    """
    print("\n" + "=" * 70)
    print("2B.2: NONCE BIT PROPAGATION MAP")
    print("=" * 70)

    # Initialize: state registers depend on their own initial bits (0..255)
    state = initial_state_deps()

    # Message schedule: only W_3 has nonce dependency
    w = initial_w_deps_mining()
    w = expand_schedule_deps(w)

    # K_t constants have no dependency
    k_deps = [set() for _ in range(32)]

    # Track per-round metrics
    round_data = []
    first_nonce_round = {}  # reg_name -> first round with nonce dependency

    print(f"\nRound-by-round nonce propagation (Block 2 compression):")
    print(f"{'Rnd':>3} | {'Nonce regs':>10} | {'Nonce bits':>10} | "
          f"{'State%':>7} | Per-register nonce deps")
    print("-" * 90)

    for t in range(64):
        # Current register aliases
        a, b, c, d_reg = state[0], state[1], state[2], state[3]
        e, f, g, h = state[4], state[5], state[6], state[7]

        # Round function
        t1 = add_mod32_multi_deps(h, sigma1_deps(e), ch_deps(e, f, g), k_deps, w[t])
        t2 = add_mod32_deps(sigma0_deps(a), maj_deps(a, b, c))

        # State update
        new_state = {
            0: add_mod32_deps(t1, t2),                          # A = T₁ + T₂
            1: [dd.copy() for dd in a],                          # B = old A
            2: [dd.copy() for dd in b],                          # C = old B
            3: [dd.copy() for dd in c],                          # D = old C
            4: add_mod32_deps([dd.copy() for dd in d_reg], t1),  # E = old D + T₁
            5: [dd.copy() for dd in e],                          # F = old E
            6: [dd.copy() for dd in f],                          # G = old F
            7: [dd.copy() for dd in g],                          # H = old G
        }
        state = new_state

        # Analyze nonce dependency per register
        per_reg_nonce = []
        nonce_regs = []
        total_nonce = set()
        total_state = set()

        for reg in range(8):
            reg_nonce = set()
            reg_state = set()
            for bit in range(32):
                reg_nonce |= (state[reg][bit] & NONCE_BITS)
                reg_state |= (state[reg][bit] & ALL_STATE_BITS)
            per_reg_nonce.append(len(reg_nonce))
            total_nonce |= reg_nonce
            total_state |= reg_state
            if len(reg_nonce) > 0:
                nonce_regs.append(REG_NAMES[reg])
                if REG_NAMES[reg] not in first_nonce_round:
                    first_nonce_round[REG_NAMES[reg]] = t

        nonce_pct = len(total_nonce) / 32 * 100
        state_pct = len(total_state) / 256 * 100

        round_data.append({
            'round': t,
            'nonce_regs': nonce_regs,
            'per_reg_nonce_bits': per_reg_nonce,
            'total_nonce_bits': len(total_nonce),
            'nonce_saturation_pct': nonce_pct,
            'state_saturation_pct': state_pct,
        })

        # Print key rounds
        if t < 15 or t % 5 == 0 or t == 63:
            per_reg_str = " ".join(f"{REG_NAMES[i]}:{per_reg_nonce[i]:>2}" for i in range(8))
            print(f"{t:3d} | {len(nonce_regs):10d} | {len(total_nonce):10d} | "
                  f"{nonce_pct:6.1f}% | {per_reg_str}")

    # Find full nonce saturation: every output bit depends on all 32 nonce bits
    full_saturation_round = None
    # Re-run to check per-bit saturation
    state2 = initial_state_deps()
    w2 = initial_w_deps_mining()
    w2 = expand_schedule_deps(w2)

    for t in range(64):
        a, b, c, d_reg = state2[0], state2[1], state2[2], state2[3]
        e, f, g, h = state2[4], state2[5], state2[6], state2[7]
        t1 = add_mod32_multi_deps(h, sigma1_deps(e), ch_deps(e, f, g), k_deps, w2[t])
        t2 = add_mod32_deps(sigma0_deps(a), maj_deps(a, b, c))
        state2 = {
            0: add_mod32_deps(t1, t2),
            1: [dd.copy() for dd in a],
            2: [dd.copy() for dd in b],
            3: [dd.copy() for dd in c],
            4: add_mod32_deps([dd.copy() for dd in d_reg], t1),
            5: [dd.copy() for dd in e],
            6: [dd.copy() for dd in f],
            7: [dd.copy() for dd in g],
        }

        all_saturated = True
        for reg in range(8):
            for bit in range(32):
                if not NONCE_BITS.issubset(state2[reg][bit]):
                    all_saturated = False
                    break
            if not all_saturated:
                break

        if all_saturated and full_saturation_round is None:
            full_saturation_round = t

    print(f"\nKey milestones:")
    print(f"  Rounds 0-2: Nonce-free (W_3 not yet consumed)")
    print(f"  Round 3: Nonce enters T₁ = H + Σ₁(E) + Ch(E,F,G) + K₃ + W₃")
    for reg_name in REG_NAMES:
        r = first_nonce_round.get(reg_name, '—')
        print(f"  Register {reg_name} first nonce-dependent: round {r}")
    if full_saturation_round is not None:
        print(f"  Full nonce saturation (every bit depends on all 32 nonce bits): round {full_saturation_round}")
    else:
        print(f"  Full nonce saturation: NOT achieved in 64 rounds")

    return {
        'nonce_free_rounds': [0, 1, 2],
        'nonce_entry_round': 3,
        'first_nonce_round_per_register': first_nonce_round,
        'full_saturation_round': full_saturation_round,
        'round_data': [{
            'round': d['round'],
            'nonce_regs': d['nonce_regs'],
            'total_nonce_bits': d['total_nonce_bits'],
            'nonce_pct': d['nonce_saturation_pct'],
        } for d in round_data],
    }


# ═══════════════════════════════════════════════════════════════
# 2B.3: CARRY CHAIN BOUNDARY INVARIANTS
# ═══════════════════════════════════════════════════════════════

def analysis_2b3():
    """
    Characterize carry chains in nonce-dependent additions.
    Each SHA-256 round has these mod-2³² additions:
      T₁ = H + Σ₁(E) + Ch(E,F,G) + K_t + W_t  (4 sequential additions)
      T₂ = Σ₀(A) + Maj(A,B,C)                  (1 addition)
      new_A = T₁ + T₂                            (1 addition)
      new_E = D + T₁                             (1 addition)
    Total: 7 additions per round.
    """
    print("\n" + "=" * 70)
    print("2B.3: CARRY CHAIN BOUNDARY INVARIANTS")
    print("=" * 70)

    # Carry probability theory
    # For random a,b: P(carry at bit i) = 1 - 1/2^(i+1) approximately
    # More precisely: P(carry_i = 1) approaches 1/2 as i → ∞
    # Average carry chain length for random 32-bit addition ≈ 2 bits

    # Empirical carry chain analysis
    import random
    random.seed(42)
    N_SAMPLES = 100000

    chain_lengths = []
    for _ in range(N_SAMPLES):
        a = random.getrandbits(32)
        b = random.getrandbits(32)
        carry = 0
        current_chain = 0
        max_chain = 0
        for bit in range(32):
            a_bit = (a >> bit) & 1
            b_bit = (b >> bit) & 1
            s = a_bit + b_bit + carry
            carry = 1 if s >= 2 else 0
            if carry:
                current_chain += 1
                max_chain = max(max_chain, current_chain)
            else:
                current_chain = 0
        chain_lengths.append(max_chain)

    avg_max_chain = sum(chain_lengths) / len(chain_lengths)
    chain_dist = defaultdict(int)
    for cl in chain_lengths:
        chain_dist[cl] += 1

    print(f"\nCarry chain statistics ({N_SAMPLES:,} random 32-bit additions):")
    print(f"  Average max carry chain length: {avg_max_chain:.2f} bits")
    print(f"  Distribution of max chain length:")
    for length in sorted(chain_dist.keys())[:12]:
        pct = chain_dist[length] / N_SAMPLES * 100
        print(f"    Length {length:2d}: {pct:6.2f}%")

    # Per-round nonce dependency of additions
    print(f"\nAdditions per round (7 total: 4 for T₁, 1 for T₂, 1 for new_A, 1 for new_E):")
    print(f"{'Round':>5} | {'Nonce-dep adds':>14} | {'Description'}")
    print("-" * 65)

    round_add_data = []
    for t in range(64):
        if t < 3:
            # State and W are both nonce-free
            nonce_adds = 0
            desc = "All additions nonce-free"
        elif t == 3:
            # W_3 = nonce enters T₁ chain. State is still nonce-free from midstate perspective
            # T₁ involves W_3 → 4 additions in T₁ chain are nonce-dep
            # T₂ uses A,B,C (still nonce-free from rounds 0-2 state) → 1 addition nonce-free
            # new_A = T₁+T₂ → nonce-dep (T₁ is)
            # new_E = D+T₁ → nonce-dep (T₁ is)
            nonce_adds = 6  # 4(T₁) + 1(new_A) + 1(new_E), T₂ is nonce-free
            desc = "Nonce enters via W₃ into T₁; T₂ still nonce-free"
        elif t == 4:
            # Now A and E from round 3 are nonce-dep, shifting through registers
            # T₂ uses A (nonce-dep from round 3) → all 7 nonce-dep
            nonce_adds = 7
            desc = "All additions nonce-dependent (A from round 3)"
        else:
            nonce_adds = 7
            desc = "All additions nonce-dependent"

        round_add_data.append({
            'round': t,
            'total_additions': 7,
            'nonce_dependent_additions': nonce_adds,
        })

        if t < 8 or t % 10 == 0 or t == 63:
            print(f"{t:5d} | {nonce_adds:14d} | {desc}")

    total_additions = sum(d['total_additions'] for d in round_add_data)
    total_nonce_adds = sum(d['nonce_dependent_additions'] for d in round_add_data)
    nonce_free_adds = total_additions - total_nonce_adds

    print(f"\nSummary:")
    print(f"  Total additions across 64 rounds: {total_additions}")
    print(f"  Nonce-dependent additions: {total_nonce_adds} ({100*total_nonce_adds/total_additions:.1f}%)")
    print(f"  Nonce-free additions: {nonce_free_adds} ({100*nonce_free_adds/total_additions:.1f}%)")
    print(f"  Expected carry bit-operations from nonce change: {total_nonce_adds * avg_max_chain:.0f}")
    print(f"  Worst-case carry bit-operations: {total_nonce_adds * 32}")

    # Carry chain depth for SHA-256 T₁ computation
    # T₁ has 4 sequential additions: each feeds carry into the next
    # Carry can propagate through ALL 32 bits of each addition
    # Sequential chain: 4 × 32 = 128 bits of carry depth per round
    print(f"\nCarry chain depth analysis:")
    print(f"  T₁ chain (4 sequential additions): max carry depth = 4 × 32 = 128 bits")
    print(f"  T₂ chain (1 addition): max carry depth = 32 bits")
    print(f"  Final adds (2 parallel): max carry depth = 32 bits each")
    print(f"  Critical path per round: 128 (T₁) + 32 (new_A or new_E) = 160 bits")
    print(f"  Over 61 nonce-dependent rounds: theoretical max = {61 * 160} bits")
    print(f"  But average case is MUCH shorter due to random carry termination")

    return {
        'additions_per_round': 7,
        'total_additions': total_additions,
        'nonce_dependent_additions': total_nonce_adds,
        'nonce_free_additions': nonce_free_adds,
        'avg_max_carry_chain': avg_max_chain,
        'carry_chain_distribution': {str(k): v for k, v in sorted(chain_dist.items())[:12]},
        'expected_carry_ops': total_nonce_adds * avg_max_chain,
        'worst_case_carry_ops': total_nonce_adds * 32,
        'round_data': round_add_data[:10],
    }


# ═══════════════════════════════════════════════════════════════
# 2B.4: DOUBLE-HASH DECOMPOSITION & BIT IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def analysis_2b4():
    """
    For the outer hash SHA-256(inner_hash), determine which bits of the
    256-bit inner hash most affect whether the outer hash has k leading zeros.

    Under the random oracle hypothesis, all 256 bits should be equally important.
    """
    print("\n" + "=" * 70)
    print("2B.4: DOUBLE-HASH DECOMPOSITION & BIT IMPORTANCE")
    print("=" * 70)

    NUM_SAMPLES = 200_000
    print(f"\nSampling {NUM_SAMPLES:,} random 256-bit inner hashes...")
    t0 = time.time()

    # Collect: for each sample, record (leading_zeros, bit_values[256])
    # Then compute point-biserial correlation between each bit and leading_zeros

    # Accumulate sufficient statistics without storing all samples:
    # For each bit position: sum of leading_zeros when bit=0, count of bit=0
    #                        sum of leading_zeros when bit=1, count of bit=1
    sum_lz_when_0 = [0.0] * 256
    sum_lz_when_1 = [0.0] * 256
    count_0 = [0] * 256
    count_1 = [0] * 256
    total_lz = 0.0
    total_lz_sq = 0.0
    lz_distribution = defaultdict(int)

    for i in range(NUM_SAMPLES):
        inner = os.urandom(32)
        outer = hashlib.sha256(inner).digest()
        lz = count_leading_zeros(outer)

        total_lz += lz
        total_lz_sq += lz * lz
        lz_distribution[lz] += 1

        # Decode bits (MSB first within each byte, matching SHA-256 bit ordering)
        for byte_idx in range(32):
            byte_val = inner[byte_idx]
            for bit_in_byte in range(8):
                bit_pos = byte_idx * 8 + bit_in_byte
                bit_val = (byte_val >> (7 - bit_in_byte)) & 1
                if bit_val == 0:
                    sum_lz_when_0[bit_pos] += lz
                    count_0[bit_pos] += 1
                else:
                    sum_lz_when_1[bit_pos] += lz
                    count_1[bit_pos] += 1

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    mean_lz = total_lz / NUM_SAMPLES
    var_lz = total_lz_sq / NUM_SAMPLES - mean_lz ** 2
    std_lz = math.sqrt(var_lz) if var_lz > 0 else 1e-10

    print(f"\nOuter hash leading-zero distribution:")
    print(f"  Mean leading zeros: {mean_lz:.4f}")
    print(f"  Std dev: {std_lz:.4f}")
    print(f"  Expected (geometric): mean ≈ 1.0, std ≈ 1.41")
    print(f"  {'k':>4} {'count':>8} {'observed':>10} {'expected':>10}")
    for k in range(min(12, max(lz_distribution.keys()) + 1)):
        obs = lz_distribution.get(k, 0) / NUM_SAMPLES
        exp = (0.5 ** k) * 0.5 if k > 0 else 0.5
        print(f"  {k:4d} {lz_distribution.get(k, 0):8d} {obs:10.6f} {exp:10.6f}")

    # Compute point-biserial correlation for each bit position
    # r_pb = (M₁ - M₀) / s_y × √(n₀n₁/n²)
    correlations = []
    for bit_pos in range(256):
        n0 = count_0[bit_pos]
        n1 = count_1[bit_pos]
        if n0 == 0 or n1 == 0:
            correlations.append(0.0)
            continue
        m0 = sum_lz_when_0[bit_pos] / n0
        m1 = sum_lz_when_1[bit_pos] / n1
        r_pb = (m1 - m0) / std_lz * math.sqrt(n0 * n1 / (NUM_SAMPLES ** 2))
        correlations.append(r_pb)

    # Statistics across all 256 bit positions
    abs_corrs = [abs(c) for c in correlations]
    mean_abs_corr = sum(abs_corrs) / 256
    max_abs_corr = max(abs_corrs)
    max_corr_bit = abs_corrs.index(max_abs_corr)

    # Under null (random oracle), each r_pb ~ N(0, 1/√N)
    # Expected |r_pb| ≈ √(2/(πN))
    expected_abs_r = math.sqrt(2 / (math.pi * NUM_SAMPLES))
    # With Bonferroni correction for 256 tests, significance threshold at p=0.01:
    # z = 2.576 + correction... use z ≈ 3.5 for 256 tests
    z_bonferroni = 3.5
    significance_threshold = z_bonferroni / math.sqrt(NUM_SAMPLES)

    print(f"\nBit-position correlation with leading zeros:")
    print(f"  Mean |correlation|: {mean_abs_corr:.6f}")
    print(f"  Expected under random oracle: {expected_abs_r:.6f}")
    print(f"  Max |correlation|: {max_abs_corr:.6f} (bit {max_corr_bit})")
    print(f"  Significance threshold (Bonferroni p<0.01): {significance_threshold:.6f}")

    any_significant = max_abs_corr > significance_threshold
    print(f"\n  Any bit significantly correlated? {'YES — investigate!' if any_significant else 'NO — consistent with random oracle'}")

    # Show top 10 and bottom 10
    indexed = sorted(enumerate(correlations), key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  Top 10 most correlated bits:")
    for rank, (bit_pos, corr) in enumerate(indexed[:10]):
        word = bit_pos // 32
        bit_in_word = bit_pos % 32
        print(f"    #{rank+1}: bit {bit_pos:3d} (word {word}, bit {bit_in_word:2d})  r = {corr:+.6f}")

    # Test uniformity: are all correlations drawn from the same distribution?
    # Chi-square test: partition 256 bits into 8 groups of 32, compare mean correlations
    print(f"\n  Per-word mean correlation (should all be ~0):")
    word_means = []
    for w in range(8):
        word_corrs = correlations[w*32:(w+1)*32]
        wm = sum(word_corrs) / 32
        word_means.append(wm)
        print(f"    Word {w}: mean r = {wm:+.6f}")

    return {
        'num_samples': NUM_SAMPLES,
        'mean_leading_zeros': mean_lz,
        'std_leading_zeros': std_lz,
        'leading_zero_distribution': {str(k): v for k, v in sorted(lz_distribution.items())},
        'mean_abs_correlation': mean_abs_corr,
        'max_abs_correlation': max_abs_corr,
        'max_corr_bit': max_corr_bit,
        'significance_threshold': significance_threshold,
        'any_significant': any_significant,
        'expected_abs_correlation': expected_abs_r,
        'top_10_bits': [(bit, corr) for bit, corr in indexed[:10]],
        'per_word_mean_correlation': word_means,
        'conclusion': 'All bits equally important (random oracle)' if not any_significant
                      else 'Some bits show significant correlation — needs investigation',
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("SHA-256 Mining: Phase 2B — Divide and Conquer Analysis")
    print("=" * 70)
    print()

    results = {}
    t_start = time.time()

    results['2b1_precomputation'] = analysis_2b1()
    results['2b2_nonce_propagation'] = analysis_2b2()
    results['2b3_carry_chains'] = analysis_2b3()
    results['2b4_double_hash'] = analysis_2b4()

    elapsed = time.time() - t_start

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'docs', 'results')
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, 'divide_and_conquer_results.json')

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"PHASE 2B COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 70}")
    print(f"Results saved to: {results_file}")
    print()
    print("Key findings:")
    r1 = results['2b1_precomputation']
    r2 = results['2b2_nonce_propagation']
    r3 = results['2b3_carry_chains']
    r4 = results['2b4_double_hash']

    print(f"  2B.1: {r1['precomputable_rounds']} of {r1['total_rounds']} rounds precomputable")
    print(f"  2B.2: Nonce enters round {r2['nonce_entry_round']}, "
          f"full saturation at round {r2['full_saturation_round']}")
    print(f"  2B.3: {r3['nonce_dependent_additions']} of {r3['total_additions']} additions nonce-dependent")
    print(f"  2B.4: Max |correlation| = {r4['max_abs_correlation']:.6f} "
          f"(threshold = {r4['significance_threshold']:.6f}) → "
          f"{'SIGNAL' if r4['any_significant'] else 'random oracle confirmed'}")


if __name__ == '__main__':
    main()
