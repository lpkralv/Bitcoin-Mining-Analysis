#!/usr/bin/env python3
"""Large-scale CMA-ES 32D nonce search benchmark. 100 headers, 1000 trials each."""
import hashlib, struct, numpy as np, json, time
from pathlib import Path

def count_lz_btc(h):
    rev = h[::-1]
    c = 0
    for b in rev:
        if b == 0: c += 8
        else: return c + (8 - b.bit_length())
    return c

def hash_nonce(stub, nonce):
    header = stub + struct.pack('<I', nonce & 0xFFFFFFFF)
    return count_lz_btc(hashlib.sha256(hashlib.sha256(header).digest()).digest())

def cmaes_32d(stub, target_lz, max_evals=100000, pop_size=50):
    dim = 32
    mean = np.full(dim, 0.5)
    sigma = 0.25
    evals = 0
    while evals < max_evals:
        pop = np.clip(np.random.normal(mean, sigma, (pop_size, dim)), 0, 1)
        fits = []
        for ind in pop:
            bits = (ind > 0.5).astype(np.uint8)
            nv = int.from_bytes(np.packbits(bits).tobytes(), 'big')
            f = hash_nonce(stub, nv)
            fits.append(f)
            evals += 1
            if f >= target_lz:
                return evals
        fits = np.array(fits)
        top_k = pop_size // 4
        top = pop[np.argsort(fits)[-top_k:]]
        mean = 0.5 * mean + 0.5 * top.mean(axis=0)
    return max_evals

def random_search(stub, target_lz, max_evals=100000):
    for i in range(max_evals):
        if hash_nonce(stub, np.random.randint(0, 2**32)) >= target_lz:
            return i + 1
    return max_evals

sandbox = Path("/mnt/d/sha256-ml-redux")
data = np.load(sandbox / "data" / "dataset_real_bitcoin.npy")
np.random.seed(42)
indices = np.random.choice(len(data), 100, replace=False)
stubs = [np.packbits(data[i, :608].astype(np.uint8)).tobytes() for i in indices]
del data

for target_lz in [8, 12]:
    print(f"\n=== Target {target_lz} LZ, 100 headers ===")
    r_list, c_list = [], []
    t0 = time.time()
    for i, stub in enumerate(stubs):
        r = random_search(stub, target_lz, 200000)
        c = cmaes_32d(stub, target_lz, 200000, pop_size=50)
        r_list.append(r)
        c_list.append(c)
        if (i+1) % 20 == 0:
            print(f"  {i+1}/100 ({time.time()-t0:.0f}s) "
                  f"random_mean={np.mean(r_list):.0f} cmaes_mean={np.mean(c_list):.0f} "
                  f"speedup={np.mean(r_list)/np.mean(c_list):.3f}x")

    rm, cm = np.mean(r_list), np.mean(c_list)
    print(f"\n  Final: random={rm:.0f}, cmaes_32d={cm:.0f}, speedup={rm/cm:.3f}x")
    print(f"  Random found: {sum(1 for x in r_list if x<200000)}/100")
    print(f"  CMA-ES found: {sum(1 for x in c_list if x<200000)}/100")

    with open(sandbox / "data" / f"cmaes_large_lz{target_lz}.json", 'w') as f:
        json.dump({'target_lz': target_lz, 'random_mean': float(rm), 'cmaes_mean': float(cm),
                   'speedup': float(rm/cm), 'n_headers': 100,
                   'random': [int(x) for x in r_list], 'cmaes': [int(x) for x in c_list]}, f)
