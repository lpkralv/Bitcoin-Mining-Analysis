# Bitcoin Mining Analysis

**Louis Slothouber** | lpslot@gmail.com | March 2026

Empirical analysis of algorithmic approaches to SHA-256 Bitcoin mining. This repository contains the research paper and all supporting code for reproducing the results. Experiments and analysis were performed using [Claude Code](https://claude.ai/claude-code) (Anthropic). [AutoResearch](https://github.com/karpathy/autoresearch) methodology by Andrej Karpathy employed to rapidly train and evaluate ML models.

## Paper

**[Algorithmic Approaches to SHA-256 Bitcoin Mining: An Empirical Analysis](paper/sha256-mining-ml-analysis.md)**

A large-scale investigation testing whether any algorithmic improvement to Bitcoin mining exists beyond the known midstate precomputation and AsicBoost optimizations. Twelve results spanning SAT solving, machine learning (seven architecture classes, 300+ hyperparameter configurations), evolutionary search, and structural analysis of the SHA-256 compression function. No additional speedup was found among the tested methods.

## Key Findings

1. **SAT solvers** (CDCL, CryptoMiniSat, Cutting Planes) cannot beat brute force
2. **Neural networks** (139M parameters) cannot approximate SHA-256 even in the forward direction
3. **Seven ML paradigms** (MLP, VAE, CLIP, diffusion, GAN, RL, transformer) find no nonce-header dependence with random nonces
4. **Real Bitcoin nonces** have rich but non-cryptographic structure (mining pool conventions)
5. **No intermediate SHA-256 state** predicts hash validity (50M-sample test, 768 state bits)
6. **Valid nonces are uniformly distributed** with no spatial clustering or easy headers
7. **CMA-ES** requires 3.8x fewer evaluations but 3.4x slower per evaluation (net ~1.1x)
8. **Message schedule sparsity** is minimal (2/48 words nonce-independent)
9. **AND-gate density** provides heuristic measure of inversion difficulty
10. **Three structural barriers** (parity, carry chains, coupling) make SAT empirically intractable
11. **No known quantum algorithm** exceeds Grover's quadratic bound for SHA-256 mining
12. **Double SHA-256** exhibits super-linear SAT coupling penalty

## Repository Structure

```
paper/                          # Research paper (Markdown)
src/
  data_acquisition/             # Bitcoin header download via Electrum protocol
    bitcoin_headers_electrum.py # Downloads ~940K real block headers (~100 seconds)
  ml_experiments/               # Machine learning experiments
    mlp_autoresearch.py         # Phase 2A: MLP hyperparameter search (80 configs)
    reduced_round_ml.py         # Phase 2C: Reduced-round ML (8 round counts)
    phase3_vae.py               # Phase 3: VAE/Autoencoder with controls
    phase3_clip.py              # Phase 3: CLIP dual-encoder
    phase3_diffusion_gan.py     # Phase 3: Conditional diffusion + WGAN-GP
    deep_investigation.py       # D2-D4: VAE controls, power tests, high-power 2C
    tier1_gaps.py               # Gap filling: R=4,5,8 + full-train diffusion/GAN
    tier2_new_directions.py     # Hash approximation, word-level transformer, timestamp
    d1_nonce_analysis.py        # Nonce structure characterization (temporal, MI)
    speedup_benchmark.py        # Model-guided vs random search benchmark
  structural_analysis/          # SHA-256 structural analysis
    divide_and_conquer_analysis.py  # Phase 2B: Precomputation, propagation, carry chains
    sha256_nonce_finder.c       # Fast C nonce finder for reduced-round data generation
    header_optimization.c       # Test whether some headers are easier to mine
    partial_eval_test.c         # Test intermediate state predictiveness
    crack4_algebraic.c          # 50M-sample intermediate state correlation test
    cmaes_64lz.c                # CMA-ES vs random at high difficulty (4B hashes)
  search_strategies/            # Alternative search strategies
    evolutionary_mining.py      # CMA-ES 1D and 32D nonce search
    cmaes_large_scale.py        # 100-header CMA-ES benchmark
    near_miss_test.py           # Near-miss gradient structure test
    overnight_experiments.py    # RL (REINFORCE) + nonce clustering
data/                           # Generated datasets (not included; see below)
```

## Reproducing Results

### Prerequisites

- Python 3.11+ with PyTorch, NumPy
- NVIDIA GPU with CUDA 12.1+ (for ML experiments)
- GCC (for C programs)

### Data Acquisition

```bash
# Download ~940K real Bitcoin block headers (~100 seconds)
python src/data_acquisition/bitcoin_headers_electrum.py --sandbox .

# Generate difficulty-stratified datasets
python src/data_acquisition/bitcoin_headers_electrum.py --sandbox . --stratified
```

### ML Experiments

```bash
# MLP AutoResearch (80-config search + full training)
python src/ml_experiments/mlp_autoresearch.py --sandbox .

# Reduced-round ML (8 round counts)
# First compile the C nonce finder:
gcc -O3 -o sha256_nonce_finder src/structural_analysis/sha256_nonce_finder.c
python src/ml_experiments/reduced_round_ml.py --sandbox .
```

### Structural Analysis (C programs)

```bash
# Compile
gcc -O3 -o header_optimization src/structural_analysis/header_optimization.c -lm
gcc -O3 -o partial_eval_test src/structural_analysis/partial_eval_test.c -lm
gcc -O3 -o crack4_algebraic src/structural_analysis/crack4_algebraic.c -lm
gcc -O3 -o cmaes_64lz src/structural_analysis/cmaes_64lz.c -lm

# Run (each reads a hex-encoded 76-byte header stub from stdin)
echo "<hex_stub>" | ./header_optimization 8 1000000
echo "<hex_stub>" | ./crack4_algebraic 50000000
echo "<hex_stub>" | ./cmaes_64lz 4000
```

## External Data

- **Bitcoin block headers**: Downloaded via Electrum protocol from public servers (no API key needed)
- **Re-mined datasets**: Generated locally using random nonce starts on real header stubs

## Hardware Used

- NVIDIA RTX 4070 Ti (12 GB VRAM) — ML training
- Apple M4 Mac Mini — analysis and coordination
- All experiments reproducible on consumer-grade hardware

## License

MIT

This research was conducted independently, on the author's own time and using personal resources. It is not affiliated with, sponsored by, or representative of any employer.

## Citation

If you use this work, please cite:

```
Slothouber, L. (2026). "Algorithmic Approaches to SHA-256 Bitcoin Mining:
An Empirical Analysis." GitHub: Bitcoin-Mining-Analysis.
```
