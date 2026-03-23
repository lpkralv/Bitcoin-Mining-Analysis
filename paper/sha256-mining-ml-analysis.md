# Algorithmic Approaches to SHA-256 Bitcoin Mining: An Empirical Analysis

**Louis Slothouber**
*March 2026*

*This article presents original empirical results on the feasibility of algorithmic improvements to [Bitcoin mining](https://en.wikipedia.org/wiki/Bitcoin_network#Mining) beyond the known [midstate optimization](https://en.bitcoin.it/wiki/Midstate). For background on SHA-256 and known attacks, see [SHA-2](https://en.wikipedia.org/wiki/SHA-2) and [AsicBoost](https://arxiv.org/abs/1604.00575).*

---

## Background

### Bitcoin Mining and SHA-256d

[Bitcoin mining](https://en.wikipedia.org/wiki/Bitcoin_network#Mining) requires finding a 32-bit nonce such that SHA-256d(block_header ‖ nonce) satisfies a difficulty target. SHA-256d denotes double hashing: SHA-256(SHA-256(x)). A [block header](https://en.bitcoin.it/wiki/Block_hashing_algorithm) is an 80-byte structure containing a version number (4 bytes), the hash of the previous block (32 bytes), the [Merkle root](https://en.wikipedia.org/wiki/Merkle_tree) of the block's transactions (32 bytes), a timestamp (4 bytes), a difficulty target (4 bytes), and the nonce (4 bytes).

Since 80 bytes exceeds the 64-byte [SHA-256 block size](https://en.wikipedia.org/wiki/SHA-2), the padded 128-byte input is processed in two blocks:

- **Block 1** (bytes 0–63): version, previous hash, and the first 28 bytes of the Merkle root. This block is entirely independent of the nonce and can be compressed once per block template (the "midstate").
- **Block 2** (bytes 64–127): the last 4 bytes of the Merkle root, timestamp, difficulty target, the **nonce** (the only variable), and deterministic [padding](https://en.wikipedia.org/wiki/SHA-2#Pseudocode).

*(For a detailed illustration of the Bitcoin block header layout and double-hash computation, see the [Block hashing algorithm](https://en.bitcoin.it/wiki/Block_hashing_algorithm) article on the Bitcoin Wiki.)*

The difficulty target requires the hash output, when interpreted as a 256-bit integer, to be below a specified threshold. In practice, this is equivalent to requiring a certain number of **leading zero bits** in the hash output. Bitcoin uses a byte-reversed representation for this comparison: the SHA-256 output words H[0]..H[7] are serialized in big-endian, then the 32-byte result is reversed byte-by-byte before comparison against the target. The [network difficulty](https://en.wikipedia.org/wiki/Bitcoin_network#Mining) adjusts approximately every two weeks; as of early 2026, the expected number of hash evaluations per valid block is approximately 2⁸⁰.

The standard assumption, formalized in the [random oracle model](https://en.wikipedia.org/wiki/Random_oracle), is that brute-force search is optimal: each nonce evaluation yields an independent, uniformly distributed hash.

### Known Optimizations

The only widely deployed optimization is **midstate precomputation**: the compression of Block 1 (the "midstate") is computed once per block template and reused across all nonce candidates. Combined with precomputation of three nonce-independent rounds in Block 2, this yields a ≈2.09× speedup over naïve double-block evaluation. The [AsicBoost](https://arxiv.org/abs/1604.00575) optimization (Hanke, 2016) achieves an additional ≈20% by exploiting invariances in the version field's effect on the message schedule, for a total of ≈2.5× over fully naïve computation.

### Scope

The question addressed here is whether additional algorithmic speedups exist — through [machine learning](https://en.wikipedia.org/wiki/Machine_learning), [SAT solving](https://en.wikipedia.org/wiki/Boolean_satisfiability_problem), [algebraic cryptanalysis](https://en.wikipedia.org/wiki/Algebraic_cryptanalysis), evolutionary search, or structural analysis of the SHA-256 [compression function](https://en.wikipedia.org/wiki/One-way_compression_function).

## Methodology

Machine learning experiments employed an **AutoResearch** methodology, inspired by the autonomous experiment framework of Karpathy (2026, [GitHub](https://github.com/karpathy/autoresearch)). For each architecture class, a random search over 40–100 hyperparameter configurations was conducted with short training runs (20–30 epochs on 50,000 samples), optimizing a validation metric. The best-performing configuration was then trained to convergence on the full dataset (up to 943,488 samples, 500 epochs with early stopping). This two-stage approach — broad automated search followed by deep training of the winner — ensures that negative results reflect the architecture class's capacity rather than an unlucky hyperparameter choice. In total, over 300 configurations were evaluated across seven architecture classes.

Training data comprised 943,488 real Bitcoin block headers obtained via the [Electrum protocol](https://electrumx.readthedocs.io/), and re-mined datasets where real header stubs were paired with nonces found from random starting positions (eliminating miner behavioral bias). Statistical significance was assessed with [Bonferroni-corrected](https://en.wikipedia.org/wiki/Bonferroni_correction) z-tests (α = 0.01/32 per nonce bit).

Structural and SAT-based experiments used custom [CNF](https://en.wikipedia.org/wiki/Conjunctive_normal_form) encoders, [CryptoMiniSat](https://github.com/msoos/cryptominisat) (with native XOR clause support), [MiniSat](https://en.wikipedia.org/wiki/MiniSat), [CaDiCaL](https://github.com/arminbiere/cadical), and the [Cutting Planes](https://en.wikipedia.org/wiki/Cutting-plane_method) solvers [RoundingSat](https://github.com/jezberg/roundingsat) and [Exact](https://gitlab.com/miao_research/Exact). All C programs were compiled with GCC 13.3 at -O3.

Experiments and analysis were performed using [Claude Code](https://claude.ai/claude-code) (Anthropic), which wrote training scripts, deployed them to the GPU server, monitored training runs autonomously, and conducted the structural and statistical analyses. The author directed the research strategy, evaluated results, and identified directions for investigation.

## Summary of Results

No additional algorithmic speedup beyond the known midstate and AsicBoost optimizations was found among any of the methods tested, spanning three broad categories: formal methods (SAT, algebraic, proof-theoretic), machine learning (seven architecture classes), and search strategies (evolutionary, gradient-based). The results below are consistent with the hypothesis that SHA-256, as used in Bitcoin mining, behaves as a random oracle at the nonce–header interface.

---

## Result 1: SAT Solvers Cannot Beat Brute Force for Mining

**Claim.** [CDCL](https://en.wikipedia.org/wiki/Conflict-driven_clause_learning) SAT solvers, XOR-aware solvers ([CryptoMiniSat](https://github.com/msoos/cryptominisat)), and [Cutting Planes](https://en.wikipedia.org/wiki/Cutting-plane_method) solvers (RoundingSat, Exact) all fail to outperform brute-force nonce search for SHA-256 mining at practical difficulty levels.

**Evidence.** SHA-256 was encoded as a CNF formula with mining-specific optimizations (constant propagation for padding bytes, nonce-only free variables). Solver performance was benchmarked at 8–32 leading zeros:

- MiniSat outperformed Glucose by 7× and CaDiCaL by 3× on mining instances.
- CryptoMiniSat's native XOR/Gaussian elimination gave 1.3× speedup at 8 leading zeros but was **2.9× slower** at 12 zeros — the XOR detection overhead exceeded the benefit.
- [Carry-save adder](https://en.wikipedia.org/wiki/Carry-save_adder) tree encoding reduced implication chain depth 4× but was **10× slower** at 12 zeros due to clause count explosion.
- Cutting Planes solvers (RoundingSat, Exact) were slower than CDCL in 4 of 5 configurations tested, despite the theoretical superiority of Cutting Planes over [resolution](https://en.wikipedia.org/wiki/Resolution_(logic)) proof systems.
- [Cube-and-conquer](https://en.wikipedia.org/wiki/Cube-and-conquer) partitioning produced **sub-linear** speedup: cubing overhead exceeded parallelism gains because SHA-256's [avalanche effect](https://en.wikipedia.org/wiki/Avalanche_effect) prevents effective constant propagation from fixed nonce bits.

**Implication.** The observed scaling of SAT solving time is empirically consistent with exponential growth in the number of leading zeros required (scaling exponent ≈ 0.99, compared to brute force's 1.0, with a 28× constant-factor overhead). This is consistent with the dominance of AND gates (73.7% of Boolean operations from carry chains in modular addition), which are expected to induce exponential branching for backward reasoning in resolution-based proof systems.

## Result 2: Neural Networks Cannot Approximate SHA-256

**Claim.** No neural network tested — from a 2-layer MLP to a 139-million-parameter deep residual network — learned to predict any bit of the SHA-256d output from its input better than random chance.

**Evidence.** A deep residual network (8 layers × 4096 units, residual connections every 2 layers) was trained on 500,000 (padded header, SHA-256d hash) pairs with [binary cross-entropy](https://en.wikipedia.org/wiki/Cross-entropy) loss and AdamW optimizer (lr = 10⁻⁴). After one full epoch, training accuracy was 50.00% and validation accuracy was 49.97% across all 256 output bits. The loss remained at the random baseline (0.6932) from the first batch to the last.

**Implication.** Within the tested architectures and training regimes, no differentiable approximation of SHA-256d was learned, suggesting that gradient-based nonce optimization through a learned surrogate is not feasible with current neural network methods. This is consistent with the concept of [computational irreducibility](https://en.wikipedia.org/wiki/Computational_irreducibility) (Wolfram, 2002) and with prior work on neural approximation of reduced-round SHA-1 (Benadjila et al., 2019), which showed partial inversion possible for 1–2 rounds but "almost uninvertible" beyond 5–6 rounds.

## Result 3: Nonce–Header Statistical Independence Under Random Nonce Selection

**Claim.** When nonces are selected uniformly at random, no statistically significant dependence was detected between header content and nonce validity at any SHA-256 round count from 4 to 64, under any of seven machine learning paradigms tested.

**Evidence.** Each architecture was selected via AutoResearch hyperparameter search (40–100 configurations per class), then trained to convergence on 50,000 re-mined Bitcoin headers at eight round counts (R = 4, 5, 8, 10, 15, 20, 32, 64). The seven paradigms test fundamentally different hypotheses about what structure might exist:

| Architecture | Question asked | AutoResearch configs | Result |
|---|---|---|---|
| MLP | Can nonce bits be predicted from header bits? | 80 | 50.07% accuracy (chance = 50%) |
| VAE/Autoencoder | Does the joint (header, nonce) distribution have compressible structure? | 40 | See below |
| CLIP dual-encoder | Do matching (header, nonce) pairs cluster together in an embedding space? | 63 | MRR = 0.008 (same as random baseline) |
| Conditional diffusion | Can a denoising model generate valid nonces from noise, conditioned on header? | 17 | 48.84% validity rate (50K test; random baseline = 50%) |
| WGAN-GP | Can an adversarial generator produce nonces indistinguishable from real ones? | 17 | 48.96% validity rate (50K test) |
| Word-level transformer | Does attending over 32-bit words (SHA-256's native granularity) reveal structure? | 2 datasets | 50.19% accuracy |
| REINFORCE policy gradient | Can a policy network learn to output nonces that produce high-quality hashes? | 1 | Policy entropy collapsed to zero (degenerate) |

All architectures were trained to convergence (up to 500 epochs, patience 20–50).

**On the VAE result.** The [variational autoencoder](https://en.wikipedia.org/wiki/Variational_autoencoder) (VAE) requires careful interpretation. An autoencoder compresses its *entire input* (header + nonce = 1024 bits) into a smaller latent representation and then reconstructs it. The best configuration (256-dimensional latent space, plain autoencoder without KL regularization) achieved 89.6% reconstruction accuracy on paired (header, nonce) data. This appears to be a strong result — but three control experiments show it is not:

1. **Shuffled control**: When headers were randomly paired with nonces from *different* samples (destroying any real association), reconstruction accuracy was 88.0% — only 1.6% lower. If the autoencoder were exploiting header–nonce correlation, shuffling should have dramatically reduced nonce reconstruction.

2. **Capacity control**: Reconstruction accuracy increased linearly with latent dimension (54.8% at dim=8, 64.3% at dim=64, 71.1% at dim=256) with no inflection point. This linear scaling is the signature of raw compression capacity, not learned structure — the autoencoder simply has enough latent dimensions to memorize portions of the input.

3. **Nonce-only control**: An autoencoder trained on *only the 32 nonce bits* (with no header input) achieved 69.2% reconstruction at latent dim=8 — substantially higher than the 54.8% achieved by the full (header + nonce) model at the same latent dimension. The header *hurts* nonce reconstruction by consuming latent capacity that could otherwise encode nonce bits.

These controls establish that the autoencoder's nonce reconstruction is a compression artifact, not evidence of header–nonce structure.

**Implication.** The negative result spans supervised prediction, unsupervised compression, contrastive matching, generative modeling (both denoising and adversarial), reinforcement learning, and attention-based architectures. Each paradigm embodies a fundamentally different hypothesis about what learnable structure might exist; the uniform null across all seven paradigms is consistent with the absence of learnable structure at the nonce–header interface, though it does not constitute a proof of impossibility.

## Result 4: Real Bitcoin Nonces Exhibit Non-Cryptographic Behavioral Structure

**Claim.** In the historical Bitcoin blockchain (943,488 block headers), 14 of 32 nonce bits are statistically predictable from header content. This structure is entirely attributable to mining-pool software conventions and carries zero information about hash validity.

**Evidence.**
- **Prediction**: An MLP (AutoResearch-selected: 2 layers × 1024 units, ReLU) achieves 51.4% overall nonce-bit accuracy on a chronological test set (*n* = 94,349), with bit 0 at 65.5% (*p* ≈ 0).
- **Temporal analysis**: Nonce bit biases are era-dependent. Bit 0 mean: 0.374 (blocks 0–189K), 0.460 (189K–377K), 0.301 (755K–943K). The structure reflects changes in mining hardware and pool software across [Bitcoin's mining eras](https://en.wikipedia.org/wiki/Bitcoin_network#Mining).
- **Feature importance**: Permutation importance shows the MLP uses block version (15.6%), previous block hash (14.8%), and difficulty target (12.4%) — all mining-era identifiers, not cryptographic inputs — to predict nonce patterns characteristic of each era.
- **Validity test**: MLP-predicted nonces produce hashes with mean 0.99 leading zeros (Bitcoin format), statistically identical to random nonces (1.01). Hamming distance between predicted and actual nonces: 15.39 (expected under independence: 16.0).
- **Mutual information**: 2,577 of 19,456 header–nonce bit pairs (13.2%) show significant mutual information (*z* > 3.5, Bonferroni, *N* = 943,488). Strongest: header bit 571 (difficulty field) ↔ nonce bit 0 (MI = 0.0103 bits).

**Implication.** Real blockchain data contains learnable structure that can mislead ML analyses if proper controls (re-mining with random nonce starts, shuffled-data baselines) are not applied. The structure reveals mining-pool operational patterns but provides no cryptographic advantage.

## Result 5: No Intermediate SHA-256 State Predicts Hash Validity

**Claim.** No individual bit of the 256-bit SHA-256 compression state at any round from 4 to 63 correlates with the validity of the final SHA-256d output, after Bonferroni correction.

**Evidence.** SHA-256 processes its input through 64 sequential rounds of a [compression function](https://en.wikipedia.org/wiki/One-way_compression_function), each transforming an 8-register (256-bit) working state. If the state at some intermediate round *R* were informative about whether the final hash will have leading zeros, a miner could inspect the state at round *R* and abort unpromising nonces early, saving the computation of rounds *R*+1 through 64.

To test this, 50 million random nonces were evaluated for a fixed header. The full 256-bit compression state was recorded at six checkpoint rounds (4, 6, 8, 60, 62, 63) — spanning from just after the nonce enters (round 4) to the final round (round 63). For each of 768 state bits (6 checkpoints × 256 bits), the conditional probability of hash validity (≥ 8 leading zeros, Bitcoin format) was computed separately for samples where the state bit was 0 vs. 1. Under the null hypothesis (no predictive value), these probabilities should be equal.

Maximum observed probability ratio across all 768 tests: 1.013 (bit 2 of the round-63 state). No bit achieved significance after Bonferroni correction (*z* > 3.5 required for 256 simultaneous tests per checkpoint). With 50 million samples, even a 0.1% conditional probability difference would have been detectable.

**Implication.** No evidence was found that would enable early termination of the compression function based on intermediate state inspection at any round tested, including the very last. This is consistent with SHA-256's [avalanche effect](https://en.wikipedia.org/wiki/Avalanche_effect): bit-level dependency tracking shows that every output bit of the compression state depends on every input bit by approximately round 8 (verified empirically via symbolic dependency propagation). The final hash value is computed by adding the round-64 working state to the input state, a step that depends on all state registers simultaneously.

## Result 6: Valid Nonces Are Uniformly Distributed

**Claim.** For any given header, the set of valid nonces is consistent with a uniform distribution over [0, 2³²), with no detectable spatial clustering. Furthermore, no tested header configuration produces systematically more or fewer valid nonces than any other.

**Evidence.** Four sub-experiments test different aspects of nonce-space structure:

*Uniformity test.* If valid nonces clustered in certain regions of the 32-bit space, a miner could focus search on those regions. For 10 headers, 10 million random nonces were sampled at difficulty 12 (≈ 1/4096 valid). The nonce space was partitioned into 1000 equal bins and the number of valid nonces per bin compared to the uniform expectation. [Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test): χ² = 962, df = 999 (critical value at 99% = 1172). The distribution is indistinguishable from uniform.

*Near-miss test.* A [hill-climbing](https://en.wikipedia.org/wiki/Hill_climbing) strategy would work if nonces producing *k*+1 leading zeros tended to be "near" (in [Hamming distance](https://en.wikipedia.org/wiki/Hamming_distance)) nonces producing *k* leading zeros — i.e., if the hash-quality landscape had gradient structure. For 5 headers, 5 million nonces were evaluated at difficulties 4–12. For each (*k*+1)-zero nonce found, the Hamming distance to the nearest *k*-zero nonce was measured and compared to the distance from a random nonce to the nearest *k*-zero nonce. The ratio was 1.00 ± 0.03 across 40 (header × difficulty) test cases. Better nonces are not spatially closer to good nonces than random nonces are — the quality landscape is flat.

*Cross-header variance.* If some headers were inherently "easier" to mine (having more valid nonces than average), miners could prioritize them. For 200 headers with 1 million nonces each at difficulty 8, the variance of valid-nonce counts (3,520) was *below* the [Poisson](https://en.wikipedia.org/wiki/Poisson_distribution) expectation (3,906). SHA-256 distributes valid nonces more uniformly across headers than even a perfectly random function would.

*Timestamp variation.* Bitcoin miners may vary the block timestamp within a ±2-hour window allowed by consensus rules. For 10 headers with 100 timestamp variants each (spanning the full window) and 100,000 nonces per variant, the variance ratio to Poisson was 0.97 ± 0.07. Timestamp choice does not affect mining difficulty.

**Implication.** Within the tested sample sizes and difficulty levels, the nonce-validity landscape shows no detectable structure: no spatial clustering, no quality gradient, no easy headers, and no exploitable variation across the miner-controllable search dimensions (nonce, timestamp, version bits). Search strategies that rely on "nearby" nonces having correlated quality — including [hill-climbing](https://en.wikipedia.org/wiki/Hill_climbing), [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing), and local search — would find no advantage over random sampling under these conditions.

## Result 7: Evolutionary Search Shows Evaluation Advantage but Practical Parity

**Claim.** [CMA-ES](https://en.wikipedia.org/wiki/CMA-ES) 32-dimensional bitwise nonce search requires 3.8× fewer hash evaluations than random search to reach the same maximum leading-zero count, but the computational overhead of population management reduces the net wall-clock advantage to ≈1.1×.

**Evidence.** For 3 headers, 4 billion hash evaluations were budgeted per method (C implementation, single-threaded):

| Metric | Random search | CMA-ES 32D |
|---|---|---|
| Hash rate | 2.8 Mh/s | 0.8 Mh/s |
| Best LZ achieved (4B budget) | 30–32 | 30–32 |
| Evals to first reach LZ = 30 | 1.26 billion | 329 million |
| Evals to first reach LZ = 8 | 433 | 56 |

The evaluation advantage (3.8× at LZ = 30, 7.7× at LZ = 8) reflects CMA-ES's superior exploration of the combinatorial bit-space through covariance adaptation. However, the per-evaluation overhead (Gaussian sampling, population selection, mean update) reduces the effective hash rate to 29% of random search, nearly canceling the evaluation advantage in wall-clock time.

**Implication.** CMA-ES reduces the number of hash evaluations needed through adaptive sampling of the 32-bit combinatorial space, but this advantage is not specific to SHA-256 — CMA-ES would show similar evaluation advantages on any function mapping 32-bit inputs to a scalar quality metric. The per-evaluation overhead of the evolutionary algorithm nearly cancels the evaluation savings in wall-clock time. For [ASIC](https://en.wikipedia.org/wiki/Application-specific_integrated_circuit) miners where hash throughput is the bottleneck and per-hash cost approaches zero, CMA-ES provides no practical benefit.

## Result 8: Message Schedule Sparsity Is Minimal

**Claim.** Under the standard Bitcoin Block 2 layout, only 2 of 48 expanded message schedule words (W₁₆ and W₁₇) are nonce-independent. All others depend on W₃ (the nonce word) through the message schedule recurrence.

**Evidence.** Dependency graph analysis of the recurrence W*ₜ* = σ₁(W*ₜ*₋₂) + W*ₜ*₋₇ + σ₀(W*ₜ*₋₁₅) + W*ₜ*₋₁₆ shows W₁₈ is the first expanded word depending on W₃. By W₂₆, every expanded word depends on all 16 input words. The nonce enters the compression function at round 3 (via W₃) and achieves full state-bit saturation by round 8.

**Implication.** Schedule precomputation beyond the midstate saves only 4.2% of the message schedule computation — negligible compared to the midstate's ≈2× savings. No novel schedule-based optimization exists beyond what AsicBoost already exploits.

## Result 9: AND-Gate Density as a Heuristic Measure of Inversion Difficulty

**Claim.** Full SHA-256 (64 rounds) contains approximately 48,128 AND-equivalent Boolean gates, of which 73.7% arise from modular addition carry chains. This gate count provides a heuristic upper bound on the branching factor faced by any backward-reasoning algorithm.

**Evidence.** The gate count was determined by analyzing the SHA-256 specification at the Boolean circuit level:

- Per round: 608 AND-equivalent gates (rounds 0–15) or 800 AND-equivalent gates (rounds 16–63, which include message schedule expansion)
- [Modular addition](https://en.wikipedia.org/wiki/Modular_arithmetic) (mod 2³²) contributes 73.7% of all AND gates through [carry chain](https://en.wikipedia.org/wiki/Carry_(arithmetic)) propagation
- The [Ch](https://en.wikipedia.org/wiki/SHA-2#Pseudocode) and [Maj](https://en.wikipedia.org/wiki/SHA-2#Pseudocode) bitwise functions contribute the remaining 26.3%
- Mining-specific constant propagation (12 of 16 Block 2 words are padding constants) eliminates only ≈11% of gates

As an informal estimate: each AND gate whose output is 0 creates a 3-way input ambiguity, suggesting an astronomically large worst-case branching factor (on the order of 3^{48,000}) for full backward traversal. This is not a formal complexity bound — actual solvers exploit structural dependencies that reduce the effective branching — but it provides intuition for why backward reasoning through SHA-256 is difficult.

**Implication.** The high AND-gate density, particularly from carry chains, explains the empirical observation that SAT solvers, algebraic methods, and meet-in-the-middle approaches all struggle with SHA-256 inversion. The carry chains in modular addition are the dominant source of non-linearity and the primary obstacle to efficient backward reasoning.

## Result 10: Three Structural Barriers to SAT-Based Mining

**Claim.** SHA-256's internal structure creates three distinct barriers that, in combination, appear to prevent [CDCL](https://en.wikipedia.org/wiki/Conflict-driven_clause_learning) SAT solvers from efficiently solving mining instances. Each barrier was identified theoretically and confirmed empirically.

**Evidence.**

*Parity barrier.* SHA-256 contains over 95,000 XOR operations. When encoded into [CNF](https://en.wikipedia.org/wiki/Conjunctive_normal_form) via the standard [Tseitin transformation](https://en.wikipedia.org/wiki/Tseitin_transformation), these create parity constraints. It is known that parity constraints require exponential-size [resolution](https://en.wikipedia.org/wiki/Resolution_(logic)) proofs (Ben-Sasson & Wigderson, 2001), and CDCL operates within the resolution proof system (Pipatsrisawat & Darwiche, 2011). Empirically, [CryptoMiniSat](https://github.com/msoos/cryptominisat)'s native XOR/Gaussian elimination — which bypasses the Tseitin encoding for XOR clauses — provided only 1.3× speedup at 8 leading zeros and was 2.9× *slower* at 12 leading zeros, confirming that resolving the parity barrier alone is insufficient.

*Carry-chain barrier.* The 48,128 AND-equivalent gates (73.7% from [modular addition](https://en.wikipedia.org/wiki/Modular_arithmetic) carry chains) create exponential backward branching when reasoning from outputs to inputs. A custom [IPASIR-UP](https://github.com/arminbiere/cadical) carry-chain propagator was implemented to supplement CNF with domain-specific carry deductions. Result: correct solutions were found, but performance was 0.77–0.81× (slower than the baseline solver), because the carry deductions were redundant with the solver's own unit propagation.

*Barrier coupling.* XOR and addition operations are interleaved in every SHA-256 round — for example, T₁ = H + Σ₁(E) + Ch(E,F,G) + K*ₜ* + W*ₜ*, where Σ₁ involves three XOR-based rotations and + denotes carry-chain addition. The empirical evidence shows that resolving either barrier independently provides no improvement: the parity barrier and carry-chain barrier must be addressed simultaneously, but no known proof system handles both efficiently.

**Implication.** The combination of interleaved XOR and carry-chain structure makes SHA-256 empirically intractable for all SAT-based approaches tested, including CDCL, XOR-augmented CDCL, IPASIR-UP propagation, carry-save adder encodings, [cube-and-conquer](https://en.wikipedia.org/wiki/Cube-and-conquer) partitioning, and [Cutting Planes](https://en.wikipedia.org/wiki/Cutting-plane_method) solvers. Whether a formal proof of exponential resolution complexity for SHA-256 mining can be constructed remains an open question in [proof complexity](https://en.wikipedia.org/wiki/Proof_complexity).

## Result 11: Quantum Mining Reduces to Grover's Bound

**Claim.** No known quantum algorithm can mine SHA-256 faster than [Grover's algorithm](https://en.wikipedia.org/wiki/Grover%27s_algorithm), which provides a quadratic speedup (O(2^{k/2}) vs O(2^k)). The structural analysis in Results 5 and 6 is consistent with the conditions under which Grover's bound is believed to be tight.

**Evidence.** Grover's algorithm achieves optimal speedup for [unstructured search](https://en.wikipedia.org/wiki/Grover%27s_algorithm#Optimality) (Bennett et al., 1997; Zalka, 1999). Better-than-Grover quantum speedups exist only for problems with exploitable structure — specifically, spatial structure enabling [quantum walks](https://en.wikipedia.org/wiki/Quantum_walk) or algebraic structure enabling [hidden subgroup](https://en.wikipedia.org/wiki/Hidden_subgroup_problem) algorithms. Our investigation tested both preconditions:

- *No spatial structure*: Result 6 shows valid nonces are uniformly distributed with no clustering or near-miss gradient (chi-squared passes uniformity, Hamming distances to near-misses equal random baseline). The search graph has no exploitable topology for quantum walk speedups.
- *No algebraic structure*: Result 5 shows intermediate compression states carry zero information about hash validity, and Result 3 shows no learnable correlation between header and nonce under any of seven ML paradigms. The mapping from nonce to hash is computationally indistinguishable from a [random oracle](https://en.wikipedia.org/wiki/Random_oracle).

Both conditions are consistent with SHA-256 mining being an *unstructured* search problem, for which Grover's bound is known to be optimal (Bennett et al., 1997).

**Implication.** Even a fault-tolerant quantum computer would achieve at most a quadratic speedup over classical mining — reducing 2^*k* hash evaluations to O(2^{k/2}) quantum evaluations. At current Bitcoin difficulty (~80 leading zeros), this reduces the search from ≈2⁸⁰ to ≈2⁴⁰ — significant, but not a break of the algorithm. Furthermore, each "quantum hash evaluation" involves implementing full SHA-256d as a reversible quantum circuit, which requires thousands of logical qubits and is far beyond current hardware capabilities (Aggarwal et al., 2018).

## Result 12: Double SHA-256 Exhibits Super-Linear Coupling

**Claim.** Bitcoin's double hashing (SHA-256d = SHA-256(SHA-256(x))) incurs a super-linear performance penalty for SAT solvers beyond the 2× increase in circuit size.

**Evidence.** The double SHA-256 CNF encoding has 318,384 variables and 1,031,528 clauses — exactly 2× the single-hash encoding. However, SAT solving time exhibits super-linear coupling:

| Difficulty (leading zeros) | Single SHA-256 (MiniSat) | Double SHA-256 (MiniSat) | Ratio |
|---|---|---|---|
| 8 | 0.9s | 14.1s | 15.7× |
| 12 | 15.0s | 52.3s | 3.5× |

The penalty is greatest at low difficulty: with 8 leading zeros, many nonces satisfy the inner hash but fail the outer hash, causing wasted search effort. At higher difficulty, the inner-hash constraint is already tight, so the second hash adds proportionally less overhead. At reduced round counts (e.g., 14 rounds), the coupling penalty reaches 76.5× because the weaker hash admits far more "false positive" inner-hash solutions.

**Implication.** The double-hash construction provides security beyond simply doubling the circuit size. The interaction between the two hash evaluations creates a coupling effect that disproportionately increases the difficulty of formal analysis.

---

## Relation to Prior Work

The best known cryptanalytic attacks on SHA-256 are preimage attacks on up to 52 of 64 rounds (Aoki & Sasaki, 2009; Khovratovich et al., 2012), using [meet-in-the-middle](https://en.wikipedia.org/wiki/Meet-in-the-middle_attack) techniques. These do not apply to mining, which requires a partial preimage (leading zeros) of the *double* hash, not inversion of a single compression.

Courtois et al. (2014) identified algebraic optimizations reducing the mining computation by a constant factor, subsequently implemented in AsicBoost. Our results show no further constant-factor improvements remain.

SAT-based mining was explored by [Heusser (2013)](https://jheusser.github.io/2013/02/03/satcoin.html) and [Nejati et al. (2020)](https://arxiv.org/abs/2005.13415) with CDCL(Crypto) solvers. Our results extend this to CryptoMiniSat with XOR clauses, carry-save encodings, cube-and-conquer, and Cutting Planes proof systems — all of which fail to improve upon brute force.

Approximate mining (Vilim et al., 2016) explored trading hash accuracy for throughput by truncating the SHA-256 computation. Our Result 5 shows that no intermediate round carries information about hash validity, ruling out informed truncation.

[Grover's algorithm](https://en.wikipedia.org/wiki/Grover%27s_algorithm) provides a provable quadratic speedup (O(2^{k/2}) vs O(2^k)) for unstructured search, but requires a fault-tolerant [quantum computer](https://en.wikipedia.org/wiki/Quantum_computing) capable of implementing SHA-256 as a quantum circuit — not achievable with current technology (Aggarwal et al., 2018).

---

## Reproducibility

All experiments were conducted on consumer-grade hardware: an NVIDIA RTX 4070 Ti GPU (12 GB VRAM) with CUDA 12.1, an Apple M4 Mac Mini for analysis, and GCC 13.3 for C programs. Bitcoin block headers were obtained via the Electrum protocol (943,488 headers, covering blocks 0 through ≈941,500 as of March 2026). Re-mined datasets used random nonce starting positions to eliminate miner behavioral bias.

Source code, generated datasets, and links to external data sources will be published in the accompanying GitHub repository.

---

## References

1. Hanke, T. (2016). "AsicBoost — A Speedup for Bitcoin Mining." [arXiv:1604.00575](https://arxiv.org/abs/1604.00575).
2. Courtois, N., Grajek, M., Naik, R. (2014). "Optimizing SHA256 in Bitcoin Mining." *CANS 2014*, Springer LNCS 8813. [Link](https://link.springer.com/chapter/10.1007/978-3-662-44893-9_12).
3. Aoki, K., Sasaki, Y. (2009). "Preimages for Step-Reduced SHA-2." [ePrint 2009/477](https://eprint.iacr.org/2009/477).
4. Khovratovich, D., Rechberger, C., Savelieva, A. (2012). "Bicliques for Preimages." FSE 2012.
5. Benadjila, R., Music, L., Parriaux, J. (2019). "Using fuzzy bits and neural networks to partially invert few rounds of some cryptographic hash functions." [arXiv:1901.02438](https://arxiv.org/abs/1901.02438).
6. Vilim, M., Duwe, H., Kumar, R. (2016). "Approximate Bitcoin Mining." DAC 2016.
7. Wolfram, S. (2002). *A New Kind of Science*. Wolfram Media.
8. Heusser, J. (2013). "SAT Solving — An Alternative to Brute Force Bitcoin Mining." [Blog post](https://jheusser.github.io/2013/02/03/satcoin.html).
9. Nejati, S., et al. (2020). "CDCL(Crypto) SAT Solvers for Cryptanalysis." [arXiv:2005.13415](https://arxiv.org/abs/2005.13415).
10. Aggarwal, D., et al. (2018). "Quantum Attacks on Bitcoin, and How to Protect Against Them." *Ledger*, 3.
11. Soos, M. (2009–present). CryptoMiniSat. [GitHub](https://github.com/msoos/cryptominisat).
12. Ben-Sasson, E., Wigderson, A. (2001). "Short proofs are narrow — resolution made simple." *Journal of the ACM*, 48(2), 149–169.
13. Pipatsrisawat, K., Darwiche, A. (2011). "On the power of clause-learning SAT solvers as resolution engines." *Artificial Intelligence*, 175(2), 512–525.
14. Bennett, C.H., Bernstein, E., Brassard, G., Vazirani, U. (1997). "Strengths and Weaknesses of Quantum Computing." *SIAM Journal on Computing*, 26(5), 1510–1523.
15. Zalka, C. (1999). "Grover's quantum searching algorithm is optimal." *Physical Review A*, 60(4), 2746.
16. Karpathy, A. (2026). "AutoResearch: AI agents running research on single-GPU nanochat training automatically." [GitHub](https://github.com/karpathy/autoresearch).
