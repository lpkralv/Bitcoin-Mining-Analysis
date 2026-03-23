/*
 * partial_eval_test.c - Test partial SHA-256 evaluation for early nonce rejection
 *
 * Tests whether intermediate states during SHA-256 compression can predict
 * whether a nonce will produce a hash with sufficient leading zeros.
 *
 * Usage: echo "<hex_stub>" | ./partial_eval_test <difficulty> <sample_size>
 *
 * Compile: gcc -O3 -o partial_eval_test partial_eval_test.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* SHA-256 constants */
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

/* SHA-256 initial hash values */
static const uint32_t H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};

/* Checkpoint rounds to analyze */
static const int CHECKPOINTS[] = {4, 8, 16, 32, 48, 56, 60, 62, 63};
static const int NUM_CHECKPOINTS = sizeof(CHECKPOINTS) / sizeof(CHECKPOINTS[0]);

/* Statistics tracking */
struct checkpoint_stats {
    uint64_t count_cond0_total;   /* Total nonces with condition=0 at this round */
    uint64_t count_cond1_total;   /* Total nonces with condition=1 at this round */
    uint64_t count_cond0_valid;   /* Valid nonces with condition=0 at this round */
    uint64_t count_cond1_valid;   /* Valid nonces with condition=1 at this round */
};

/* Fast xorshift PRNG state */
static uint64_t xorshift_state = 1;

/* Rotate right */
static inline uint32_t ror(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

/* Fast xorshift64 PRNG */
static inline uint32_t xorshift32(void) {
    xorshift_state ^= xorshift_state << 13;
    xorshift_state ^= xorshift_state >> 7;
    xorshift_state ^= xorshift_state << 17;
    return (uint32_t)xorshift_state;
}

/* Convert hex character to value */
static int hex_char_to_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Parse hex string into bytes */
static int parse_hex(const char *hex, uint8_t *out, int max_bytes) {
    int len = strlen(hex);
    if (len % 2 != 0) return -1;

    int bytes = len / 2;
    if (bytes > max_bytes) return -1;

    for (int i = 0; i < bytes; i++) {
        int h = hex_char_to_val(hex[2*i]);
        int l = hex_char_to_val(hex[2*i+1]);
        if (h < 0 || l < 0) return -1;
        out[i] = (h << 4) | l;
    }

    return bytes;
}

/* SHA-256 compression with checkpoint recording */
static void sha256_compress_with_checkpoints(const uint32_t *W, uint32_t *hash,
                                           uint32_t checkpoints[NUM_CHECKPOINTS]) {
    uint32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3];
    uint32_t e = hash[4], f = hash[5], g = hash[6], h = hash[7];

    int checkpoint_idx = 0;

    for (int t = 0; t < 64; t++) {
        uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K[t] + W[t];
        uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;

        /* Record checkpoint if this is a checkpoint round */
        if (checkpoint_idx < NUM_CHECKPOINTS && t == CHECKPOINTS[checkpoint_idx] - 1) {
            checkpoints[checkpoint_idx] = a;  /* Record A register value */
            checkpoint_idx++;
        }
    }

    hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
    hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
}

/* Standard SHA-256 compression (for outer hash) */
static void sha256_compress(const uint32_t *W, uint32_t *hash) {
    uint32_t a = hash[0], b = hash[1], c = hash[2], d = hash[3];
    uint32_t e = hash[4], f = hash[5], g = hash[6], h = hash[7];

    for (int t = 0; t < 64; t++) {
        uint32_t S1 = ror(e, 6) ^ ror(e, 11) ^ ror(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + K[t] + W[t];
        uint32_t S0 = ror(a, 2) ^ ror(a, 13) ^ ror(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    hash[0] += a; hash[1] += b; hash[2] += c; hash[3] += d;
    hash[4] += e; hash[5] += f; hash[6] += g; hash[7] += h;
}

/* Prepare message schedule W from 512-bit block */
static void prepare_message_schedule(const uint8_t *block, uint32_t *W) {
    /* First 16 words from block (big-endian) */
    for (int t = 0; t < 16; t++) {
        W[t] = ((uint32_t)block[4*t] << 24) | ((uint32_t)block[4*t+1] << 16) |
               ((uint32_t)block[4*t+2] << 8) | (uint32_t)block[4*t+3];
    }

    /* Extend to 64 words */
    for (int t = 16; t < 64; t++) {
        uint32_t s0 = ror(W[t-15], 7) ^ ror(W[t-15], 18) ^ (W[t-15] >> 3);
        uint32_t s1 = ror(W[t-2], 17) ^ ror(W[t-2], 19) ^ (W[t-2] >> 10);
        W[t] = W[t-16] + s0 + W[t-7] + s1;
    }
}

/* Count leading zero bits in hash (Bitcoin format) */
static int count_leading_zeros(const uint32_t *hash) {
    int zeros = 0;

    for (int i = 0; i < 8; i++) {
        uint32_t word = hash[i];
        if (word == 0) {
            zeros += 32;
        } else {
            /* Count leading zeros in this word */
            zeros += __builtin_clz(word);
            break;
        }
    }

    return zeros;
}

/* Test a single nonce */
static void test_nonce(const uint8_t *stub, uint32_t nonce, int difficulty,
                      struct checkpoint_stats *stats, uint32_t *midstate) {
    uint8_t block2[64];
    uint32_t W[64];
    uint32_t hash[8];
    uint32_t checkpoints[NUM_CHECKPOINTS];

    /* Prepare block 2: nonce + padding */
    memset(block2, 0, 64);

    /* Copy last 12 bytes of stub */
    memcpy(block2, stub + 64, 12);

    /* Add nonce (little-endian) */
    block2[12] = nonce & 0xff;
    block2[13] = (nonce >> 8) & 0xff;
    block2[14] = (nonce >> 16) & 0xff;
    block2[15] = (nonce >> 24) & 0xff;

    /* SHA-256 padding */
    block2[16] = 0x80;
    /* Zeros from 17-55 already set by memset */

    /* Length in bits: 640 (80 bytes * 8) - big-endian */
    block2[56] = 0x00; block2[57] = 0x00; block2[58] = 0x00; block2[59] = 0x00;
    block2[60] = 0x00; block2[61] = 0x00; block2[62] = 0x02; block2[63] = 0x80;

    /* Compute inner hash with checkpoint recording */
    memcpy(hash, midstate, 32);
    prepare_message_schedule(block2, W);
    sha256_compress_with_checkpoints(W, hash, checkpoints);

    /* Compute outer hash */
    uint32_t outer_hash[8];
    memcpy(outer_hash, H0, 32);

    /* Prepare outer block: inner hash + padding */
    uint8_t outer_block[64];
    memset(outer_block, 0, 64);

    /* Inner hash as input (big-endian) */
    for (int i = 0; i < 8; i++) {
        outer_block[4*i] = (hash[i] >> 24) & 0xff;
        outer_block[4*i+1] = (hash[i] >> 16) & 0xff;
        outer_block[4*i+2] = (hash[i] >> 8) & 0xff;
        outer_block[4*i+3] = hash[i] & 0xff;
    }

    /* Padding for 256-bit input */
    outer_block[32] = 0x80;
    outer_block[56] = 0x00; outer_block[57] = 0x00; outer_block[58] = 0x00; outer_block[59] = 0x00;
    outer_block[60] = 0x00; outer_block[61] = 0x00; outer_block[62] = 0x01; outer_block[63] = 0x00;

    prepare_message_schedule(outer_block, W);
    sha256_compress(W, outer_hash);

    /* Check if hash meets difficulty requirement */
    int leading_zeros = count_leading_zeros(outer_hash);
    int is_valid = (leading_zeros >= difficulty);

    /* Update statistics for each checkpoint */
    for (int i = 0; i < NUM_CHECKPOINTS; i++) {
        /* Use MSB of A register as the condition */
        int condition = (checkpoints[i] >> 31) & 1;  /* 1 if MSB set, 0 if clear */

        if (condition == 0) {
            stats[i].count_cond0_total++;
            if (is_valid) stats[i].count_cond0_valid++;
        } else {
            stats[i].count_cond1_total++;
            if (is_valid) stats[i].count_cond1_valid++;
        }
    }
}

/* Compute midstate from first block */
static void compute_midstate(const uint8_t *stub, uint32_t *midstate) {
    uint8_t block1[64];
    uint32_t W[64];

    /* Copy first 64 bytes as block 1 */
    memcpy(block1, stub, 64);

    /* Initialize with SHA-256 IV */
    memcpy(midstate, H0, 32);

    /* Compress block 1 */
    prepare_message_schedule(block1, W);
    sha256_compress(W, midstate);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: echo \"<hex_stub>\" | %s <difficulty> <sample_size>\n", argv[0]);
        fprintf(stderr, "Example: echo \"00010203...\" | %s 4 1000000\n", argv[0]);
        return 1;
    }

    int difficulty = atoi(argv[1]);
    uint64_t sample_size = strtoull(argv[2], NULL, 10);

    if (difficulty < 1 || difficulty > 256) {
        fprintf(stderr, "Error: difficulty must be 1-256\n");
        return 1;
    }

    if (sample_size < 1000) {
        fprintf(stderr, "Error: sample_size must be >= 1000\n");
        return 1;
    }

    /* Read hex stub from stdin */
    char hex_input[256];
    if (!fgets(hex_input, sizeof(hex_input), stdin)) {
        fprintf(stderr, "Error: failed to read hex stub from stdin\n");
        return 1;
    }

    /* Remove newline */
    char *newline = strchr(hex_input, '\n');
    if (newline) *newline = '\0';

    /* Parse hex stub */
    uint8_t stub[80];
    int stub_len = parse_hex(hex_input, stub, 80);
    if (stub_len != 76) {
        fprintf(stderr, "Error: expected 76-byte (152 hex char) stub, got %d bytes\n", stub_len);
        return 1;
    }

    printf("Partial SHA-256 Evaluation Test\n");
    printf("================================\n");
    printf("Difficulty: %d leading zero bits\n", difficulty);
    printf("Sample size: %llu nonces\n", (unsigned long long)sample_size);
    printf("Analyzing checkpoints at rounds: ");
    for (int i = 0; i < NUM_CHECKPOINTS; i++) {
        printf("%d", CHECKPOINTS[i]);
        if (i < NUM_CHECKPOINTS - 1) printf(", ");
    }
    printf("\n\n");

    /* Compute midstate */
    uint32_t midstate[8];
    compute_midstate(stub, midstate);

    /* Initialize statistics */
    struct checkpoint_stats stats[NUM_CHECKPOINTS];
    memset(stats, 0, sizeof(stats));

    /* Initialize PRNG */
    xorshift_state = time(NULL) ^ 0x123456789ABCDEFULL;

    /* Test nonces */
    clock_t start_time = clock();
    printf("Testing nonces... ");
    fflush(stdout);

    for (uint64_t i = 0; i < sample_size; i++) {
        uint32_t nonce = xorshift32();
        test_nonce(stub, nonce, difficulty, stats, midstate);

        /* Progress indicator */
        if ((i + 1) % (sample_size / 20) == 0) {
            printf(".");
            fflush(stdout);
        }
    }

    clock_t end_time = clock();
    double elapsed = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\nCompleted in %.2f seconds (%.0f nonces/sec)\n\n",
           elapsed, sample_size / elapsed);

    /* Analyze results */
    printf("Results:\n");
    printf("Round  P(valid|MSB=0)  P(valid|MSB=1)    Ratio  Predictive?\n");
    printf("-----  --------------  --------------  -------  -----------\n");

    for (int i = 0; i < NUM_CHECKPOINTS; i++) {
        double p_valid_cond0 = 0.0, p_valid_cond1 = 0.0;
        double ratio = 1.0;
        const char *predictive = "NO";

        if (stats[i].count_cond0_total > 0) {
            p_valid_cond0 = (double)stats[i].count_cond0_valid / stats[i].count_cond0_total;
        }

        if (stats[i].count_cond1_total > 0) {
            p_valid_cond1 = (double)stats[i].count_cond1_valid / stats[i].count_cond1_total;
        }

        if (p_valid_cond1 > 0.0) {
            ratio = p_valid_cond0 / p_valid_cond1;
        }

        /* Consider predictive if ratio differs significantly from 1.0 */
        if (ratio < 0.95 || ratio > 1.05) {
            predictive = "YES";
        }

        printf("%5d     %8.6f       %8.6f    %7.3f       %s\n",
               CHECKPOINTS[i], p_valid_cond0, p_valid_cond1, ratio, predictive);
    }

    printf("\nInterpretation:\n");
    printf("- MSB=0 means the most significant bit of register A is clear\n");
    printf("- MSB=1 means the most significant bit of register A is set\n");
    printf("- Ratio significantly != 1.0 indicates predictive value\n");
    printf("- Early rounds with predictive value enable optimization\n");

    /* Calculate overall statistics */
    uint64_t total_valid = 0;
    for (int i = 0; i < NUM_CHECKPOINTS; i++) {
        total_valid += stats[i].count_cond0_valid + stats[i].count_cond1_valid;
        break; /* Just need one checkpoint for total */
    }

    printf("\nOverall statistics:\n");
    printf("Total valid nonces: %llu / %llu (%.6f%%)\n",
           (unsigned long long)total_valid, (unsigned long long)sample_size,
           100.0 * total_valid / sample_size);
    printf("Expected for difficulty %d: %.6f%%\n",
           difficulty, 100.0 / (1ULL << difficulty));

    return 0;
}