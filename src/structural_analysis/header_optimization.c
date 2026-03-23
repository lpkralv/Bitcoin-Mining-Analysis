/*
 * Bitcoin header optimization analyzer.
 *
 * Tests whether different Bitcoin block headers produce varying numbers
 * of valid nonces, indicating potential structure in the hash function.
 *
 * Reads 76-byte header stubs (hex) from stdin, one per line.
 * For each header, samples random nonces and counts how many produce
 * SHA-256d hashes with >= K leading zero bits (Bitcoin reversed format).
 *
 * Usage: cat stubs.txt | ./header_optimization <difficulty> <sample_size>
 * Example: ./header_optimization 8 1000000
 *
 * Compile: gcc -O3 -o header_optimization header_optimization.c -lm
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <math.h>

/* SHA-256 constants and macros from reference implementation */
static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

static const uint32_t H0[8] = {
    0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
    0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19
};

#define ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(e,f,g) (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c) (((a)&(b))^((a)&(c))^((b)&(c)))
#define S0(a) (ROTR(a,2)^ROTR(a,13)^ROTR(a,22))
#define S1(e) (ROTR(e,6)^ROTR(e,11)^ROTR(e,25))
#define s0(x) (ROTR(x,7)^ROTR(x,18)^((x)>>3))
#define s1(x) (ROTR(x,17)^ROTR(x,19)^((x)>>10))

/* Fast xorshift PRNG - much faster than rand() */
typedef struct {
    uint32_t state;
} xorshift_t;

static inline uint32_t xorshift_next(xorshift_t *rng) {
    uint32_t x = rng->state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng->state = x;
    return x;
}

static void xorshift_seed(xorshift_t *rng, uint32_t seed) {
    rng->state = seed ? seed : 1;
}

/* SHA-256 compression function */
static void compress(uint32_t h[8], const uint32_t m[16]) {
    uint32_t W[64];
    int i;

    /* Prepare message schedule */
    for(i = 0; i < 16; i++) W[i] = m[i];
    for(i = 16; i < 64; i++) W[i] = s1(W[i-2]) + W[i-7] + s0(W[i-15]) + W[i-16];

    /* Initialize working variables */
    uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
    uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

    /* Main compression loop */
    for(i = 0; i < 64; i++){
        uint32_t t1 = hh + S1(e) + CH(e,f,g) + K[i] + W[i];
        uint32_t t2 = S0(a) + MAJ(a,b,c);
        hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }

    /* Add to hash state */
    h[0] += a; h[1] += b; h[2] += c; h[3] += d;
    h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
}

/* Convert bytes to 32-bit words (big-endian) */
static void bytes_to_words(const uint8_t *b, uint32_t *w, int n) {
    for(int i = 0; i < n; i++)
        w[i] = ((uint32_t)b[4*i] << 24) | ((uint32_t)b[4*i+1] << 16) |
               ((uint32_t)b[4*i+2] << 8) | b[4*i+3];
}

/* Hex decoding utilities */
static int hex_val(char c) {
    if(c >= '0' && c <= '9') return c - '0';
    if(c >= 'a' && c <= 'f') return c - 'a' + 10;
    if(c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int hex_decode(const char *hex, uint8_t *out, int max_bytes) {
    int i = 0;
    while(hex[2*i] && hex[2*i+1] && i < max_bytes) {
        int hi = hex_val(hex[2*i]), lo = hex_val(hex[2*i+1]);
        if(hi < 0 || lo < 0) return -1;
        out[i] = (hi << 4) | lo;
        i++;
    }
    return i;
}

/* SHA-256d: double SHA-256 hash */
static void sha256d(uint32_t result[8], const uint32_t midstate[8],
                    const uint32_t block2[16]) {
    uint32_t inner[8];
    memcpy(inner, midstate, 32);
    compress(inner, block2);

    /* Prepare outer hash input: 32-byte inner hash, padded to 64 bytes */
    uint8_t buf[64];
    for(int i = 0; i < 8; i++){
        buf[4*i] = (inner[i] >> 24) & 0xFF;
        buf[4*i+1] = (inner[i] >> 16) & 0xFF;
        buf[4*i+2] = (inner[i] >> 8) & 0xFF;
        buf[4*i+3] = inner[i] & 0xFF;
    }
    buf[32] = 0x80;
    memset(buf + 33, 0, 23);
    /* Length = 256 bits = 0x100 */
    buf[62] = 0x01; buf[63] = 0x00;

    uint32_t om[16];
    bytes_to_words(buf, om, 16);
    memcpy(result, H0, 32);
    compress(result, om);
}

/* Count leading zero bits in Bitcoin format (reversed byte order) */
static int count_leading_zeros_bitcoin(const uint32_t hash[8]) {
    /* Convert hash to bytes in big-endian order */
    uint8_t bytes[32];
    for(int i = 0; i < 8; i++){
        bytes[4*i] = (hash[i] >> 24) & 0xFF;
        bytes[4*i+1] = (hash[i] >> 16) & 0xFF;
        bytes[4*i+2] = (hash[i] >> 8) & 0xFF;
        bytes[4*i+3] = hash[i] & 0xFF;
    }

    /* Reverse bytes for Bitcoin format */
    uint8_t btc_bytes[32];
    for(int i = 0; i < 32; i++){
        btc_bytes[i] = bytes[31 - i];
    }

    /* Count leading zero bits */
    int zeros = 0;
    for(int i = 0; i < 32; i++){
        if(btc_bytes[i] == 0){
            zeros += 8;
        } else {
            zeros += __builtin_clz((uint32_t)btc_bytes[i]) - 24;
            break;
        }
    }
    return zeros;
}

/* Statistics tracking */
typedef struct {
    int count;
    double sum;
    double sum_sq;
    int min_val;
    int max_val;
} stats_t;

static void stats_init(stats_t *s) {
    s->count = 0;
    s->sum = 0.0;
    s->sum_sq = 0.0;
    s->min_val = INT32_MAX;
    s->max_val = INT32_MIN;
}

static void stats_add(stats_t *s, int val) {
    s->count++;
    s->sum += val;
    s->sum_sq += (double)val * val;
    if(val < s->min_val) s->min_val = val;
    if(val > s->max_val) s->max_val = val;
}

static double stats_mean(const stats_t *s) {
    return s->count > 0 ? s->sum / s->count : 0.0;
}

static double stats_stddev(const stats_t *s) {
    if(s->count <= 1) return 0.0;
    double mean = stats_mean(s);
    return sqrt((s->sum_sq - s->count * mean * mean) / (s->count - 1));
}

int main(int argc, char **argv) {
    if(argc != 3){
        fprintf(stderr, "Usage: %s <difficulty> <sample_size>\n", argv[0]);
        fprintf(stderr, "Example: %s 8 1000000\n", argv[0]);
        return 1;
    }

    int difficulty = atoi(argv[1]);
    int sample_size = atoi(argv[2]);

    if(difficulty < 1 || difficulty > 32){
        fprintf(stderr, "Difficulty must be 1-32 (leading zero bits)\n");
        return 1;
    }

    if(sample_size < 1000 || sample_size > 100000000){
        fprintf(stderr, "Sample size must be 1000-100000000\n");
        return 1;
    }

    double expected_count = (double)sample_size / (1ULL << difficulty);

    char line[256];
    int header_idx = 0;
    stats_t stats;
    stats_init(&stats);

    /* Seed PRNG with current time and PID */
    uint32_t base_seed = (uint32_t)time(NULL) ^ (uint32_t)getpid();

    printf("# Testing Bitcoin header optimization\n");
    printf("# Difficulty: %d bits, Sample size: %d, Expected count: %.3f\n",
           difficulty, sample_size, expected_count);
    printf("# Format: header_idx valid_count sample_size expected_count\n");

    while(fgets(line, sizeof(line), stdin)){
        line[strcspn(line, "\r\n")] = 0;
        if(!line[0]) continue;

        uint8_t hdr[76];
        if(hex_decode(line, hdr, 76) != 76){
            fprintf(stderr, "Header %d: expected 152 hex chars (76 bytes)\n", header_idx);
            continue;
        }

        /* Pad header to 128 bytes */
        uint8_t padded[128];
        memcpy(padded, hdr, 76);
        memset(padded + 76, 0, 4); /* nonce placeholder */
        padded[80] = 0x80;
        memset(padded + 81, 0, 39);
        padded[126] = 0x02; padded[127] = 0x80; /* length=640 bits */

        /* Parse blocks */
        uint32_t m1[16], m2[16];
        bytes_to_words(padded, m1, 16);
        bytes_to_words(padded + 64, m2, 16);

        /* Compute midstate: compress block 1 with full 64 rounds */
        uint32_t midstate[8];
        memcpy(midstate, H0, 32);
        compress(midstate, m1);

        /* Initialize PRNG for this header */
        xorshift_t rng;
        xorshift_seed(&rng, base_seed ^ (header_idx * 0x517CC1B7u));

        /* Sample random nonces and count valid ones */
        int valid_count = 0;
        for(int i = 0; i < sample_size; i++){
            uint32_t nonce = xorshift_next(&rng);

            /* Insert nonce (little-endian) into block2 word 3 (big-endian) */
            m2[3] = ((nonce & 0xFF) << 24) | (((nonce >> 8) & 0xFF) << 16) |
                    (((nonce >> 16) & 0xFF) << 8) | ((nonce >> 24) & 0xFF);

            uint32_t hash[8];
            sha256d(hash, midstate, m2);

            if(count_leading_zeros_bitcoin(hash) >= difficulty){
                valid_count++;
            }
        }

        printf("%d %d %d %.3f\n", header_idx, valid_count, sample_size, expected_count);
        fflush(stdout);

        stats_add(&stats, valid_count);
        header_idx++;
    }

    /* Print summary statistics */
    if(stats.count > 0){
        printf("\n# Summary statistics for %d headers:\n", stats.count);
        printf("# Mean: %.3f, StdDev: %.3f, Min: %d, Max: %d\n",
               stats_mean(&stats), stats_stddev(&stats), stats.min_val, stats.max_val);

        /* Chi-squared test for uniformity */
        if(stats.count > 1){
            double observed_var = stats_stddev(&stats) * stats_stddev(&stats);
            double expected_var = expected_count; /* Poisson variance = mean */
            double chi_sq = (stats.count - 1) * observed_var / expected_var;

            printf("# Chi-squared test: observed_var=%.3f, expected_var=%.3f, chi_sq=%.3f\n",
                   observed_var, expected_var, chi_sq);
            printf("# (Higher chi_sq suggests non-uniform distribution across headers)\n");
        }
    }

    return 0;
}