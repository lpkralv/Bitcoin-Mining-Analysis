/*
 * CMA-ES vs Random Search at High Difficulty
 *
 * Since 64 LZ with 32-bit nonce is nearly impossible (probability ~2^{-32}),
 * we measure: given a budget of N hash evaluations, which method achieves
 * the highest maximum leading zeros?
 *
 * CMA-ES: population of 32-bit vectors, selection, covariance adaptation
 * Random: pure random nonce sampling
 *
 * Tests both at realistic scale (billions of hashes in C).
 *
 * Compile: gcc -O3 -o cmaes_64lz cmaes_64lz.c -lm
 * Usage: echo "<hex_stub>" | ./cmaes_64lz <budget_millions>
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

/* SHA-256 implementation (same as other C programs) */
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

static void compress(uint32_t h[8], const uint32_t m[16]) {
    uint32_t W[64];
    for (int i = 0; i < 16; i++) W[i] = m[i];
    for (int i = 16; i < 64; i++) W[i] = s1(W[i-2]) + W[i-7] + s0(W[i-15]) + W[i-16];
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t t1 = hh+S1(e)+CH(e,f,g)+K[i]+W[i];
        uint32_t t2 = S0(a)+MAJ(a,b,c);
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
}

static void bytes_to_words(const uint8_t *b, uint32_t *w, int n) {
    for (int i = 0; i < n; i++)
        w[i] = ((uint32_t)b[4*i]<<24)|((uint32_t)b[4*i+1]<<16)|((uint32_t)b[4*i+2]<<8)|b[4*i+3];
}

static int count_lz_btc(const uint32_t hash[8]) {
    uint8_t bytes[32];
    for (int i = 0; i < 8; i++) {
        bytes[4*i]=(hash[i]>>24)&0xFF; bytes[4*i+1]=(hash[i]>>16)&0xFF;
        bytes[4*i+2]=(hash[i]>>8)&0xFF; bytes[4*i+3]=hash[i]&0xFF;
    }
    int c = 0;
    for (int i = 31; i >= 0; i--) {
        if (bytes[i] == 0) { c += 8; continue; }
        for (int b = 0; b < 8; b++) {
            if (bytes[i] & (1 << b)) return c;
            c++;
        }
    }
    return c;
}

static int hash_nonce(const uint32_t mid[8], uint32_t m2[16], uint32_t nonce) {
    /* Insert nonce (little-endian) into block2 word 3 (big-endian) */
    m2[3] = ((nonce&0xFF)<<24) | (((nonce>>8)&0xFF)<<16) |
             (((nonce>>16)&0xFF)<<8) | ((nonce>>24)&0xFF);

    uint32_t inner[8];
    memcpy(inner, mid, 32);
    compress(inner, m2);

    /* Outer hash */
    uint8_t buf[64];
    for (int i = 0; i < 8; i++) {
        buf[4*i]=(inner[i]>>24)&0xFF; buf[4*i+1]=(inner[i]>>16)&0xFF;
        buf[4*i+2]=(inner[i]>>8)&0xFF; buf[4*i+3]=inner[i]&0xFF;
    }
    buf[32]=0x80; memset(buf+33,0,23); buf[62]=0x01; buf[63]=0x00;

    uint32_t om[16];
    bytes_to_words(buf, om, 16);
    uint32_t oh[8];
    memcpy(oh, H0, 32);
    compress(oh, om);

    return count_lz_btc(oh);
}

/* Simple xorshift PRNG */
static uint64_t rng_state;
static uint32_t xrand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state);
}

/* Gaussian random (Box-Muller) */
static double grand(void) {
    double u1 = (xrand() + 1.0) / 4294967297.0;
    double u2 = (xrand() + 1.0) / 4294967297.0;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
}

static int hex_val(char c) {
    if (c>='0'&&c<='9') return c-'0';
    if (c>='a'&&c<='f') return c-'a'+10;
    if (c>='A'&&c<='F') return c-'A'+10;
    return -1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: echo '<hex_stub>' | %s <budget_millions>\n", argv[0]);
        return 1;
    }
    long budget = atol(argv[1]) * 1000000L;

    char line[256];
    if (!fgets(line, sizeof(line), stdin)) return 1;
    line[strcspn(line, "\r\n")] = 0;

    uint8_t hdr[76];
    for (int i = 0; i < 76; i++) {
        int hi = hex_val(line[2*i]), lo = hex_val(line[2*i+1]);
        hdr[i] = (hi<<4)|lo;
    }

    /* Pad and compute midstate */
    uint8_t padded[128];
    memcpy(padded, hdr, 76);
    memset(padded+76, 0, 4);
    padded[80]=0x80; memset(padded+81,0,39);
    padded[126]=0x02; padded[127]=0x80;

    uint32_t m1[16], m2_template[16];
    bytes_to_words(padded, m1, 16);
    bytes_to_words(padded+64, m2_template, 16);

    uint32_t mid[8];
    memcpy(mid, H0, 32);
    compress(mid, m1);

    rng_state = (uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32);

    printf("CMA-ES vs Random at high difficulty\n");
    printf("Budget: %ld million hashes per method\n\n", budget / 1000000);

    /* === Random Search === */
    printf("Random search...\n");
    uint32_t m2r[16];
    memcpy(m2r, m2_template, 64);

    int random_best_lz = 0;
    long random_best_at = 0;
    time_t t0 = time(NULL);

    /* Track LZ milestones */
    long random_milestones[33] = {0}; /* first time reaching each LZ */

    for (long i = 0; i < budget; i++) {
        uint32_t nonce = xrand();
        int lz = hash_nonce(mid, m2r, nonce);
        if (lz > random_best_lz) {
            random_best_lz = lz;
            random_best_at = i + 1;
            if (lz < 33) random_milestones[lz] = i + 1;
        }
        if ((i+1) % (budget/10) == 0) {
            fprintf(stderr, "  Random: %ld%% best_lz=%d\n", (i+1)*100/budget, random_best_lz);
        }
    }
    time_t t1 = time(NULL);
    double random_time = difftime(t1, t0);
    double random_rate = budget / random_time / 1e6;

    printf("  Best LZ: %d (first reached at eval %ld)\n", random_best_lz, random_best_at);
    printf("  Rate: %.1f M hashes/sec, Time: %.0f sec\n\n", random_rate, random_time);

    /* === CMA-ES 32D === */
    printf("CMA-ES 32D search...\n");
    uint32_t m2c[16];
    memcpy(m2c, m2_template, 64);

    int cmaes_best_lz = 0;
    long cmaes_best_at = 0;
    long cmaes_milestones[33] = {0};

    int pop_size = 200;
    double mean[32], sigma = 0.25;
    for (int i = 0; i < 32; i++) mean[i] = 0.5;

    long evals = 0;
    t0 = time(NULL);

    while (evals < budget) {
        /* Sample population */
        double pop[200][32];
        int fits[200];

        for (int p = 0; p < pop_size && evals < budget; p++) {
            /* Sample individual */
            uint32_t nonce = 0;
            for (int b = 0; b < 32; b++) {
                double val = mean[b] + sigma * grand();
                if (val < 0) val = 0;
                if (val > 1) val = 1;
                pop[p][b] = val;
                if (val > 0.5) nonce |= (1 << (31-b));
            }

            fits[p] = hash_nonce(mid, m2c, nonce);
            evals++;

            if (fits[p] > cmaes_best_lz) {
                cmaes_best_lz = fits[p];
                cmaes_best_at = evals;
                if (fits[p] < 33) cmaes_milestones[fits[p]] = evals;
            }
        }

        /* Select top quartile */
        int top_k = pop_size / 4;
        /* Simple selection: find top_k indices */
        int selected[50];
        for (int k = 0; k < top_k; k++) {
            int best_idx = -1, best_fit = -1;
            for (int p = 0; p < pop_size; p++) {
                int already = 0;
                for (int j = 0; j < k; j++) if (selected[j] == p) already = 1;
                if (!already && fits[p] > best_fit) {
                    best_fit = fits[p]; best_idx = p;
                }
            }
            selected[k] = best_idx;
        }

        /* Update mean */
        for (int b = 0; b < 32; b++) {
            double new_mean = 0;
            for (int k = 0; k < top_k; k++) new_mean += pop[selected[k]][b];
            new_mean /= top_k;
            mean[b] = 0.5 * mean[b] + 0.5 * new_mean;
        }

        if (evals % (budget/10) < pop_size) {
            fprintf(stderr, "  CMA-ES: %ld%% best_lz=%d\n", evals*100/budget, cmaes_best_lz);
        }
    }

    t1 = time(NULL);
    double cmaes_time = difftime(t1, t0);
    double cmaes_rate = budget / cmaes_time / 1e6;

    printf("  Best LZ: %d (first reached at eval %ld)\n", cmaes_best_lz, cmaes_best_at);
    printf("  Rate: %.1f M hashes/sec, Time: %.0f sec\n\n", cmaes_rate, cmaes_time);

    /* === Summary === */
    printf("=== COMPARISON ===\n");
    printf("Budget: %ld M hashes each\n", budget / 1000000);
    printf("Random: best_lz=%d, rate=%.1f Mh/s\n", random_best_lz, random_rate);
    printf("CMA-ES: best_lz=%d, rate=%.1f Mh/s\n", cmaes_best_lz, cmaes_rate);
    printf("LZ advantage: %+d\n", cmaes_best_lz - random_best_lz);
    printf("Rate ratio: %.2fx (CMA-ES / Random)\n", cmaes_rate / random_rate);

    /* Milestone comparison */
    printf("\nMilestones (first eval to reach each LZ):\n");
    printf("LZ   Random        CMA-ES       Ratio\n");
    int max_lz = random_best_lz > cmaes_best_lz ? random_best_lz : cmaes_best_lz;
    for (int lz = 8; lz <= max_lz; lz += 4) {
        long rm = random_milestones[lz < 33 ? lz : 32];
        long cm = cmaes_milestones[lz < 33 ? lz : 32];
        if (rm > 0 && cm > 0) {
            printf("%2d   %12ld  %12ld  %.3fx\n", lz, rm, cm, (double)rm / cm);
        } else if (rm > 0) {
            printf("%2d   %12ld  not reached\n", lz, rm);
        } else if (cm > 0) {
            printf("%2d   not reached    %12ld\n", lz, cm);
        }
    }

    return 0;
}
