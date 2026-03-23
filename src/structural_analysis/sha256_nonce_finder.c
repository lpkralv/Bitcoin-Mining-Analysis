/*
 * Reduced-round SHA-256 nonce finder for ML training data generation.
 *
 * Reads 76-byte header stubs (hex) from stdin, one per line.
 * Finds a nonce producing >=1 leading zero bit in reduced-round SHA-256d.
 * Outputs nonce as decimal uint32, one per line.
 *
 * Block 1 (bytes 0-63) always uses 64 rounds.
 * Block 2 (bytes 64-127) and outer hash use configurable rounds.
 * Nonce is at bytes 76-79 (little-endian) = W[3] of block 2.
 * Nonce first affects output at round 4 (W[3] used in round 3, 0-indexed).
 *
 * Usage: cat stubs.txt | ./sha256_nonce_finder <num_rounds>
 * Compile: gcc -O3 -o sha256_nonce_finder sha256_nonce_finder.c
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

static void compress(uint32_t h[8], const uint32_t m[16], int nr) {
    uint32_t W[64];
    int i;
    for(i=0;i<16;i++) W[i]=m[i];
    for(i=16;i<nr;i++) W[i]=s1(W[i-2])+W[i-7]+s0(W[i-15])+W[i-16];
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for(i=0;i<nr;i++){
        uint32_t t1=hh+S1(e)+CH(e,f,g)+K[i]+W[i];
        uint32_t t2=S0(a)+MAJ(a,b,c);
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
}

static void bytes_to_words(const uint8_t *b, uint32_t *w, int n) {
    for(int i=0;i<n;i++)
        w[i]=((uint32_t)b[4*i]<<24)|((uint32_t)b[4*i+1]<<16)|((uint32_t)b[4*i+2]<<8)|b[4*i+3];
}

static int hex_val(char c) {
    if(c>='0'&&c<='9') return c-'0';
    if(c>='a'&&c<='f') return c-'a'+10;
    if(c>='A'&&c<='F') return c-'A'+10;
    return -1;
}

static int hex_decode(const char *hex, uint8_t *out, int max_bytes) {
    int i=0;
    while(hex[2*i] && hex[2*i+1] && i<max_bytes) {
        int hi=hex_val(hex[2*i]), lo=hex_val(hex[2*i+1]);
        if(hi<0||lo<0) return -1;
        out[i]=(hi<<4)|lo;
        i++;
    }
    return i;
}

/* SHA-256d with reduced rounds on block2 and outer hash.
 * Block 1 always uses 64 rounds. Returns hash in standard big-endian word order. */
static void sha256d_reduced(uint32_t result[8], const uint32_t midstate[8],
                            const uint32_t block2[16], int nr) {
    uint32_t inner[8];
    memcpy(inner, midstate, 32);
    compress(inner, block2, nr);

    /* Prepare outer hash input: 32-byte inner hash, padded to 64 bytes */
    uint8_t buf[64];
    int i;
    for(i=0;i<8;i++){
        buf[4*i]=(inner[i]>>24)&0xFF; buf[4*i+1]=(inner[i]>>16)&0xFF;
        buf[4*i+2]=(inner[i]>>8)&0xFF; buf[4*i+3]=inner[i]&0xFF;
    }
    buf[32]=0x80;
    memset(buf+33,0,23);
    /* Length = 256 bits = 0x100 */
    buf[62]=0x01; buf[63]=0x00;

    uint32_t om[16];
    bytes_to_words(buf, om, 16);
    memcpy(result, H0, 32);
    compress(result, om, nr);
}

static int leading_zeros(const uint32_t h[8]) {
    for(int i=0;i<8;i++){
        if(h[i]==0) continue;
        return i*32+__builtin_clz(h[i]);
    }
    return 256;
}

int main(int argc, char **argv) {
    if(argc!=2){
        fprintf(stderr,"Usage: %s <num_rounds>\n",argv[0]);
        return 1;
    }
    int nr = atoi(argv[1]);
    if(nr<4||nr>64){
        fprintf(stderr,"num_rounds must be 4-64 (nonce enters at round 3)\n");
        return 1;
    }

    char line[256];
    int lineno=0;
    uint32_t seed = (uint32_t)time(NULL) ^ (uint32_t)getpid();

    while(fgets(line, sizeof(line), stdin)){
        lineno++;
        line[strcspn(line,"\r\n")]=0;
        if(!line[0]) continue;

        uint8_t hdr[76];
        if(hex_decode(line, hdr, 76)!=76){
            fprintf(stderr,"Line %d: expected 152 hex chars (76 bytes)\n",lineno);
            continue;
        }

        /* Pad header to 128 bytes */
        uint8_t padded[128];
        memcpy(padded, hdr, 76);
        memset(padded+76, 0, 4); /* nonce placeholder */
        padded[80]=0x80;
        memset(padded+81,0,39);
        padded[126]=0x02; padded[127]=0x80; /* length=640 */

        /* Parse blocks */
        uint32_t m1[16], m2[16];
        bytes_to_words(padded, m1, 16);
        bytes_to_words(padded+64, m2, 16);

        /* Midstate: compress block 1 with 64 rounds (always) */
        uint32_t mid[8];
        memcpy(mid, H0, 32);
        compress(mid, m1, 64);

        /* Find nonce */
        uint32_t start = seed * 0x9E3779B9u + lineno * 0x517CC1B7u;
        uint32_t nonce = start;
        uint32_t hash[8];

        do {
            /* Insert nonce (little-endian) into block2 word 3 (big-endian) */
            m2[3] = ((nonce&0xFF)<<24) | (((nonce>>8)&0xFF)<<16) |
                     (((nonce>>16)&0xFF)<<8) | ((nonce>>24)&0xFF);

            sha256d_reduced(hash, mid, m2, nr);

            if(leading_zeros(hash) >= 1){
                printf("%u\n", nonce);
                fflush(stdout);
                break;
            }
            nonce++;
        } while(nonce != start);
    }
    return 0;
}
