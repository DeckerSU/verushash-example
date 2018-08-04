/******************************************************************************
 * Copyright © 2014-2018 The SuperNET Developers.                             *
 *                                                                            *
 * See the AUTHORS, DEVELOPER-AGREEMENT and LICENSE files at                  *
 * the top-level directory of this distribution for the individual copyright  *
 * holder information and the developer policies on copyright and licensing.  *
 *                                                                            *
 * Unless otherwise agreed in a custom licensing agreement, no part of the    *
 * SuperNET software, including this file may be copied, modified, propagated *
 * or distributed except according to the terms contained in the LICENSE file *
 *                                                                            *
 * Removal or modification of this copyright notice is prohibited.            *
 *                                                                            *
 ******************************************************************************/

#include "curve25519.h"

// following is ported from libtom

#define STORE32L(x, y)                                                                     \
{ (y)[3] = (uint8_t)(((x)>>24)&255); (y)[2] = (uint8_t)(((x)>>16)&255);   \
(y)[1] = (uint8_t)(((x)>>8)&255); (y)[0] = (uint8_t)((x)&255); }

#define LOAD32L(x, y)                            \
{ x = (uint32_t)(((uint64_t)((y)[3] & 255)<<24) | \
((uint32_t)((y)[2] & 255)<<16) | \
((uint32_t)((y)[1] & 255)<<8)  | \
((uint32_t)((y)[0] & 255))); }

#define STORE64L(x, y)                                                                     \
{ (y)[7] = (uint8_t)(((x)>>56)&255); (y)[6] = (uint8_t)(((x)>>48)&255);   \
(y)[5] = (uint8_t)(((x)>>40)&255); (y)[4] = (uint8_t)(((x)>>32)&255);   \
(y)[3] = (uint8_t)(((x)>>24)&255); (y)[2] = (uint8_t)(((x)>>16)&255);   \
(y)[1] = (uint8_t)(((x)>>8)&255); (y)[0] = (uint8_t)((x)&255); }

#define LOAD64L(x, y)                                                       \
{ x = (((uint64_t)((y)[7] & 255))<<56)|(((uint64_t)((y)[6] & 255))<<48)| \
(((uint64_t)((y)[5] & 255))<<40)|(((uint64_t)((y)[4] & 255))<<32)| \
(((uint64_t)((y)[3] & 255))<<24)|(((uint64_t)((y)[2] & 255))<<16)| \
(((uint64_t)((y)[1] & 255))<<8)|(((uint64_t)((y)[0] & 255))); }

#define STORE32H(x, y)                                                                     \
{ (y)[0] = (uint8_t)(((x)>>24)&255); (y)[1] = (uint8_t)(((x)>>16)&255);   \
(y)[2] = (uint8_t)(((x)>>8)&255); (y)[3] = (uint8_t)((x)&255); }

#define LOAD32H(x, y)                            \
{ x = (uint32_t)(((uint64_t)((y)[0] & 255)<<24) | \
((uint32_t)((y)[1] & 255)<<16) | \
((uint32_t)((y)[2] & 255)<<8)  | \
((uint32_t)((y)[3] & 255))); }

#define STORE64H(x, y)                                                                     \
{ (y)[0] = (uint8_t)(((x)>>56)&255); (y)[1] = (uint8_t)(((x)>>48)&255);     \
(y)[2] = (uint8_t)(((x)>>40)&255); (y)[3] = (uint8_t)(((x)>>32)&255);     \
(y)[4] = (uint8_t)(((x)>>24)&255); (y)[5] = (uint8_t)(((x)>>16)&255);     \
(y)[6] = (uint8_t)(((x)>>8)&255); (y)[7] = (uint8_t)((x)&255); }

#define LOAD64H(x, y)                                                      \
{ x = (((uint64_t)((y)[0] & 255))<<56)|(((uint64_t)((y)[1] & 255))<<48) | \
(((uint64_t)((y)[2] & 255))<<40)|(((uint64_t)((y)[3] & 255))<<32) | \
(((uint64_t)((y)[4] & 255))<<24)|(((uint64_t)((y)[5] & 255))<<16) | \
(((uint64_t)((y)[6] & 255))<<8)|(((uint64_t)((y)[7] & 255))); }

// Various logical functions
#define RORc(x, y) ( ((((uint32_t)(x)&0xFFFFFFFFUL)>>(uint32_t)((y)&31)) | ((uint32_t)(x)<<(uint32_t)(32-((y)&31)))) & 0xFFFFFFFFUL)
#define Ch(x,y,z)       (z ^ (x & (y ^ z)))
#define Maj(x,y,z)      (((x | y) & z) | (x & y))
#define S(x, n)         RORc((x),(n))
#define R(x, n)         (((x)&0xFFFFFFFFUL)>>(n))
#define Sigma0(x)       (S(x, 2) ^ S(x, 13) ^ S(x, 22))
#define Sigma1(x)       (S(x, 6) ^ S(x, 11) ^ S(x, 25))
#define Gamma0(x)       (S(x, 7) ^ S(x, 18) ^ R(x, 3))
#define Gamma1(x)       (S(x, 17) ^ S(x, 19) ^ R(x, 10))
#define MIN(x, y) ( ((x)<(y))?(x):(y) )

static inline int32_t sha256_vcompress(struct sha256_vstate * md,uint8_t *buf)
{
    uint32_t S[8],W[64],t0,t1,i;
    for (i=0; i<8; i++) // copy state into S
        S[i] = md->state[i];
    for (i=0; i<16; i++) // copy the state into 512-bits into W[0..15]
        LOAD32H(W[i],buf + (4*i));
    for (i=16; i<64; i++) // fill W[16..63]
        W[i] = Gamma1(W[i - 2]) + W[i - 7] + Gamma0(W[i - 15]) + W[i - 16];

#define RND(a,b,c,d,e,f,g,h,i,ki)                    \
t0 = h + Sigma1(e) + Ch(e, f, g) + ki + W[i];   \
t1 = Sigma0(a) + Maj(a, b, c);                  \
d += t0;                                        \
h  = t0 + t1;

    RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],0,0x428a2f98);
    RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],1,0x71374491);
    RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],2,0xb5c0fbcf);
    RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],3,0xe9b5dba5);
    RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],4,0x3956c25b);
    RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],5,0x59f111f1);
    RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],6,0x923f82a4);
    RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],7,0xab1c5ed5);
    RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],8,0xd807aa98);
    RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],9,0x12835b01);
    RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],10,0x243185be);
    RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],11,0x550c7dc3);
    RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],12,0x72be5d74);
    RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],13,0x80deb1fe);
    RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],14,0x9bdc06a7);
    RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],15,0xc19bf174);
    RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],16,0xe49b69c1);
    RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],17,0xefbe4786);
    RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],18,0x0fc19dc6);
    RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],19,0x240ca1cc);
    RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],20,0x2de92c6f);
    RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],21,0x4a7484aa);
    RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],22,0x5cb0a9dc);
    RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],23,0x76f988da);
    RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],24,0x983e5152);
    RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],25,0xa831c66d);
    RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],26,0xb00327c8);
    RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],27,0xbf597fc7);
    RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],28,0xc6e00bf3);
    RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],29,0xd5a79147);
    RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],30,0x06ca6351);
    RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],31,0x14292967);
    RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],32,0x27b70a85);
    RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],33,0x2e1b2138);
    RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],34,0x4d2c6dfc);
    RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],35,0x53380d13);
    RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],36,0x650a7354);
    RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],37,0x766a0abb);
    RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],38,0x81c2c92e);
    RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],39,0x92722c85);
    RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],40,0xa2bfe8a1);
    RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],41,0xa81a664b);
    RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],42,0xc24b8b70);
    RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],43,0xc76c51a3);
    RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],44,0xd192e819);
    RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],45,0xd6990624);
    RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],46,0xf40e3585);
    RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],47,0x106aa070);
    RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],48,0x19a4c116);
    RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],49,0x1e376c08);
    RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],50,0x2748774c);
    RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],51,0x34b0bcb5);
    RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],52,0x391c0cb3);
    RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],53,0x4ed8aa4a);
    RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],54,0x5b9cca4f);
    RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],55,0x682e6ff3);
    RND(S[0],S[1],S[2],S[3],S[4],S[5],S[6],S[7],56,0x748f82ee);
    RND(S[7],S[0],S[1],S[2],S[3],S[4],S[5],S[6],57,0x78a5636f);
    RND(S[6],S[7],S[0],S[1],S[2],S[3],S[4],S[5],58,0x84c87814);
    RND(S[5],S[6],S[7],S[0],S[1],S[2],S[3],S[4],59,0x8cc70208);
    RND(S[4],S[5],S[6],S[7],S[0],S[1],S[2],S[3],60,0x90befffa);
    RND(S[3],S[4],S[5],S[6],S[7],S[0],S[1],S[2],61,0xa4506ceb);
    RND(S[2],S[3],S[4],S[5],S[6],S[7],S[0],S[1],62,0xbef9a3f7);
    RND(S[1],S[2],S[3],S[4],S[5],S[6],S[7],S[0],63,0xc67178f2);
#undef RND
    for (i=0; i<8; i++) // feedback
        md->state[i] = md->state[i] + S[i];
    return(0);
}

#undef RORc
#undef Ch
#undef Maj
#undef S
#undef R
#undef Sigma0
#undef Sigma1
#undef Gamma0
#undef Gamma1

static inline void sha256_vinit(struct sha256_vstate * md)
{
    md->curlen = 0;
    md->length = 0;
    md->state[0] = 0x6A09E667UL;
    md->state[1] = 0xBB67AE85UL;
    md->state[2] = 0x3C6EF372UL;
    md->state[3] = 0xA54FF53AUL;
    md->state[4] = 0x510E527FUL;
    md->state[5] = 0x9B05688CUL;
    md->state[6] = 0x1F83D9ABUL;
    md->state[7] = 0x5BE0CD19UL;
}

static inline int32_t sha256_vprocess(struct sha256_vstate *md,const uint8_t *in,uint64_t inlen)
{
    uint64_t n; int32_t err;
    if ( md->curlen > sizeof(md->buf) )
        return(-1);
    while ( inlen > 0 )
    {
        if ( md->curlen == 0 && inlen >= 64 )
        {
            if ( (err= sha256_vcompress(md,(uint8_t *)in)) != 0 )
                return(err);
            md->length += 64 * 8, in += 64, inlen -= 64;
        }
        else
        {
            n = MIN(inlen,64 - md->curlen);
            memcpy(md->buf + md->curlen,in,(size_t)n);
            md->curlen += n, in += n, inlen -= n;
            if ( md->curlen == 64 )
            {
                if ( (err= sha256_vcompress(md,md->buf)) != 0 )
                    return(err);
                md->length += 8*64;
                md->curlen = 0;
            }
        }
    }
    return(0);
}

static inline int32_t sha256_vdone(struct sha256_vstate *md,uint8_t *out)
{
    int32_t i;
    if ( md->curlen >= sizeof(md->buf) )
        return(-1);
    md->length += md->curlen * 8; // increase the length of the message
    md->buf[md->curlen++] = (uint8_t)0x80; // append the '1' bit
    // if len > 56 bytes we append zeros then compress.  Then we can fall back to padding zeros and length encoding like normal.
    if ( md->curlen > 56 )
    {
        while ( md->curlen < 64 )
            md->buf[md->curlen++] = (uint8_t)0;
        sha256_vcompress(md,md->buf);
        md->curlen = 0;
    }
    while ( md->curlen < 56 ) // pad upto 56 bytes of zeroes
        md->buf[md->curlen++] = (uint8_t)0;
    STORE64H(md->length,md->buf+56); // store length
    sha256_vcompress(md,md->buf);
    for (i=0; i<8; i++) // copy output
        STORE32H(md->state[i],out+(4*i));
    return(0);
}

int32_t init_hexbytes_noT(char *hexbytes,uint8_t *message,long len);

void vcalc_sha256(char hashstr[(256 >> 3) * 2 + 1],uint8_t hash[256 >> 3],uint8_t *src,int32_t len)
{
    struct sha256_vstate md;
    sha256_vinit(&md);
    sha256_vprocess(&md,src,len);
    sha256_vdone(&md,hash);
    if ( hashstr != 0 )
        init_hexbytes_noT(hashstr,hash,256 >> 3);
}
