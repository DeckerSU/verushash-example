#ifndef SPX_HARAKA_H
#define SPX_HARAKA_H

#include <sys/types.h>
#include <stdint.h>
#include "blocks.h"

/* Tweak constants with seed */
__device__ void tweak_constants(const unsigned char *pk_seed, const unsigned char *sk_seed, 
	                 unsigned long long seed_length);

/* Haraka Sponge */
__device__ void haraka_S(unsigned char *out, unsigned long long outlen,
              const unsigned char *in, unsigned long long inlen);

/* Applies the 512-bit Haraka permutation to in. */
__device__ void haraka512_perm(unsigned char *out, const unsigned char *in);

/* Implementation of Haraka-512 */
__device__ void haraka512(unsigned char *out, const unsigned char *in);

/* Implementation of Haraka-256 */
__device__ void haraka256(unsigned char *out, const unsigned char *in);

/* Implementation of Haraka-256 using sk.seed constants */
__device__ void haraka256_sk(unsigned char *out, const unsigned char *in);

__global__ void VerusHash_GPU(void *result, const void *data/*, size_t len*/);
//__global__ void VerusHash_GPU(unsigned char *result, unsigned char *data, int len);

#endif