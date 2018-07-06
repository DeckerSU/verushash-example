#ifndef SPX_HARAKA_H
#define SPX_HARAKA_H

#include <sys/types.h>
#include <stdint.h>
#include "blocks.h"

/* Tweak constants with seed */
//__device__ void tweak_constants(const unsigned char *pk_seed, const unsigned char *sk_seed, 
//	                 unsigned long long seed_length);

/* Haraka Sponge */
//__device__ void haraka_S(unsigned char *out, unsigned long long outlen,
//              const unsigned char *in, unsigned long long inlen);

/* Applies the 512-bit Haraka permutation to in. */
//__global__ void haraka512_perm(unsigned char *out, const unsigned char *in);

/* Implementation of Haraka-512 */
//__global__ void haraka512(unsigned char *out, const unsigned char *in);

/* Implementation of Haraka-256 */
//__global__ void haraka256(unsigned char *out, const unsigned char *in);

/* Implementation of Haraka-256 using sk.seed constants */
//__global__ void haraka256_sk(unsigned char *out, const unsigned char *in);

//__global__ void VerusHash_GPU(void *result, const void *data/*, size_t len*/);
//__global__ void VerusHash_GPU(unsigned char *result, unsigned char *data, int len);

//__global__ void haraka512_gpu(unsigned char *out_arr, const unsigned char *in_arr);
//__global__ void haraka512_gpu(unsigned char *out_arr, uint32_t first0, uint32_t first1, uint32_t first2, uint32_t first3, int k);
__global__ void haraka512_gpu(unsigned char *out_arr, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3, uint32_t b4, uint32_t b5, uint32_t b6, uint32_t b7, uint32_t start);

#endif