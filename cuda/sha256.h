#ifndef _SHA256_H_
#define _SHA256_H_

#include <sys/types.h>
#include <stdint.h>
#include "blocks.h"

typedef struct SHA256Context {
	uint32_t state[8];
	uint32_t count[2];
	unsigned char buf[64];
} SHA256_CTX;

__global__ void SHA256(char *str, int len, unsigned char dgst[32]);
__global__ void  functest( uint32_t *data );

#endif /* !_SHA256_H_ */
