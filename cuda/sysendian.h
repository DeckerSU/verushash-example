#ifndef _SYSENDIAN_H_
#define _SYSENDIAN_H_

/* If we don't have be64enc, the <sys/endian.h> we have isn't usable. */
#if !HAVE_DECL_BE64ENC
#undef HAVE_SYS_ENDIAN_H
#endif

#ifdef HAVE_SYS_ENDIAN_H

#include <sys/endian.h>

#else

#include <stdint.h>

__device__ static inline uint32_t be32dec_GPU(const void *pp)
{
	const uint8_t *p = (uint8_t const *)pp;

	return ((uint32_t)(p[3]) + ((uint32_t)(p[2]) << 8) +
	    ((uint32_t)(p[1]) << 16) + ((uint32_t)(p[0]) << 24));
}


__device__ static inline void be32enc_GPU(void *pp, uint32_t x)
{
	uint8_t * p = (uint8_t *)pp;

	p[3] = x & 0xff;
	p[2] = (x >> 8) & 0xff;
	p[1] = (x >> 16) & 0xff;
	p[0] = (x >> 24) & 0xff;
}

__device__ static inline uint32_t le32dec_GPU(const void *pp)
{
        const uint8_t *p = (uint8_t const *)pp;

        return ((uint32_t)(p[0]) + ((uint32_t)(p[1]) << 8) +
            ((uint32_t)(p[2]) << 16) + ((uint32_t)(p[3]) << 24));
}

__device__ static inline void le32enc_GPU(void *pp, uint32_t x)
{

	uint8_t * p = (uint8_t *)pp;

	p[0] = x & 0xff;
	p[1] = (x >> 8) & 0xff;
	p[2] = (x >> 16) & 0xff;
	p[3] = (x >> 24) & 0xff;
}

#endif /* !HAVE_SYS_ENDIAN_H */

#endif /* !_SYSENDIAN_H_ */
