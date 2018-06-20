//
// Just for debug
#define IMPRIME1(vetor, texto)  \
    uint8_t *ptr = (uint8_t *)vetor;  \
    int iPTR = 0; \
    printf("%s%2d: %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x \n", texto, threadNumber, ptr[iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR]);  \

//
#define IMPRIME2(vetor, texto)  \
    ptr = (uint8_t *)vetor;  \
    iPTR = 0; \
    printf("%s%2d: %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x %x \n", texto, threadNumber, \
    ptr[iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], \
    ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR], ptr[++iPTR]);  \

__device__ static void blkcpy_GPU(uint32_t * dest, const uint32_t * src, size_t count, unsigned int totalPasswords)
{
        do {
            *dest++ = *src++; *dest++ = *src++;
			*dest++ = *src++; *dest++ = *src++;
		} while (count -= 4);
}

__device__ static inline void blkcpy_GPU64(uint64_t * dest, const uint64_t * src, size_t count)
{
        do {
            *dest++ = *src++; *dest++ = *src++;
			*dest++ = *src++; *dest++ = *src++;
		} while (count -= 4);
}

