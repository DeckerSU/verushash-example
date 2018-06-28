#include <stdio.h>
#include <stdint.h>

// Simulate _mm_unpacklo_epi32
__device__ void unpacklo32(unsigned char *t, unsigned char *a, unsigned char *b) 
{
    unsigned char tmp[16];
    memcpy(tmp, a, 4);
    memcpy(tmp + 4, b, 4);
    memcpy(tmp + 8, a + 4, 4);
    memcpy(tmp + 12, b + 4, 4);
    memcpy(t, tmp, 16);
}

// Simulate _mm_unpackhi_epi32
__device__ void unpackhi32(unsigned char *t, unsigned char *a, unsigned char *b) 
{
    unsigned char tmp[16];
    memcpy(tmp, a + 8, 4);
    memcpy(tmp + 4, b + 8, 4);
    memcpy(tmp + 8, a + 12, 4);
    memcpy(tmp + 12, b + 12, 4);
    memcpy(t, tmp, 16);
}

__global__ void printme(unsigned char *t, unsigned char *a, unsigned char *b) {
	int i;
	printf("T: "); for (i=0; i<16; i++) printf("%02x", t[i]); printf("\n");
	printf("A: "); for (i=0; i<16; i++) printf("%02x", a[i]); printf("\n");
	printf("B: "); for (i=0; i<16; i++) printf("%02x", b[i]); printf("\n");

        unpacklo32(t, a, b);

	printf("T: "); for (i=0; i<16; i++) printf("%02x", t[i]); printf("\n");
	printf("A: "); for (i=0; i<16; i++) printf("%02x", a[i]); printf("\n");
	printf("B: "); for (i=0; i<16; i++) printf("%02x", b[i]); printf("\n");
}
int main() {

  unsigned char *t = NULL;
  unsigned char *t_cuda = NULL;
  unsigned char *a = NULL;
  unsigned char *a_cuda = NULL;
  unsigned char *b = NULL;
  unsigned char *b_cuda = NULL;


  // ToDo: https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/

  // a = (unsigned char *) malloc (16);
  cudaMallocHost((void**)&a, 16);
  cudaMalloc(&a_cuda, 16);
  cudaMallocHost((void**)&b, 16);
  cudaMalloc(&b_cuda, 16);
  cudaMallocHost((void**)&t, 16);
  cudaMalloc(&t_cuda, 16);
  
  int i;
  for (i=0; i<16; i++) t[i] = 0x00;
  for (i=0; i<16; i++) a[i] = 0xa0 | i;
  for (i=0; i<16; i++) b[i] = 0xb0 | i;

  cudaMemcpy(a_cuda, a, 16, cudaMemcpyHostToDevice);
  cudaMemcpy(b_cuda, b, 16, cudaMemcpyHostToDevice);
  cudaMemcpy(t_cuda, t, 16, cudaMemcpyHostToDevice);

  printme<<< 1, 1 >>>(t_cuda, a_cuda, b_cuda);
  cudaDeviceSynchronize();
  return  0;
}