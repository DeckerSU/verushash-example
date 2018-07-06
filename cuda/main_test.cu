#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#include <sys/times.h>
#include <sys/time.h>
#endif

#include <time.h>
#include "sha256.h"
#include "haraka.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <stdint.h> // portable: uint64_t   MSVC: __int64 

// MSVC defines this in winsock2.h!?
typedef struct timeval {
    long tv_sec;
    long tv_usec;
} timeval;

int gettimeofday(struct timeval * tp, struct timezone * tzp)
{
    // Note: some broken versions only have 8 trailing zero's, the correct epoch has 9 trailing zero's
    // This magic number is the number of 100 nanosecond intervals since January 1, 1601 (UTC)
    // until 00:00:00 January 1, 1970 
    static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime( &system_time );
    SystemTimeToFileTime( &system_time, &file_time );
    time =  ((uint64_t)file_time.dwLowDateTime )      ;
    time += ((uint64_t)file_time.dwHighDateTime) << 32;

    tp->tv_sec  = (long) ((time - EPOCH) / 10000000L);
    tp->tv_usec = (long) (system_time.wMilliseconds * 1000);
    return 0;
}
#endif

int main(int argc, char *argv[])
{

  printf("VerusHash Bruteforcer v0.02 by \x1B[01;32mDecker\x1B[0m and \x1B[01;33mOcean\x1B[0m (q) 2018\n\n");
  printf("[+] It's just a beginning ... \n");
  printf("[*] NTHREAD.%d \n", NTHREAD);

  typedef unsigned int beu32;

  cudaError_t err;
  int         device = (argc == 1) ? 0 : atoi(argv[1]);
 
  cudaDeviceProp props;
  err = cudaGetDeviceProperties(&props,device);
 
  if (err) 
    return -1;
 
  printf("%s (%2d)\n",props.name,props.multiProcessorCount);

  cudaSetDevice(device);
  //cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
  //cudaDeviceReset();
  //cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

  /* Check VerusHash */

  int i,k;

  struct timeval  tv1, tv2;

  gettimeofday(&tv1, NULL);

  //cudaFuncSetCacheConfig(VerusHash_GPU, cudaFuncCachePreferShared);
  //cudaFuncSetCacheConfig(VerusHash_GPU, cudaFuncCachePreferL1);

  /* haraka512 gpu speed test */
  // out_arr - 32, in_arr - 64
  // https://jhui.github.io/2017/03/06/CUDA/ - nice CUDA turorial

  unsigned char *haraka_out_arr      = NULL;
  unsigned char *haraka_out_arr_cuda = NULL;
//  unsigned char *haraka_in_arr       = NULL;
//  unsigned char *haraka_in_arr_cuda   = NULL;

  cudaMallocHost((void**)&haraka_out_arr, 32 * NTHREAD);
  cudaMalloc(&haraka_out_arr_cuda,        32 * NTHREAD);

//  cudaMallocHost((void**)&haraka_in_arr,  64 * NTHREAD);
//  cudaMalloc(&haraka_in_arr_cuda,         64 * NTHREAD);

  uint32_t nmax = 256 * 1000000;
//  uint32_t nmax = NTHREAD * 10000; 
  uint32_t value;

  k = 0;
  for (k=0; k < nmax / NTHREAD; k++) 
  { // main loop

//  memset(haraka_in_arr, 0x00, 64 * NTHREAD);

//  for (i=0; i<NTHREAD; i++) {
	// fill arrays in local host memory
//  	value = (k * NTHREAD + i);
//        haraka_in_arr[i * 64 + 32] = value;
        //printf("GPU indata[%d]: ", i); for (int z=0; z < 64; z++) { printf("%02x", *(haraka_in_arr + i * 64 + z)); } printf("\n");
//  }

//  cudaMemcpy(haraka_in_arr_cuda, haraka_in_arr , 64 * NTHREAD, cudaMemcpyHostToDevice);

  uint32_t b0 = 0xb0b0b0b0;
  uint32_t b1 = 0xb1b1b1b1;
  uint32_t b2 = 0xb2b2b2b2;
  uint32_t b3 = 0xb3b3b3b3;
  uint32_t b4 = 0xb4b4b4b4;
  uint32_t b5 = 0xb5b5b5b5;
  uint32_t b6 = 0xb6b6b6b6;
  uint32_t b7 = 0xb7b7b7b7;

  //haraka512_gpu<<<BLOCKS,THREADS>>>(haraka_out_arr_cuda, b0, b1, b2, b3, b4, b5, b6, b7, k * NTHREAD);
  haraka512_gpu<<<BLOCKS,THREADS>>>(haraka_out_arr_cuda, b0, b1, b2, b3, b4, b5, b6, b7, k * NTHREAD);

  err = cudaDeviceSynchronize();
  if (err) {
    printf("Err = %d\n",err);
    exit(err);
  }

  cudaMemcpy(haraka_out_arr, haraka_out_arr_cuda , 32 * NTHREAD, cudaMemcpyDeviceToHost);

//  for (i=0; i<NTHREAD; i++) {
        //printf("GPU result[%d]: ", i); for (int z=0; z < 32; z++) { printf("%02x", *(haraka_out_arr + i * 32 + z)); } printf("\n");
//  }


  } // main loop

  double time_elapsed;
  gettimeofday(&tv2, NULL);
  time_elapsed = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec);
  printf ("in %f seconds, %f H/s\n",
         time_elapsed, (double) (nmax / time_elapsed));


  // print first 9 result from last iter to make sure it equal test results vector
  for (i=0; i<9; i++) {
        printf("GPU result[%d]: ", i); for (int z=0; z < 32; z++) { printf("%02x", *(haraka_out_arr + i * 32 + z)); } printf("\n");
  }

  cudaFreeHost(haraka_out_arr);
//  cudaFree(haraka_out_arr_cuda);
//  cudaFreeHost(haraka_in_arr);
//  cudaFree(haraka_in_arr_cuda);
}

/*

Test data:

GPU indata[0]: 00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000  
GPU indata[1]: 00000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000  
GPU indata[2]: 00000000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000  
GPU indata[3]: 00000000000000000000000000000000000000000000000000000000000000000300000000000000000000000000000000000000000000000000000000000000  
GPU indata[4]: 00000000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000000000  
GPU indata[5]: 00000000000000000000000000000000000000000000000000000000000000000500000000000000000000000000000000000000000000000000000000000000  
GPU indata[6]: 00000000000000000000000000000000000000000000000000000000000000000600000000000000000000000000000000000000000000000000000000000000  
GPU indata[7]: 00000000000000000000000000000000000000000000000000000000000000000700000000000000000000000000000000000000000000000000000000000000  
GPU indata[8]: 00000000000000000000000000000000000000000000000000000000000000000800000000000000000000000000000000000000000000000000000000000000  
GPU result[0]: 3636363636363636363636363636363636363636363636363636363636363636                                                                  
GPU result[1]: c7c3bc8108084e24d0cb4b02300f35aa8bce0f0065f269ebe4378e78fcef8e1f                                                                  
GPU result[2]: 26a9f97a73692253918382bd401f4e3d502b9f8c1f5142b1cb37399e77ec456e                                                                  
GPU result[3]: f838071ca2bc19ac3a6277a9786b9d3bdf827274847de835de2c9da035fe29a5                                                                  
GPU result[4]: f2c8e2ed4b68798e9de1071f891d3e34da2a87314f0dfa38a8fc5dfa8e2fca84                                                                  
GPU result[5]: f8f8aee17130e77cf4aee7da016fb00aba05d4d5a8550371b6ce8fdf74202740                                                                  
GPU result[6]: d170aadcb072dd3c0e9dbed1eae567dbbe8c9d82032649dff0da9fbae3989913                                                                  
GPU result[7]: 127f8873e5e4e92612ff6d359ff2a7d315f64dd0c03545964a4e2bb2039441ad                                                                  
GPU result[8]: fdce98e3981c442e3166143aeb2c86d5804be296af76486ef95214c2eb4bdd52                                                                  

*/