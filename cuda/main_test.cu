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

  printf("VerusHash Bruteforcer v0.01 by \x1B[01;32mDecker\x1B[0m (q) 2018\n\n");
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

  int i,j,k;

  struct timeval  tv1, tv2;

  gettimeofday(&tv1, NULL);

  //cudaFuncSetCacheConfig(VerusHash_GPU, cudaFuncCachePreferShared);
  //cudaFuncSetCacheConfig(VerusHash_GPU, cudaFuncCachePreferL1);

  /* haraka512 gpu speed test */
  // out_arr - 32, in_arr - 64
  // https://jhui.github.io/2017/03/06/CUDA/ - nice CUDA turorial

  unsigned char *haraka_out_arr      = NULL;
  unsigned char *haraka_out_arr_cuda = NULL;
  unsigned char *haraka_in_arr       = NULL;
  unsigned char *haraka_in_arr_cuda  = NULL;

  cudaMallocHost((void**)&haraka_out_arr, 32 * NTHREAD);
  cudaMalloc(&haraka_out_arr_cuda,        32 * NTHREAD);
  cudaMallocHost((void**)&haraka_in_arr,  64 * NTHREAD);
  cudaMalloc(&haraka_in_arr_cuda,         64 * NTHREAD);

  uint32_t nmax = 256 * 1000000;
  uint32_t value;

  for (k=0; k < nmax / NTHREAD; k++) { // main loop

  for (i=0; i<NTHREAD; i++) {
  	// fill arrays in local host memory
  	value = (k * NTHREAD + i);
        haraka_in_arr[i * 64 + 32] = value;
  }

  cudaMemcpy(haraka_in_arr_cuda, haraka_in_arr , 64 * NTHREAD, cudaMemcpyHostToDevice);

  haraka512_gpu<<<BLOCKS,THREADS>>>(haraka_out_arr_cuda, haraka_in_arr_cuda);

  err = cudaDeviceSynchronize();
  if (err) {
    printf("Err = %d\n",err);
    exit(err);
  }

  cudaMemcpy(haraka_out_arr, haraka_out_arr_cuda , 64 * NTHREAD, cudaMemcpyDeviceToHost);

  } // main loop

  double time_elapsed;
  gettimeofday(&tv2, NULL);
  time_elapsed = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec);
  printf ("in %f seconds, %f H/s\n",
         time_elapsed, (double) (nmax / time_elapsed));

}