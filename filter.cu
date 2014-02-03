#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>

#define CUDA(fn) do {                           \
        cudaError_t err = fn;                   \
        if (err == cudaSuccess) break;          \
        printf("CUDA Error: %d\n", err);        \
        return -(int)err;                       \
    } while (0)


#define __NAIVE__ 0
#define __USING_SMEM__ 1
#define __DEBUG__ 1
#define __PROFILE__ 1
#define BLOCK_SIZE 1024

__global__ static void
filter_device_kernel(const float *input, int ilen,
                     const float *kernel, int klen,
                     float *output, float *debug1, float *debug2, float *debug3, float *debug4)
{
    float sum = 0;
    // TODO: WRITE FILTER CODE HERE

    int w = klen / 2;

#if __NAIVE__
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Checking for edge conditions
    int start = tid < w ? 0 : (tid - w);
    int end = tid >= (ilen - w) ? ilen /*(ilen - 1)*/ : (tid + w); //Should use ilen because the loop upper bound below is <

    // Perform the filter operation
    for (int i = start; i < end; i++) {
        sum += input[i] * kernel[i - tid + w];
        if(tid == 0){
          debug1[i] = input[i];
          debug2[i] = kernel[i - tid + w];
          debug3[i] = input[i] * kernel[i - tid + w];
          debug4[i] = sum;
        }
    }

    if (tid < ilen) output[tid] = sum;
#endif
}

__global__ static void
filter_device_kernel_smem(const float *input, int ilen,
                     const float *kernel, int klen,
                     float *output, float *debug1, float *debug2, float *debug3, float *debug4)
{
    float sum = 0;
    // TODO: WRITE FILTER CODE HERE

    int w = klen / 2;

#if __USING_SMEM__
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int bid = blockIdx.x;
    int local_tid = threadIdx.x;
    int nb = (ilen+BLOCK_SIZE-1)/BLOCK_SIZE;
 
    extern __shared__ float input_shared[]; 

    if(local_tid>0 && local_tid<BLOCK_SIZE-1)
      input_shared[local_tid+w] = input[tid];
    else if (local_tid == 0)
      if (bid>0)
        for (unsigned i=0; i<=w; i++)
          input_shared[w-i] = input[tid-i];
      else //bid == 0
        input_shared[w] = input[tid]; //input_shared[0:w-1] unused
    else // local_tid == BLOCK_SIZE-1
      if(bid<nb-1)
        for (unsigned i=0; i<=w; i++)
          input_shared[local_tid+w+i] = input[tid+i];
      else //bid == nb-1
        input_shared[local_tid+w] = input[tid]; //input_shared[BLOCK_SIZE:BLOCK_SIZE+w-1] unused

    __syncthreads();

    //int start = tid < w ? 0 : (tid - w);
    //int end = tid >= (ilen - w) ? ilen /*(ilen - 1)*/ : (tid + w);

    int start = tid < w ? w : local_tid;
    int end = tid >= (ilen - w) ? BLOCK_SIZE + w - 1 : local_tid + 2*w;

    if(tid == 0){
      for(int i = 0; i < klen; i++){
        debug1[i] = 0; debug2[i] = 0; debug3[i] = 0; debug4[i] = 0;
      }
    }

    for (int i = start; i < end; i++) {
        //sum += input_shared[i] * kernel[i - local_tid + w];
        float res = input_shared[i] * kernel[i - local_tid];
        //if(local_tid == BLOCK_SIZE-1)
          sum = sum + res;
        if(tid == 0){
          debug1[i] = input_shared[i];
          debug2[i] = kernel[i - local_tid];
          debug3[i] = input_shared[i] * kernel[i - local_tid];
          debug4[i] = sum;
        }
    }

    if (tid < ilen) output[tid] = sum;
#endif

}

cudaError_t filter_device(const float *input , int ilen,
                          const float *kernel, int klen,
                          float *output, float *debug1, float *debug2, float *debug3, float *debug4)
{
    dim3 blocks(1,1), threads(1,1);
    // TODO: CALCULATE OPTIMUM BLOCKS AND THREADS HERE
#if __NAIVE__
    blocks.x = ilen/klen; blocks.y = 1;
    threads.x = klen; threads.y = 1;

    filter_device_kernel<<<blocks, threads>>>(input, ilen,
                                              kernel, klen,
                                              output, debug1, debug2, debug3, debug4);
#endif

    return cudaSuccess;
}

cudaError_t filter_device_smem(const float *input , int ilen,
                          const float *kernel, int klen,
                          float *output, float *debug1, float *debug2, float *debug3, float *debug4)
{
    dim3 blocks(1,1), threads(1,1);
    // TODO: CALCULATE OPTIMUM BLOCKS AND THREADS HERE

#if __USING_SMEM__
    blocks.x = (ilen+BLOCK_SIZE - 1)/BLOCK_SIZE; blocks.y = 1;
    threads.x = BLOCK_SIZE; threads.y = 1;

    filter_device_kernel_smem<<<blocks, threads, (BLOCK_SIZE+klen)*sizeof(float)>>>(input, ilen,
                                              kernel, klen,
                                              output, debug1, debug2, debug3, debug4);
#endif 

    return cudaSuccess;
}

cudaError_t filter_host(const float *input , int ilen,
                        const float *kernel, int klen,
                        float *output)
{
    int w = klen / 2;

    for (int idx = 0; idx < ilen; idx++) {

        // Checking for edge conditions
        int start = idx < w ? 0 : (idx - w);
        int end = idx >= (ilen - w) ? (ilen - 1) : (idx + w);

        // Perform the filter operation
        float sum = 0;
        for (int i = start; i < end; i++) {
            sum += input[i] * kernel[i - idx + w];
        }

        output[idx] = sum;
    }

    return cudaSuccess;
}


#define ILEN 1024 * 1024
#define KLEN 32

int main()
{

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
      printf("No CUDA compatible device found\n");
      exit(EXIT_FAILURE);
    }
    printf("%d CUDA compatible devices found\n", deviceCount);


    cudaDeviceProp prop;
    int dev = 0;

    while(dev < deviceCount) {
      if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess)
        printf("Device %d %s compute v%d.%d sharedMemPerBlock %d maxThreadsPerBlock %d maxThreadsDim.x %d maxGridSize.x %d multiProcessorCount %d maxThreadsPerMultiProcessor %d\n", dev, prop.name, prop.major, prop.minor, prop.sharedMemPerBlock, prop.maxThreadsPerBlock, prop.maxThreadsDim[0], prop.maxGridSize[0], prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
      dev++;
    }

    dev = 0; //dev0:C2070 dev1:C2050 on shiva
    cudaSetDevice(dev);

    float *h_input, *h_kernel;
    float *h_output, *h_result;

#if __DEBUG__
    float *h_debug1, *h_debug2, *h_debug3, *h_debug4;
#endif    

    float *d_input, *d_kernel;
    float *d_output;

#if __DEBUG__
    float *d_debug1, *d_debug2, *d_debug3, *d_debug4;
#endif

    size_t ibytes = ILEN * sizeof(float);
    size_t kbytes = KLEN * sizeof(float);

    // Allocate memory
    h_input  = (float *)malloc(ibytes);
    h_kernel = (float *)malloc(kbytes);
    h_output = (float *)malloc(ibytes);
    h_result = (float *)malloc(ibytes);
#if __DEBUG__
    h_debug1 = (float *)malloc(kbytes);
    h_debug2 = (float *)malloc(kbytes);
    h_debug3 = (float *)malloc(kbytes);
    h_debug4 = (float *)malloc(kbytes);
#endif

    CUDA(cudaMalloc(&d_input , ibytes));
    CUDA(cudaMalloc(&d_kernel, kbytes));
    CUDA(cudaMalloc(&d_output, ibytes));
#if __DEBUG__
    CUDA(cudaMalloc(&d_debug1, kbytes));
    CUDA(cudaMalloc(&d_debug2, kbytes));
    CUDA(cudaMalloc(&d_debug3, kbytes));
    CUDA(cudaMalloc(&d_debug4, kbytes));
#endif

    // Generate data
    for (int i = 0; i < ILEN; i++) h_input[i]  = (float)(rand() % 100) / 100.0;
    for (int i = 0; i < KLEN; i++) h_kernel[i] = (float)(rand() % 100) / 100.0;

    // Send data to GPU
    CUDA(cudaMemcpy(d_input, h_input, ibytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_kernel, h_kernel, kbytes, cudaMemcpyHostToDevice));

#if __PROFILE__
    timeval start, stop, diff;

    // Calculate
    gettimeofday(&start, NULL);
#endif
    CUDA(filter_host(h_input, ILEN, h_kernel, KLEN, h_output));
#if __PROFILE__
    gettimeofday(&stop, NULL);
    timersub(&stop, &start, &diff);
    printf("Time on CPU : %ld us\n", diff.tv_sec * 1000000 + diff.tv_usec);
#endif

#if __NAIVE__
    printf("--------Naive-------------\n");
#if __PROFILE__
    gettimeofday(&start, NULL);
#endif
    CUDA(filter_device(d_input, ILEN, d_kernel, KLEN, d_output, d_debug1, d_debug2, d_debug3, d_debug4));
    CUDA(cudaDeviceSynchronize());
#if __PROFILE__
    gettimeofday(&stop, NULL);
    timersub(&stop, &start, &diff);
    printf("Time on GPU : %ld us\n", diff.tv_sec * 1000000 + diff.tv_usec);
#endif

    // Copy data back from GPU
    CUDA(cudaMemcpy(h_result, d_output, ibytes, cudaMemcpyDeviceToHost));
#if __DEBUG__
    CUDA(cudaMemcpy(h_debug1, d_debug1, kbytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(h_debug2, d_debug2, kbytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(h_debug3, d_debug3, kbytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(h_debug4, d_debug4, kbytes, cudaMemcpyDeviceToHost));
  
    for(int i = 0; i < KLEN; i++) {
        printf("input=%lf, kernel=%lf, inputxkernel=%lf, sum=%lf\n", h_debug1[i], h_debug2[i], h_debug3[i], h_debug4[i]);
    }
#endif

    // Error checking
    float err = 0.0;
    for (int i = 0; i < ILEN; i++) {
#if __DEBUG__
        if(i < 10)
	  printf("h_result[%d]=%lf, h_output[%d]=%lf\n", i, h_result[i], i, h_output[i]);
#endif
        float diff = h_result[i] - h_output[i];
        err = err + diff * diff;
    }
    err = err / ILEN;
    printf("Error: %lf\n", err);
#endif

#if __USING_SMEM__
    printf("----------Using SMEM-------------\n");
#if __PROFILE__
    gettimeofday(&start, NULL);
#endif
    CUDA(filter_device_smem(d_input, ILEN, d_kernel, KLEN, d_output, d_debug1, d_debug2, d_debug3, d_debug4));
    CUDA(cudaDeviceSynchronize());
#if __PROFILE__
    gettimeofday(&stop, NULL);
    timersub(&stop, &start, &diff);
    printf("Time on GPU : %ld us\n", diff.tv_sec * 1000000 + diff.tv_usec);
#endif

    // Copy data back from GPU
    CUDA(cudaMemcpy(h_result, d_output, ibytes, cudaMemcpyDeviceToHost));
#if __DEBUG__
    for (int i = 0; i < KLEN; i++){
      h_debug1[i] = 0;
      h_debug2[i] = 0;
      h_debug3[i] = 0;
      h_debug4[i] = 0;
    }
    CUDA(cudaMemcpy(h_debug1, d_debug1, kbytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(h_debug2, d_debug2, kbytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(h_debug3, d_debug3, kbytes, cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(h_debug4, d_debug4, kbytes, cudaMemcpyDeviceToHost));

    for(int i = 0; i < KLEN; i++) {
        printf("input=%lf, kernel=%lf, inputxkernel=%lf, sum=%lf\n", h_debug1[i], h_debug2[i], h_debug3[i], h_debug4[i]);
    }
#endif

    // Error checking
#if __NAIVE__
    err = 0.0;
#else
    float err = 0.0;
#endif

    for (int i = 0; i < ILEN; i++) {
#if __DEBUG__
        if(i < 10)
          printf("h_result[%d]=%lf, h_output[%d]=%lf\n", i, h_result[i], i, h_output[i]);
#endif
        float diff = h_result[i] - h_output[i];
        err = err + diff * diff;
    }
    err = err / ILEN;
    printf("Error: %lf\n", err);
#endif
    // TODO: BENCHMARK filter_host and filter_device

    free(h_input);
    free(h_kernel);
    free(h_output);
    free(h_result);
#if __DEBUG__
    free(h_debug1);
    free(h_debug2);
    free(h_debug3);
    free(h_debug4);
#endif

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
#if __DEBUG__
    cudaFree(d_debug1);
    cudaFree(d_debug2);
    cudaFree(d_debug3);
    cudaFree(d_debug4);
#endif
    return 0;
}
