#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>

#define CUDA(fn) do {                           \
        cudaError_t err = fn;                   \
        if (err == cudaSuccess) break;          \
        printf("CUDA Error: %d\n", err);        \
        return -(int)err;                       \
    } while (0)

__global__ static void
filter_device_kernel(const float *input, int ilen,
                     const float *kernel, int klen,
                     float *output)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float sum = 0;
    // TODO: WRITE FILTER CODE HERE
    int w = klen / 2;

    // Checking for edge conditions
    int start = tid < w ? 0 : (tid - w);
    int end = tid >= (ilen - w) ? ilen /*(ilen - 1)*/ : (tid + w); //Should use ilen because the loop upper bound below is <

    // Perform the filter operation
    for (int i = start; i < end; i++) {
        sum += input[i] * kernel[i - tid + w];
    }

    if (tid < ilen) output[tid] = sum;
}

cudaError_t filter_device(const float *input , int ilen,
                          const float *kernel, int klen,
                          float *output)
{
    dim3 blocks(1,1), threads(1,1);
    // TODO: CALCULATE OPTIMUM BLOCKS AND THREADS HERE
    blocks.x = ilen/klen; blocks.y = 1;
    threads.x = klen; threads.y = 1; 

    filter_device_kernel<<<blocks, threads>>>(input, ilen,
                                              kernel, klen,
                                              output);

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
    float *h_input, *h_kernel;
    float *h_output, *h_result;

    float *d_input, *d_kernel;
    float *d_output;

    size_t ibytes = ILEN * sizeof(float);
    size_t kbytes = KLEN * sizeof(float);

    // Allocate memory
    h_input  = (float *)malloc(ibytes);
    h_kernel = (float *)malloc(kbytes);
    h_output = (float *)malloc(ibytes);
    h_result = (float *)malloc(ibytes);

    CUDA(cudaMalloc(&d_input , ibytes));
    CUDA(cudaMalloc(&d_kernel, kbytes));
    CUDA(cudaMalloc(&d_output, ibytes));

    // Generate data
    for (int i = 0; i < ILEN; i++) h_input[i]  = (float)(rand() % 100) / 100.0;
    for (int i = 0; i < KLEN; i++) h_kernel[i] = (float)(rand() % 100) / 100.0;

    // Send data to GPU
    CUDA(cudaMemcpy(d_input, h_input, ibytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_kernel, h_kernel, kbytes, cudaMemcpyHostToDevice));

    timeval start, stop, diff;

    // Calculate
    gettimeofday(&start, NULL);
    CUDA(filter_host(h_input, ILEN, h_kernel, KLEN, h_output));
    gettimeofday(&stop, NULL);
    timersub(&stop, &start, &diff);
    printf("Time on CPU : %ld us\n", diff.tv_sec * 1000000 + diff.tv_usec);

    gettimeofday(&start, NULL);
    CUDA(filter_device(d_input, ILEN, d_kernel, KLEN, d_output));
    gettimeofday(&stop, NULL);
    timersub(&stop, &start, &diff);
    printf("Time on GPU : %ld us\n", diff.tv_sec * 1000000 + diff.tv_usec);

    // Copy data back from GPU
    CUDA(cudaMemcpy(h_result, d_output, ibytes, cudaMemcpyDeviceToHost));

    // Error checking
    float err = 0.0;
    for (int i = 0; i < ILEN; i++) {
        //if(i < 10)
	  //printf("h_result[%d]=%lf, h_output[%d]=%lf\n", i, h_result[i], i, h_output[i]);
        float diff = h_result[i] - h_output[i];
        err = err + diff * diff;
    }
    err = err / ILEN;
    printf("Error: %lf\n", err);

    // TODO: BENCHMARK filter_host and filter_device

    free(h_input);
    free(h_kernel);
    free(h_output);
    free(h_result);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
    return 0;
}
