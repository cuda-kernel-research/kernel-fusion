#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARMUP_ITERS 10
#define BENCHMARK_ITERS 100

__global__ void map_kernel_fp16(const half *a, const half *b, half *temp, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        temp[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void reduce_kernel_mixed_naive(const half *temp, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float val = __half2float(temp[idx]);
        atomicAdd(result, val);
    }
}

__global__ void reduce_kernel_mixed_block(const half *temp, float *result, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        sdata[tid] = __half2float(temp[idx]);
    }
    else
    {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(result, sdata[0]);
    }
}

__global__ void mapreduce_fused_mixed_naive(const half *a, const half *b, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        half partial_h = __hmul(a[idx], b[idx]);
        float partial_f = __half2float(partial_h);
        atomicAdd(result, partial_f);
    }
}

__global__ void mapreduce_fused_mixed_block(const half *a, const half *b, float *result, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        half partial_h = __hmul(a[idx], b[idx]);
        sdata[tid] = __half2float(partial_h);
    }
    else
    {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(result, sdata[0]);
    }
}

float benchmark_unfused_mixed_naive(half *d_a, half *d_b, float *d_result,
                                    half *d_temp, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_mixed_naive<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_mixed_naive<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_fused_mixed_naive(half *d_a, half *d_b, float *d_result, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_mixed_naive<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_mixed_naive<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_unfused_mixed_block(half *d_a, half *d_b, float *d_result,
                                    half *d_temp, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_mixed_block<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_mixed_block<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_fused_mixed_block(half *d_a, half *d_b, float *d_result, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_mixed_block<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_mixed_block<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

int main()
{
    const int sizes[] = {1000, 10000, 100000, 1000000, 10000000, 100000000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    FILE *data_file_naive = fopen("data_map_reduce_mixed_naive.csv", "a");
    if (data_file_naive == NULL)
    {
        printf("ERROR: Cannot open naive data file!\n");
        return 1;
    }

    FILE *data_file_block = fopen("data_map_reduce_mixed_block.csv", "a");
    if (data_file_block == NULL)
    {
        printf("ERROR: Cannot open block data file!\n");
        fclose(data_file_naive);
        return 1;
    }

    // ========== NAIVE IMPLEMENTATION ==========
    printf("Map-Reduce sum(A*B) Mixed Precision: NAIVE Implementation\n");
    printf("===============================================================================================================\n");
    printf("%-10s | %-12s | %-12s | %-9s | %-12s | %-10s | %-12s\n",
           "Size",
           "Unfused Time (μs)",
           "Fused Time (μs)",
           "Speedup",
           "BW Unfused",
           "BW Fused",
           "Memory Saved");
    printf("%-10s | %-17s | %-15s | %-9s | %-12s | %-10s | %-12s\n",
           "",
           "",
           "",
           "",
           "(GB/s)",
           "(GB/s)",
           "(KiB)");
    printf("===============================================================================================================\n");

    for (int s = 0; s < num_sizes; s++)
    {
        int n = sizes[s];
        size_t bytes_fp16 = n * sizeof(half);

        float *h_a_float = new float[n];
        float *h_b_float = new float[n];
        half *h_a = new half[n];
        half *h_b = new half[n];

        for (int i = 0; i < n; i++)
        {
            h_a_float[i] = 1.0f + (i % 100) * 0.01f;
            h_b_float[i] = 2.0f + (i % 100) * 0.01f;
            h_a[i] = __float2half(h_a_float[i]);
            h_b[i] = __float2half(h_b_float[i]);
        }

        half *d_a, *d_b, *d_temp;
        float *d_result;
        cudaMalloc(&d_a, bytes_fp16);
        cudaMalloc(&d_b, bytes_fp16);
        cudaMalloc(&d_result, sizeof(float));
        cudaMalloc(&d_temp, bytes_fp16);

        cudaMemcpy(d_a, h_a, bytes_fp16, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes_fp16, cudaMemcpyHostToDevice);

        float time_unfused_naive = benchmark_unfused_mixed_naive(d_a, d_b, d_result, d_temp, n);
        float time_fused_naive = benchmark_fused_mixed_naive(d_a, d_b, d_result, n);

        float speedup_naive = time_unfused_naive / time_fused_naive;
        float bw_unfused_naive = (4.0f * bytes_fp16) / (time_unfused_naive * 1e-6) / 1e9;
        float bw_fused_naive = (2.0f * bytes_fp16) / (time_fused_naive * 1e-6) / 1e9;
        float memory_saved_kib = bytes_fp16 / 1024.0f;

        printf("%-10d | %17.2f | %15.2f | %8.2fx | %12.2f | %10.2f | %12.1f\n",
               n, time_unfused_naive, time_fused_naive, speedup_naive,
               bw_unfused_naive, bw_fused_naive, memory_saved_kib);

        // Append to naive data file
        fprintf(data_file_naive, "%d,%.2f,%.2f,%.2f,%.2f,%.2f\n", n, time_unfused_naive, time_fused_naive, speedup_naive, bw_unfused_naive, bw_fused_naive);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        cudaFree(d_temp);
        delete[] h_a;
        delete[] h_b;
        delete[] h_a_float;
        delete[] h_b_float;
    }

    // ========== BLOCK-LEVEL IMPLEMENTATION ==========
    printf("\n\nMap-Reduce sum(A*B) Mixed Precision: BLOCK-LEVEL Implementation\n");
    printf("===============================================================================================================\n");
    printf("%-10s | %-12s | %-12s | %-9s | %-12s | %-10s | %-12s\n",
           "Size",
           "Unfused Time (μs)",
           "Fused Time (μs)",
           "Speedup",
           "BW Unfused",
           "BW Fused",
           "Memory Saved");
    printf("%-10s | %-17s | %-15s | %-9s | %-12s | %-10s | %-12s\n",
           "",
           "",
           "",
           "",
           "(GB/s)",
           "(GB/s)",
           "(KiB)");
    printf("===============================================================================================================\n");

    for (int s = 0; s < num_sizes; s++)
    {
        int n = sizes[s];
        size_t bytes_fp16 = n * sizeof(half);

        float *h_a_float = new float[n];
        float *h_b_float = new float[n];
        half *h_a = new half[n];
        half *h_b = new half[n];

        for (int i = 0; i < n; i++)
        {
            h_a_float[i] = 1.0f + (i % 100) * 0.01f;
            h_b_float[i] = 2.0f + (i % 100) * 0.01f;
            h_a[i] = __float2half(h_a_float[i]);
            h_b[i] = __float2half(h_b_float[i]);
        }

        half *d_a, *d_b, *d_temp;
        float *d_result;
        cudaMalloc(&d_a, bytes_fp16);
        cudaMalloc(&d_b, bytes_fp16);
        cudaMalloc(&d_result, sizeof(float));
        cudaMalloc(&d_temp, bytes_fp16);

        cudaMemcpy(d_a, h_a, bytes_fp16, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes_fp16, cudaMemcpyHostToDevice);

        float time_unfused_block = benchmark_unfused_mixed_block(d_a, d_b, d_result, d_temp, n);
        float time_fused_block = benchmark_fused_mixed_block(d_a, d_b, d_result, n);

        float speedup_block = time_unfused_block / time_fused_block;
        float bw_unfused_block = (4.0f * bytes_fp16) / (time_unfused_block * 1e-6) / 1e9;
        float bw_fused_block = (2.0f * bytes_fp16) / (time_fused_block * 1e-6) / 1e9;
        float memory_saved_kib = bytes_fp16 / 1024.0f;

        printf("%-10d | %17.2f | %15.2f | %8.2fx | %12.2f | %10.2f | %12.1f\n",
               n, time_unfused_block, time_fused_block, speedup_block,
               bw_unfused_block, bw_fused_block, memory_saved_kib);

        // Append to block data file
        fprintf(data_file_block, "%d,%.2f,%.2f,%.2f,%.2f,%.2f\n", n, time_unfused_block, time_fused_block, speedup_block, bw_unfused_block, bw_fused_block);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        cudaFree(d_temp);
        delete[] h_a;
        delete[] h_b;
        delete[] h_a_float;
        delete[] h_b_float;
    }

    fclose(data_file_naive);
    fclose(data_file_block);
    printf("\nData appended to data_map_reduce_mixed_naive.csv and data_map_reduce_mixed_block.csv\n");

    return 0;
}