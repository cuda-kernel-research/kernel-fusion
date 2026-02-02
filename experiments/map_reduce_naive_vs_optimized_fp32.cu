#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARMUP_ITERS 10
#define BENCHMARK_ITERS 100

__global__ void map_kernel(const float *a, const float *b, float *temp, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        temp[idx] = a[idx] * b[idx];
    }
}

__global__ void reduce_kernel_naive(const float *temp, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        atomicAdd(result, temp[idx]);
    }
}

__global__ void mapreduce_fused_naive(const float *a, const float *b, float *result, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        float partial = a[idx] * b[idx];
        atomicAdd(result, partial);
    }
}

__global__ void reduce_kernel_block(const float *temp, float *result, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? temp[idx] : 0.0f;
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

__global__ void mapreduce_fused_block(const float *a, const float *b, float *result, int n)
{
    __shared__ float sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
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

float benchmark_unfused_naive(float *d_a, float *d_b, float *d_result,
                              float *d_temp, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_naive<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_naive<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_fused_naive(float *d_a, float *d_b, float *d_result, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_naive<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_naive<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_unfused_block(float *d_a, float *d_b, float *d_result,
                              float *d_temp, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_block<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_block<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_fused_block(float *d_a, float *d_b, float *d_result, int n)
{
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_block<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++)
    {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_block<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
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

    FILE *results_file = fopen("results_map_reduce_fp32.txt", "w");
    FILE *data_file_naive = fopen("data_map_reduce_fp32_naive.csv", "a");
    FILE *data_file_block = fopen("data_map_reduce_fp32_block.csv", "a");

    if (results_file == NULL || data_file_naive == NULL || data_file_block == NULL)
    {
        printf("ERROR: Cannot open results files\n");
        return 1;
    }

    printf("Map-Reduce sum(A*B) FP32: NAIVE Implementation\n");
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
    printf("================================================================================\n");

    fprintf(results_file, "Map-Reduce sum(A*B) FP32: NAIVE Implementation\n");
    fprintf(results_file, "===============================================================================================================\n");
    fprintf(results_file, "%-10s | %-12s | %-12s | %-9s | %-12s | %-10s | %-12s\n",
            "Size",
            "Unfused Time (μs)",
            "Fused Time (μs)",
            "Speedup",
            "BW Unfused",
            "BW Fused",
            "Memory Saved");
    fprintf(results_file, "%-10s | %-17s | %-15s | %-9s | %-12s | %-10s | %-12s\n",
            "",
            "",
            "",
            "",
            "(GB/s)",
            "(GB/s)",
            "(KiB)");
    fprintf(results_file, "================================================================================\n");
    for (int s = 0; s < num_sizes; s++)
    {
        int n = sizes[s];
        size_t bytes = n * sizeof(float);

        float *h_a = new float[n];
        float *h_b = new float[n];

        for (int i = 0; i < n; i++)
        {
            h_a[i] = 1.0f + (i % 100) * 0.01f;
            h_b[i] = 2.0f + (i % 100) * 0.01f;
        }

        float *d_a, *d_b, *d_result, *d_temp;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_result, sizeof(float));
        cudaMalloc(&d_temp, bytes);

        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        float time_unfused = benchmark_unfused_naive(d_a, d_b, d_result, d_temp, n);
        float h_result_unfused = 0.0f;
        cudaMemcpy(&h_result_unfused, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        float time_fused = benchmark_fused_naive(d_a, d_b, d_result, n);
        float h_result_fused = 0.0f;
        cudaMemcpy(&h_result_fused, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        float speedup = time_unfused / time_fused;

        float bw_unfused = (4.0f * bytes) / (time_unfused * 1e-6) / 1e9;
        float bw_fused = (2.0f * bytes) / (time_fused * 1e-6) / 1e9;

        float expected = 0.0f;
        for (int i = 0; i < n; i++)
        {
            expected += h_a[i] * h_b[i];
        }

        bool correct = true;
        float rel_tolerance = (n >= 1000000) ? 0.005f : 0.001f;
        float tolerance = fabs(expected) * rel_tolerance;

        if (fabs(h_result_unfused - expected) > tolerance)
        {
            correct = false;
            printf("ERROR (unfused): expected=%.2f, got=%.2f (diff=%.2f, %.4f%%)\n",
                   expected, h_result_unfused,
                   fabs(h_result_unfused - expected),
                   fabs(h_result_unfused - expected) / fabs(expected) * 100);
        }
        if (fabs(h_result_fused - expected) > tolerance)
        {
            correct = false;
            printf("ERROR (fused): expected=%.2f, got=%.2f (diff=%.2f, %.4f%%)\n",
                   expected, h_result_fused,
                   fabs(h_result_fused - expected),
                   fabs(h_result_fused - expected) / fabs(expected) * 100);
        }
        float memory_saved_kib = bytes / 1024.0f;
        printf("%-10d | %17.2f | %15.2f | %8.2fx | %12.2f | %10.2f | %12.1f\n",
               n,
               time_unfused,
               time_fused,
               speedup,
               bw_unfused,
               bw_fused,
               memory_saved_kib);
        fprintf(results_file, "%-10d | %17.2f | %15.2f | %8.2fx | %12.2f | %10.2f | %12.1f\n",
                n,
                time_unfused,
                time_fused,
                speedup,
                bw_unfused,
                bw_fused,
                memory_saved_kib);

        fprintf(data_file_naive, "%d,%.2f,%.2f,%.2f,%.2f,%.2f\n", n, time_unfused, time_fused, speedup, bw_unfused, bw_fused);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        cudaFree(d_temp);
        delete[] h_a;
        delete[] h_b;
    }

    printf("\n\nMap-Reduce sum(A*B) FP32: BLOCK-LEVEL Implementation\n");
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
    printf("================================================================================\n");

    fprintf(results_file, "\n\nMap-Reduce sum(A*B) FP32: BLOCK-LEVEL Implementation\n");
    fprintf(results_file, "===============================================================================================================\n");
    fprintf(results_file, "%-10s | %-12s | %-12s | %-9s | %-12s | %-10s | %-12s\n",
            "Size",
            "Unfused Time (μs)",
            "Fused Time (μs)",
            "Speedup",
            "BW Unfused",
            "BW Fused",
            "Memory Saved");
    fprintf(results_file, "%-10s | %-17s | %-15s | %-9s | %-12s | %-10s | %-12s\n",
            "",
            "",
            "",
            "",
            "(GB/s)",
            "(GB/s)",
            "(KiB)");
    fprintf(results_file, "================================================================================\n");

    for (int s = 0; s < num_sizes; s++)
    {
        int n = sizes[s];
        size_t bytes = n * sizeof(float);

        float *h_a = new float[n];
        float *h_b = new float[n];

        for (int i = 0; i < n; i++)
        {
            h_a[i] = 1.0f + (i % 100) * 0.01f;
            h_b[i] = 2.0f + (i % 100) * 0.01f;
        }

        float *d_a, *d_b, *d_result, *d_temp;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_result, sizeof(float));
        cudaMalloc(&d_temp, bytes);

        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        float time_unfused = benchmark_unfused_block(d_a, d_b, d_result, d_temp, n);
        float h_result_unfused = 0.0f;
        cudaMemcpy(&h_result_unfused, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        float time_fused = benchmark_fused_block(d_a, d_b, d_result, n);
        float h_result_fused = 0.0f;
        cudaMemcpy(&h_result_fused, d_result, sizeof(float), cudaMemcpyDeviceToHost);

        float speedup = time_unfused / time_fused;

        float bw_unfused = (4.0f * bytes) / (time_unfused * 1e-6) / 1e9;
        float bw_fused = (2.0f * bytes) / (time_fused * 1e-6) / 1e9;

        float expected = 0.0f;
        for (int i = 0; i < n; i++)
        {
            expected += h_a[i] * h_b[i];
        }

        bool correct = true;
        float rel_tolerance = (n >= 1000000) ? 0.005f : 0.001f;
        float tolerance = fabs(expected) * rel_tolerance;

        if (fabs(h_result_unfused - expected) > tolerance)
        {
            correct = false;
            printf("ERROR (unfused): expected=%.2f, got=%.2f (diff=%.2f, %.4f%%)\n",
                   expected, h_result_unfused,
                   fabs(h_result_unfused - expected),
                   fabs(h_result_unfused - expected) / fabs(expected) * 100);
        }
        if (fabs(h_result_fused - expected) > tolerance)
        {
            correct = false;
            printf("ERROR (fused): expected=%.2f, got=%.2f (diff=%.2f, %.4f%%)\n",
                   expected, h_result_fused,
                   fabs(h_result_fused - expected),
                   fabs(h_result_fused - expected) / fabs(expected) * 100);
        }

        float memory_saved_kib = bytes / 1024.0f;
        printf("%-10d | %17.2f | %15.2f | %8.2fx | %12.2f | %10.2f | %12.1f\n",
               n,
               time_unfused,
               time_fused,
               speedup,
               bw_unfused,
               bw_fused,
               memory_saved_kib);
        fprintf(results_file, "%-10d | %17.2f | %15.2f | %8.2fx | %12.2f | %10.2f | %12.1f\n",
                n,
                time_unfused,
                time_fused,
                speedup,
                bw_unfused,
                bw_fused,
                memory_saved_kib);

        fprintf(data_file_block, "%d,%.2f,%.2f,%.2f,%.2f,%.2f\n", n, time_unfused, time_fused, speedup, bw_unfused, bw_fused);

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        cudaFree(d_temp);
        delete[] h_a;
        delete[] h_b;
    }
    fclose(data_file_naive);
    fclose(data_file_block);
    fclose(results_file);
    return 0;
}