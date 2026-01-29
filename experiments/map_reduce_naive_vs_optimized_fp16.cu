#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARMUP_ITERS 10
#define BENCHMARK_ITERS 100

__global__ void map_kernel_fp16(const half* a, const half* b, half* temp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp[idx] = __hmul(a[idx], b[idx]);
    }
}

__global__ void reduce_kernel_fp16_naive(const half* temp, half* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, temp[idx]);
    }
}

__global__ void mapreduce_fused_fp16_naive(const half* a, const half* b, half* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half partial_h = __hmul(a[idx], b[idx]);
        atomicAdd(result, partial_h);
    }
}

__global__ void reduce_kernel_fp16_block(const half* temp, half* result, int n) {
    __shared__ half sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? temp[idx] : __float2half(0.0f);
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

__global__ void mapreduce_fused_fp16_block(const half* a, const half* b, half* result, int n) {
    __shared__ half sdata[BLOCK_SIZE];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? a[idx] * b[idx] : __float2half(0.0f);
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

float benchmark_unfused_fp16_naive(half* d_a, half* d_b, half* d_result, 
                                    half* d_temp, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int i = 0; i < WARMUP_ITERS; i++) {
        cudaMemset(d_result, 0, sizeof(half));
        map_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_fp16_naive<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        cudaMemset(d_result, 0, sizeof(half));
        map_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_fp16_naive<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_fused_fp16_naive(half* d_a, half* d_b, half* d_result, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int i = 0; i < WARMUP_ITERS; i++) {
        cudaMemset(d_result, 0, sizeof(half));
        mapreduce_fused_fp16_naive<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        cudaMemset(d_result, 0, sizeof(half));
        mapreduce_fused_fp16_naive<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_unfused_fp16_block(half* d_a, half* d_b, half* d_result, 
                               half* d_temp, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < WARMUP_ITERS; i++) {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_fp16_block<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_fp16_block<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_fused_fp16_block(half* d_a, half* d_b, half* d_result, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < WARMUP_ITERS; i++) {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_fp16_block<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_fp16_block<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

int main() {
    const int sizes[] = {1024, 10240, 102400, 1024000, 10240000, 102400000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    FILE* results_file = fopen("results_map_reduce_fp16.txt", "w");
    FILE* data_file_naive = fopen("data_map_reduce_fp16_naive.csv", "a");
    FILE* data_file_block = fopen("data_map_reduce_fp16_block.csv", "a");

    if (results_file == NULL || data_file_naive == NULL || data_file_block == NULL) {
        printf("ERROR: Cannot open results files\n");
        return 1;
    }

    printf("Map-Reduce sum(A*B) FP16: NAIVE Implementation\n");
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

    fprintf(results_file, "Map-Reduce sum(A*B) FP16: NAIVE Implementation\n");
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
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        size_t bytes_fp16 = n * sizeof(half);
        
        float *h_a_float = new float[n];
        float *h_b_float = new float[n];
        half *h_a = new half[n];
        half *h_b = new half[n];
        
        for (int i = 0; i < n; i++) {
            h_a_float[i] = 1.0f + (i % 100) * 0.01f;
            h_b_float[i] = 2.0f + (i % 100) * 0.01f;
            h_a[i] = __float2half(h_a_float[i]);
            h_b[i] = __float2half(h_b_float[i]);
        }
        
        half *d_a, *d_b, *d_temp, *d_result;
        cudaMalloc(&d_a, bytes_fp16);
        cudaMalloc(&d_b, bytes_fp16);
        cudaMalloc(&d_result, sizeof(half));
        cudaMalloc(&d_temp, bytes_fp16);
        
        cudaMemcpy(d_a, h_a, bytes_fp16, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes_fp16, cudaMemcpyHostToDevice);
        
        float time_unfused_naive = benchmark_unfused_fp16_naive(d_a, d_b, d_result, d_temp, n);
        half h_result_unfused_naive;
        cudaMemcpy(&h_result_unfused_naive, d_result, sizeof(half), cudaMemcpyDeviceToHost);
        
        float time_fused_naive = benchmark_fused_fp16_naive(d_a, d_b, d_result, n);
        half h_result_fused_naive;
        cudaMemcpy(&h_result_fused_naive, d_result, sizeof(half), cudaMemcpyDeviceToHost);
        
        float speedup_naive = time_unfused_naive / time_fused_naive;
        float bw_unfused_naive = (4.0f * bytes_fp16) / (time_unfused_naive * 1e-6) / 1e9;
        float bw_fused_naive = (2.0f * bytes_fp16) / (time_fused_naive * 1e-6) / 1e9;        
        float memory_saved_kib = bytes_fp16 / 1024.0f;
        printf("%-10d | %17.2f | %15.2f | %8.2fx | %12.2f | %10.2f | %12.1f\n",
            n,
            time_unfused_naive,
            time_fused_naive,
            speedup_naive,
            bw_unfused_naive,
            bw_fused_naive,
            memory_saved_kib);
        fprintf(results_file, "%-10d | %17.2f | %15.2f | %8.2fx | %12.2f | %10.2f | %12.1f\n",
            n,
            time_unfused_naive,
            time_fused_naive,
            speedup_naive,
            bw_unfused_naive,
            bw_fused_naive,
            memory_saved_kib);
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
    
    printf("\n\nMap-Reduce sum(A*B) FP16: BLOCK-LEVEL Implementation\n");
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

    fprintf(results_file, "\n\nMap-Reduce sum(A*B) FP16: BLOCK-LEVEL Implementation\n");
    fprintf(results_file, "===============================================================================================================\n");
    fprintf(results_file, "%-10s | %-12s | %-12s | %-9s | %-12s | %-10s | %-12s\n",
        "Size",
        "Unfused Time (μs)",
        "Fused Time (μs)",
        "Speedup",
        "BW Unfused",
        "BW Fused",
        "Memory Saved");
    fprintf(results_file,"%-10s | %-17s | %-15s | %-9s | %-12s | %-10s | %-12s\n",
        "",
        "",
        "",
        "",
        "(GB/s)",
        "(GB/s)",
        "(KiB)");
    fprintf(results_file,"================================================================================\n");

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        size_t bytes_fp16 = n * sizeof(half);
        
        float *h_a_float = new float[n];
        float *h_b_float = new float[n];
        half *h_a = new half[n];
        half *h_b = new half[n];
        
        for (int i = 0; i < n; i++) {
            h_a_float[i] = 1.0f + (i % 100) * 0.01f;
            h_b_float[i] = 2.0f + (i % 100) * 0.01f;
            h_a[i] = __float2half(h_a_float[i]);
            h_b[i] = __float2half(h_b_float[i]);
        }
        
        half *d_a, *d_b, *d_result, *d_temp;
        cudaMalloc(&d_a, bytes_fp16);
        cudaMalloc(&d_b, bytes_fp16);
        cudaMalloc(&d_result, sizeof(half));
        cudaMalloc(&d_temp, bytes_fp16);
        
        cudaMemcpy(d_a, h_a, bytes_fp16, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes_fp16, cudaMemcpyHostToDevice);
        
        float time_unfused_block = benchmark_unfused_fp16_block(d_a, d_b, d_result, d_temp, n);
        half h_result_unfused_block;
        cudaMemcpy(&h_result_unfused_block, d_result, sizeof(half), cudaMemcpyDeviceToHost);
        
        float time_fused_block = benchmark_fused_fp16_block(d_a, d_b, d_result, n);
        half h_result_fused_block;
        cudaMemcpy(&h_result_fused_block, d_result, sizeof(half), cudaMemcpyDeviceToHost);
        
        float speedup_block = time_unfused_block / time_fused_block;
        float bw_unfused_block = (4.0f * bytes_fp16) / (time_unfused_block * 1e-6) / 1e9;
        float bw_fused_block = (2.0f * bytes_fp16) / (time_fused_block * 1e-6) / 1e9;

        float memory_saved_kib = bytes_fp16 / 1024.0f;
        printf("%-10d | %17.2f | %15.2f | %8.2fx | %12.2f | %10.2f | %12.1f\n",
            n,
            time_unfused_block,
            time_fused_block,
            speedup_block,
            bw_unfused_block,
            bw_fused_block,
            memory_saved_kib);
        fprintf(results_file,"%-10d | %17.2f | %15.2f | %8.2fx | %12.2f | %10.2f | %12.1f\n",
            n,
            time_unfused_block,
            time_fused_block,
            speedup_block,
            bw_unfused_block,
            bw_fused_block,
            memory_saved_kib);
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
    fclose(results_file);
    return 0;
}