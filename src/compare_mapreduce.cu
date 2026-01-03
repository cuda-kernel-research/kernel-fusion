#include <stdio.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARMUP_ITERS 10
#define BENCHMARK_ITERS 100

__global__ void map_kernel(const float* a, const float* b, float* temp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp[idx] = a[idx] * b[idx];
    }
}

__global__ void reduce_kernel_naive(const float* temp, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(result, temp[idx]);
    }
}

__global__ void mapreduce_fused_naive(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float partial = a[idx] * b[idx];
        atomicAdd(result, partial);
    }
}

float benchmark_unfused(float* d_a, float* d_b, float* d_result, 
                        float* d_temp, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < WARMUP_ITERS; i++) {
        cudaMemset(d_result, 0, sizeof(float));
        map_kernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        reduce_kernel_naive<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
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

float benchmark_fused(float* d_a, float* d_b, float* d_result, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    for (int i = 0; i < WARMUP_ITERS; i++) {
        cudaMemset(d_result, 0, sizeof(float));
        mapreduce_fused_naive<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
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

int main() {
    const int sizes[] = {1024, 10240, 102400, 1024000, 10240000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("Map-Reduce sum(A*B): Fused vs Unfused Comparison (Naive Implementation)\n");
    printf("================================================================================\n");
    printf("Size\t\tUnfused(μs)\tFused(μs)\tSpeedup\t\tBW Unfused\tBW Fused\tMemory Saved\n");
    printf("\t\t\t\t\t\t\t(GB/s)\t\t(GB/s)\n");
    printf("================================================================================\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        size_t bytes = n * sizeof(float);
        
        float *h_a = new float[n];
        float *h_b = new float[n];
        
        for (int i = 0; i < n; i++) {
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
        
        float time_unfused = benchmark_unfused(d_a, d_b, d_result, d_temp, n);
        float h_result_unfused = 0.0f;
        cudaMemcpy(&h_result_unfused, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        float time_fused = benchmark_fused(d_a, d_b, d_result, n);
        float h_result_fused = 0.0f;
        cudaMemcpy(&h_result_fused, d_result, sizeof(float), cudaMemcpyDeviceToHost);
        
        float speedup = time_unfused / time_fused;
        
        float bw_unfused = (4.0f * bytes) / (time_unfused * 1e-6) / 1e9;
        float bw_fused = (2.0f * bytes) / (time_fused * 1e-6) / 1e9;
        
        float expected = 0.0f;
        for (int i = 0; i < n; i++) {
            expected += h_a[i] * h_b[i];
        }
        
        bool correct = true;
        const float rel_tolerance = 0.001f;  
        float tolerance = fabs(expected) * rel_tolerance;
        if (fabs(h_result_unfused - expected) > tolerance || 
            fabs(h_result_fused - expected) > tolerance) {
            correct = false;
            printf("ERROR: expected=%.2f, unfused=%.2f, fused=%.2f\n",
                   expected, h_result_unfused, h_result_fused);
        }
        
        float memory_saved_mb = (float)bytes / (1024 * 1024);
        if (memory_saved_mb < 0.1) {
            float memory_saved_kb = (float)bytes / 1024;
            printf("%d\t\t%.2f\t\t%.2f\t\t%.2fx\t\t%.2f\t\t%.2f\t\t%.1f KB\n",
                   n, time_unfused, time_fused, speedup, bw_unfused, bw_fused, memory_saved_kb);
        } else {
            printf("%d\t\t%.2f\t\t%.2f\t\t%.2fx\t\t%.2f\t\t%.2f\t\t%.1f MB\n",
                   n, time_unfused, time_fused, speedup, bw_unfused, bw_fused, memory_saved_mb);
        }

        if (!correct) {
            printf("WARNING: Verification failed for size %d\n", n);
        }
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        cudaFree(d_temp);
        delete[] h_a;
        delete[] h_b;
    }
    
    return 0;
}