#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARMUP_ITERS 10
#define BENCHMARK_ITERS 100

__global__ void scale_kernel(float alpha, const float* a, float* temp1, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp1[idx] = alpha * a[idx];
    }
}

__global__ void add_kernel(const float* temp1, const float* b, float* temp2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp2[idx] = temp1[idx] + b[idx];
    }
}

__global__ void relu_kernel(const float* temp2, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = fmaxf(0.0f, temp2[idx]);
    }
}

__global__ void scaled_add_relu_fused(float alpha, const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = alpha * a[idx] + b[idx];
        result[idx] = fmaxf(0.0f, val);
    }
}

float benchmark_unfused(float alpha, float* d_a, float* d_b, float* d_result, 
                        float* d_temp1, float* d_temp2, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; i++) {
        scale_kernel<<<gridSize, BLOCK_SIZE>>>(alpha, d_a, d_temp1, n);
        add_kernel<<<gridSize, BLOCK_SIZE>>>(d_temp1, d_b, d_temp2, n);
        relu_kernel<<<gridSize, BLOCK_SIZE>>>(d_temp2, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        scale_kernel<<<gridSize, BLOCK_SIZE>>>(alpha, d_a, d_temp1, n);
        add_kernel<<<gridSize, BLOCK_SIZE>>>(d_temp1, d_b, d_temp2, n);
        relu_kernel<<<gridSize, BLOCK_SIZE>>>(d_temp2, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (milliseconds * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_fused(float alpha, float* d_a, float* d_b, float* d_result, 
                      int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERS; i++) {
        scaled_add_relu_fused<<<gridSize, BLOCK_SIZE>>>(alpha, d_a, d_b, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        scaled_add_relu_fused<<<gridSize, BLOCK_SIZE>>>(alpha, d_a, d_b, d_result, n);
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
    const float alpha = 0.01f;

    printf("Scaled Add + ReLU: Fused vs Unfused Comparison\n");
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
            h_a[i] = -10.0f + i * 0.002f; 
            h_b[i] = -5.0f + i * 0.001f;  
        }

        float *d_a, *d_b, *d_result, *d_temp1, *d_temp2;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_result, bytes);
        cudaMalloc(&d_temp1, bytes); 
        cudaMalloc(&d_temp2, bytes); 

        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        float time_unfused = benchmark_unfused(alpha, d_a, d_b, d_result, d_temp1, d_temp2, n);
        float time_fused = benchmark_fused(alpha, d_a, d_b, d_result, n);

        float speedup = time_unfused / time_fused;
        float bw_unfused = (7.0f * bytes) / (time_unfused * 1e-6) / 1e9;
        float bw_fused = (3.0f * bytes) / (time_fused * 1e-6) / 1e9;
        float memory_saved_mb = (2.0f * bytes) / (1024 * 1024);
        
        float *h_result = new float[std::min(10, n)];
        cudaMemcpy(h_result, d_result, std::min(10, n) * sizeof(float), 
                   cudaMemcpyDeviceToHost);

        bool correct = true;
        const float tolerance = 1e-5f;
        for (int i = 0; i < std::min(10, n); i++) {
            float expected = fmaxf(0.0f, alpha * h_a[i] + h_b[i]);
            if (fabs(h_result[i] - expected) > tolerance) {
                correct = false;
                printf("ERROR at index %d: expected=%.6f, got=%.6f\n",
                       i, expected, h_result[i]);
                break;
            }
        }

        if (memory_saved_mb < 0.1) {
            float memory_saved_kb = (2.0f * bytes) / 1024;
            printf("%d\t\t%.2f\t\t%.2f\t\t%.2fx\t\t%.2f\t\t%.2f\t\t%.1f KB\n",
                n, time_unfused, time_fused, speedup, bw_unfused, bw_fused, memory_saved_kb);
        } else {
            printf("%d\t\t%.2f\t\t%.2f\t\t%.2fx\t\t%.2f\t\t%.2f\t\t%.1f MB\n",
                n, time_unfused, time_fused, speedup, bw_unfused, bw_fused, memory_saved_mb);
        }

        delete[] h_result;

        if (!correct) {
            printf("WARNING: Verification failed for size %d\n", n);
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        cudaFree(d_temp1);
        cudaFree(d_temp2);
        delete[] h_a;
        delete[] h_b;
    }

    return 0;
}