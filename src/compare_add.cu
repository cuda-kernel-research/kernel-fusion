#include <stdio.h>
#include <cuda_runtime.h>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARMUP_ITERS 10
#define BENCHMARK_ITERS 100

__global__ void add_kernel(const float* a, const float* b, float* temp, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp[idx] = a[idx] + b[idx];
    }
}

__global__ void copy_kernel(const float* temp, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = temp[idx];
    }
}

__global__ void add_fused_kernel(const float* a, const float* b, float* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}

float benchmark_unfused(float* d_a, float* d_b, float* d_result, float* d_temp, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < WARMUP_ITERS; i++) {
        add_kernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        copy_kernel<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        add_kernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp, n);
        copy_kernel<<<gridSize, BLOCK_SIZE>>>(d_temp, d_result, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (elapsed_ms * 1000.0f) / BENCHMARK_ITERS;
}

float benchmark_fused(float* d_a, float* d_b, float* d_result, int n) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < WARMUP_ITERS; i++) {
        add_fused_kernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < BENCHMARK_ITERS; i++) {
        add_fused_kernel<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_result, n);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return (elapsed_ms * 1000.0f) / BENCHMARK_ITERS;
}

int main() {
    const int sizes[] = {1024, 10240, 102400, 1024000, 10240000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("Element-wise Add: Fused vs Unfused Comparison\n");
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
            h_a[i] = 1.0f + i * 0.001f;
            h_b[i] = 2.0f + i * 0.001f;
        }

        float *d_a, *d_b, *d_result, *d_temp;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_result, bytes);
        cudaMalloc(&d_temp, bytes);

        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        float time_unfused = benchmark_unfused(d_a, d_b, d_result, d_temp, n);
        float time_fused = benchmark_fused(d_a, d_b, d_result, n);

        float speedup = time_unfused / time_fused;
        float bw_unfused = (5.0f * bytes) / (time_unfused * 1e-6) / 1e9;
        float bw_fused = (3.0f * bytes) / (time_fused * 1e-6) / 1e9;

        float *h_result = new float[std::min(10, n)];
        cudaMemcpy(h_result, d_result, std::min(10, n) * sizeof(float), 
                   cudaMemcpyDeviceToHost);

        bool correct = true;
        const float tolerance = 1e-5f;
        for (int i = 0; i < std::min(10, n); i++) {
            float expected = h_a[i] + h_b[i];
            if (fabs(h_result[i] - expected) > tolerance) {
                correct = false;
                printf("ERROR at index %d: expected=%.6f, got=%.6f\n",
                       i, expected, h_result[i]);
                break;
            }
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

        delete[] h_result;

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        cudaFree(d_temp);
        delete[] h_a;
        delete[] h_b;
    }

    return 0;
}