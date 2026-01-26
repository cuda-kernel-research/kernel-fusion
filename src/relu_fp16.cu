#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARMUP_ITERS 10
#define BENCHMARK_ITERS 100
#define NUM_BATCHES 5

__global__ void scale_kernel_fp16(float alpha, const half* a, half* temp1, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half alpha_h = __float2half(alpha);
        temp1[idx] = __hmul(alpha_h, a[idx]);
    }
}

__global__ void add_kernel_fp16(const half* temp1, const half* b, half* temp2, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp2[idx] = __hadd(temp1[idx], b[idx]);
    }
}

__global__ void relu_kernel_fp16(const half* temp2, half* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half zero = __float2half(0.0f);
        result[idx] = __hmax(temp2[idx], zero);
    }
}

__global__ void scaled_add_relu_fused_fp16(float alpha, const half* a, const half* b, half* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half alpha_h = __float2half(alpha);
        half val = __hfma(alpha_h, a[idx], b[idx]);
        half zero = __float2half(0.0f);
        result[idx] = __hmax(val, zero);
    }
}

float benchmark_unfused_fp16(float alpha, half* d_a, half* d_b, half* d_result, 
                              half* d_temp1, half* d_temp2, int n, float* std_dev) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ITERS_PER_BATCH = BENCHMARK_ITERS / NUM_BATCHES;

    for (int i = 0; i < WARMUP_ITERS; i++) {
        scale_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(alpha, d_a, d_temp1, n);
        add_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_temp1, d_b, d_temp2, n);
        relu_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_temp2, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float batch_times[NUM_BATCHES];

    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        cudaEventRecord(start);
        
        for (int i = 0; i < ITERS_PER_BATCH; i++) {
            scale_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(alpha, d_a, d_temp1, n);
            add_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_temp1, d_b, d_temp2, n);
            relu_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_temp2, d_result, n);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float batch_elapsed_ms;
        cudaEventElapsedTime(&batch_elapsed_ms, start, stop);
        batch_times[batch] = (batch_elapsed_ms * 1000.0f) / ITERS_PER_BATCH;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float mean = 0.0f;
    for (int i = 0; i < NUM_BATCHES; i++) {
        mean += batch_times[i];
    }
    mean /= NUM_BATCHES;

    float variance = 0.0f;
    for (int i = 0; i < NUM_BATCHES; i++) {
        float diff = batch_times[i] - mean;
        variance += diff * diff;
    }
    variance /= NUM_BATCHES;
    *std_dev = sqrt(variance);

    return mean;
}

float benchmark_fused_fp16(float alpha, half* d_a, half* d_b, half* d_result, 
                            int n, float* std_dev) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ITERS_PER_BATCH = BENCHMARK_ITERS / NUM_BATCHES;

    for (int i = 0; i < WARMUP_ITERS; i++) {
        scaled_add_relu_fused_fp16<<<gridSize, BLOCK_SIZE>>>(alpha, d_a, d_b, d_result, n);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float batch_times[NUM_BATCHES];

    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        cudaEventRecord(start);
        
        for (int i = 0; i < ITERS_PER_BATCH; i++) {
            scaled_add_relu_fused_fp16<<<gridSize, BLOCK_SIZE>>>(alpha, d_a, d_b, d_result, n);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float batch_elapsed_ms;
        cudaEventElapsedTime(&batch_elapsed_ms, start, stop);
        batch_times[batch] = (batch_elapsed_ms * 1000.0f) / ITERS_PER_BATCH;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float mean = 0.0f;
    for (int i = 0; i < NUM_BATCHES; i++) {
        mean += batch_times[i];
    }
    mean /= NUM_BATCHES;

    float variance = 0.0f;
    for (int i = 0; i < NUM_BATCHES; i++) {
        float diff = batch_times[i] - mean;
        variance += diff * diff;
    }
    variance /= NUM_BATCHES;
    *std_dev = sqrt(variance);

    return mean;
}

int main() {
    const int sizes[] = {1024, 10240, 102400, 1024000, 10240000};
    const int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const float alpha = 0.01f;

    printf("Scaled Add + ReLU FP16: Fused vs Unfused Comparison\n");
    printf("=========================================================================================================================\n");
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
    printf("-------------------------------------------------------------------------------------------------------------------------\n");

    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        size_t bytes = n * sizeof(half);

        float *h_a_float = new float[n];
        float *h_b_float = new float[n];
        half *h_a = new half[n];
        half *h_b = new half[n];

        for (int i = 0; i < n; i++) {
            h_a_float[i] = -10.0f + i * 0.002f; 
            h_b_float[i] = -5.0f + i * 0.001f;
            h_a[i] = __float2half(h_a_float[i]);
            h_b[i] = __float2half(h_b_float[i]);
        }

        half *d_a, *d_b, *d_result, *d_temp1, *d_temp2;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_result, bytes);
        cudaMalloc(&d_temp1, bytes); 
        cudaMalloc(&d_temp2, bytes); 

        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        float std_unfused, std_fused;
        float time_unfused = benchmark_unfused_fp16(alpha, d_a, d_b, d_result, d_temp1, d_temp2, n, &std_unfused);
        float time_fused = benchmark_fused_fp16(alpha, d_a, d_b, d_result, n, &std_fused);

        float cv_unfused = (std_unfused / time_unfused) * 100.0f;
        float cv_fused = (std_fused / time_fused) * 100.0f;

        float speedup = time_unfused / time_fused;
        float bw_unfused = (7.0f * bytes) / (time_unfused * 1e-6) / 1e9;
        float bw_fused = (3.0f * bytes) / (time_fused * 1e-6) / 1e9;
        float memory_saved_kib = (2.0f * bytes) / 1024.0f;
        
        half *h_result = new half[std::min(10, n)];
        cudaMemcpy(h_result, d_result, std::min(10, n) * sizeof(half), 
                   cudaMemcpyDeviceToHost);

        bool correct = true;
        const float rel_tolerance = 1e-2f;
        for (int i = 0; i < std::min(10, n); i++) {
            float expected = fmaxf(0.0f, alpha * h_a_float[i] + h_b_float[i]);
            float result_float = __half2float(h_result[i]);
            float rel_error = fabs(result_float - expected) / (fabs(expected) + 1e-7f);
            if (rel_error > rel_tolerance) {
                correct = false;
                printf("ERROR at index %d: expected=%.6f, got=%.6f (rel_error=%.6f)\n",
                       i, expected, result_float, rel_error);
                break;
            }
        }

        printf("%-10d | %7.2f ± %4.2f (%3.1f%%) | %7.2f ± %4.2f (%3.1f%%) | %8.2fx | %12.2f | %10.2f | %12.1f\n",
            n,
            time_unfused, std_unfused, cv_unfused,
            time_fused, std_fused, cv_fused,
            speedup,
            bw_unfused,
            bw_fused,
            memory_saved_kib);

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
        delete[] h_a_float;
        delete[] h_b_float;
    }

    return 0;
}

// nvcc -O3 -arch=sm_75 relu_fp16.cu -o izlaz