#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // ← DODATO za FP16
#include <cmath>
#include <algorithm>

#define BLOCK_SIZE 256
#define WARMUP_ITERS 10
#define BENCHMARK_ITERS 100
#define NUM_BATCHES 5

// ← FP16 KERNELS
__global__ void multiply_kernel_fp16(const half* a, const half* b, half* temp1, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        temp1[idx] = __hmul(a[idx], b[idx]);  // FP16 multiply
    }
}

__global__ void add_kernel_fp16(const half* temp1, const half* d, half* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = __hadd(temp1[idx], d[idx]);  // FP16 add
    }
}

__global__ void fma_fused_kernel_fp16(const half* a, const half* b, const half* d, half* result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = __hfma(a[idx], b[idx], d[idx]);  // FP16 FMA
    }
}

float benchmark_unfused_fp16(half* d_a, half* d_b, half* d_d, half* d_result, 
                              half* d_temp1, int n, float* std_dev) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ITERS_PER_BATCH = BENCHMARK_ITERS / NUM_BATCHES;
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        multiply_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp1, n);
        add_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_temp1, d_d, d_result, n);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float batch_times[NUM_BATCHES];
    
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        cudaEventRecord(start);
        
        for (int i = 0; i < ITERS_PER_BATCH; i++) {
            multiply_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_temp1, n);
            add_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_temp1, d_d, d_result, n);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float batch_elapsed_ms;
        cudaEventElapsedTime(&batch_elapsed_ms, start, stop);
        batch_times[batch] = (batch_elapsed_ms * 1000.0f) / ITERS_PER_BATCH;
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < NUM_BATCHES; i++) {
        mean += batch_times[i];
    }
    mean /= NUM_BATCHES;
    
    // Calculate std dev
    float variance = 0.0f;
    for (int i = 0; i < NUM_BATCHES; i++) {
        float diff = batch_times[i] - mean;
        variance += diff * diff;
    }
    variance /= NUM_BATCHES;
    *std_dev = sqrt(variance);
    
    return mean;
}

float benchmark_fused_fp16(half* d_a, half* d_b, half* d_d, half* d_result, 
                            int n, float* std_dev) {
    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int ITERS_PER_BATCH = BENCHMARK_ITERS / NUM_BATCHES;
    
    // Warmup
    for (int i = 0; i < WARMUP_ITERS; i++) {
        fma_fused_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_d, d_result, n);
    }
    cudaDeviceSynchronize();
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float batch_times[NUM_BATCHES];
    
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        cudaEventRecord(start);
        
        for (int i = 0; i < ITERS_PER_BATCH; i++) {
            fma_fused_kernel_fp16<<<gridSize, BLOCK_SIZE>>>(d_a, d_b, d_d, d_result, n);
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float batch_elapsed_ms;
        cudaEventElapsedTime(&batch_elapsed_ms, start, stop);
        batch_times[batch] = (batch_elapsed_ms * 1000.0f) / ITERS_PER_BATCH;
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < NUM_BATCHES; i++) {
        mean += batch_times[i];
    }
    mean /= NUM_BATCHES;
    
    // Calculate std dev
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
    
    printf("\nFused Multiply-Add (FMA) FP16: Fused vs Unfused Comparison\n");
    printf("=========================================================================================================================\n");
    printf("%-10s | %-21s | %-21s | %-9s | %-12s | %-10s | %-12s\n",
           "Size",
           "Unfused Time (μs)",
           "Fused Time (μs)",
           "Speedup",
           "BW Unfused",
           "BW Fused",
           "Memory Saved");
    printf("%-10s | %-21s | %-21s | %-9s | %-12s | %-10s | %-12s\n",
           "",
           "(mean ± std (CV%))",
           "(mean ± std (CV%))",
           "",
           "(GB/s)",
           "(GB/s)",
           "(KiB)");
    printf("-------------------------------------------------------------------------------------------------------------------------\n");
    
    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        size_t bytes = n * sizeof(half);  // ← 2 bytes per element
        
        // ← Host arrays (FP32 for initialization, then convert to FP16)
        float *h_a_float = new float[n];
        float *h_b_float = new float[n];
        float *h_d_float = new float[n];
        half *h_a = new half[n];
        half *h_b = new half[n];
        half *h_d = new half[n];
        
        for (int i = 0; i < n; i++) {
            h_a_float[i] = 1.0f + i * 0.001f;
            h_b_float[i] = 2.0f + i * 0.001f;
            h_d_float[i] = 0.5f + i * 0.0005f;
            
            // ← Convert to FP16
            h_a[i] = __float2half(h_a_float[i]);
            h_b[i] = __float2half(h_b_float[i]);
            h_d[i] = __float2half(h_d_float[i]);
        }
        
        // ← Device arrays (FP16)
        half *d_a, *d_b, *d_d, *d_result, *d_temp1;
        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_d, bytes);
        cudaMalloc(&d_result, bytes);
        cudaMalloc(&d_temp1, bytes);
        
        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_d, h_d, bytes, cudaMemcpyHostToDevice);
        
        float std_unfused, std_fused;
        float time_unfused = benchmark_unfused_fp16(d_a, d_b, d_d, d_result, d_temp1, n, &std_unfused);
        float time_fused = benchmark_fused_fp16(d_a, d_b, d_d, d_result, n, &std_fused);
        
        float cv_unfused = (std_unfused / time_unfused) * 100.0f;
        float cv_fused = (std_fused / time_fused) * 100.0f;
        
        float speedup = time_unfused / time_fused;
        
        // ← Bandwidth (6 reads + 2 writes unfused, 4 reads + 1 write fused)
        float bw_unfused = (6.0f * bytes) / (time_unfused * 1e-6) / 1e9;
        float bw_fused = (4.0f * bytes) / (time_fused * 1e-6) / 1e9;
        float memory_saved_kib = bytes / 1024.0f;
        
        // ← Verification (relative error for FP16)
        half *h_result = new half[std::min(10, n)];
        cudaMemcpy(h_result, d_result, std::min(10, n) * sizeof(half), 
                   cudaMemcpyDeviceToHost);
        
        bool correct = true;
        const float rel_tolerance = 1e-2f;  // ← 1% relative error for FP16
        for (int i = 0; i < std::min(10, n); i++) {
            float expected = h_a_float[i] * h_b_float[i] + h_d_float[i];
            float result_float = __half2float(h_result[i]);
            float rel_error = fabs(result_float - expected) / fabs(expected);
            if (rel_error > rel_tolerance) {
                correct = false;
                printf("ERROR at index %d: expected=%.6f, got=%.6f (rel_error=%.6f)\n",
                       i, expected, result_float, rel_error);
                break;
            }
        }
        
        delete[] h_result;
        
        printf("%-10d | %6.2f ± %4.2f (%4.1f%%) | %6.2f ± %4.2f (%4.1f%%) | %7.2fx | %12.2f | %10.2f | %12.1f\n",
               n,
               time_unfused, std_unfused, cv_unfused,
               time_fused, std_fused, cv_fused,
               speedup,
               bw_unfused,
               bw_fused,
               memory_saved_kib);
        
        if (!correct) {
            printf("WARNING: Verification failed for size %d\n", n);
        }
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_d);
        cudaFree(d_result);
        cudaFree(d_temp1);
        delete[] h_a;
        delete[] h_b;
        delete[] h_d;
        delete[] h_a_float;
        delete[] h_b_float;
        delete[] h_d_float;
    }
    
    return 0;
}

// nvcc -O3 -arch=sm_75 compare_fma_fp16.cu -o izlaz