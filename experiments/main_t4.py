#!/usr/bin/env python3
import subprocess
import csv
import numpy as np
import os

NUMBER_OF_RUNS = 10

def run_benchmark_N_times(source_file, executable, data_filename):
    """Compile and run benchmark 10 times"""
    
    # Compile
    print(f"Compiling {source_file}...")
    result = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_75", source_file, "-o", executable],
        capture_output=True
    )
    
    if result.returncode != 0:
        print("ERROR: Compilation failed!")
        return None
    
    print(" Compiled\n")
    
    # Delete old data file (clean start)
    if os.path.exists(data_filename):
        os.remove(data_filename)
    
    # Run 10 times (C++ will append to data file each time)
    for i in range(NUMBER_OF_RUNS):
        print(f"Run {i+1}/{NUMBER_OF_RUNS}...", end=' ', flush=True)
        subprocess.run([f"./{executable}"], capture_output=True)
        print("correct")
    
    # Clean up executable
    subprocess.run(["rm", executable])
    
    # Read all data at once
    return read_all_data(data_filename)

def read_all_data(filename):
    """Read all runs from data file"""
    data = {}
    
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found!")
        return data
    
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            
            size = int(row[0])
            unfused = float(row[1])
            fused = float(row[2])
            
            if size not in data:
                data[size] = {'unfused': [], 'fused': []}
            
            data[size]['unfused'].append(unfused)
            data[size]['fused'].append(fused)
    
    return data

def print_statistics(data, title):
    """Print statistics table"""
    print(f"\n{'='*90}")
    print(f"{title} - Statistics from {NUMBER_OF_RUNS} runs")
    print('='*90)
    print(f"{'Size':<10} | {'Unfused (μs)':<25} | {'Fused (μs)':<25} | {'Speedup'}")
    print(f"{'':10} | {'mean ± std':<25} | {'mean ± std':<25} |")
    print('-'*90)
    
    for size in sorted(data.keys()):
        u = np.array(data[size]['unfused'])
        f = np.array(data[size]['fused'])
        
        u_m, u_s = np.mean(u), np.std(u, ddof=1)
        f_m, f_s = np.mean(f), np.std(f, ddof=1)

        speedup = u_m / f_m

        print(f"{size:<10} | {u_m:6.2f} ± {u_s:4.2f} | "
              f"{f_m:6.2f} ± {f_s:4.2f} | {speedup:6.2f}x")

def save_stats(data, filename, title):
    """Save statistics to CSV"""
    with open(filename, 'w') as f:
        f.write(f"# {title}\n")
        f.write("Size,Unfused_Mean,Unfused_Std,Fused_Mean,Fused_Std,Speedup\n")
        
        for size in sorted(data.keys()):
            u_m = np.mean(data[size]['unfused'])
            u_s = np.std(data[size]['unfused'], ddof=1)
            f_m = np.mean(data[size]['fused'])
            f_s = np.std(data[size]['fused'], ddof=1)
            
            f.write(f"{size},{u_m:.2f},{u_s:.2f},{f_m:.2f},{f_s:.2f},{u_m/f_m:.2f}\n")

# ===== MAIN =====
if __name__ == "__main__":
    benchmarks = [
        ("compare_add_fp32.cu", "add_fp32", "data_add_fp32.csv", "Add FP32"),
        ("compare_add_fp16.cu", "add_fp16", "data_add_fp16.csv", "Add FP16"),
        ("compare_fma_fp32.cu", "fma_fp32", "data_fma_fp32.csv", "FMA FP32"),
        ("compare_fma_fp16.cu", "fma_fp16", "data_fma_fp16.csv", "FMA FP16"),
        ("relu_fp32.cu", "relu_fp32", "data_relu_fp32.csv", "ReLU FP32"),
        ("relu_fp16.cu", "relu_fp16", "data_relu_fp16.csv", "ReLU FP16"),
        # ("map_reduce_naive_vs_optimized_fp32.cu", "map_reduce_fp32", "data_map_reduce_fp32.csv", "Map Reduce FP32"),
        # ("map_reduce_naive_vs_optimized_fp16.cu", "map_reduce_fp16", "data_map_reduce_fp16.csv", "Map Reduce FP16"),
    ]

    for source, exe, data_file, title in benchmarks:
        print(f"\n{'#'*90}")
        print(f"# {title}")
        print('#'*90)
        
        # Run 10 times
        data = run_benchmark_N_times(source, exe, data_file)
        
        if not data:
            continue
        
        # Print stats
        print_statistics(data, title)
        
        # Save stats
        save_stats(data, f"stats_{exe}.csv", title)
        print(f" Saved to stats_{exe}.csv")

    print("\n Done!")