#!/usr/bin/env python3
import subprocess
import csv
import numpy as np
import os

NUMBER_OF_RUNS = 10

def run_benchmark_n_times(source_file, executable, data_file_naive, data_file_block):
    """Compile and run benchmark NUMBER_OF_RUNS times"""
    
    # Compile
    print(f"Compiling {source_file}...")
    result = subprocess.run(
        ["nvcc", "-O3", "-arch=sm_80", source_file, "-o", executable],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    if result.returncode != 0:
        print("ERROR: Compilation failed!")
        print(result.stderr.decode())
        print(result.stdout.decode())
        return None, None
    
    print("✓ Compiled\n")
    
    # Delete old data files (clean start)
    if os.path.exists(data_file_naive):
        os.remove(data_file_naive)
    if os.path.exists(data_file_block):
        os.remove(data_file_block)
    
    # Run NUMBER_OF_RUNS times (C++ will append to data files each time)
    for i in range(NUMBER_OF_RUNS):
        print(f"Run {i+1}/{NUMBER_OF_RUNS}...", end=' ', flush=True)
        result = subprocess.run([f"./{executable}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print(f"ERROR (exit code {result.returncode})!")
            print("STDERR:", result.stderr.decode())
            print("STDOUT:", result.stdout.decode())
            return None, None
        print("✓")
    
    # Clean up executable
    subprocess.run(["rm", executable], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Read data from separate files
    data_naive = read_all_data(data_file_naive)
    data_block = read_all_data(data_file_block)
    
    return data_naive, data_block

def read_all_data(filename):
    """Read all runs from data file"""
    data = {}
    
    if not os.path.exists(filename):
        print(f"ERROR: {filename} not found!")
        return data
    
    with open(filename, 'r') as infile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) != 6:
                continue
            
            try:
                size = int(row[0])
                unfused = float(row[1])
                fused = float(row[2])
                speedup = float(row[3])
                bw_unfused = float(row[4])
                bw_fused = float(row[5])
            except ValueError:
                continue
            
            if size not in data:
                data[size] = {'unfused': [], 'fused': [], 'speedup': [], 'bw_unfused': [], 'bw_fused': []}
            
            data[size]['unfused'].append(unfused)
            data[size]['fused'].append(fused)
            data[size]['speedup'].append(speedup)
            data[size]['bw_unfused'].append(bw_unfused)
            data[size]['bw_fused'].append(bw_fused)
    
    return data

def print_statistics(data, title):
    """Print statistics table"""
    print(f"\n{'='*150}")
    print(f"{title} - Statistics from {NUMBER_OF_RUNS} runs")
    print('='*150)
    print(f"{'Size':<10} | {'Unfused (μs)':<20} | {'Fused (μs)':<20} | {'Speedup':<15} | {'BW Unfused (GB/s)':<20} | {'BW Fused (GB/s)':<20}")
    print(f"{'':10} | {'mean ± std':<20} | {'mean ± std':<20} | {'mean ± std':<15} | {'mean ± std':<20} | {'mean ± std':<20}")
    print('-'*150)
    
    for size in sorted(data.keys()):
        u = np.array(data[size]['unfused'])
        f = np.array(data[size]['fused'])
        s = np.array(data[size]['speedup'])
        bw_u = np.array(data[size]['bw_unfused'])
        bw_f = np.array(data[size]['bw_fused'])
        
        if len(u) < 2:
            print(f"{size:<10} | WARNING: Only {len(u)} sample(s)!")
            continue
        
        u_m, u_s = np.mean(u), np.std(u, ddof=1)
        f_m, f_s = np.mean(f), np.std(f, ddof=1)
        s_m, s_s = np.mean(s), np.std(s, ddof=1)
        bw_u_m, bw_u_s = np.mean(bw_u), np.std(bw_u, ddof=1)
        bw_f_m, bw_f_s = np.mean(bw_f), np.std(bw_f, ddof=1)

        print(f"{size:<10} | {u_m:7.2f} ± {u_s:5.2f} | {f_m:7.2f} ± {f_s:5.2f} | "
              f"{s_m:5.2f} ± {s_s:4.2f} | {bw_u_m:7.2f} ± {bw_u_s:5.2f} | {bw_f_m:7.2f} ± {bw_f_s:5.2f}")

def save_stats(data, filename, title):
    """Save statistics to CSV"""
    with open(filename, 'w') as outfile:
        outfile.write(f"# {title}\n")
        outfile.write("Size,Unfused_Mean,Unfused_Std,Fused_Mean,Fused_Std,Speedup_Mean,Speedup_Std,BW_Unfused_Mean,BW_Unfused_Std,BW_Fused_Mean,BW_Fused_Std,Num_Samples\n")
        
        for size in sorted(data.keys()):
            u = np.array(data[size]['unfused'])
            f = np.array(data[size]['fused'])
            s = np.array(data[size]['speedup'])
            bw_u = np.array(data[size]['bw_unfused'])
            bw_f = np.array(data[size]['bw_fused'])
            
            if len(u) < 2:
                continue
            
            u_m = np.mean(u)
            u_s = np.std(u, ddof=1)
            f_m = np.mean(f)
            f_s = np.std(f, ddof=1)
            s_m = np.mean(s)
            s_s = np.std(s, ddof=1)
            bw_u_m = np.mean(bw_u)
            bw_u_s = np.std(bw_u, ddof=1)
            bw_f_m = np.mean(bw_f)
            bw_f_s = np.std(bw_f, ddof=1)
            
            outfile.write(f"{size},{u_m:.2f},{u_s:.2f},{f_m:.2f},{f_s:.2f},{s_m:.2f},{s_s:.2f},{bw_u_m:.2f},{bw_u_s:.2f},{bw_f_m:.2f},{bw_f_s:.2f},{len(u)}\n")

# ===== MAIN =====
if __name__ == "__main__":
    benchmarks = [
        ("map_reduce_naive_vs_optimized_fp32.cu", "map_reduce_fp32", 
         "data_map_reduce_fp32_naive.csv", "data_map_reduce_fp32_block.csv", "Map Reduce FP32"),
        ("map_reduce_naive_vs_optimized_fp16.cu", "map_reduce_fp16", 
         "data_map_reduce_fp16_naive.csv", "data_map_reduce_fp16_block.csv", "Map Reduce FP16"),
    ]

    for source, exe, data_file_naive, data_file_block, title in benchmarks:
        print(f"\n{'#'*90}")
        print(f"# {title}")
        print('#'*90)
        
        # Run NUMBER_OF_RUNS times
        data_naive, data_block = run_benchmark_n_times(source, exe, data_file_naive, data_file_block)
        
        if not data_naive or not data_block:
            print("ERROR: No data collected!")
            continue
        
        # Print and save NAIVE stats
        print_statistics(data_naive, f"{title} - NAIVE")
        save_stats(data_naive, f"stats_{exe}_naive.csv", f"{title} - NAIVE")
        print(f"✓ Saved to stats_{exe}_naive.csv")
        
        # Print and save BLOCK-LEVEL stats
        print_statistics(data_block, f"{title} - BLOCK-LEVEL")
        save_stats(data_block, f"stats_{exe}_block.csv", f"{title} - BLOCK-LEVEL")
        print(f"✓ Saved to stats_{exe}_block.csv")

    print("\n✓ Done!")
