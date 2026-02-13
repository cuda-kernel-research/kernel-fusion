import torch
import torchvision.models as models
import time
import numpy as np

def benchmark_model(model, input_tensor, name, num_warmup=10, num_iterations=100):
    model = model.cuda().eval()
    input_tensor = input_tensor.cuda()
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            output = model(input_tensor)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    unfused_time = np.mean(times)
    unfused_std = np.std(times)
    
    model_jit = torch.jit.script(model)
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model_jit(input_tensor)
    torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            output = model_jit(input_tensor)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
    
    fused_time = np.mean(times)
    fused_std = np.std(times)
    
    speedup = unfused_time / fused_time
    
    print(f"{name:20s} | Unfused: {unfused_time:7.2f}±{unfused_std:5.2f} ms | "
          f"Fused: {fused_time:7.2f}±{fused_std:5.2f} ms | Speedup: {speedup:.2f}x")
    
    return {
        'name': name,
        'unfused_time': unfused_time,
        'unfused_std': unfused_std,
        'fused_time': fused_time,
        'fused_std': fused_std,
        'speedup': speedup
    }

def main():
    print("PyTorch Model Fusion Benchmark")
    print("=" * 90)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 90)
    print()
    
    results = []
    
    models_to_test = [
        ('ResNet18', models.resnet18(weights=None), (1, 3, 224, 224)),
        ('ResNet50', models.resnet50(weights=None), (1, 3, 224, 224)),
        ('MobileNetV2', models.mobilenet_v2(weights=None), (1, 3, 224, 224)),
        ('EfficientNet-B0', models.efficientnet_b0(weights=None), (1, 3, 224, 224)),
    ]
    
    for name, model, input_shape in models_to_test:
        input_tensor = torch.randn(*input_shape)
        result = benchmark_model(model, input_tensor, name)
        results.append(result)
        print()
    
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Model':<20} | {'Unfused (ms)':<15} | {'Fused (ms)':<15} | {'Speedup':<10}")
    print("-" * 90)
    for r in results:
        print(f"{r['name']:<20} | {r['unfused_time']:>7.2f}±{r['unfused_std']:>5.2f} | "
              f"{r['fused_time']:>7.2f}±{r['fused_std']:>5.2f} | {r['speedup']:>6.2f}x")
    print("=" * 90)
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    print(f"\nAverage speedup: {avg_speedup:.2f}x")

if __name__ == "__main__":
    main()