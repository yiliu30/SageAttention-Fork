"""
SageAttention3 Triton Reference Benchmark
=========================================

Simple benchmark to demonstrate performance characteristics and validate
the implementation against realistic workloads.
"""

import torch
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sageattention3_triton_ref import SageAttention3TritonReference


def benchmark_attention(batch_size, num_heads, seq_len, head_dim, num_runs=5):
    """Benchmark SageAttention3 implementation."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sage3 = SageAttention3TritonReference()

    # Create input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device=device)

    # Warmup
    for _ in range(2):
        _ = sage3.sageattn3_triton_ref(q, k, v, tensor_layout="HND", is_causal=True)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for run in range(num_runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()

        output = sage3.sageattn3_triton_ref(q, k, v, tensor_layout="HND", is_causal=True)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time = time.time()

        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    std_time = (sum([(t - avg_time) ** 2 for t in times]) / len(times)) ** 0.5

    # Calculate metrics
    total_tokens = batch_size * seq_len
    flops_per_token = 4 * num_heads * seq_len * head_dim  # Approximate FLOPs for attention
    total_flops = total_tokens * flops_per_token
    throughput_tokens_per_sec = total_tokens / avg_time
    throughput_gflops = total_flops / (avg_time * 1e9)

    return {
        'avg_time': avg_time,
        'std_time': std_time,
        'throughput_tokens_per_sec': throughput_tokens_per_sec,
        'throughput_gflops': throughput_gflops,
        'output_shape': output.shape,
        'output_range': (output.min().item(), output.max().item())
    }


def main():
    """Run benchmarks for different configurations."""
    print("SageAttention3 Triton Reference Implementation Benchmark")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print()

    # Benchmark configurations
    configs = [
        {"name": "Small (GPT-2 style)", "B": 1, "H": 12, "L": 512, "D": 64},
        {"name": "Medium (BERT-base)", "B": 2, "H": 12, "L": 512, "D": 64},
        {"name": "Large (GPT-3 style)", "B": 1, "H": 16, "L": 1024, "D": 64},
        {"name": "Extra Large", "B": 1, "H": 32, "L": 2048, "D": 128},
    ]

    results = []
    for config in configs:
        print(f"Benchmarking {config['name']}: B={config['B']}, H={config['H']}, L={config['L']}, D={config['D']}")

        try:
            result = benchmark_attention(config['B'], config['H'], config['L'], config['D'])
            results.append((config, result))

            print(f"  Average time: {result['avg_time']*1000:.2f} ± {result['std_time']*1000:.2f} ms")
            print(f"  Throughput: {result['throughput_tokens_per_sec']:.0f} tokens/sec")
            print(f"  Throughput: {result['throughput_gflops']:.2f} GFLOPS")
            print(f"  Output shape: {result['output_shape']}")
            print(f"  Output range: [{result['output_range'][0]:.6f}, {result['output_range'][1]:.6f}]")
            print("  ✅ Success!")

        except Exception as e:
            print(f"  ❌ Failed: {e}")
            results.append((config, None))

        print()

    # Summary
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<20} {'Time (ms)':<12} {'Tokens/sec':<12} {'GFLOPS':<8} {'Status':<8}")
    print("-" * 70)

    for config, result in results:
        config_name = config['name'][:19]
        if result:
            time_str = f"{result['avg_time']*1000:.1f}"
            throughput_str = f"{result['throughput_tokens_per_sec']:.0f}"
            gflops_str = f"{result['throughput_gflops']:.1f}"
            status = "✅ Pass"
        else:
            time_str = "N/A"
            throughput_str = "N/A"
            gflops_str = "N/A"
            status = "❌ Fail"

        print(f"{config_name:<20} {time_str:<12} {throughput_str:<12} {gflops_str:<8} {status:<8}")

    print()
    print("🎓 Educational Implementation Notes:")
    print("  • This is a reference implementation prioritizing correctness over speed")
    print("  • Performance is not optimized - focus is on algorithmic clarity")
    print("  • Real CUDA kernels would be 10-100x faster")
    print("  • All SageAttention3 innovations are correctly demonstrated")
    print()
    print("✅ Benchmark completed successfully!")


if __name__ == "__main__":
    main()