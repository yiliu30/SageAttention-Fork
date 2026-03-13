"""
Analysis of Cumulative Operations Complexity for Online Softmax Optimization

This script analyzes the computational and memory complexity of torch.cumprod and torch.cummax,
which are relevant for eliminating the explicit loop in online softmax.
"""

import torch
import time
import numpy as np


def analyze_theoretical_complexity():
    """Analyze the theoretical complexity of cumulative operations."""
    print("🔍 THEORETICAL COMPLEXITY ANALYSIS")
    print("=" * 50)

    print("\n📊 COMPUTATIONAL COMPLEXITY:")
    print("torch.cummax(x, dim=-1):")
    print("  • Time Complexity: O(N) - single pass through data")
    print("  • Space Complexity: O(N) - output tensor same size as input")
    print("  • Algorithm: Sequential scan, each element depends on previous max")
    print("  • Parallelization: Limited due to data dependencies")

    print("\ntorch.cumprod(x, dim=-1):")
    print("  • Time Complexity: O(N) - single pass through data")
    print("  • Space Complexity: O(N) - output tensor same size as input")
    print("  • Algorithm: Sequential scan, each element depends on previous product")
    print("  • Parallelization: Limited due to data dependencies")
    print("  • Numerical Issues: Potential overflow/underflow with long sequences")


def benchmark_scaling_behavior():
    """Benchmark how operations scale with input size."""
    print("\n⚡ SCALING BEHAVIOR BENCHMARK")
    print("=" * 40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test sizes from small to large
    sizes = [100, 500, 1000, 5000, 10000, 50000, 100000]
    if device.type == 'cuda':
        sizes.extend([500000, 1000000])  # Only test large sizes on GPU

    results = {'sizes': sizes, 'cummax': [], 'cumprod': [], 'naive_loop': []}

    print(f"\n{'Size':<10} {'Cummax (μs)':<12} {'Cumprod (μs)':<13} {'Naive Loop (μs)':<16} {'Cummax vs Loop':<15}")
    print("-" * 80)

    for size in sizes:
        # Create test data
        x = torch.randn(size, device=device, dtype=torch.float32)

        # Benchmark cummax
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        for _ in range(10):
            result_cummax = torch.cummax(x, dim=0)[0]
        torch.cuda.synchronize() if device.type == 'cuda' else None
        cummax_time = (time.perf_counter() - start) * 100  # μs per op

        # Benchmark cumprod
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        for _ in range(10):
            result_cumprod = torch.cumprod(x, dim=0)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        cumprod_time = (time.perf_counter() - start) * 100  # μs per op

        # Benchmark naive loop equivalent (only for smaller sizes)
        if size <= 10000:
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start = time.perf_counter()
            for _ in range(10):
                # Simulate naive cummax loop
                result_naive = torch.zeros_like(x)
                current_max = float('-inf')
                for i in range(size):
                    current_max = max(current_max, x[i].item())
                    result_naive[i] = current_max
            torch.cuda.synchronize() if device.type == 'cuda' else None
            naive_time = (time.perf_counter() - start) * 100  # μs per op
        else:
            naive_time = float('inf')  # Too slow to measure

        results['cummax'].append(cummax_time)
        results['cumprod'].append(cumprod_time)
        results['naive_loop'].append(naive_time)

        speedup = naive_time / cummax_time if naive_time != float('inf') else float('inf')
        speedup_str = f"{speedup:.1f}x" if speedup != float('inf') else "∞"

        print(f"{size:<10} {cummax_time:<12.2f} {cumprod_time:<13.2f} {naive_time:<16.2f} {speedup_str:<15}")

    return results


def analyze_memory_access_patterns():
    """Analyze memory access patterns and cache efficiency."""
    print("\n🧠 MEMORY ACCESS PATTERNS")
    print("=" * 35)

    print("CUMULATIVE OPERATIONS:")
    print("  • Sequential Access: Read elements in order (cache-friendly)")
    print("  • Data Dependencies: Each output depends on previous output")
    print("  • Memory Bandwidth: Limited by sequential nature, not bandwidth")
    print("  • Cache Locality: Excellent for input, moderate for output")

    print("\nGPU IMPLEMENTATION CHALLENGES:")
    print("  • Warp Divergence: Minimal (all threads do same operation)")
    print("  • Parallel Reduction: Cannot fully parallelize due to dependencies")
    print("  • Memory Coalescing: Good for sequential access")
    print("  • Shared Memory: Limited benefit due to dependency chain")


def analyze_for_online_softmax():
    """Analyze how these operations help with online softmax."""
    print("\n🎯 APPLICATION TO ONLINE SOFTMAX")
    print("=" * 40)

    print("CURRENT LOOP-BASED APPROACH:")
    print("  for k_idx in range(seq_len):")
    print("    m_new = max(m_old, scores[k_idx])      # O(N) per iteration")
    print("    alpha = exp(m_old - m_new)             # O(N) per iteration")
    print("    probs[:k_idx] *= alpha                 # O(k) per iteration")
    print("    # ... more operations")
    print("  Total: O(N²) due to growing slice operations")

    print("\nCUMULATIVE OPERATIONS APPROACH:")
    print("  cummax_scores = torch.cummax(scores, dim=-1)     # O(N)")
    print("  alpha = exp(cummax[:-1] - cummax[1:])           # O(N)")
    print("  corrections = torch.cumprod(alpha.flip(-1))     # O(N)")
    print("  # ... vectorized operations")
    print("  Total: O(N) - linear scaling!")

    print("\nTRADE-OFFS:")
    print("  ✅ Advantages:")
    print("    • Linear O(N) complexity vs O(N²)")
    print("    • Vectorized operations (better GPU utilization)")
    print("    • No explicit Python loops")
    print("    • Better memory access patterns")

    print("  ⚠️ Limitations:")
    print("    • cummax/cumprod have sequential dependencies")
    print("    • Limited parallelization within operation")
    print("    • Potential numerical issues with very long sequences")
    print("    • More complex to understand than loop-based approach")


def benchmark_attention_relevant_shapes():
    """Benchmark with shapes relevant to attention computation."""
    print("\n🔍 ATTENTION-RELEVANT BENCHMARKS")
    print("=" * 40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Typical attention shapes: [batch, heads, seq_len_q, seq_len_k]
    configs = [
        (1, 1, 128, 128),     # Small
        (2, 8, 512, 512),     # Medium
        (1, 12, 2048, 2048),  # Large
        (4, 16, 1024, 1024),  # Many heads
    ]

    print(f"{'Config':<20} {'Cummax (ms)':<12} {'Cumprod (ms)':<13} {'Ratio':<8}")
    print("-" * 55)

    for batch, heads, seq_q, seq_k in configs:
        # Simulate attention scores tensor
        scores = torch.randn(batch, heads, seq_q, seq_k, device=device)

        # Benchmark cummax along key dimension
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        for _ in range(5):
            _ = torch.cummax(scores, dim=-1)[0]
        torch.cuda.synchronize() if device.type == 'cuda' else None
        cummax_time = (time.perf_counter() - start) * 200  # ms per op

        # Benchmark cumprod
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        for _ in range(5):
            _ = torch.cumprod(torch.abs(scores), dim=-1)  # abs to avoid negative products
        torch.cuda.synchronize() if device.type == 'cuda' else None
        cumprod_time = (time.perf_counter() - start) * 200  # ms per op

        ratio = cumprod_time / cummax_time
        config_str = f"({batch},{heads},{seq_q},{seq_k})"
        print(f"{config_str:<20} {cummax_time:<12.3f} {cumprod_time:<13.3f} {ratio:<8.2f}x")


def compare_with_alternatives():
    """Compare cumulative ops with alternative implementations."""
    print("\n🏁 COMPARISON WITH ALTERNATIVES")
    print("=" * 40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 2048

    x = torch.randn(seq_len, device=device)

    print(f"Sequence length: {seq_len}")
    print(f"{'Method':<25} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 50)

    # 1. torch.cummax (optimized)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.perf_counter()
    for _ in range(100):
        result1 = torch.cummax(x, dim=0)[0]
    torch.cuda.synchronize() if device.type == 'cuda' else None
    cummax_time = (time.perf_counter() - start) * 10

    print(f"{'torch.cummax':<25} {cummax_time:<12.3f} {'1.0x':<10}")

    # 2. Manual loop (baseline)
    if seq_len <= 5000:  # Only for reasonable sizes
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.perf_counter()
        for _ in range(100):
            result2 = torch.zeros_like(x)
            running_max = float('-inf')
            for i in range(seq_len):
                running_max = max(running_max, x[i].item())
                result2[i] = running_max
        torch.cuda.synchronize() if device.type == 'cuda' else None
        loop_time = (time.perf_counter() - start) * 10

        loop_speedup = loop_time / cummax_time
        print(f"{'Manual loop':<25} {loop_time:<12.3f} {loop_speedup:<10.1f}x")

    # 3. Vectorized alternative using scatter/gather
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.perf_counter()
    for _ in range(100):
        # This doesn't actually compute cummax correctly, just for timing comparison
        indices = torch.arange(seq_len, device=device).unsqueeze(0).expand(seq_len, -1)
        mask = indices <= torch.arange(seq_len, device=device).unsqueeze(1)
        # This is O(N²) and not correct, just showing why cummax is better
        result3 = torch.where(mask, x.unsqueeze(0), float('-inf')).max(dim=1)[0]
    torch.cuda.synchronize() if device.type == 'cuda' else None
    vectorized_time = (time.perf_counter() - start) * 10

    vectorized_speedup = vectorized_time / cummax_time
    print(f"{'Naive vectorized O(N²)':<25} {vectorized_time:<12.3f} {vectorized_speedup:<10.1f}x")


def main():
    """Run all analyses."""
    try:
        analyze_theoretical_complexity()
        results = benchmark_scaling_behavior()
        analyze_memory_access_patterns()
        analyze_for_online_softmax()
        benchmark_attention_relevant_shapes()
        compare_with_alternatives()

        print(f"\n📋 SUMMARY:")
        print(f"  • Both cummax and cumprod are O(N) operations")
        print(f"  • cummax is generally faster and more numerically stable")
        print(f"  • Sequential dependencies limit parallelization")
        print(f"  • Still much better than O(N²) loop-based approaches")
        print(f"  • Best suited for eliminating explicit loops in attention")

        print(f"\n💡 RECOMMENDATION FOR ONLINE SOFTMAX:")
        print(f"  • Use torch.cummax for running maximum computation")
        print(f"  • Avoid torch.cumprod for long sequences (numerical issues)")
        print(f"  • Consider block-wise processing for very long sequences")
        print(f"  • Hybrid approach: cumulative ops + tiling for best performance")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()