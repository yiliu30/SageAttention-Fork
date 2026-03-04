"""
Analysis of Online Softmax Performance Impact

This script analyzes why the online softmax loop is slow and demonstrates
the performance differences between vectorized and sequential operations.
"""

import torch
import time
import numpy as np
from naive_attention import naive_attention_standard, naive_attention_online_softmax, naive_attention_scan_like

def benchmark_sequence_lengths():
    """Benchmark performance across different sequence lengths."""
    print("🔍 ANALYZING ONLINE SOFTMAX PERFORMANCE IMPACT")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Test different sequence lengths
    seq_lengths = [16, 32, 64, 128, 256, 512]
    batch_size, num_heads, head_dim = 1, 1, 32  # Keep other dims small

    results = {
        'seq_lengths': seq_lengths,
        'standard_times': [],
        'online_times': [],
        'scan_like_times': [],
        'online_slowdown_factors': [],
        'scan_like_slowdown_factors': []
    }

    print(f"\nTesting with batch_size={batch_size}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"{'Seq Len':<8} {'Standard (ms)':<15} {'Online (ms)':<15} {'Scan-like (ms)':<15} {'Online vs Std':<12} {'Scan vs Std':<12}")
    print("-" * 90)

    for seq_len in seq_lengths:
        # Create test tensors
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # Benchmark standard softmax (vectorized)
        times = []
        for _ in range(10):  # Multiple runs for average
            start = time.perf_counter()
            _ = naive_attention_standard(q, k, v)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        standard_time = np.mean(times)

        # Benchmark online softmax (sequential)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = naive_attention_online_softmax(q, k, v)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        online_time = np.mean(times)

        # Benchmark scan-like softmax (optimized)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            _ = naive_attention_scan_like(q, k, v)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
        scan_like_time = np.mean(times)

        online_slowdown = online_time / standard_time
        scan_like_slowdown = scan_like_time / standard_time

        results['standard_times'].append(standard_time)
        results['online_times'].append(online_time)
        results['scan_like_times'].append(scan_like_time)
        results['online_slowdown_factors'].append(online_slowdown)
        results['scan_like_slowdown_factors'].append(scan_like_slowdown)

        print(f"{seq_len:<8} {standard_time:<15.3f} {online_time:<15.3f} {scan_like_time:<15.3f} {online_slowdown:<12.1f}x {scan_like_slowdown:<12.1f}x")

    return results

def analyze_complexity():
    """Analyze the computational complexity differences."""
    print("\n🧮 COMPUTATIONAL COMPLEXITY ANALYSIS")
    print("=" * 50)

    print("Standard Softmax (Vectorized):")
    print("  1. Q @ K^T:           O(N²·D) - single matrix multiplication")
    print("  2. Softmax:           O(N²) - vectorized across all positions")
    print("  3. Probs @ V:         O(N²·D) - single matrix multiplication")
    print("  Total: O(N²·D) with high parallelism")

    print("\nOnline Softmax (Sequential):")
    print("  For each of N key positions:")
    print("    1. Extract scores:   O(N) - slice operation")
    print("    2. Update max:       O(N) - element-wise maximum")
    print("    3. Rescale probs:    O(N²) - multiply previous probabilities")
    print("    4. Compute new prob: O(N) - exponential")
    print("    5. Update sum:       O(N) - element-wise addition")
    print("  Total loop: N × O(N²) = O(N³)")
    print("  Final P @ V: O(N²·D)")
    print("  Overall: O(N³ + N²·D) with low parallelism")

    print("\nScan-like Softmax (Optimized Sequential):")
    print("  1. Q @ K^T:           O(N²·D) - single matrix multiplication")
    print("  2. Cumulative max:    O(N²) - vectorized cummax operation")
    print("  3. Online loop:       O(N) iterations with O(N) work each = O(N²)")
    print("  4. Final P @ V:       O(N²·D) - single matrix multiplication")
    print("  Total: O(N²·D + N²) with better GPU utilization")
    print("  Improvement: Eliminates O(N³) term, reduces constant factors")

def demonstrate_loop_overhead():
    """Demonstrate the specific overhead of the sequential loop."""
    print("\n⚡ LOOP OVERHEAD DEMONSTRATION")
    print("=" * 40)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_len = 128
    batch_size, num_heads, head_dim = 1, 1, 32

    # Create test data
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Compute full attention scores (same for both methods)
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    print(f"Sequence length: {seq_len}")
    print(f"Attention scores shape: {scores.shape}")

    # Time the vectorized softmax
    start = time.perf_counter()
    vectorized_probs = torch.softmax(scores, dim=-1)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    vectorized_time = (time.perf_counter() - start) * 1000

    # Time the online softmax loop (just the softmax part)
    start = time.perf_counter()

    # Initialize online softmax variables
    m = torch.full((batch_size, num_heads, seq_len), float('-inf'),
                   device=device, dtype=torch.float32)
    l = torch.zeros((batch_size, num_heads, seq_len),
                    device=device, dtype=torch.float32)
    online_probs = torch.zeros_like(scores, dtype=torch.float32)

    # The expensive loop
    for k_idx in range(seq_len):
        current_scores = scores[:, :, :, k_idx]
        m_old = m.clone()
        m_new = torch.maximum(m, current_scores)
        alpha = torch.exp(m_old - m_new)

        if k_idx > 0:
            online_probs[:, :, :, :k_idx] *= alpha.unsqueeze(-1)

        online_probs[:, :, :, k_idx] = torch.exp(current_scores - m_new)
        l = l * alpha + online_probs[:, :, :, k_idx]
        m = m_new

    online_probs = online_probs / l.unsqueeze(-1)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    loop_time = (time.perf_counter() - start) * 1000

    print(f"\nSoftmax timing:")
    print(f"  Vectorized softmax: {vectorized_time:.3f} ms")
    print(f"  Online softmax loop: {loop_time:.3f} ms")
    print(f"  Loop overhead: {loop_time/vectorized_time:.1f}x slower")

    # Verify they produce the same result
    max_diff = torch.abs(vectorized_probs - online_probs).max().item()
    print(f"  Max difference: {max_diff:.2e} (should be ~0)")

def explain_why_slow():
    """Explain the specific reasons why the loop is slow."""
    print("\n🐌 WHY THE ONLINE SOFTMAX LOOP IS SLOW")
    print("=" * 45)

    print("1. SEQUENTIAL PROCESSING:")
    print("   • Each iteration depends on previous results")
    print("   • Cannot parallelize across key positions")
    print("   • GPU cores are underutilized")

    print("\n2. MEMORY ACCESS PATTERNS:")
    print("   • Each iteration accesses different memory locations")
    print("   • Poor cache locality compared to vectorized operations")
    print("   • Multiple scatter/gather operations per iteration")

    print("\n3. KERNEL LAUNCH OVERHEAD:")
    print("   • Each tensor operation launches a separate GPU kernel")
    print("   • Kernel launch overhead ~5-10μs per operation")
    print("   • With N iterations × multiple ops = significant overhead")

    print("\n4. PYTHON LOOP OVERHEAD:")
    print("   • Python for-loop has per-iteration overhead")
    print("   • CPU-GPU synchronization points")
    print("   • Not compiled/optimized like vectorized operations")

    print("\n5. TENSOR OPERATIONS SCALING:")
    print("   • online_probs[:,:,:,:k_idx] *= alpha grows with k_idx")
    print("   • O(k) work per iteration → O(N²) total")
    print("   • Vectorized version: O(N²) work done once")

def suggest_optimizations():
    """Suggest how this could be optimized."""
    print("\n🚀 HOW TO OPTIMIZE (FLASHATTENTION APPROACH)")
    print("=" * 50)

    print("1. BLOCK-WISE PROCESSING:")
    print("   • Process multiple keys per iteration (e.g., 64-128 keys)")
    print("   • Reduces loop iterations from N to N/block_size")
    print("   • Better GPU utilization")

    print("\n2. FUSED KERNELS:")
    print("   • Write custom CUDA kernels that fuse operations")
    print("   • Eliminate intermediate memory reads/writes")
    print("   • Reduce kernel launch overhead")

    print("\n3. MEMORY OPTIMIZATION:")
    print("   • Tile computation to fit in faster memory (shared/cache)")
    print("   • Recompute instead of store intermediate results")
    print("   • Minimize global memory access")

    print("\n4. COMPILER OPTIMIZATIONS:")
    print("   • Use torch.compile() or custom autograd functions")
    print("   • JIT compilation can optimize the loop")
    print("   • Loop unrolling and vectorization")

    print("\n5. ALGORITHMIC IMPROVEMENTS:")
    print("   • Process queries in blocks too (block-sparse)")
    print("   • Use approximation techniques for very long sequences")
    print("   • Leverage attention patterns (local, strided, etc.)")

def main():
    """Run all analyses."""
    try:
        # Run benchmarks
        results = benchmark_sequence_lengths()

        # Additional analyses
        analyze_complexity()
        demonstrate_loop_overhead()
        explain_why_slow()
        suggest_optimizations()

        print(f"\n📊 SUMMARY:")
        online_final_slowdown = results['online_slowdown_factors'][-1]
        scan_like_final_slowdown = results['scan_like_slowdown_factors'][-1]

        print(f"  • Online softmax is {online_final_slowdown:.0f}x slower than standard at seq_len=512")
        print(f"  • Scan-like softmax is {scan_like_final_slowdown:.0f}x slower than standard at seq_len=512")
        improvement = online_final_slowdown / scan_like_final_slowdown
        print(f"  • Scan-like is {improvement:.1f}x faster than online softmax")
        print(f"  • Slowdown grows roughly O(N) with sequence length")
        print(f"  • Main causes: sequential processing + loop overhead")
        print(f"  • Solution: Block-wise processing (FlashAttention)")

        print(f"\n✅ This educational implementation demonstrates the concept")
        print(f"   Scan-like approach shows how to optimize while maintaining clarity!")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()