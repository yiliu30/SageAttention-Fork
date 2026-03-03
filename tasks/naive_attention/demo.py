"""
Educational Demo for Naive Attention with Online Softmax

This demo script provides an educational walkthrough of naive attention implementation,
comparing standard torch.softmax with the online softmax algorithm from FlashAttention.

The demo includes:
1. Step-by-step explanation of attention computation
2. Visual comparison of outputs between methods
3. Detailed analysis of numerical differences
4. Educational insights about the online softmax algorithm
5. Performance timing comparisons

Run this script to understand how attention works at a fundamental level and see
how the online softmax algorithm achieves the same results with better numerical stability.
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our implementations
from naive_attention import (
    naive_attention,
    naive_attention_standard,
    naive_attention_online_softmax,
    compare_attention_methods
)


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f"{title.center(70)}")
    print("="*70)


def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{title}")
    print("-" * len(title))


def demonstrate_tensor_shapes():
    """Demonstrate tensor shapes throughout the attention computation."""
    print_section_header("TENSOR SHAPES IN ATTENTION COMPUTATION")

    # Setup parameters
    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Setup: batch_size={batch_size}, num_heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"Device: {device}")

    # Create input tensors
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print_subsection("Input Tensors")
    print(f"Query (Q):  {q.shape} - [batch, heads, seq_len_q, head_dim]")
    print(f"Key (K):    {k.shape} - [batch, heads, seq_len_k, head_dim]")
    print(f"Value (V):  {v.shape} - [batch, heads, seq_len_v, head_dim]")

    print_subsection("Step 1: Attention Scores (Q @ K^T)")
    scores = torch.matmul(q, k.transpose(-2, -1))
    print(f"Scores:     {scores.shape} - [batch, heads, seq_len_q, seq_len_k]")
    print(f"Each element scores[b,h,i,j] = similarity between query i and key j")

    print_subsection("Step 2: Attention Probabilities (Softmax)")
    probs = F.softmax(scores, dim=-1)
    print(f"Probs:      {probs.shape} - [batch, heads, seq_len_q, seq_len_k]")
    print(f"Each row sums to 1.0: {probs[0, 0, 0].sum().item():.6f}")

    print_subsection("Step 3: Output (Probs @ V)")
    output = torch.matmul(probs, v)
    print(f"Output:     {output.shape} - [batch, heads, seq_len_q, head_dim]")
    print(f"Same shape as Query - each query position gets a weighted combination of values")


def demonstrate_online_softmax_algorithm():
    """Demonstrate the online softmax algorithm step by step."""
    print_section_header("ONLINE SOFTMAX ALGORITHM DEMONSTRATION")

    # Use a small example for clarity
    batch_size, num_heads, seq_len, head_dim = 1, 1, 4, 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(123)  # For reproducible demo
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print(f"Demo with small example: {seq_len}x{seq_len} attention matrix")

    # Compute full attention scores
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    print_subsection("Full Attention Scores Matrix")
    scores_np = scores[0, 0].cpu().numpy()
    print("Scores (Q @ K^T / sqrt(d)):")
    for i in range(seq_len):
        row_str = " ".join([f"{scores_np[i, j]:6.3f}" for j in range(seq_len)])
        print(f"  [{row_str}]")

    print_subsection("Standard Softmax")
    standard_probs = F.softmax(scores, dim=-1)
    standard_probs_np = standard_probs[0, 0].cpu().numpy()
    print("Standard softmax probabilities:")
    for i in range(seq_len):
        row_str = " ".join([f"{standard_probs_np[i, j]:6.3f}" for j in range(seq_len)])
        row_sum = standard_probs_np[i].sum()
        print(f"  [{row_str}] sum={row_sum:.3f}")

    print_subsection("Online Softmax Step-by-Step")
    print("Processing one key position at a time...")

    # Initialize online softmax variables
    m = torch.full((batch_size, num_heads, seq_len), float('-inf'), device=device, dtype=torch.float32)
    l = torch.zeros((batch_size, num_heads, seq_len), device=device, dtype=torch.float32)
    online_probs = torch.zeros_like(scores, dtype=torch.float32)

    for k_idx in range(seq_len):
        print(f"\nStep {k_idx + 1}: Processing key position {k_idx}")

        # Current scores for this key position
        current_scores = scores[:, :, :, k_idx]
        print(f"  Current scores: {current_scores[0, 0].cpu().numpy()}")

        # Update running maximum
        m_old = m.clone()
        m_new = torch.maximum(m, current_scores)
        print(f"  Running max: {m_old[0, 0].cpu().numpy()} -> {m_new[0, 0].cpu().numpy()}")

        # Compute correction factor
        alpha = torch.exp(m_old - m_new)
        print(f"  Correction factor (alpha): {alpha[0, 0].cpu().numpy()}")

        # Rescale previous probabilities
        if k_idx > 0:
            online_probs[:, :, :, :k_idx] *= alpha.unsqueeze(-1)
            print(f"  Rescaled previous probabilities by alpha")

        # Compute new probabilities
        online_probs[:, :, :, k_idx] = torch.exp(current_scores - m_new)
        print(f"  New probabilities: {online_probs[0, 0, :, k_idx].cpu().numpy()}")

        # Update running sum
        l = l * alpha + online_probs[:, :, :, k_idx]
        print(f"  Running sum: {l[0, 0].cpu().numpy()}")

        # Show current unnormalized probabilities
        current_probs = online_probs[0, 0, :, :k_idx+1].cpu().numpy()
        print(f"  Current unnormalized probs: {current_probs}")

    # Final normalization
    online_probs = online_probs / l.unsqueeze(-1)

    print_subsection("Final Comparison")
    online_probs_np = online_probs[0, 0].cpu().numpy()
    print("Online softmax probabilities:")
    for i in range(seq_len):
        row_str = " ".join([f"{online_probs_np[i, j]:6.3f}" for j in range(seq_len)])
        row_sum = online_probs_np[i].sum()
        print(f"  [{row_str}] sum={row_sum:.3f}")

    # Check they're identical
    max_diff = torch.abs(standard_probs - online_probs).max().item()
    print(f"\nMaximum difference between methods: {max_diff:.2e}")
    print("✓ Methods produce identical results!" if max_diff < 1e-6 else "✗ Methods differ!")


def demonstrate_numerical_stability():
    """Demonstrate numerical stability benefits of online softmax."""
    print_section_header("NUMERICAL STABILITY DEMONSTRATION")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, num_heads, seq_len, head_dim = 1, 1, 16, 32

    print("Testing with extreme values that could cause overflow...")

    # Create inputs that would cause overflow with naive softmax
    torch.manual_seed(456)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device) * 20  # Large values
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device) * 20
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    # Compute attention scores (these will be very large)
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    max_score = scores.max().item()
    min_score = scores.min().item()

    print(f"Attention scores range: [{min_score:.1f}, {max_score:.1f}]")
    print(f"exp({max_score:.1f}) = {np.exp(max_score):.2e} (potential overflow)")

    # Test standard softmax
    try:
        standard_output = naive_attention_standard(q, k, v)
        standard_has_nan = torch.isnan(standard_output).any()
        standard_has_inf = torch.isinf(standard_output).any()
        print(f"Standard softmax - NaN: {standard_has_nan}, Inf: {standard_has_inf}")
    except Exception as e:
        print(f"Standard softmax failed: {e}")
        standard_output = None

    # Test online softmax
    try:
        online_output = naive_attention_online_softmax(q, k, v)
        online_has_nan = torch.isnan(online_output).any()
        online_has_inf = torch.isinf(online_output).any()
        print(f"Online softmax - NaN: {online_has_nan}, Inf: {online_has_inf}")
    except Exception as e:
        print(f"Online softmax failed: {e}")
        online_output = None

    # Compare if both worked
    if standard_output is not None and online_output is not None:
        max_diff = torch.abs(standard_output - online_output).max().item()
        print(f"Maximum difference between methods: {max_diff:.2e}")

        cos_sim = F.cosine_similarity(
            standard_output.flatten().unsqueeze(0),
            online_output.flatten().unsqueeze(0)
        ).item()
        print(f"Cosine similarity: {cos_sim:.6f}")


def run_comprehensive_comparison():
    """Run a comprehensive comparison across different configurations."""
    print_section_header("COMPREHENSIVE COMPARISON")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Test configurations
    configs = [
        (1, 1, 16, 32),      # Small
        (2, 4, 64, 64),      # Medium
        (1, 8, 128, 96),     # Large sequence
        (4, 12, 32, 48),     # Many heads
    ]

    print_subsection("Testing Different Configurations")
    print(f"{'Config':<20} {'Cosine Sim':<12} {'Max Diff':<12} {'Mean Diff':<12} {'Status'}")
    print("-" * 70)

    all_passed = True

    for batch_size, num_heads, seq_len, head_dim in configs:
        torch.manual_seed(42)
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

        # Run comparison
        pytorch_out, standard_out, online_out, metrics = compare_attention_methods(q, k, v)

        # Extract key metrics
        cos_sim = metrics['cosine_sim_standard_vs_online']
        max_diff = metrics['max_abs_diff_standard_vs_online']
        mean_diff = metrics['mean_abs_diff_standard_vs_online']

        # Check if passed
        passed = cos_sim > 0.95 and max_diff < 1e-4 and mean_diff < 1e-5
        status = "✓ PASS" if passed else "✗ FAIL"
        if not passed:
            all_passed = False

        config_str = f"({batch_size},{num_heads},{seq_len},{head_dim})"
        print(f"{config_str:<20} {cos_sim:<12.6f} {max_diff:<12.2e} {mean_diff:<12.2e} {status}")

    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")


def benchmark_performance():
    """Benchmark performance of different attention methods."""
    print_section_header("PERFORMANCE BENCHMARK")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Benchmark configuration
    batch_size = 4
    num_heads = 8
    seq_len = 256
    head_dim = 64
    num_warmup = 5
    num_trials = 20

    print(f"Configuration: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"Trials: {num_trials} (after {num_warmup} warmup)\n")

    # Create test data
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

    # Methods to benchmark
    methods = {
        'PyTorch SDPA': lambda: F.scaled_dot_product_attention(q, k, v),
        'Standard Softmax': lambda: naive_attention_standard(q, k, v),
        'Online Softmax': lambda: naive_attention_online_softmax(q, k, v)
    }

    results = {}

    for method_name, method_func in methods.items():
        # Warmup
        for _ in range(num_warmup):
            _ = method_func()
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_trials):
            start_time = time.perf_counter()
            output = method_func()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms

        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)

        results[method_name] = {
            'avg': avg_time, 'std': std_time, 'min': min_time, 'max': max_time
        }

    # Display results
    print(f"{'Method':<20} {'Avg (ms)':<10} {'Std (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10}")
    print("-" * 70)

    for method_name, timing in results.items():
        print(f"{method_name:<20} {timing['avg']:<10.2f} {timing['std']:<10.2f} "
              f"{timing['min']:<10.2f} {timing['max']:<10.2f}")

    # Compute relative performance
    pytorch_time = results['PyTorch SDPA']['avg']
    print(f"\nRelative to PyTorch SDPA:")
    for method_name, timing in results.items():
        if method_name != 'PyTorch SDPA':
            relative = timing['avg'] / pytorch_time
            print(f"  {method_name:<18}: {relative:.1f}x {'faster' if relative < 1 else 'slower'}")


def demonstrate_causal_attention():
    """Demonstrate causal attention masking."""
    print_section_header("CAUSAL ATTENTION DEMONSTRATION")

    # Small example for visualization
    batch_size, num_heads, seq_len, head_dim = 1, 1, 6, 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(789)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print(f"Demonstrating causal masking with {seq_len}x{seq_len} attention matrix")

    # Compute attention scores
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    print_subsection("Original Attention Scores")
    scores_np = scores[0, 0].cpu().numpy()
    for i in range(seq_len):
        row_str = " ".join([f"{scores_np[i, j]:5.2f}" for j in range(seq_len)])
        print(f"  [{row_str}]")

    print_subsection("After Causal Masking (upper triangle = -inf)")
    # Apply causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    masked_scores = scores.masked_fill(~mask, float('-inf'))
    masked_scores_np = masked_scores[0, 0].cpu().numpy()

    for i in range(seq_len):
        row_str = " ".join([f"{masked_scores_np[i, j]:5.2f}" if j <= i else "  -inf"
                           for j in range(seq_len)])
        print(f"  [{row_str}]")

    print_subsection("Causal Attention Probabilities")
    causal_probs = F.softmax(masked_scores, dim=-1)
    causal_probs_np = causal_probs[0, 0].cpu().numpy()

    for i in range(seq_len):
        row_str = " ".join([f"{causal_probs_np[i, j]:5.3f}" for j in range(seq_len)])
        row_sum = causal_probs_np[i, :i+1].sum()  # Sum only valid positions
        print(f"  [{row_str}] sum={row_sum:.3f}")

    # Compare our implementation
    print_subsection("Verification Against Our Implementation")
    our_output = naive_attention_online_softmax(q, k, v, is_causal=True)
    pytorch_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    max_diff = torch.abs(our_output - pytorch_output).max().item()
    cos_sim = F.cosine_similarity(our_output.flatten().unsqueeze(0),
                                pytorch_output.flatten().unsqueeze(0)).item()

    print(f"Max difference from PyTorch: {max_diff:.2e}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print("✓ Causal masking works correctly!" if max_diff < 1e-5 else "✗ Causal masking issue!")


def main():
    """Main demo function."""
    print("🧠 NAIVE ATTENTION WITH ONLINE SOFTMAX - EDUCATIONAL DEMO")
    print("Understanding attention computation from first principles")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    try:
        # Run all demonstrations
        demonstrate_tensor_shapes()
        demonstrate_online_softmax_algorithm()
        demonstrate_numerical_stability()
        demonstrate_causal_attention()
        run_comprehensive_comparison()
        benchmark_performance()

        print_section_header("DEMO COMPLETED SUCCESSFULLY! 🎉")
        print("Key Takeaways:")
        print("1. Online softmax produces identical results to standard softmax")
        print("2. Online softmax provides better numerical stability")
        print("3. The algorithm processes attention incrementally (key by key)")
        print("4. This foundation enables memory-efficient attention (FlashAttention)")
        print("5. Both causal and non-causal attention work correctly")
        print("\nNext steps:")
        print("- Study FlashAttention paper for block-wise processing")
        print("- Explore tiled/blocked attention implementations")
        print("- Learn about memory optimization techniques")

    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()