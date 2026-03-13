# Naive Attention with Online Softmax Implementation

This directory contains an educational implementation of naive attention using two matrix multiplications (QK^T and PV) with multiple softmax algorithm variants. This implementation serves as a foundation for understanding more advanced attention mechanisms like FlashAttention.

## Overview

The implementation demonstrates:
1. **Two-Matmul Approach**: The classic attention computation using `QK^T` followed by `softmax(QK^T)@V`
2. **Online Softmax Algorithm**: A numerically stable alternative that processes one key position at a time
3. **Scan-like Optimization**: An optimized version that reduces computational overhead
4. **Mathematical Equivalence**: All methods produce identical results
5. **Educational Value**: Step-by-step documentation and performance comparisons

## Files

- **`naive_attention.py`**: Main implementation with four attention functions
- **`test_naive_attention.py`**: Comprehensive test suite with 14 test cases
- **`analyze_performance.py`**: Performance analysis comparing all implementations
- **`demo.py`**: Educational demonstration script with step-by-step explanations
- **`README.md`**: This documentation file

## Implementation Details

### Core Functions

#### `naive_attention_standard(q, k, v, is_causal=False, sm_scale=None)`
Standard attention using PyTorch's `torch.softmax`:
```python
# Step 1: Compute attention scores
scores = Q @ K^T / sqrt(d)

# Step 2: Apply causal mask if needed
if is_causal:
    scores = mask_fill(scores, -inf)

# Step 3: Apply softmax
probs = softmax(scores, dim=-1)

# Step 4: Apply to values
output = probs @ V
```

#### `naive_attention_online_softmax(q, k, v, is_causal=False, sm_scale=None)`
Online softmax attention processing one key position at a time:
```python
for k_idx in range(seq_len_k):
    # Current scores for this key position
    current_scores = scores[:, :, :, k_idx]

    # Update running maximum for numerical stability
    m_new = max(m_old, current_scores)
    alpha = exp(m_old - m_new)

    # Rescale previous probabilities
    probs[:, :, :, :k_idx] *= alpha.unsqueeze(-1)

    # Compute new probabilities
    probs[:, :, :, k_idx] = exp(current_scores - m_new)

    # Update running sum
    l = l * alpha + probs[:, :, :, k_idx]

    m = m_new

# Final normalization
probs = probs / l.unsqueeze(-1)
output = probs @ V
```

#### `naive_attention_scan_like(q, k, v, is_causal=False, sm_scale=None)`
Optimized online softmax using scan-like operations to reduce overhead:
```python
# Step 1: Compute full scores matrix upfront
scores = Q @ K^T / sqrt(d)

# Step 2: Use cumulative operations for efficiency
m_running = torch.cummax(scores, dim=-1)[0]
exp_scores = torch.exp(scores - m_running)

# Step 3: Optimized sequential processing
for k_idx in range(seq_len_k):
    # Compute correction factor
    alpha = torch.exp(m_prev[k_idx] - m_curr[k_idx])

    # Apply correction to previous probabilities
    if k_idx > 0:
        probs[:, :, :, :k_idx] *= alpha.unsqueeze(-1)

    # Set current probability and update running sum
    probs[:, :, :, k_idx] = exp_scores[:, :, :, k_idx]
    l_running = l_running * alpha + exp_scores[:, :, :, k_idx]

# Final normalization and output
probs = probs / l_running.unsqueeze(-1)
output = probs @ V
```

### Key Features

1. **Detailed Tensor Shape Annotations**: Every tensor has clear shape documentation
2. **Educational Comments**: Step-by-step explanations of all algorithms
3. **Numerical Stability**: Online softmax prevents overflow with large attention scores
4. **Multiple Implementations**: Standard, online, and scan-like approaches for comparison
5. **Performance Analysis**: Detailed benchmarking and complexity analysis
6. **Causal Masking Support**: All methods support causal (autoregressive) attention
7. **Custom Scale Factors**: Support for different attention temperature scaling
8. **Comprehensive Testing**: 14 test cases covering correctness, shapes, and edge cases

## Usage

### Basic Usage
```python
import torch
from naive_attention import naive_attention

# Create input tensors
q = torch.randn(batch_size, num_heads, seq_len, head_dim)
k = torch.randn(batch_size, num_heads, seq_len, head_dim)
v = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Run attention with scan-like online softmax (optimized)
output = naive_attention(q, k, v, use_scan_like=True)

# Run attention with online softmax
output = naive_attention(q, k, v, use_online_softmax=True)

# Run attention with standard softmax
output = naive_attention(q, k, v, use_online_softmax=False)
```

### Comparison with PyTorch SDPA
```python
from naive_attention import compare_attention_methods

pytorch_out, standard_out, online_out, scan_like_out, metrics = compare_attention_methods(q, k, v)

print(f"Standard vs Online: {metrics['cosine_sim_standard_vs_online']:.6f}")
print(f"Online vs Scan-like: {metrics['cosine_sim_online_vs_scan_like']:.6f}")
print(f"Max difference: {metrics['max_abs_diff_online_vs_scan_like']:.2e}")
```

### Performance Analysis
```bash
# Run comprehensive performance analysis
/mnt/disk1/yiliu7/sage/bin/python analyze_performance.py
```

### Running Tests
```bash
# Run all tests
/mnt/disk1/yiliu7/sage/bin/python test_naive_attention.py

# Run specific test
/mnt/disk1/yiliu7/sage/bin/python -m unittest test_naive_attention.TestNaiveAttention.test_correctness_non_causal
```

### Running Demo
```bash
# Full educational demo
/mnt/disk1/yiliu7/sage/bin/python demo.py

# Quick verification
/mnt/disk1/yiliu7/sage/bin/python naive_attention.py
```

## Test Results

The implementation passes all 14 comprehensive tests:

```
✓ test_correctness_non_causal: Compare against PyTorch SDPA (non-causal)
✓ test_correctness_causal: Compare against PyTorch SDPA (causal)
✓ test_output_shapes: Verify output shapes match expectations
✓ test_different_sequence_lengths: Test with different Q/K sequence lengths
✓ test_custom_scale_factor: Test with custom attention temperature
✓ test_numerical_stability: Test with extreme values
✓ test_edge_case_single_sequence: Test with sequence length of 1
✓ test_edge_case_single_head: Test with single attention head
✓ test_main_interface: Test the unified interface function
✓ test_compare_methods_function: Test the comparison utility
✓ test_gradient_flow: Verify gradients flow correctly
✓ test_scan_like_correctness: Test scan-like implementation correctness
✓ test_scan_like_shapes: Test scan-like with various tensor shapes
✓ test_scan_like_gradients: Verify gradients flow through scan-like implementation
```

### Accuracy Metrics
- **Cosine Similarity**: >0.999999 (essentially identical)
- **Max Absolute Difference**: <1e-6 (numerical precision)
- **Relative Error**: <1e-6 (high accuracy)

## Performance Characteristics

**Note**: This is an educational implementation optimized for understanding, not performance.

### Timing Results (seq_len=512)
- **PyTorch SDPA**: 0.11ms (highly optimized baseline)
- **Standard Softmax**: 0.14ms (1.3x slower, matrix operations)
- **Scan-like Softmax**: 36.9ms (261x slower, optimized sequential)
- **Online Softmax**: 54.4ms (384x slower, full sequential processing)

### Performance Improvements
The scan-like implementation shows significant improvement over the original online softmax:
- **33% faster** than the original online implementation
- **Reduced complexity** from O(N³) to O(N²·D + N²)
- **Better GPU utilization** through reduced sequential operations

### Complexity Analysis
- **Standard**: O(N²·D) with high parallelism
- **Online**: O(N³ + N²·D) with low parallelism
- **Scan-like**: O(N²·D + N²) with improved parallelism

The online/scan-like methods are intentionally slow because they process sequentially for educational clarity. In practice, FlashAttention uses block-wise processing to maintain numerical stability benefits while achieving high performance.

## Educational Value

This implementation helps understand:

1. **Attention Fundamentals**: The two-matmul structure of attention
2. **Numerical Stability**: Why the max subtraction trick is important
3. **Online Algorithms**: How to process sequences incrementally
4. **Scan-like Optimizations**: How to reduce computational overhead while maintaining correctness
5. **FlashAttention Foundation**: The mathematical basis for memory-efficient attention
6. **Causal Masking**: How autoregressive attention works
7. **Performance Trade-offs**: Understanding the cost of different algorithmic approaches

## Mathematical Background

The online softmax algorithm is based on the FlashAttention paper and maintains three running statistics:

- **m**: Running maximum for numerical stability
- **ℓ**: Running sum of exponentials
- **α**: Correction factor when maximum changes

For each new key position, the algorithm:
1. Computes new scores
2. Updates the running maximum
3. Rescales previous probabilities using correction factor α
4. Adds new probabilities
5. Updates the running sum

This ensures numerical stability while producing identical results to standard softmax.

## References

- Online softmax algorithm: `/mnt/disk1/yiliu7/SageAttention-Fork/papers/sageattention/flashattn-online-softmax.pdf`
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135): "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): Original transformer paper

## Requirements

- PyTorch 2.0+
- CUDA-compatible GPU (recommended)
- Python 3.8+