"""
Test Suite for Naive Attention Implementation

This module provides comprehensive tests for the naive attention implementation,
verifying correctness against PyTorch's scaled_dot_product_attention and testing
various edge cases and configurations.

Test Categories:
1. Correctness tests: Compare against PyTorch SDPA
2. Shape tests: Verify output shapes match expectations
3. Edge case tests: Test with different sequence lengths, dimensions
4. Numerical stability tests: Test with extreme values
5. Causal masking tests: Verify causal attention correctness
6. Performance comparison: Basic timing comparisons
"""

import torch
import torch.nn.functional as F
import unittest
import time
import sys
import os

# Add the current directory to path to import naive_attention
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from naive_attention import (
    naive_attention,
    naive_attention_standard,
    naive_attention_online_softmax,
    naive_attention_scan_like,
    compare_attention_methods
)


class TestNaiveAttention(unittest.TestCase):
    """Test suite for naive attention implementations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        torch.manual_seed(42)  # For reproducible tests
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32

        # Standard test dimensions
        self.batch_size = 2
        self.num_heads = 4
        self.seq_len = 32
        self.head_dim = 64

        # Tolerance levels for numerical comparison
        self.cosine_threshold = 0.95
        self.abs_diff_threshold = 1e-4
        self.rel_error_threshold = 1e-4

    def create_test_tensors(self, batch_size=None, num_heads=None, seq_len_q=None,
                          seq_len_k=None, head_dim=None, dtype=None):
        """Create test tensors with specified or default dimensions."""
        batch_size = batch_size or self.batch_size
        num_heads = num_heads or self.num_heads
        seq_len_q = seq_len_q or self.seq_len
        seq_len_k = seq_len_k or seq_len_q  # Default to same length
        head_dim = head_dim or self.head_dim
        dtype = dtype or self.dtype

        q = torch.randn(batch_size, num_heads, seq_len_q, head_dim,
                       device=self.device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len_k, head_dim,
                       device=self.device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len_k, head_dim,
                       device=self.device, dtype=dtype)

        return q, k, v

    def assert_tensors_close(self, tensor1, tensor2, name1="tensor1", name2="tensor2"):
        """Assert that two tensors are numerically close using multiple metrics."""
        # Flatten tensors for easier computation
        flat1 = tensor1.flatten()
        flat2 = tensor2.flatten()

        # Cosine similarity
        cos_sim = F.cosine_similarity(flat1.unsqueeze(0), flat2.unsqueeze(0), dim=1).item()
        self.assertGreater(cos_sim, self.cosine_threshold,
                          f"Cosine similarity between {name1} and {name2} too low: {cos_sim:.6f}")

        # Maximum absolute difference
        max_abs_diff = torch.abs(flat1 - flat2).max().item()
        self.assertLess(max_abs_diff, self.abs_diff_threshold,
                       f"Max absolute difference between {name1} and {name2} too high: {max_abs_diff:.2e}")

        # Relative error
        l2_diff = torch.norm(flat1 - flat2).item()
        l2_ref = torch.norm(flat1).item()
        rel_error = l2_diff / (l2_ref + 1e-8)
        self.assertLess(rel_error, self.rel_error_threshold,
                       f"Relative error between {name1} and {name2} too high: {rel_error:.2e}")

    def test_correctness_non_causal(self):
        """Test correctness against PyTorch SDPA for non-causal attention."""
        q, k, v = self.create_test_tensors()

        # Get reference output from PyTorch
        pytorch_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Test standard softmax version
        standard_output = naive_attention_standard(q, k, v, is_causal=False)
        self.assert_tensors_close(pytorch_output, standard_output, "PyTorch", "Standard")

        # Test online softmax version
        online_output = naive_attention_online_softmax(q, k, v, is_causal=False)
        self.assert_tensors_close(pytorch_output, online_output, "PyTorch", "Online")

        # Test scan-like online softmax version
        scan_like_output = naive_attention_scan_like(q, k, v, is_causal=False)
        self.assert_tensors_close(pytorch_output, scan_like_output, "PyTorch", "Scan-like")

        # Test that standard and online produce identical results
        self.assert_tensors_close(standard_output, online_output, "Standard", "Online")

        # Test that online and scan-like produce identical results
        self.assert_tensors_close(online_output, scan_like_output, "Online", "Scan-like")

    def test_correctness_causal(self):
        """Test correctness against PyTorch SDPA for causal attention."""
        q, k, v = self.create_test_tensors()

        # Get reference output from PyTorch
        pytorch_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Test standard softmax version
        standard_output = naive_attention_standard(q, k, v, is_causal=True)
        self.assert_tensors_close(pytorch_output, standard_output, "PyTorch", "Standard (causal)")

        # Test online softmax version
        online_output = naive_attention_online_softmax(q, k, v, is_causal=True)
        self.assert_tensors_close(pytorch_output, online_output, "PyTorch", "Online (causal)")

        # Test scan-like online softmax version
        scan_like_output = naive_attention_scan_like(q, k, v, is_causal=True)
        self.assert_tensors_close(pytorch_output, scan_like_output, "PyTorch", "Scan-like (causal)")

        # Test that standard and online produce identical results
        self.assert_tensors_close(standard_output, online_output, "Standard (causal)", "Online (causal)")

        # Test that online and scan-like produce identical results
        self.assert_tensors_close(online_output, scan_like_output, "Online (causal)", "Scan-like (causal)")

    def test_output_shapes(self):
        """Test that output shapes match input query shapes."""
        test_configs = [
            (1, 1, 8, 32),      # Small single head
            (2, 4, 16, 64),     # Standard config
            (1, 8, 128, 96),    # Longer sequence
            (3, 12, 64, 48),    # Different dimensions
        ]

        for batch_size, num_heads, seq_len, head_dim in test_configs:
            with self.subTest(batch=batch_size, heads=num_heads, seq=seq_len, dim=head_dim):
                q, k, v = self.create_test_tensors(batch_size, num_heads, seq_len,
                                                 seq_len, head_dim)

                expected_shape = (batch_size, num_heads, seq_len, head_dim)

                # Test standard version
                standard_output = naive_attention_standard(q, k, v)
                self.assertEqual(standard_output.shape, expected_shape)

                # Test online version
                online_output = naive_attention_online_softmax(q, k, v)
                self.assertEqual(online_output.shape, expected_shape)

                # Test scan-like version
                scan_like_output = naive_attention_scan_like(q, k, v)
                self.assertEqual(scan_like_output.shape, expected_shape)

    def test_different_sequence_lengths(self):
        """Test attention with different query and key sequence lengths."""
        seq_len_q = 32
        seq_len_k = 48

        q, k, v = self.create_test_tensors(seq_len_q=seq_len_q, seq_len_k=seq_len_k)

        # PyTorch reference
        pytorch_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Our implementations
        standard_output = naive_attention_standard(q, k, v, is_causal=False)
        online_output = naive_attention_online_softmax(q, k, v, is_causal=False)
        scan_like_output = naive_attention_scan_like(q, k, v, is_causal=False)

        # Check shapes
        expected_shape = (self.batch_size, self.num_heads, seq_len_q, self.head_dim)
        self.assertEqual(pytorch_output.shape, expected_shape)
        self.assertEqual(standard_output.shape, expected_shape)
        self.assertEqual(online_output.shape, expected_shape)
        self.assertEqual(scan_like_output.shape, expected_shape)

        # Check correctness
        self.assert_tensors_close(pytorch_output, standard_output, "PyTorch", "Standard (diff seq)")
        self.assert_tensors_close(pytorch_output, online_output, "PyTorch", "Online (diff seq)")
        self.assert_tensors_close(pytorch_output, scan_like_output, "PyTorch", "Scan-like (diff seq)")

    def test_custom_scale_factor(self):
        """Test attention with custom scale factors."""
        q, k, v = self.create_test_tensors()

        custom_scales = [0.1, 0.5, 1.0, 2.0]

        for scale in custom_scales:
            with self.subTest(scale=scale):
                # PyTorch reference
                pytorch_output = F.scaled_dot_product_attention(q, k, v, scale=scale)

                # Our implementations
                standard_output = naive_attention_standard(q, k, v, sm_scale=scale)
                online_output = naive_attention_online_softmax(q, k, v, sm_scale=scale)
                scan_like_output = naive_attention_scan_like(q, k, v, sm_scale=scale)

                # Check correctness
                self.assert_tensors_close(pytorch_output, standard_output,
                                        f"PyTorch (scale={scale})", f"Standard (scale={scale})")
                self.assert_tensors_close(pytorch_output, online_output,
                                        f"PyTorch (scale={scale})", f"Online (scale={scale})")
                self.assert_tensors_close(pytorch_output, scan_like_output,
                                        f"PyTorch (scale={scale})", f"Scan-like (scale={scale})")

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large values (potential overflow)
        q, k, v = self.create_test_tensors()
        q = q * 10  # Scale up queries
        k = k * 10  # Scale up keys

        # Should still produce valid results without NaN or Inf
        standard_output = naive_attention_standard(q, k, v)
        online_output = naive_attention_online_softmax(q, k, v)
        scan_like_output = naive_attention_scan_like(q, k, v)

        # Check for NaN or Inf values
        self.assertFalse(torch.isnan(standard_output).any(), "Standard output contains NaN")
        self.assertFalse(torch.isinf(standard_output).any(), "Standard output contains Inf")
        self.assertFalse(torch.isnan(online_output).any(), "Online output contains NaN")
        self.assertFalse(torch.isinf(online_output).any(), "Online output contains Inf")
        self.assertFalse(torch.isnan(scan_like_output).any(), "Scan-like output contains NaN")
        self.assertFalse(torch.isinf(scan_like_output).any(), "Scan-like output contains Inf")

        # Online and scan-like softmax should be more numerically stable
        self.assert_tensors_close(standard_output, online_output, "Standard (large values)", "Online (large values)")
        self.assert_tensors_close(online_output, scan_like_output, "Online (large values)", "Scan-like (large values)")

    def test_edge_case_single_sequence(self):
        """Test with sequence length of 1."""
        q, k, v = self.create_test_tensors(seq_len_q=1, seq_len_k=1)

        pytorch_output = F.scaled_dot_product_attention(q, k, v)
        standard_output = naive_attention_standard(q, k, v)
        online_output = naive_attention_online_softmax(q, k, v)
        scan_like_output = naive_attention_scan_like(q, k, v)

        self.assert_tensors_close(pytorch_output, standard_output, "PyTorch (seq=1)", "Standard (seq=1)")
        self.assert_tensors_close(pytorch_output, online_output, "PyTorch (seq=1)", "Online (seq=1)")
        self.assert_tensors_close(pytorch_output, scan_like_output, "PyTorch (seq=1)", "Scan-like (seq=1)")

    def test_edge_case_single_head(self):
        """Test with single attention head."""
        q, k, v = self.create_test_tensors(num_heads=1)

        pytorch_output = F.scaled_dot_product_attention(q, k, v)
        standard_output = naive_attention_standard(q, k, v)
        online_output = naive_attention_online_softmax(q, k, v)
        scan_like_output = naive_attention_scan_like(q, k, v)

        self.assert_tensors_close(pytorch_output, standard_output, "PyTorch (1 head)", "Standard (1 head)")
        self.assert_tensors_close(pytorch_output, online_output, "PyTorch (1 head)", "Online (1 head)")
        self.assert_tensors_close(pytorch_output, scan_like_output, "PyTorch (1 head)", "Scan-like (1 head)")

    def test_main_interface(self):
        """Test the main naive_attention interface function."""
        q, k, v = self.create_test_tensors()

        # Test with use_online_softmax=False
        output_standard = naive_attention(q, k, v, use_online_softmax=False)
        expected_standard = naive_attention_standard(q, k, v)
        self.assertTrue(torch.allclose(output_standard, expected_standard))

        # Test with use_online_softmax=True
        output_online = naive_attention(q, k, v, use_online_softmax=True)
        expected_online = naive_attention_online_softmax(q, k, v)
        self.assertTrue(torch.allclose(output_online, expected_online))

        # Test with use_scan_like=True (should override use_online_softmax)
        output_scan_like = naive_attention(q, k, v, use_scan_like=True)
        expected_scan_like = naive_attention_scan_like(q, k, v)
        self.assertTrue(torch.allclose(output_scan_like, expected_scan_like))

        # Test that use_scan_like=True overrides use_online_softmax=False
        output_scan_override = naive_attention(q, k, v, use_online_softmax=False, use_scan_like=True)
        self.assertTrue(torch.allclose(output_scan_override, expected_scan_like))

    def test_compare_methods_function(self):
        """Test the compare_attention_methods utility function."""
        q, k, v = self.create_test_tensors()

        pytorch_out, standard_out, online_out, scan_like_out, metrics = compare_attention_methods(q, k, v)

        # Check that outputs are returned
        self.assertEqual(pytorch_out.shape, (self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        self.assertEqual(standard_out.shape, (self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        self.assertEqual(online_out.shape, (self.batch_size, self.num_heads, self.seq_len, self.head_dim))
        self.assertEqual(scan_like_out.shape, (self.batch_size, self.num_heads, self.seq_len, self.head_dim))

        # Check that metrics are computed
        expected_metric_keys = [
            'cosine_sim_pytorch_vs_standard',
            'cosine_sim_pytorch_vs_online',
            'cosine_sim_pytorch_vs_scan_like',
            'cosine_sim_standard_vs_online',
            'cosine_sim_standard_vs_scan_like',
            'cosine_sim_online_vs_scan_like',
            'max_abs_diff_pytorch_vs_standard',
            'max_abs_diff_pytorch_vs_online',
            'max_abs_diff_pytorch_vs_scan_like',
            'max_abs_diff_standard_vs_online',
            'max_abs_diff_standard_vs_scan_like',
            'max_abs_diff_online_vs_scan_like',
            'mean_abs_diff_pytorch_vs_standard',
            'mean_abs_diff_pytorch_vs_online',
            'mean_abs_diff_pytorch_vs_scan_like',
            'mean_abs_diff_standard_vs_online',
            'mean_abs_diff_standard_vs_scan_like',
            'mean_abs_diff_online_vs_scan_like',
            'rel_error_pytorch_vs_standard',
            'rel_error_pytorch_vs_online',
            'rel_error_pytorch_vs_scan_like',
            'rel_error_standard_vs_online',
            'rel_error_standard_vs_scan_like',
            'rel_error_online_vs_scan_like'
        ]

        for key in expected_metric_keys:
            self.assertIn(key, metrics, f"Missing metric: {key}")
            self.assertIsInstance(metrics[key], float, f"Metric {key} should be float")

        # Specific test: online and scan-like should be identical
        self.assertAlmostEqual(metrics['cosine_sim_online_vs_scan_like'], 1.0, places=6)
        self.assertLess(metrics['max_abs_diff_online_vs_scan_like'], 1e-10)

    def test_scan_like_correctness(self):
        """Test scan-like implementation correctness in detail."""
        q, k, v = self.create_test_tensors()

        # Test both causal and non-causal
        for is_causal in [False, True]:
            with self.subTest(causal=is_causal):
                # Compare with online implementation (should be identical)
                online_output = naive_attention_online_softmax(q, k, v, is_causal=is_causal)
                scan_like_output = naive_attention_scan_like(q, k, v, is_causal=is_causal)

                # Should be exactly identical
                self.assertTrue(torch.allclose(online_output, scan_like_output, rtol=1e-10, atol=1e-10),
                              f"Scan-like and online outputs should be identical for causal={is_causal}")

    def test_scan_like_shapes(self):
        """Test scan-like implementation with various tensor shapes."""
        test_configs = [
            (1, 1, 4, 32),      # Minimal case
            (2, 4, 16, 64),     # Standard case
            (1, 8, 64, 96),     # Longer sequence
            (3, 6, 32, 48),     # Non-standard dimensions
        ]

        for batch_size, num_heads, seq_len, head_dim in test_configs:
            with self.subTest(batch=batch_size, heads=num_heads, seq=seq_len, dim=head_dim):
                q, k, v = self.create_test_tensors(batch_size, num_heads, seq_len, seq_len, head_dim)

                output = naive_attention_scan_like(q, k, v)
                expected_shape = (batch_size, num_heads, seq_len, head_dim)
                self.assertEqual(output.shape, expected_shape)

                # Should not contain NaN or Inf
                self.assertFalse(torch.isnan(output).any())
                self.assertFalse(torch.isinf(output).any())

    def test_scan_like_gradients(self):
        """Test that gradients flow correctly through scan-like implementation."""
        q, k, v = self.create_test_tensors()
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        # Forward and backward pass
        output = naive_attention_scan_like(q, k, v)
        loss = output.sum()
        loss.backward()

        # Check gradients exist and are finite
        self.assertIsNotNone(q.grad, "Query gradients should not be None")
        self.assertIsNotNone(k.grad, "Key gradients should not be None")
        self.assertIsNotNone(v.grad, "Value gradients should not be None")

        self.assertTrue(torch.isfinite(q.grad).all(), "Query gradients should be finite")
        self.assertTrue(torch.isfinite(k.grad).all(), "Key gradients should be finite")
        self.assertTrue(torch.isfinite(v.grad).all(), "Value gradients should be finite")

        # Compare gradients with online softmax
        q2, k2, v2 = self.create_test_tensors()
        q2.requires_grad_(True)
        k2.requires_grad_(True)
        v2.requires_grad_(True)

        # Make sure we use the same data
        q2.data.copy_(q.data)
        k2.data.copy_(k.data)
        v2.data.copy_(v.data)

        output2 = naive_attention_online_softmax(q2, k2, v2)
        loss2 = output2.sum()
        loss2.backward()

        # Gradients should be close between scan-like and online
        self.assertTrue(torch.allclose(q.grad, q2.grad, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(k.grad, k2.grad, rtol=1e-5, atol=1e-6))
        self.assertTrue(torch.allclose(v.grad, v2.grad, rtol=1e-5, atol=1e-6))

    def test_gradient_flow(self):
        """Test that gradients flow correctly through both implementations."""
        q, k, v = self.create_test_tensors()
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)

        # Forward pass
        standard_output = naive_attention_standard(q, k, v)
        online_output = naive_attention_online_softmax(q.detach().requires_grad_(True),
                                                     k.detach().requires_grad_(True),
                                                     v.detach().requires_grad_(True))

        # Backward pass
        loss_standard = standard_output.sum()
        loss_online = online_output.sum()

        loss_standard.backward()
        loss_online.backward()

        # Check that gradients were computed
        self.assertIsNotNone(q.grad, "Query gradients should not be None")
        self.assertIsNotNone(k.grad, "Key gradients should not be None")
        self.assertIsNotNone(v.grad, "Value gradients should not be None")


def run_performance_comparison():
    """Run a basic performance comparison between methods."""
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Test configuration
    batch_size = 4
    num_heads = 8
    seq_len = 512
    head_dim = 64
    num_warmup = 5
    num_trials = 20

    print(f"Configuration: batch={batch_size}, heads={num_heads}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"Trials: {num_trials} (after {num_warmup} warmup)")

    # Create test data
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

    methods = {
        'PyTorch SDPA': lambda: F.scaled_dot_product_attention(q, k, v),
        'Standard Softmax': lambda: naive_attention_standard(q, k, v),
        'Online Softmax': lambda: naive_attention_online_softmax(q, k, v),
        'Scan-like Softmax': lambda: naive_attention_scan_like(q, k, v)
    }

    results = {}

    for method_name, method_func in methods.items():
        # Warmup
        for _ in range(num_warmup):
            _ = method_func()
            if device.type == 'cuda':
                torch.cuda.synchronize()

        # Timing
        times = []
        for _ in range(num_trials):
            start_time = time.perf_counter()
            _ = method_func()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to milliseconds

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5

        results[method_name] = {'avg': avg_time, 'std': std_time}
        print(f"{method_name:20}: {avg_time:.2f} ± {std_time:.2f} ms")

    # Compute relative performance
    pytorch_time = results['PyTorch SDPA']['avg']
    print(f"\nRelative to PyTorch SDPA:")
    for method_name, timing in results.items():
        if method_name != 'PyTorch SDPA':
            relative = timing['avg'] / pytorch_time
            print(f"{method_name:20}: {relative:.2f}x slower")


if __name__ == '__main__':
    print("Running Naive Attention Test Suite")
    print("="*50)

    # Run unit tests
    unittest.main(verbosity=2, exit=False)

    # Run performance comparison
    run_performance_comparison()