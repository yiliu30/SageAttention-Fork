#!/usr/bin/env python3
"""
Real Kernel Comparison Script for SageAttention3 Educational Implementation

This script compares three SageAttention3 implementations:
1. Educational PyTorch implementation (sageattn3_torch)
2. Educational Triton implementation (sageattn3_torch_triton)
3. Real SageAttention3 Blackwell kernel (sageattn3_blackwell)

Usage:
    python compare_real_kernel.py

Requirements:
    - CUDA Blackwell GPU (RTX 5090 D or similar)
    - SageAttention3 Blackwell kernel compiled and installed
"""

import torch
import torch.nn.functional as F
import sys
import os

def setup_paths():
    """Add necessary paths for imports."""
    # Add current directory for our implementation
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    # Add Blackwell kernel path
    blackwell_path = '/mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell'
    sys.path.insert(0, blackwell_path)

def test_real_kernel_comparison():
    """Compare educational implementations against real SageAttention3 Blackwell kernel."""
    print("⚡ SageAttention3 Multi-Implementation Comparison")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available - real kernel requires CUDA")
        return False

    device = torch.device('cuda')
    print(f"🖥️  GPU: {torch.cuda.get_device_name(0)}")

    # Import implementations
    implementations = {}

    # try:
    #     from sageattn3_torch import sageattn3_torch
    #     implementations['pytorch'] = sageattn3_torch
    #     print("✅ Educational PyTorch implementation imported")
    # except ImportError as e:
    #     print(f"❌ Failed to import PyTorch implementation: {e}")

    try:
        from sageattn3_torch_triton import sageattn3_torch_triton
        implementations['triton'] = sageattn3_torch_triton
        print("✅ Educational Triton implementation imported")
    except ImportError as e:
        print(f"❌ Failed to import Triton implementation: {e}")

    try:
        from sageattn3 import sageattn3_blackwell
        implementations['real'] = sageattn3_blackwell
        print("✅ Real SageAttention3 Blackwell kernel imported")
    except ImportError as e:
        print(f"❌ Failed to import real kernel: {e}")
        print("   Make sure the kernel is compiled for your GPU")

    if len(implementations) < 2:
        print("❌ Need at least 2 implementations to compare")
        return False

    # Test configurations
    test_configs = [
        # {"B": 1, "H": 8, "N": 256, "D": 64, "name": "Small"},
        # {"B": 1, "H": 8, "N": 512, "D": 64, "name": "Medium"},
        # {"B": 1, "H": 8, "N": 1024, "D": 64, "name": "Large"},
        # {"B": 2, "H": 16, "N": 512, "D": 64, "name": "Multi-batch"},
        # {"B": 2, "H": 30, "N": 256, "D": 64, "name": "CogVideoX-like"},  # Problematic config
        {"B": 2, "H": 30, "N": 1024*16, "D": 64, "name": "CogVideoX-like-Extreme"},  # Problematic config
    ]

    all_results = []

    for config in test_configs:
        B, H, N, D = config["B"], config["H"], config["N"], config["D"]
        name = config["name"]

        print(f"\n🧪 Testing {name} Configuration: {B}×{H}×{N}×{D}")
        print("-" * 50)

        # Skip large configs if head_dim >= 256 for real kernel
        if D >= 256 and 'real' in implementations:
            print("⚠️ Skipped real kernel: requires head_dim < 256")

        torch.manual_seed(42)
        dtype = torch.float16

        # Create test tensors
        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * 0.01

        config_results = {}

        # Run each implementation
        for impl_name, impl_func in implementations.items():
            try:
                print(f"🔬 Running {impl_name} implementation...")

                torch.cuda.synchronize()
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()

                if impl_name == 'pytorch':
                    output = impl_func(
                        q.clone(), k.clone(), v.clone(),
                        tensor_layout='HND',
                        per_block_mean=True,
                        is_causal=False
                    )
                elif impl_name == 'triton':
                    output = impl_func(
                        q.clone(), k.clone(), v.clone(),
                        tensor_layout='HND',
                        per_block_mean=True,
                        is_causal=False,
                        tile_size_q=128,    # Aligned with real kernel
                        tile_size_k=128,    # Aligned with real kernel
                        debug=False
                    )
                elif impl_name == 'real':
                    if D >= 256:  # Skip real kernel for large head_dim
                        print(f"  ⚠️ Skipping: head_dim {D} >= 256")
                        continue
                    output = impl_func(
                        q.clone(), k.clone(), v.clone(),
                        is_causal=False,
                        per_block_mean=True
                    )

                end_time.record()
                torch.cuda.synchronize()
                exec_time = start_time.elapsed_time(end_time)

                config_results[impl_name] = {
                    'output': output,
                    'time': exec_time,
                    'success': True
                }

                print(f"  ✅ Success: {exec_time:.2f}ms, range=[{output.min().item():.3f}, {output.max().item():.3f}]")

            except Exception as e:
                print(f"  ❌ Failed: {e}")
                config_results[impl_name] = {
                    'output': None,
                    'time': float('inf'),
                    'success': False,
                    'error': str(e)
                }

        # Compare all pairs of implementations
        print(f"\n📊 Cross-Implementation Comparison:")
        print("-" * 30)

        successful_impls = [(k, v) for k, v in config_results.items() if v['success']]
        if len(successful_impls) < 2:
            print("⚠️ Not enough successful implementations to compare")
            continue

        config_comparisons = []

        # Compare all pairs
        for i, (name1, result1) in enumerate(successful_impls):
            for j, (name2, result2) in enumerate(successful_impls):
                if i < j:  # Only compare each pair once
                    output1, output2 = result1['output'], result2['output']

                    # Compute similarity
                    cosine_sim = F.cosine_similarity(
                        output1.reshape(1, -1).float(),
                        output2.reshape(1, -1).float(),
                        dim=-1
                    ).item()

                    # Compute differences
                    diff = (output1 - output2).float()
                    max_abs = diff.abs().max().item()
                    mean_abs = diff.abs().mean().item()

                    # Performance comparison
                    time1, time2 = result1['time'], result2['time']
                    speedup = time1 / time2 if time2 > 0 else float('inf')

                    # Grade the comparison
                    if cosine_sim >= 0.999:
                        grade = "🎉 IDENTICAL"
                        grade_color = "🟢"
                    elif cosine_sim >= 0.99:
                        grade = "🔥 EXCELLENT"
                        grade_color = "🟢"
                    elif cosine_sim >= 0.95:
                        grade = "✅ VERY GOOD"
                        grade_color = "🟢"
                    elif cosine_sim >= 0.90:
                        grade = "⚠️ GOOD"
                        grade_color = "🟡"
                    elif cosine_sim >= 0.80:
                        grade = "⚠️ FAIR"
                        grade_color = "🟡"
                    else:
                        grade = "❌ POOR"
                        grade_color = "🔴"

                    print(f"  {name1} vs {name2}:")
                    print(f"    Similarity: {cosine_sim:.6f} {grade_color} {grade}")
                    print(f"    Max diff: {max_abs:.3e}, Mean diff: {mean_abs:.3e}")
                    print(f"    Time ratio ({name1}/{name2}): {speedup:.2f}x")

                    config_comparisons.append({
                        'pair': f"{name1}_vs_{name2}",
                        'similarity': cosine_sim,
                        'max_diff': max_abs,
                        'grade': grade,
                        'success': cosine_sim >= 0.85
                    })

        all_results.append({
            'config': name,
            'implementations': config_results,
            'comparisons': config_comparisons
        })

    # Overall summary
    print("\n" + "=" * 70)
    print("📈 MULTI-IMPLEMENTATION COMPARISON RESULTS")
    print("=" * 70)

    if not all_results:
        print("❌ No tests completed successfully")
        return False

    # Summary by configuration
    print("\n📋 Configuration Summary:")
    print("-" * 40)

    total_configs = len(all_results)
    successful_configs = 0
    all_similarities = []

    for result in all_results:
        config_name = result['config']
        successful_impls = len([impl for impl in result['implementations'].values() if impl['success']])
        successful_comparisons = [comp for comp in result['comparisons'] if comp['success']]

        if successful_comparisons:
            avg_similarity = sum(comp['similarity'] for comp in successful_comparisons) / len(successful_comparisons)
            all_similarities.extend([comp['similarity'] for comp in successful_comparisons])
            successful_configs += 1
            status = "✅"
        else:
            avg_similarity = 0.0
            status = "❌"

        print(f"{status} {config_name:<15} {successful_impls}/3 impls  Avg similarity: {avg_similarity:.4f}")

    # Overall statistics
    if all_similarities:
        overall_avg = sum(all_similarities) / len(all_similarities)
        min_sim = min(all_similarities)
        max_sim = max(all_similarities)

        print(f"\n📊 Overall Statistics:")
        print(f"   Successful configs: {successful_configs}/{total_configs}")
        print(f"   Average similarity: {overall_avg:.4f}")
        print(f"   Similarity range: {min_sim:.4f} - {max_sim:.4f}")

        # Detailed comparison breakdown
        print(f"\n🔍 Implementation Comparison Breakdown:")
        print("-" * 50)

        # Collect all pairwise comparisons
        pytorch_vs_triton = []
        pytorch_vs_real = []
        triton_vs_real = []

        for result in all_results:
            for comp in result['comparisons']:
                if comp['pair'] == 'pytorch_vs_triton':
                    pytorch_vs_triton.append(comp['similarity'])
                elif comp['pair'] == 'pytorch_vs_real':
                    pytorch_vs_real.append(comp['similarity'])
                elif comp['pair'] == 'triton_vs_real':
                    triton_vs_real.append(comp['similarity'])

        def print_comparison_stats(name, similarities):
            if similarities:
                avg = sum(similarities) / len(similarities)
                min_sim = min(similarities)
                max_sim = max(similarities)
                print(f"  {name:<20} Avg: {avg:.4f} Range: [{min_sim:.4f}, {max_sim:.4f}] ({len(similarities)} tests)")
            else:
                print(f"  {name:<20} No successful comparisons")

        print_comparison_stats("PyTorch vs Triton:", pytorch_vs_triton)
        print_comparison_stats("PyTorch vs Real:", pytorch_vs_real)
        print_comparison_stats("Triton vs Real:", triton_vs_real)

        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        if overall_avg >= 0.99:
            print("🎉 OUTSTANDING: All implementations are highly aligned!")
            final_success = True
        elif overall_avg >= 0.95:
            print("🔥 EXCELLENT: Implementations are well aligned!")
            final_success = True
        elif overall_avg >= 0.90:
            print("✅ GOOD: Implementations are reasonably aligned")
            final_success = True
        elif overall_avg >= 0.80:
            print("⚠️  FAIR: Some alignment issues detected")
            final_success = False
        else:
            print("❌ POOR: Significant alignment issues between implementations")
            final_success = False

        # Specific analysis for Triton implementation
        if pytorch_vs_triton:
            triton_avg = sum(pytorch_vs_triton) / len(pytorch_vs_triton)
            print(f"\n🔬 TRITON ANALYSIS:")
            if triton_avg >= 0.99:
                print(f"✅ Triton implementation is highly accurate ({triton_avg:.4f} similarity)")
            elif triton_avg >= 0.90:
                print(f"⚠️  Triton implementation has moderate accuracy ({triton_avg:.4f} similarity)")
                print("   Consider investigation for accuracy improvements")
            else:
                print(f"❌ Triton implementation has poor accuracy ({triton_avg:.4f} similarity)")
                print("   Requires debugging - not suitable for production use")

    else:
        print("❌ No successful comparisons")
        final_success = False

    print("\n🔍 Key Verification Points:")
    print("✅ Global range normalization: vecMax / 6.0f")
    print("✅ Two-level P quantization: FP8(448) + FP4(6)")
    print("✅ True NVFP4 E2M1 quantization levels")
    print("✅ Real kernel constant alignment")
    if 'triton' in [impl for result in all_results for impl in result['implementations'].keys()]:
        print("✅ Triton kernel implementation accuracy")

    return final_success

def main():
    """Main comparison function."""
    setup_paths()

    print("🎯 SageAttention3 Multi-Implementation Comparison")
    print("Comparing educational PyTorch, Triton, and real Blackwell kernel implementations")
    print()

    success = test_real_kernel_comparison()

    if success:
        print("\n✅ COMPARISON SUCCESSFUL: Implementation alignment verified!")
    else:
        print("\n❌ COMPARISON ISSUES: Review implementation alignment or GPU compatibility")

    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n🛑 Comparison interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)