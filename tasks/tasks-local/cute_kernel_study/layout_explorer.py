#!/usr/bin/env python3
"""
CuTe Layout Explorer for SageAttention3 Study

This script helps you understand CuTe layout algebra by computing
concrete examples from the SA3 kernel. No GPU required.

Usage: python layout_explorer.py
"""

import sys
from typing import List, Tuple


def layout_offset(shape: tuple, stride: tuple, coord: tuple) -> int:
    """Compute the physical offset for a logical coordinate."""
    if isinstance(shape, int):
        return coord * stride
    offset = 0
    for s, st, c in zip(shape, stride, coord):
        if isinstance(s, tuple):
            offset += layout_offset(s, st, c)
        else:
            offset += c * st
    return offset


def layout_size(shape) -> int:
    """Compute total number of elements in a layout."""
    if isinstance(shape, int):
        return shape
    total = 1
    for s in shape:
        total *= layout_size(s)
    return total


def print_layout_1d(shape: int, stride: int, name: str = ""):
    """Print a 1D layout mapping."""
    print(f"  {name}Layout<{shape}, {stride}>:")
    offsets = [i * stride for i in range(shape)]
    print(f"    indices 0..{shape-1} -> offsets {offsets}")


def print_layout_2d(shape: tuple, stride: tuple, name: str = ""):
    """Print a 2D layout as a grid."""
    rows, cols = shape if isinstance(shape[0], int) else (layout_size(shape[0]), layout_size(shape[1]))
    print(f"  {name}Layout<{shape}, {stride}>  ({rows}×{cols} = {rows*cols} elements):")

    # For simple shapes, show the offset grid
    if isinstance(shape[0], int) and isinstance(shape[1], int):
        max_col_show = min(cols, 16)
        for i in range(min(rows, 8)):
            offsets = [i * stride[0] + j * stride[1] for j in range(max_col_show)]
            suffix = "..." if cols > 16 else ""
            print(f"    row {i:3d}: {offsets}{suffix}")
        if rows > 8:
            print(f"    ... ({rows - 8} more rows)")


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# ============================================================
# Section 1: Basic Layout Concepts
# ============================================================
def explore_basic_layouts():
    section("1. Basic Layout Concepts")

    print("A CuTe Layout maps logical coordinates to physical offsets.")
    print("Layout = (Shape, Stride)")
    print("offset(coord) = sum(coord_i * stride_i)")
    print()

    # Row-major 4×8
    print("--- Row-major 4×8 ---")
    print_layout_2d((4, 8), (8, 1), "RowMajor: ")

    print()

    # Column-major 4×8
    print("--- Column-major 4×8 ---")
    print_layout_2d((4, 8), (1, 4), "ColMajor: ")

    print()

    # Strided (skip every other element)
    print("--- Strided 4×4, stride=(8, 2) --- (elements at even positions)")
    print_layout_2d((4, 4), (8, 2))


# ============================================================
# Section 2: Hierarchical Shapes
# ============================================================
def explore_hierarchical():
    section("2. Hierarchical (Nested) Shapes")

    print("CuTe shapes can be nested: Shape<Shape<_2, _4>, _8>")
    print("This represents 2 groups of (2×4) = 16 elements per group, 8 groups = 128 total")
    print()

    # MMA CLayout example from SM120_16x32x64_TN_VS_NVFP4
    print("--- MMA CLayout from SA3's NVFP4 atom ---")
    print("Shape:  ((4, 8), ((2, 4), 2))")
    print("Stride: ((32, 1), ((16, 128), 8))")
    print()
    print("This maps (Thread_32, Value_16) -> (M_16, N_32)")
    print()

    # Show the mapping for a few threads
    for tid in range(4):
        t0 = tid % 4   # Shape[0][0] = 4
        t1 = tid // 4  # Shape[0][1] = 8

        vals = []
        for v_2 in range(2):       # ((2, 4), 2) outer
            for v_1 in range(4):   # ((2, 4), 2) middle
                for v_0 in range(2): # ((2, 4), 2) inner
                    offset = t0 * 32 + t1 * 1 + v_0 * 16 + v_1 * 128 + v_2 * 8
                    m = offset % 16    # Assuming M is fastest-varying
                    n = offset // 16
                    vals.append(f"({offset})")

        print(f"  Thread {tid}: value offsets = {vals}")


# ============================================================
# Section 3: SA3 Concrete Dimensions
# ============================================================
def explore_sa3_dimensions():
    section("3. SA3 Kernel Dimensions")

    kBlockM = 128
    kBlockN = 128
    kHeadDim = 128  # Also kBlockK
    kStages = 3
    kNWarps = 12
    kNThreads = kNWarps * 32
    NumMmaThreads = kNThreads - 128  # Minus producer WG
    SFVectorSize = 16

    print(f"Tile shape (M×N×K): {kBlockM}×{kBlockN}×{kHeadDim}")
    print(f"MMA atom: 16×32×64 (NVFP4 block-scaled)")
    print(f"Atoms tiled along M: 8 (8 × 16 = 128)")
    print(f"")
    print(f"GEMM-I (Q×K → S):")
    print(f"  Q tile: {kBlockM}×{kHeadDim} (stays in registers)")
    print(f"  K tile: {kBlockN}×{kHeadDim} (loaded per n_block)")
    print(f"  S tile: {kBlockM}×{kBlockN} (accumulator)")
    print(f"  k_block iterations: {kHeadDim}÷64 = {kHeadDim // 64}")
    print(f"")
    print(f"GEMM-II (P×V → O):")
    print(f"  P tile: {kBlockM}×{kBlockN} (quantized to FP4 in registers)")
    print(f"  V tile: {kHeadDim}×{kBlockN} (V^T, loaded per n_block)")
    print(f"  O tile: {kBlockM}×{kHeadDim} (accumulator)")
    print(f"  v_block iterations: {kBlockN}÷64 = {kBlockN // 64}")
    print(f"")
    print(f"Threads: {kNThreads} ({kNWarps} warps)")
    print(f"  WG0 (Producer):  128 threads, 24 registers each")
    print(f"  WG1 (Consumer0): 128 threads, 232 registers each")
    print(f"  WG2 (Consumer1): 128 threads, 232 registers each")
    print(f"  Consumer MMA threads: {NumMmaThreads}")
    print(f"")
    print(f"Scale factors:")
    print(f"  SFQ/SFK: {kHeadDim // SFVectorSize} per row (1 SF per {SFVectorSize} elements)")
    print(f"  SFP: {kBlockN // SFVectorSize} per row (computed, not loaded)")
    print(f"  SFV: {kBlockN // SFVectorSize} per row")


# ============================================================
# Section 4: Shared Memory Budget
# ============================================================
def explore_smem_budget():
    section("4. Shared Memory Budget")

    kBlockM = 128
    kBlockN = 128
    kHeadDim = 128
    kStages = 3

    # FP4 = 4 bits = 0.5 bytes per element
    # E4M3 = 8 bits = 1 byte per element
    # float = 4 bytes
    # BF16 = 2 bytes

    q_data = kBlockM * kHeadDim // 2          # FP4, 1 stage
    k_data = kBlockN * kHeadDim // 2 * kStages  # FP4, 3 stages
    v_data = kHeadDim * kBlockN // 2 * kStages  # FP4, 3 stages (transposed)

    sfq = kBlockM * (kHeadDim // 16)            # E4M3, 1 stage
    sfk = kBlockN * (kHeadDim // 16) * kStages  # E4M3, 3 stages
    sfv = (kHeadDim // 16) * kBlockN * kStages  # E4M3, 3 stages (transposed)

    ds = kBlockM * kBlockN * 4 * kStages        # float32, 3 stages
    o_data = kBlockM * kHeadDim * 2              # BF16, 1 stage (epilogue)

    # Pipeline barriers (rough estimate)
    pipeline_overhead = 512  # barriers, semaphores, alignment padding

    total = q_data + k_data + v_data + sfq + sfk + sfv + ds + o_data + pipeline_overhead

    print(f"{'Component':<25} {'Size (bytes)':>12} {'Size (KB)':>10}")
    print(f"{'-'*47}")
    print(f"{'Q data (FP4, 1 stage)':<25} {q_data:>12,} {q_data/1024:>10.1f}")
    print(f"{'K data (FP4, 3 stages)':<25} {k_data:>12,} {k_data/1024:>10.1f}")
    print(f"{'V data (FP4, 3 stages)':<25} {v_data:>12,} {v_data/1024:>10.1f}")
    print(f"{'SFQ (E4M3, 1 stage)':<25} {sfq:>12,} {sfq/1024:>10.1f}")
    print(f"{'SFK (E4M3, 3 stages)':<25} {sfk:>12,} {sfk/1024:>10.1f}")
    print(f"{'SFV (E4M3, 3 stages)':<25} {sfv:>12,} {sfv/1024:>10.1f}")
    print(f"{'DeltaS (f32, 3 stages)':<25} {ds:>12,} {ds/1024:>10.1f}")
    print(f"{'O (BF16, epilogue)':<25} {o_data:>12,} {o_data/1024:>10.1f}")
    print(f"{'Pipeline overhead':<25} {pipeline_overhead:>12,} {pipeline_overhead/1024:>10.1f}")
    print(f"{'-'*47}")
    print(f"{'TOTAL':<25} {total:>12,} {total/1024:>10.1f}")
    print()
    print(f"Note: Blackwell B200 has 228 KB shared memory per SM")
    print(f"      DeltaS dominates! ({ds} bytes = {ds/1024:.1f} KB)")
    print(f"      FP4 data is very compact ({q_data + k_data + v_data} bytes total)")


# ============================================================
# Section 5: Tile Scheduling
# ============================================================
def explore_tile_scheduling():
    section("5. Tile Scheduling (StaticPersistentTileScheduler)")

    seqlen_q = 4096
    num_heads = 32
    batch_size = 2
    kBlockM = 128
    num_sm = 170  # B200

    num_blocks_m = (seqlen_q + kBlockM - 1) // kBlockM
    total_tiles = num_blocks_m * num_heads * batch_size

    print(f"Configuration:")
    print(f"  seqlen_q = {seqlen_q}, num_heads = {num_heads}, batch_size = {batch_size}")
    print(f"  kBlockM = {kBlockM}")
    print(f"  num_blocks_m = {num_blocks_m}")
    print(f"  total_tiles = {num_blocks_m} × {num_heads} × {batch_size} = {total_tiles}")
    print(f"  num_SMs = {num_sm}")
    print(f"  tiles_per_SM = {total_tiles / num_sm:.1f}")
    print()

    # Show tile assignment for first few SMs
    print("Tile assignment (SM → tile_idx → (m_block, head, batch)):")
    for sm in range(min(4, num_sm)):
        tiles = []
        tile_idx = sm
        while tile_idx < total_tiles:
            m_block = tile_idx % num_blocks_m
            remainder = tile_idx // num_blocks_m
            head = remainder % num_heads
            batch = remainder // num_heads
            tiles.append(f"({m_block},{head},{batch})")
            tile_idx += num_sm
            if len(tiles) >= 5:
                tiles.append("...")
                break
        print(f"  SM {sm}: {', '.join(tiles)}")


# ============================================================
# Section 6: Online Softmax Simulation
# ============================================================
def explore_online_softmax():
    section("6. Online Softmax Simulation")

    import math

    # Small example: 4 rows × 8 columns, split into 2 tiles of 4 columns each
    rows, cols = 4, 8
    tile_n = 4

    # Random scores
    import random
    random.seed(42)
    S = [[random.gauss(0, 2) for _ in range(cols)] for _ in range(rows)]

    print(f"Attention scores S ({rows}×{cols}):")
    for i in range(rows):
        print(f"  row {i}: [{', '.join(f'{x:6.2f}' for x in S[i])}]")

    # Standard softmax
    print(f"\n--- Standard softmax ---")
    P_standard = []
    for i in range(rows):
        max_val = max(S[i])
        exp_vals = [math.exp(x - max_val) for x in S[i]]
        sum_exp = sum(exp_vals)
        P_standard.append([x / sum_exp for x in exp_vals])
        print(f"  row {i}: [{', '.join(f'{x:.4f}' for x in P_standard[i])}]  sum={sum(P_standard[i]):.4f}")

    # Online softmax (tile by tile)
    print(f"\n--- Online softmax (tile_n={tile_n}) ---")
    row_max = [float('-inf')] * rows
    row_sum = [0.0] * rows
    O = [[0.0] * cols for _ in range(rows)]  # Placeholder for attention output

    for tile in range(cols // tile_n):
        start = tile * tile_n
        end = start + tile_n
        print(f"\n  Tile {tile} (columns {start}:{end}):")

        for i in range(rows):
            tile_vals = S[i][start:end]
            tile_max = max(tile_vals)
            new_max = max(row_max[i], tile_max)

            if row_max[i] > float('-inf'):
                scale = math.exp(row_max[i] - new_max)
            else:
                scale = 1.0

            exp_vals = [math.exp(x - new_max) for x in tile_vals]
            tile_sum = sum(exp_vals)

            row_sum[i] = row_sum[i] * scale + tile_sum
            row_max[i] = new_max

            print(f"    row {i}: tile_max={tile_max:.2f}, new_max={new_max:.2f}, "
                  f"scale={scale:.4f}, tile_sum={tile_sum:.4f}, total_sum={row_sum[i]:.4f}")

    print(f"\n  Final row sums: {[f'{s:.4f}' for s in row_sum]}")
    print(f"  Standard sums:  {[f'{sum(P_standard[i]):.4f}' for i in range(rows)]}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  CuTe Layout Explorer for SageAttention3 Study")
    print("=" * 70)

    explore_basic_layouts()
    explore_hierarchical()
    explore_sa3_dimensions()
    explore_smem_budget()
    explore_tile_scheduling()
    explore_online_softmax()

    print("\n" + "=" * 70)
    print("  Done! Review the output above to build intuition for CuTe layouts.")
    print("  Next: Read PHASE1_CUTE_FOUNDATIONS.md for the full theory.")
    print("=" * 70)
