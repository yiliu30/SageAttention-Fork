#!/usr/bin/env python3
"""Quantify the numerical impact of each bug independently."""

import torch
import torch.nn.functional as F
import math
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sageattn3_torch import apply_qk_smoothing, educational_quantize, educational_quantize_p_two_level

def reference_attention(q, k, v, sm_scale):
    """Standard SDPA (ground truth)."""
    return F.scaled_dot_product_attention(q, k, v, scale=sm_scale)

def tiled_attention(q, k, v, delta_s, sm_scale, bug1=False, bug2=False, tile_q=128, tile_k=128):
    """
    Tiled online attention with toggleable bugs.
    
    bug1=True: scale QK^T before adding delta_s (wrong order)
    bug2=True: multiply tile_sum and pv by beta (double-counting)
    """
    B, H, N, D = q.shape
    device = q.device
    output = torch.zeros_like(q)
    num_q_tiles = (N + tile_q - 1) // tile_q
    num_k_tiles = (N + tile_k - 1) // tile_k

    for qi in range(num_q_tiles):
        qs, qe = qi * tile_q, min((qi + 1) * tile_q, N)
        q_tile = q[:, :, qs:qe, :]
        running_max = torch.full((B, H, qe - qs), -torch.inf, dtype=torch.float32, device=device)
        running_sum = torch.zeros((B, H, qe - qs), dtype=torch.float32, device=device)
        out_tile = torch.zeros((B, H, qe - qs, D), dtype=q.dtype, device=device)

        for ki in range(num_k_tiles):
            ks, ke = ki * tile_k, min((ki + 1) * tile_k, N)
            k_tile = k[:, :, ks:ke, :]
            v_tile = v[:, :, ks:ke, :]

            qk = torch.matmul(q_tile, k_tile.transpose(-2, -1))

            if delta_s is not None:
                gid = qi if delta_s.size(2) > 1 else 0
                gid = min(gid, delta_s.size(2) - 1)
                ds = delta_s[:, :, gid, ks:ke]
                ds_bc = ds.unsqueeze(2).expand(-1, -1, qe - qs, -1)

                if bug1:
                    # BUG 1: scale first, then add delta_s (wrong)
                    qk = qk * sm_scale + ds_bc
                else:
                    # CORRECT: add delta_s first, then scale
                    qk = (qk + ds_bc) * sm_scale
            else:
                qk = qk * sm_scale

            tile_max = qk.max(dim=-1)[0].float()
            old_max = running_max.clone()
            new_max = torch.maximum(running_max, tile_max)
            alpha = torch.exp(old_max - new_max)

            out_tile = out_tile * alpha.unsqueeze(-1).to(q.dtype)

            p = torch.exp(qk - new_max.unsqueeze(-1).to(q.dtype))
            pv = torch.matmul(p.to(v_tile.dtype), v_tile)

            if bug2:
                # BUG 2: multiply by beta (double-counting)
                beta = torch.exp(tile_max - new_max)
                running_sum = running_sum * alpha + p.sum(dim=-1).float() * beta
                out_tile = out_tile + pv * beta.unsqueeze(-1).to(q.dtype)
            else:
                # CORRECT: direct accumulation
                running_sum = running_sum * alpha + p.sum(dim=-1).float()
                out_tile = out_tile + pv

            running_max = new_max

        out_tile = out_tile / (running_sum.unsqueeze(-1).to(q.dtype) + 1e-8)
        output[:, :, qs:qe, :] = out_tile

    return output

def cosine_sim(a, b):
    return F.cosine_similarity(a.reshape(1, -1).float(), b.reshape(1, -1).float()).item()

def max_abs_diff(a, b):
    return (a.float() - b.float()).abs().max().item()

def main():
    device = 'cuda'
    dtype = torch.float16

    configs = [
        {"B": 1, "H": 8, "N": 256, "D": 64, "scale": 0.1, "name": "small rand (0.1)"},
        {"B": 1, "H": 8, "N": 256, "D": 64, "scale": 1.0, "name": "unit rand (1.0)"},
        {"B": 1, "H": 8, "N": 256, "D": 64, "scale": 3.0, "name": "large rand (3.0)"},
        {"B": 1, "H": 8, "N": 1024, "D": 64, "scale": 1.0, "name": "long seq 1024"},
        {"B": 1, "H": 8, "N": 4096, "D": 64, "scale": 1.0, "name": "long seq 4096"},
        {"B": 1, "H": 8, "N": 256, "D": 128, "scale": 1.0, "name": "D=128 (sm_scale smaller)"},
    ]

    print("=" * 80)
    print("Impact Analysis: Each Bug Isolated")
    print("=" * 80)

    for cfg in configs:
        B, H, N, D = cfg["B"], cfg["H"], cfg["N"], cfg["D"]
        s = cfg["scale"]
        sm_scale = 1.0 / math.sqrt(D)
        torch.manual_seed(42)

        q = torch.randn(B, H, N, D, device=device, dtype=dtype) * s
        k = torch.randn(B, H, N, D, device=device, dtype=dtype) * s
        v = torch.randn(B, H, N, D, device=device, dtype=dtype) * s

        # Ground truth (no quantization, no smoothing)
        ref = reference_attention(q, k, v, sm_scale)

        # QK smoothing
        q_s, k_s, delta_s = apply_qk_smoothing(q, k)

        print(f"\n--- {cfg['name']} ({B}x{H}x{N}x{D}) ---")
        print(f"  delta_s magnitude: mean={delta_s.abs().mean().item():.4e}, max={delta_s.abs().max().item():.4e}")
        print(f"  QK^T magnitude (approx): ~{(q.float() @ k.float().transpose(-2,-1)).abs().mean().item():.4e}")
        print(f"  sm_scale = {sm_scale:.4f}")
        print(f"  delta_s * sm_scale vs delta_s ratio: {sm_scale:.4f}x")
        print()

        # 4 variants: no bugs, bug1 only, bug2 only, both bugs
        correct    = tiled_attention(q_s, k_s, v, delta_s, sm_scale, bug1=False, bug2=False)
        with_bug1  = tiled_attention(q_s, k_s, v, delta_s, sm_scale, bug1=True,  bug2=False)
        with_bug2  = tiled_attention(q_s, k_s, v, delta_s, sm_scale, bug1=False, bug2=True)
        with_both  = tiled_attention(q_s, k_s, v, delta_s, sm_scale, bug1=True,  bug2=True)

        # Also: correct without smoothing (to see if smoothing helps at all)
        no_smooth  = tiled_attention(q, k, v, None, sm_scale, bug1=False, bug2=False)

        print(f"  {'Variant':<35} {'vs SDPA':>12} {'vs Correct':>12} {'MaxDiff':>12}")
        print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}")

        variants = [
            ("No smoothing (baseline)",         no_smooth,  None),
            ("Correct (both bugs fixed)",        correct,    None),
            ("Bug1 only (delta_s unscaled)",     with_bug1,  correct),
            ("Bug2 only (beta double-count)",    with_bug2,  correct),
            ("Both bugs (original code)",        with_both,  correct),
        ]

        for name, out, vs in variants:
            sim_ref = cosine_sim(out, ref)
            diff_ref = max_abs_diff(out, ref)
            if vs is not None:
                sim_correct = cosine_sim(out, vs)
                print(f"  {name:<35} {sim_ref:>11.6f}  {sim_correct:>11.6f}  {diff_ref:>11.3e}")
            else:
                print(f"  {name:<35} {sim_ref:>11.6f}  {'—':>12}  {diff_ref:>11.3e}")

    print("\n" + "=" * 80)
    print("Key:")
    print("  'vs SDPA'    = cosine similarity to torch SDPA (ground truth, no quant)")
    print("  'vs Correct' = cosine similarity to the fully-fixed tiled implementation")
    print("  Bug1: delta_s added AFTER sm_scale (wrong order, off by 1/sqrt(D))")
    print("  Bug2: tile_sum & pv multiplied by beta (double-counting max shift)")
    print("=" * 80)

if __name__ == "__main__":
    main()
