"""
Chunked Scaled Dot-Product Attention (bf16).

Chunks along the Q sequence dimension so the full attention map is never
materialised at once.  Softmax is exact (each chunk sees the full K-row).

Usage:
    python chunked_sdpa.py          # quick smoke-test with random tensors
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional


def chunked_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
    sm_scale: Optional[float] = None,
    is_causal: bool = False,
) -> torch.Tensor:
    """
    Chunked scaled dot-product attention.

    Args:
        q: [B, H, N_q, d]  (bf16 / fp16 / fp32)
        k: [B, H, N_kv, d]
        v: [B, H, N_kv, d]
        chunk_size: number of Q-rows per chunk
        sm_scale: softmax scale, defaults to 1/sqrt(d)
        is_causal: apply causal mask

    Returns:
        o: [B, H, N_q, d]  same dtype as q
    """
    B, H, N_q, d = q.shape
    N_kv = k.shape[2]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)

    o = torch.empty_like(q)

    for start in range(0, N_q, chunk_size):
        end = min(start + chunk_size, N_q)
        q_chunk = q[:, :, start:end]                          # [B, H, C, d]

        # S = Q_chunk @ K^T, computed in fp32 for precision
        s = torch.matmul(
            q_chunk.float(), k.float().transpose(-1, -2)
        )                                                      # [B, H, C, N_kv]
        s *= sm_scale

        if is_causal:
            # rows [start, end), cols [0, N_kv)
            # mask: col > row  (row index is start + local_row)
            row_idx = torch.arange(start, end, device=s.device).unsqueeze(1)
            col_idx = torch.arange(N_kv, device=s.device).unsqueeze(0)
            causal_mask = col_idx > row_idx                    # [C, N_kv]
            s.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        p = F.softmax(s, dim=-1)                               # [B, H, C, N_kv]  fp32

        # O_chunk = P @ V
        o_chunk = torch.matmul(p, v.float())                   # [B, H, C, d]  fp32
        o[:, :, start:end] = o_chunk.to(q.dtype)

    return o


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _run_smoke_test():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    B, H, N, d = 1, 8, 1024, 64
    q = torch.randn(B, H, N, d, device=device, dtype=dtype)
    k = torch.randn(B, H, N, d, device=device, dtype=dtype)
    v = torch.randn(B, H, N, d, device=device, dtype=dtype)

    # Reference: PyTorch sdpa
    ref = F.scaled_dot_product_attention(q, k, v)

    for chunk_size in [64, 128, 256, N]:
        out = chunked_sdpa(q, k, v, chunk_size=chunk_size)

        cos_sim = F.cosine_similarity(
            ref.flatten().float(), out.flatten().float(), dim=0
        ).item()
        max_diff = (ref.float() - out.float()).abs().max().item()
        print(f"chunk_size={chunk_size:5d}  cos_sim={cos_sim:.8f}  max_diff={max_diff:.6e}")

    # Causal test
    ref_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out_causal = chunked_sdpa(q, k, v, chunk_size=128, is_causal=True)
    cos_sim_c = F.cosine_similarity(
        ref_causal.flatten().float(), out_causal.flatten().float(), dim=0
    ).item()
    print(f"causal   chunk=128   cos_sim={cos_sim_c:.8f}")

    print("\nSmoke test passed." if cos_sim > 0.99999 else "\nWARNING: low similarity!")


if __name__ == "__main__":
    _run_smoke_test()
