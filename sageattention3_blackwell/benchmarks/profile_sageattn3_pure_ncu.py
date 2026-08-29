import os

import torch

from sageattn3.api import (
    blockscaled_fp4_attn,
    preprocess_qkv,
    scale_and_quant_fp4,
    scale_and_quant_fp4_permute,
    scale_and_quant_fp4_transpose,
    scale_and_quant_fp8_transpose,
)

BATCH, HEADS, SEQUENCE_LENGTH, HEAD_DIM = 1, 40, 16384, 128
DEVICE = "cuda:0"
KERNEL_VARIANT = os.environ.get("SAGEATTN3_KERNEL_VARIANT", "baseline")
if KERNEL_VARIANT not in ("baseline", "two_cta", "p_pack_bypass", "fp8_pv"):
    raise ValueError(f"Unknown kernel variant: {KERNEL_VARIANT}")
USE_TWO_CTA = KERNEL_VARIANT == "two_cta"
BYPASS_P_PACKING = KERNEL_VARIANT == "p_pack_bypass"
USE_FP8_PV = KERNEL_VARIANT == "fp8_pv"

torch.manual_seed(0)
q = torch.randn(BATCH, HEADS, SEQUENCE_LENGTH, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)
q, k, v, delta_s = preprocess_qkv(
    q,
    k,
    v,
    q_group_size=64 if USE_TWO_CTA else 128,
)
q_list = scale_and_quant_fp4(q)
k_list = scale_and_quant_fp4_permute(k)
v_list = (
    scale_and_quant_fp8_transpose(v)
    if USE_FP8_PV
    else scale_and_quant_fp4_transpose(v)
)


def pure_attention():
    return blockscaled_fp4_attn(
        q_list,
        k_list,
        v_list,
        delta_s,
        SEQUENCE_LENGTH,
        is_causal=False,
        use_two_cta=USE_TWO_CTA,
        bypass_p_packing=BYPASS_P_PACKING,
        fp8_pv=USE_FP8_PV,
    )


for _ in range(3):
    pure_attention()
torch.cuda.synchronize()

cudart = torch.cuda.cudart()
cudart.cudaProfilerStart()
pure_attention()
torch.cuda.synchronize()
cudart.cudaProfilerStop()
