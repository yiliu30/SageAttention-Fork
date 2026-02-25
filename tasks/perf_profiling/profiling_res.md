
# Profiling configurations:

| Parameter | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 D (32 GB VRAM) |
| Driver | 580.105.08 |
| Model | CogVideoX-2b (`cogvideox-2b`) |
| Model dtype | float16 |
| Num frames | 49 |
| Num inference steps | 5 (smoke mode) |
| Guidance scale | 6 |
| Prompt | "A dog is running in the park." |
| Warmup iterations | 3 |
| CPU offload | Disabled (full model on CUDA) |
| Profiling method | Async CUDA event timing (GPU device time) |


### Results:
============================================================
Attention Proportion Report (sage3)
============================================================
  Total pipeline GPU time :     9664.2 ms
  Transformer GPU time    :     4651.0 ms
  Attention kernel time   :     1625.4 ms
  ---- Proportions ----
  Attn / Pipeline         :       16.8 %
  Attn / Transformer      :       34.9 %
  Transformer / Pipeline  :       48.1 %
  ---- Counts ----
  Attention calls         :        150
  Transformer fwd calls   :          5
  Avg attn per call       :      10.84 ms
  ---- Sage3 Sub-step Breakdown ----
  preprocess_qkv                :    317.1 ms ( 19.5% of attn)
  scale_and_quant_fp4           :     16.9 ms (  1.0% of attn)
  scale_and_quant_fp4_permute   :     16.3 ms (  1.0% of attn)
  scale_and_quant_fp4_transpose :     16.4 ms (  1.0% of attn)
  blockscaled_fp4_attn          :   1232.2 ms ( 75.8% of attn)
============================================================


============================================================
Attention Proportion Report (sdpa)
============================================================
  Total pipeline GPU time :    11350.6 ms
  Transformer GPU time    :     6290.7 ms
  Attention kernel time   :     3216.1 ms
  ---- Proportions ----
  Attn / Pipeline         :       28.3 %
  Attn / Transformer      :       51.1 %
  Transformer / Pipeline  :       55.4 %
  ---- Counts ----
  Attention calls         :        150
  Transformer fwd calls   :          5
  Avg attn per call       :      21.44 ms
============================================================