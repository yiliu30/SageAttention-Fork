# CogVideoX-2b Workload Profile
Config: 480×720, 49 frames, 50 steps, batch=2, attention=sage3 (SageAttention3 (CUTE/Blackwell))
GPU: NVIDIA GeForce RTX 5090 D

## Timing Summary
| Component                                | GPU Time (ms) | % Pipeline | % Transformer |
|------------------------------------------|---------------|------------|---------------|
| **Pipeline total**                       |       52357.1 |     100.0% |            — |
| **Transformer total**                    |       47425.8 |      90.6% |        100.0% |
| Attention kernel                         |       16580.8 |      31.7% |         35.0% |
| Attn QKV projection                      |        5571.7 |      10.6% |         11.7% |
| Attn output projection                   |        1852.8 |       3.5% |          3.9% |
| FFN (up + down)                          |       15209.0 |      29.0% |         32.1% |
| Norm linear                              |          38.4 |       0.1% |          0.1% |
| Other linear                             |          12.9 |       0.0% |          0.0% |
| **All linear total**                     |       22684.8 |      43.3% |         47.8% |
| Unaccounted (norms, activations, RoPE, reshapes) |        8160.3 |      15.6% |         17.2% |

**Sanity check**: attn + all_linear = 39265.6 ms = 82.8% of transformer

## Wall-Clock Check
| Wall-clock (ms) |  GPU pipeline (ms) |  Host overhead % |
|-----------------|--------------------|------------------|
|         52355.6 |            52357.1 |            -0.0% |

## Per-Call Statistics
| Component                      |  Total Calls |  Avg ms/call |
|--------------------------------|--------------|--------------|
| Transformer forward            |           50 |       948.52 |
| Attention kernel               |         1500 |        11.05 |
| Attn QKV projection (90 layers) |         4500 |         1.24 |
| Attn output projection (30 layers) |         1500 |         1.24 |
| FFN (up + down) (60 layers)    |         3000 |         5.07 |
| Norm linear (61 layers)        |         3050 |         0.01 |
| Other linear (4 layers)        |          200 |         0.06 |

## Linear Layer Discovery
Total nn.Linear layers discovered: 245
  - attn_qkv: 90
  - attn_out: 30
  - ffn: 60
  - norm_linear: 61
  - other_linear: 4
