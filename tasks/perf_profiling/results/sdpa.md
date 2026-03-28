# CogVideoX-2b Workload Profile
Config: 480×720, 49 frames, 50 steps, batch=2, attention=sdpa (PyTorch SDPA)
GPU: NVIDIA GeForce RTX 5090 D

## Timing Summary
| Component                                | GPU Time (ms) | % Pipeline | % Transformer |
|------------------------------------------|---------------|------------|---------------|
| **Pipeline total**                       |       68463.7 |     100.0% |            — |
| **Transformer total**                    |       63462.3 |      92.7% |        100.0% |
| Attention kernel                         |       32444.0 |      47.4% |         51.1% |
| Attn QKV projection                      |        5607.4 |       8.2% |          8.8% |
| Attn output projection                   |        1884.4 |       2.8% |          3.0% |
| FFN (up + down)                          |       15320.5 |      22.4% |         24.1% |
| Norm linear                              |          38.3 |       0.1% |          0.1% |
| Other linear                             |          12.9 |       0.0% |          0.0% |
| **All linear total**                     |       22863.5 |      33.4% |         36.0% |
| Unaccounted (norms, activations, RoPE, reshapes) |        8154.8 |      11.9% |         12.8% |

**Sanity check**: attn + all_linear = 55307.5 ms = 87.2% of transformer

## Wall-Clock Check
| Wall-clock (ms) |  GPU pipeline (ms) |  Host overhead % |
|-----------------|--------------------|------------------|
|         68461.7 |            68463.7 |            -0.0% |

## Per-Call Statistics
| Component                      |  Total Calls |  Avg ms/call |
|--------------------------------|--------------|--------------|
| Transformer forward            |           50 |      1269.25 |
| Attention kernel               |         1500 |        21.63 |
| Attn QKV projection (90 layers) |         4500 |         1.25 |
| Attn output projection (30 layers) |         1500 |         1.26 |
| FFN (up + down) (60 layers)    |         3000 |         5.11 |
| Norm linear (61 layers)        |         3050 |         0.01 |
| Other linear (4 layers)        |          200 |         0.06 |

## Linear Layer Discovery
Total nn.Linear layers discovered: 245
  - attn_qkv: 90
  - attn_out: 30
  - ffn: 60
  - norm_linear: 61
  - other_linear: 4
