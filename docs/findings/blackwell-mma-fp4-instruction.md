# Blackwell FP4 MMA Instruction — PTX Reference

**Date:** 2026-03-17
**GPU:** NVIDIA Blackwell (SM120)
**Source:** `sageattention3_blackwell/sageattn3/blackwell/` kernel source (CUTLASS CuTe)

## Overview

SageAttention3's Blackwell kernel uses a single MMA atom for both QK and PV GEMMs:

```
SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4
```

This lowers to the PTX instruction documented below.

---

## Full PTX Instruction

```asm
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
    {%0,  %1,  %2,  %3},        // D: output (4 x f32)
    {%4,  %5,  %6,  %7},        // A: input  (4 x u32, packed FP4)
    {%8,  %9},                   // B: input  (2 x u32, packed FP4)
    {%10, %11, %12, %13},        // C: accumulator input (4 x f32)
    {%14},                        // SFA: scale factor for A (u32, packed ue4m3)
    {%15, %16},                   // A block ID, thread ID (u16, u16)
    {%17},                        // SFB: scale factor for B (u32, packed ue4m3)
    {%18, %19};                   // B block ID, thread ID (u16, u16)
```

With output/input constraints:

```
:  "=f"(d0),  "=f"(d1),  "=f"(d8),  "=f"(d9)        // D: 4 x f32 output
:   "r"(a0),   "r"(a1),   "r"(a2),   "r"(a3),        // A: 4 x u32 (32 FP4 elements)
    "r"(b0),   "r"(b1),                                // B: 2 x u32 (16 FP4 elements)
    "f"(c0),   "f"(c1),   "f"(c8),   "f"(c9),        // C: 4 x f32 accumulator
    "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA),        // A scale factor + addressing
    "r"(uint32_t(sfb0)), "h"(bidB), "h"(tidB0)        // B scale factor + addressing
```

---

## Mnemonic Breakdown

```
mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
│   │    │       │               │             │         │        │   │   │   │    │    │   │
│   │    │       │               │             │         │        │   │   │   │    │    │   └─ Scale factor type
│   │    │       │               │             │         │        │   │   │   │    │    └───── C (accum input) type
│   │    │       │               │             │         │        │   │   │   │    └────────── B element type
│   │    │       │               │             │         │        │   │   │   └─────────────── A element type
│   │    │       │               │             │         │        │   │   └──────────────────── D (output) type
│   │    │       │               │             │         │        │   └─────────────────────── B layout
│   │    │       │               │             │         │        └──────────────────────────── A layout
│   │    │       │               │             │         └───────────────────────────────────── Tile shape
│   │    │       │               │             └─────────────────────────────────────────────── Scale vector granularity
│   │    │       │               └───────────────────────────────────────────────────────────── Scaling mode
│   │    │       └───────────────────────────────────────────────────────────────────────────── MMA kind/format
│   │    └───────────────────────────────────────────────────────────────────────────────────── Alignment
│   └────────────────────────────────────────────────────────────────────────────────────────── Synchronization
└────────────────────────────────────────────────────────────────────────────────────────────── Instruction
```

### Each field explained

| Field | Value | Meaning |
|-------|-------|---------|
| `mma` | — | Matrix multiply-accumulate instruction |
| `sync` | — | Synchronous execution across the warp (all 32 threads participate) |
| `aligned` | — | Operand addresses are naturally aligned |
| `kind::mxf4nvf4` | — | Microscaling FP4, NVIDIA variant. Two FP4 formats exist on Blackwell: MX-compliant (`mxf4`) and NVIDIA-native (`nvf4`). This variant uses NVFP4 |
| `block_scale` | — | Block-level scale factors (one scale per group of elements, not per-element) |
| `scale_vec::4X` | — | Scale vector granularity: 1 UE4M3 scale factor covers a 16-element block; `4X` means 4 scale factors are packed together, covering 64 elements along K |
| `m16n8k64` | — | Tile shape: M=16 rows, N=8 columns output; K=64 reduction dimension. Each instruction reduces 64 FP4 elements |
| `row` | — | Matrix A is in row-major layout |
| `col` | — | Matrix B is in column-major layout |
| `f32` (1st) | — | **D (output)** type: FP32 |
| `e2m1` (1st) | — | **A (input)** type: E2M1 = FP4 (1 sign + 2 exponent + 1 mantissa bits) |
| `e2m1` (2nd) | — | **B (input)** type: E2M1 = FP4 |
| `f32` (2nd) | — | **C (accumulator input)** type: FP32 |
| `ue4m3` | — | **Scale factor** type: unsigned E4M3 = FP8 (0 sign + 4 exponent + 3 mantissa bits). Unsigned because scales are always positive |

---

## Operand Details

### D — Output accumulator (4 x FP32)

```
{%0, %1, %2, %3}  →  "=f"(d0), "=f"(d1), "=f"(d8), "=f"(d9)
```

4 FP32 values per thread. The 16x8 output tile is distributed across the warp's 32 threads, with each thread owning 4 elements of the result matrix.

### A — Matrix A (4 x u32 = 32 FP4 elements)

```
{%4, %5, %6, %7}  →  "r"(a0), "r"(a1), "r"(a2), "r"(a3)
```

4 x 32-bit registers = 128 bits = 32 FP4 elements per thread. With 32 threads cooperating in the warp, this covers the full 16x64 A tile. Packed as 8 FP4 values per 32-bit register.

### B — Matrix B (2 x u32 = 16 FP4 elements)

```
{%8, %9}  →  "r"(b0), "r"(b1)
```

2 x 32-bit registers = 64 bits = 16 FP4 elements per thread. Covers the 8x64 B tile (column-major, so K=64 is the leading dimension).

### C — Accumulator input (4 x FP32)

```
{%10, %11, %12, %13}  →  "f"(c0), "f"(c1), "f"(c8), "f"(c9)
```

Same layout as D. The MMA computes `D = A × B + C`, so C carries the running accumulation from previous K-tiles.

### Scale factors and addressing

```
{%14}          →  "r"(uint32_t(sfa0))   // SFA: packed ue4m3 scale factors for A
{%15, %16}     →  "h"(bidA), "h"(tidA)  // Block ID and thread ID for A scale lookup
{%17}          →  "r"(uint32_t(sfb0))   // SFB: packed ue4m3 scale factors for B
{%18, %19}     →  "h"(bidB), "h"(tidB0) // Block ID and thread ID for B scale lookup
```

- **SFA/SFB**: 32-bit registers containing packed UE4M3 scale factors (4 x 8-bit = 4 scale factors per register). Each scale factor covers a 16-element block along K.
- **bidA/bidB**: Block index — selects which group of scale factors to use (u16).
- **tidA/tidB**: Thread index — selects which thread's data this scale applies to (u16).

The hardware uses these to look up the correct scale factor for each thread's data. The effective dequantization is:

```
A_real[i] = A_fp4[i] × 2^(decode(SFA[block_of(i)]))
```

---

## Computation

The instruction computes:

```
D (f32) = (A_fp4 × SFA_scale) × (B_fp4 × SFB_scale) + C (f32)
```

Expanding:
- Each FP4 (E2M1) element has range {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0} × {+1, -1}
- The UE4M3 scale factor provides a shared exponent per 16-element block
- The products are accumulated in full FP32 precision — no intermediate rounding

### Throughput advantage

K=64 per instruction means FP4 processes **2x more elements** per MMA than FP8 (which uses K=32) and **4x more** than FP16 (K=16). This is the fundamental reason FP4 attention is faster: fewer MMA instructions for the same head dimension.

---

## Usage in SageAttention3

The kernel uses this instruction for both GEMM stages of attention:

### GEMM-I: QK (S = Q × Kᵀ)

| Operand | Role | Type |
|---------|------|------|
| A | Q (query) | FP4 (e2m1) + ue4m3 scales |
| B | K (key) | FP4 (e2m1) + ue4m3 scales |
| C/D | S (attention scores) | FP32 accumulator |

Q and K are pre-quantized to FP4 with per-block scale factors before the kernel launch.

### GEMM-II: PV (O = P × V)

| Operand | Role | Type |
|---------|------|------|
| A | P (softmax output) | FP4 (e2m1) + ue4m3 scales |
| B | V (value) | FP4 (e2m1) + ue4m3 scales |
| C/D | O (attention output) | FP32 accumulator |

P is **dynamically quantized** in-register: after softmax produces FP32 attention weights, the kernel computes per-block absmax, derives UE4M3 scale factors, and converts the FP32 values to FP4 — all in registers, no memory round-trip.

### Full pipeline

```
FP4 Q × FP4 K  ──MMA(f32)──►  FP32 S  ──softmax──►  FP32 P  ──quantize──►  FP4 P
                                                                                │
                                                            FP4 P × FP4 V  ──MMA(f32)──►  FP32 O
```

Both GEMMs accumulate in FP32. The only precision loss occurs at the FP4 quantization boundaries (Q, K, V pre-quantized; P dynamically quantized).

---

## Data Type Reference

| Type | Bits | Format | Range | Usage |
|------|------|--------|-------|-------|
| `e2m1` (FP4) | 4 | 1s + 2e + 1m | ±6.0 | Q, K, V, P matrix elements |
| `ue4m3` (FP8) | 8 | 0s + 4e + 3m | 0–448 | Block scale factors (always positive) |
| `f32` (FP32) | 32 | IEEE 754 | ±3.4e38 | Accumulators (C, D) |

## References

- [NVIDIA PTX ISA — mma instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/)
- [OCP Microscaling (MX) Specification](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
