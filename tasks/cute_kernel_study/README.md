# CuTe Kernel Study — Zero to Hero via SageAttention3

## Overview

This directory contains a comprehensive learning guide for mastering CuTe/CUTLASS kernel development by studying the SageAttention3 Blackwell CUTE kernel (~3,800 lines of custom code across 14 files).

**Prerequisites**: CUDA kernel experience + PTX familiarity. No CUTLASS/CuTe experience needed.

## Study Materials

| File | Phase | Topic | Est. Time |
|------|-------|-------|-----------|
| [PHASE1_CUTE_FOUNDATIONS.md](PHASE1_CUTE_FOUNDATIONS.md) | 1 | CuTe layout algebra, tensors, tiling, TMA, MMA atoms | 4-6 hours |
| [PHASE2_FMHA_IN_CUTEDSL.md](PHASE2_FMHA_IN_CUTEDSL.md) | 2 | Flash attention algorithm, CuTeDSL study, mapping to C++ | 4-6 hours |
| [PHASE3_SA3_ARCHITECTURE.md](PHASE3_SA3_ARCHITECTURE.md) | 3 | SA3 kernel architecture: warp specialization, tile scheduling, pipelines | 4-6 hours |
| [PHASE4_MAINLOOP_LINE_BY_LINE.md](PHASE4_MAINLOOP_LINE_BY_LINE.md) | 4 | Line-by-line analysis of mainloop_tma_ws.h (the 908-line kernel core) | 6-8 hours |
| [PHASE5_HANDS_ON_MASTERY.md](PHASE5_HANDS_ON_MASTERY.md) | 5 | Exercises, modifications, benchmarks, verification checklists | 4-6 hours |
| [layout_explorer.py](layout_explorer.py) | 1-3 | Interactive Python script for exploring CuTe concepts | Run anytime |

**Total estimated study time: 25-35 hours**

## Quick Start

```bash
# 1. Run the layout explorer (no GPU needed)
python layout_explorer.py

# 2. Start reading Phase 1
# Open PHASE1_CUTE_FOUNDATIONS.md

# 3. Set up CuTeDSL (optional, for CuTeDSL exercises)
cd sageattention3_blackwell/csrc/cutlass
pip install -e python/ --no-build-isolation
```

## SA3 Kernel File Map

```
sageattention3_blackwell/sageattn3/blackwell/
├── api.cu              (364 lines) Entry point: PyTorch ↔ CUDA
├── params.h            (178 lines) Flash_fwd_params struct
├── static_switch.h      (83 lines) Compile-time branching macro
├── launch.h            (118 lines) Template dispatch + kernel launch
├── kernel_traits.h     (202 lines) Tile shapes, MMA atoms, smem layouts
├── kernel_ws.h         (202 lines) Warp specialization (producer/consumer)
├── tile_scheduler.h    (304 lines) Persistent tile scheduling
├── named_barrier.h     (118 lines) Producer↔consumer sync barriers
├── mainloop_tma_ws.h   (908 lines) THE CORE: TMA loads + GEMM-I/II + softmax
├── softmax_fused.h     (173 lines) Online softmax fused with FP4 quantization
├── epilogue_tma_ws.h   (222 lines) Output store (registers → smem → gmem)
├── cute_extension.h    (326 lines) Custom SM120 NVFP4 MMA atom + SF helpers
├── utils.h             (408 lines) PTX intrinsics, reduction ops, layout transforms
├── blockscaled_layout.h(149 lines) Scale factor memory layouts
└── block_info.h         (60 lines) Block coordinate helpers
```

## Key Concepts Quick Reference

| Concept | CuTe Term | SA3 Example |
|---------|-----------|-------------|
| Data layout | `Layout<Shape, Stride>` | `SmemLayoutQ`, `LayoutP`, `LayoutSFP` |
| Memory view | `Tensor(ptr, layout)` | `sQ = make_tensor(smem_ptr, SmemLayoutQ{})` |
| Block extraction | `local_tile(tensor, shape, coord)` | `gQ = local_tile(mQ, (128,128), (m_block, 0))` |
| Thread distribution | `partition_fragment_A/B/C` | `tSrQ = thread_mma.partition_fragment_A(sQ)` |
| HW DMA | `SM90_TMA_LOAD` | `copy(tma_load_Q.with(barrier), src, dst)` |
| HW matmul | `SM120_16x32x64_TN_VS_NVFP4` | `gemm(mma, zip(Q,SFQ), zip(K,SFK), S)` |
| Async pipeline | `PipelineTmaAsync<3>` | `pipeline_k.producer_acquire(stage)` |
| Warp specialization | `warpgroup_reg_alloc<232>()` | Producer 24 regs, Consumer 232 regs |

## Related Materials

- Paper summaries: `papers/sageattention/paper_summaries.md`
- SA3 pseudocode: `tasks/sage3_impl_pseudocode/sageattention3_pseudocode_v3.py`
- SA3 flow diagrams: `tasks/sage3_flow_vis/sage3-flash-flow-v1-annotated.excalidraw`
- CuTeDSL examples: `sageattention3_blackwell/csrc/cutlass/examples/python/CuTeDSL/`
- SA3 paper: `papers/sageattention/SageAttention3-2505.11594.pdf`
