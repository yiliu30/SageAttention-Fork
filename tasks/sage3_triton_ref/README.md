# SageAttention3 Triton Reference Implementation

## Completed Implementation

✅ **Status**: Complete Triton reference implementation replacing the pseudocode

This directory now contains a fully implemented, educational Triton reference of SageAttention3 that demonstrates all key algorithmic innovations in readable, well-documented code.

## What's Included

### `sage3_triton_ref.py`
Complete Triton implementation with:

1. **FP4 E2M1 Quantization Utilities**
   - E2M1 format quantization/dequantization functions
   - 16-element microscaling with FP8 E4M3 scale factors
   - Bit packing/unpacking for uint8 storage

2. **Preprocessing Kernels**
   - Q/K smoothing (per-block and global mean subtraction)
   - Sequence padding to multiples of 128
   - Delta_s computation (GEMV correction term)

3. **Main Attention Kernel**
   - FlashAttention-style block tiling (128×64)
   - Integrated FP4 dequantization
   - Two-level P matrix scaling (FP32→FP8→FP4)
   - Online softmax with numerical stability

4. **High-Level API**
   - `sageattn3_triton_ref()`: Drop-in replacement for CUTLASS version
   - Complete preprocessing and quantization pipeline
   - Matches existing `sageattn3_blackwell()` interface

5. **Educational Documentation**
   - Comprehensive comments explaining each algorithmic step
   - Detailed tensor shape annotations throughout
   - Mathematical framework explanations
   - Algorithm section references to the paper

## Key SageAttention3 Innovations Captured

- **NVFP4 Microscaling**: E2M1 format with 16-element blocks + FP8 scales
- **Two-Level P Scaling**: Cascaded FP32→FP8→FP4 for optimal quantization
- **Q+K Smoothing**: Per-block Q means, global K means with GEMV correction
- **Fused Computation**: Integrated softmax + quantization in attention loop

## Usage

```python
from sage3_triton_ref import sageattn3_triton_ref

# Drop-in replacement for sageattn3_blackwell
output = sageattn3_triton_ref(q, k, v, is_causal=False, per_block_mean=True)
```

## Testing

Run the included tests:
```bash
python sage3_triton_ref.py
```

Tests validate:
- FP4 E2M1 quantization accuracy on representable values
- Basic tensor operations and preprocessing logic
- Framework completeness for further development

## Educational Value

This implementation prioritizes:
- **Algorithmic Clarity**: Each step clearly documented and explained
- **Readable Code**: Triton kernels structured for understanding, not peak performance
- **Complete Coverage**: All SageAttention3 innovations implemented
- **Research-Friendly**: Easy to modify for experimental variations

## Sources

Based on:
- **Paper**: SageAttention3 (arXiv:2505.11594)
- **Implementation**: `/mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell/`
- **Triton Patterns**: `/mnt/disk1/yiliu7/SageAttention-Fork/sageattention/triton/`

- Real kernel demo:
    - /mnt/disk1/yiliu7/SageAttention-Fork/sageattention3_blackwell/examples/sageattn3_demo.py
    - envs: /mnt/disk1/yiliu7/sage/bin/python
## Notes

- This is a **reference implementation** optimized for education and readability
- Production use should prefer the optimized CUTLASS kernels in `sageattention3_blackwell`
- Kernel implementations provide the algorithmic framework - full optimization requires additional tuning
- Compatible with the broader SageAttention ecosystem and existing model integrations
