 # SageAttention3 Triton Reference Implementation

## ✅ Completed Implementation

A comprehensive, educational Triton reference implementation of SageAttention3 that demonstrates all key algorithmic innovations in readable, well-documented code.

## 🚀 Features Implemented

### Core Components
- **`sageattention3_triton_ref.py`** - Main implementation with complete SageAttention3 pipeline
- **`test_sageattention3.py`** - Comprehensive test suite validating all features

### Key Algorithmic Innovations
✅ **NVFP4 Microscaling Quantization**
- E2M1 format with representable values {0, 0.5, 1, 2, 3, 4, 6}
- Block size 1×16 with FP8 E4M3 scale factors
- Full quantization/dequantization pipeline

✅ **Two-Level P Matrix Scaling**
- Level 1: Per-token FP32 scaling to [0, 448×6] range
- Level 2: NVFP4 microscaling within blocks
- Fused with online softmax for memory efficiency

✅ **FP4MM Instruction Simulation**
- Hardware FP4×FP4 matrix multiplication simulation
- Direct scale factor integration
- FP32 accumulation matching hardware behavior

✅ **K Column Permutation**
- Architecture-specific column reordering
- Matches FP4MM accumulator layout
- Eliminates costly thread shuffle operations

✅ **V Transposition**
- seq_len ↔ head_dim dimension swap
- Optimizes PV matrix multiplication
- Pre-quantization transposition for efficiency

✅ **Q+K Smoothing**
- Per-channel smoothing factors
- Per-block Q mean subtraction (128-token blocks)
- Extended SageAttention2 approach

✅ **Fused Online Softmax with Quantization**
- Streaming softmax computation
- Integrated P matrix quantization
- Memory-efficient persistent kernel simulation

## 🔧 Usage

### Basic Usage
```python
from sageattention3_triton_ref import SageAttention3TritonReference

# Initialize
sage3 = SageAttention3TritonReference()

# Create input tensors (HND layout)
q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
k = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')
v = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.float16, device='cuda')

# Run SageAttention3
output = sage3.sageattn3_triton_ref(
    q, k, v,
    tensor_layout="HND",
    is_causal=True,
    smooth_k=True,
    smooth_q=True,
    per_block_q_mean_sub=True
)
```

### Supported Configurations
- **Layouts**: HND `[B, H, L, D]` and NHD `[B, L, H, D]`
- **Attention Types**: Causal and non-causal
- **Head Dimensions**: 64, 128 (divisible by 16 for microscaling)
- **Sequence Lengths**: Any length (automatic padding)
- **Batch Sizes**: Any batch size

## 🧪 Testing

### Run Example
```bash
python sageattention3_triton_ref.py
```

### Run Comprehensive Tests
```bash
python test_sageattention3.py
```

### Test Coverage
- FP4 quantization and dequantization
- K column permutation patterns
- V transposition mechanics
- Preprocessing pipeline with smoothing
- Complete attention computation
- All algorithmic innovations
- Multiple tensor layouts and configurations

## 📚 Educational Value

### Architecture Simulation
The implementation simulates the warp-specialized persistent kernel architecture:
```
WarpGroup 0 (Producer): TMA loads Q, K, V + scale factors
WarpGroup 1-2 (Consumer): FP4MM QK^T → Online softmax → FP4MM PV
Communication: Async TMA pipeline (3-stage K/V)
```

### Algorithm Flow
1. **Preprocessing**: Smoothing + per-block Q mean subtraction + padding
2. **Quantization**: Q/K/V → FP4 with specialized patterns (permute K, transpose V)
3. **Attention Core**:
   - QK^T via FP4MM simulation
   - Online softmax with two-level P quantization
   - PV via FP4MM with quantized P
4. **Postprocessing**: Unpad and restore original tensor layout

### Key Design Principles
- **Educational Focus**: Prioritizes readability over performance
- **Algorithmic Accuracy**: Maintains mathematical correctness
- **Hardware Simulation**: Models actual FP4MM behavior
- **Comprehensive Documentation**: Explains each innovation clearly

## 📖 References

### Source Materials
- **Paper**: SageAttention3 (arXiv 2505.11594)
- **Codebase**: `/mnt/disk1/yiliu7/SageAttention-Fork`
- **Pseudocode**: `tasks/sage3_impl_pseudocode/sageattention3_pseudocode_v3.py`

### Related Implementations
- SageAttention (v1): INT8 QK^T quantization
- SageAttention2: FP8 PV + two-level accumulation
- SageAttention2++: Enhanced with fp32+fp16 strategy
- SageAttention3: FP4 microscaling for Blackwell GPUs

## 🎯 Implementation Quality

### ✅ Requirements Met
- **Data Flow Consistency**: Matches real CUDA kernel architecture
- **Core Function Alignment**: Compatible API signatures with pseudocode
- **Required Kernels**: All specified Triton kernels implemented
  - `blockscaled_fp4_attn` - Main attention kernel
  - `mma` - FP4 matrix multiplication simulation
  - `quantize` - FP4 quantization functions
  - `online_softmax_with_quant` - Fused softmax with quantization
- **Performance**: Educational focus (correctness over speed)

### 🧪 Validation
- All unit tests passing
- Multiple configuration support
- Numerical stability verified
- Memory layout correctness confirmed
- End-to-end pipeline validated

## 🔮 Future Extensions

This reference implementation provides a foundation for:
- Performance optimization studies
- Hardware-specific kernel development
- Algorithm modification experiments
- Integration with other attention mechanisms
- Educational material for FP4 quantization techniques

---

**Status**: ✅ **COMPLETE** - Fully functional educational reference ready for use!