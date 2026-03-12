# SageAttention3 Blackwell — Install Guide

## Prerequisites

- Python >= 3.12
- PyTorch >= 2.8.0 (with CUDA >= 12.8)
- A Blackwell GPU (sm_100, sm_120, or sm_121)
- `ninja` build system

## Steps

```bash
# 1. Clone the repo
git clone https://github.com/thu-ml/SageAttention
cd SageAttention/sageattention3_blackwell

# 2. Initialize the CUTLASS submodule (required for headers)
#    If csrc/cutlass/ is empty, clone it manually:
rm -rf csrc/cutlass
git clone --depth 1 https://github.com/NVIDIA/cutlass.git csrc/cutlass

# 3. Install in editable mode (using your target venv python)
pip install -e . --no-build-isolation

# 4. Verify
python -c "from sageattn3 import sageattn3_blackwell; print('OK')"
```

## Troubleshooting

- **`fatal error: cutlass/numeric_types.h: No such file or directory`** — The `csrc/cutlass/` directory exists but is empty. Remove it and re-clone CUTLASS (step 2).
- Compilation takes ~3 minutes (two CUDA extensions: `fp4attn_cuda` and `fp4quant_cuda`).
- Only `sm_100a`, `sm_120a`, and `sm_121a` GPU architectures are supported.
