#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
CUTLASS_DIR="${REPO_DIR}/csrc/cutlass"
EXPECTED_CUTLASS_COMMIT="dc45f979ae336a235da1676b311f35efeb30149a"
SOURCE="${SCRIPT_DIR}/sm120_fp8_mma_accum_bench.cu"
BINARY="${SCRIPT_DIR}/sm120_fp8_mma_accum_bench"

if [[ ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
  echo "nvcc not found at ${CUDA_HOME}/bin/nvcc" >&2
  exit 1
fi
if [[ ! -d "${CUTLASS_DIR}/include" ]]; then
  echo "CUTLASS checkout not found at ${CUTLASS_DIR}" >&2
  exit 1
fi
if [[ "$(git -C "${CUTLASS_DIR}" rev-parse HEAD)" != "${EXPECTED_CUTLASS_COMMIT}" ]]; then
  echo "CUTLASS must be checked out at ${EXPECTED_CUTLASS_COMMIT}" >&2
  exit 1
fi

"${CUDA_HOME}/bin/nvcc" \
  -std=c++17 \
  -O3 \
  -lineinfo \
  -gencode=arch=compute_120a,code=sm_120a \
  -Xptxas=-v \
  -I"${CUTLASS_DIR}/include" \
  -I"${CUTLASS_DIR}/tools/util/include" \
  "${SOURCE}" \
  -o "${BINARY}"

if [[ "${1:-}" == "--build-only" ]]; then
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec "${BINARY}" "$@"
