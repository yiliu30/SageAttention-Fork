#!/bin/sh
set -eu

repo_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
env_dir="/home/yiliu7/workspace/venvs/sageattention3-nvfp4"
ncu="/home/yiliu7/workspace/tools/nsight-compute-2026.2.1/opt/nvidia/nsight-compute/2026.2.1/ncu"
report_dir="${HOME}/.local/state/sageattention3"
kernel_variant="${1:-baseline}"
case "${kernel_variant}" in
  baseline|two_cta|p_pack_bypass|fp8_pv) ;;
  *)
    echo "usage: $0 [baseline|two_cta|p_pack_bypass|fp8_pv]" >&2
    exit 2
    ;;
esac
report_base="${report_dir}/sageattn3_pure_16k_${kernel_variant}"

mkdir -p "${report_dir}"

/usr/bin/env CUDA_VISIBLE_DEVICES=0 SAGEATTN3_KERNEL_VARIANT="${kernel_variant}" "${ncu}" \
  --set full \
  --devices 0 \
  --target-processes application-only \
  --profile-from-start off \
  --force-overwrite \
  --export "${report_base}" \
  "${env_dir}/bin/python" \
  "${repo_dir}/benchmarks/profile_sageattn3_pure_ncu.py"
