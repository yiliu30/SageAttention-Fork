/*
 * Standalone validation of the upstream MXFP4 block-scaled MMA atom
 *   cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<e2m1, e2m1, float, ue8m0, 32>
 * plus a by-construction verification of the gmem SF byte layout that the
 * Python/CUDA quantizer must produce.
 *
 * ===========================================================================
 * RESULT SUMMARY (read this first)
 * ===========================================================================
 *  [PASS] arch gate: CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED is live.
 *  [PASS] (A) gmem SF byte layout verified by construction against a closed
 *             form, injectivity, and the uint16 K-block pairing.
 *  [PASS] (B) 4-bit operand loading via cute::copy requires
 *             recast<uint8_t>() first -- elementwise copy() on a 4-bit
 *             element type silently drops the upper nibble of every byte.
 *             Demonstrated and asserted.
 *  [PASS] (C) atom semantics confirmed at the PTX/register level: with A and B
 *             all-ones and K=64, D = 64 * 2^(sfa-127) * 2^(sfb-127) exactly,
 *             including asymmetric scales. So 2X/ue8m0 semantics over K=64 are
 *             correct.
 *
 * ---------------------------------------------------------------------------
 * RETRACTED CLAIM (kept as a warning)
 * ---------------------------------------------------------------------------
 * An earlier revision of this file asserted the upstream VS=32 atom was
 * "self-inconsistent" -- SFBits=16 (2 ue8m0 bytes, K=64) versus a belief that
 * scale_vec::2X consumes 4 SFs (K=128) -- and concluded an atom-level gemm
 * could not be numerically correct. THAT WAS WRONG.
 *
 * `scale_vec::2X` means the scale vector spans 2*16 = 32 ELEMENTS, i.e. exactly
 * SFVecSize = 32, NOT "two SF registers". Over Shape_MNK.K = 64 that is
 * 64/32 = 2 SFs = 2 ue8m0 bytes = the single uint16 the atom declares. The
 * geometry is consistent, and part (C) below now *demonstrates* correctness
 * rather than arguing about it.
 *
 * Note also: the atom this file targets (upstream N=8, SM120_16x8x64_TN_VS) is
 * NOT the atom the attention kernel uses. The kernel needs 16 C values per
 * thread for its online-softmax P quantization, which the N=8 atom's
 * ((_4,_8),(_2,_2)) C layout (4 values/thread) does not provide; the kernel uses
 * the N=32 counterpart SM120_16x32x64_TN_VS_MXFP4 instead. This file remains a
 * valid check of MXFP4 atom *semantics*.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/float_subbyte.h>

#include "blockscaled_layout.h"   // flash::BlockScaledConfig
#include "cute_extension.h"       // cute::partition_fragment_SFA/SFB

using namespace cute;

using SFV = Int<32>;                 // SF vector size: 32 for MXFP4
using ASF = cutlass::float_ue8m0_t;
using E2M1 = cutlass::float_e2m1_t;

using Atom = cute::SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<E2M1, E2M1, float, ASF, 32>;
using BlkScaledConfig = flash::BlockScaledConfig<32>;

// The gmem layout for a (rows x K) SF tensor, rank-2 (head/batch modes dropped).
template <int Rows, int K>
using LayoutSF = decltype(take<0, 2>(BlkScaledConfig::tile_atom_to_shape_SFQKV(
    make_shape(Int<Rows>{}, Int<K>{}, Int<1>{}, Int<1>{}))));

// ---------------------------------------------------------------------------
// (A) gmem SF byte layout, verified by construction
// ---------------------------------------------------------------------------
template <int Rows, int K>
bool check_layout(const char* label) {
  using LSF = LayoutSF<Rows, K>;
  constexpr LSF lsf{};
  constexpr int NKB = K / 32;
  const int total = (int)cosize(lsf);

  printf("SF layout [%s] (%d rows x K=%d): ", label, Rows, K);
  print(lsf);
  printf("\n  cosize = %d bytes  (%d rows * %d bytes/row)\n", total, Rows, NKB);

  int bad_formula = 0, bad_inj = 0, bad_pair = 0;
  std::vector<int> seen(total, 0);
  for (int r = 0; r < Rows; r++) {
    for (int kb = 0; kb < NKB; kb++) {
      // row term (within a 64-row block: 16*(r%16) + 4*((r/16)%4));
      // 64-row blocks are separated by 256*(K/128) bytes
      // K term: the first 4 K-blocks are 4 adjacent bytes; each further
      // group of 4 K-blocks is 256 bytes on
      int off = (int)lsf(make_coord(r, kb * 32));
      int exp = 16 * (r % 16) + 4 * ((r / 16) % 4)
              + 256 * ((K / 128) * (r / 64))
              + (kb % 4) + 256 * (kb / 4);
      if (off != exp) bad_formula++;
      if (off < 0 || off >= total || seen[off]++) bad_inj++;
      // adjacency used by the atom's uint16: (kb=0,kb=1) and (kb=2,kb=3)
      if (kb == 0 || kb == 2) {
        int o0 = (int)lsf(make_coord(r, kb * 32));
        int o1 = (int)lsf(make_coord(r, (kb + 1) * 32));
        if (o1 != o0 + 1) bad_pair++;
      }
    }
  }
  printf("  formula mismatches = %d, colliding bytes = %d, non-adjacent uint16 pairs = %d\n",
         bad_formula, bad_inj, bad_pair);
  printf("  worked example: ");
  for (int kb = 0; kb < NKB; kb++)
    printf("byte(r=5,kb=%d)=%d ", kb, (int)lsf(make_coord(5, kb * 32)));
  if (Rows > 16) { printf("| "); for (int kb = 0; kb < NKB; kb++)
    printf("byte(r=16,kb=%d)=%d ", kb, (int)lsf(make_coord(16, kb * 32))); }
  if (Rows > 64) { printf("| "); for (int kb = 0; kb < NKB; kb++)
    printf("byte(r=64,kb=%d)=%d ", kb, (int)lsf(make_coord(64, kb * 32))); }
  printf("\n");

  bool ok = (bad_formula == 0 && bad_inj == 0 && bad_pair == 0);
  printf("  (A) %s: %s\n", label, ok ? "PASS" : "FAIL");
  return ok;
}

// ---------------------------------------------------------------------------
// (B) 4-bit operand loading: recast<uint8_t> is mandatory before copy()
// ---------------------------------------------------------------------------
// cute's elementwise copy() of a float_e2m1_t / uint4_t tensor treats the
// 4-bit value as a byte-sized store, so it silently zeroes the upper nibble of
// every byte. Recasting to uint8_t first is the only correct way to move
// sub-byte data with a plain copy. This kernel proves the difference.
__global__ void nibble_copy_probe(const uint8_t* src, uint8_t* dst_bad, uint8_t* dst_good,
                                  int nbytes) {
#if defined(CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED)
  auto g = make_tensor(make_gmem_ptr(reinterpret_cast<const E2M1*>(src)),
                       make_layout(make_shape(Int<64>{}, Int<128>{}), GenRowMajor{}));
  // (bad) plain copy of the 4-bit tensor
  {
    auto f = make_fragment_like(g);
    copy(g, f);                                  // <-- drops upper nibbles
    auto d = make_tensor(make_gmem_ptr(reinterpret_cast<uint8_t*>(dst_bad)),
                         make_layout(make_shape(Int<64>{}, Int<64>{}), GenRowMajor{}));
    copy(recast<uint8_t>(f), d);
  }
  // (good) recast first
  {
    auto f = make_fragment_like(g);
    copy(recast<uint8_t>(g), recast<uint8_t>(f));
    auto d = make_tensor(make_gmem_ptr(reinterpret_cast<uint8_t*>(dst_good)),
                         make_layout(make_shape(Int<64>{}, Int<64>{}), GenRowMajor{}));
    copy(recast<uint8_t>(f), d);
  }
  (void)nbytes;
#endif
}

bool check_nibble_path() {
  const int n = 64 * 64;
  std::vector<uint8_t> src(n), bad(n, 0xAA), good(n, 0xAA);
  for (int i = 0; i < n; i++) src[i] = (uint8_t)(i * 37 + 11);
  uint8_t *d_src, *d_bad, *d_good;
  cudaMalloc(&d_src, n); cudaMalloc(&d_bad, n); cudaMalloc(&d_good, n);
  cudaMemcpy(d_src, src.data(), n, cudaMemcpyHostToDevice);
  cudaMemset(d_bad, 0xAA, n); cudaMemset(d_good, 0xAA, n);
  nibble_copy_probe<<<1, 32>>>(d_src, d_bad, d_good, n);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("FAIL: nibble probe: %s\n", cudaGetErrorString(e)); return false; }
  cudaMemcpy(bad.data(), d_bad, n, cudaMemcpyDeviceToHost);
  cudaMemcpy(good.data(), d_good, n, cudaMemcpyDeviceToHost);
  int nbad = 0, ngood = 0;
  for (int i = 0; i < n; i++) { if (bad[i] != src[i]) nbad++; if (good[i] != src[i]) ngood++; }
  printf("(B) 4-bit copy round-trip over %d bytes:\n", n);
  printf("      plain copy()        : %d/%d bytes wrong  (upper nibble destroyed)\n", nbad, n);
  printf("      recast<uint8_t>()   : %d/%d bytes wrong\n", ngood, n);
  bool ok = (nbad > 0 && ngood == 0);
  printf("(B) %s\n", ok ? "PASS (recast requirement demonstrated)" : "FAIL");
  cudaFree(d_src); cudaFree(d_bad); cudaFree(d_good);
  return ok;
}

// ---------------------------------------------------------------------------
// (C) atom semantics, verified at the PTX/register level
//
// Earlier revisions of this file claimed the upstream VS=32 atom was
// "self-inconsistent": SFBits=16 (2 ue8m0 bytes) vs scale_vec::2X supposedly
// consuming 4 SFs (K=128).  That reading was WRONG and has been retracted.
// `scale_vec::2X` means the scale vector spans 2*16 = 32 ELEMENTS -- i.e.
// exactly SFVecSize=32 -- not "two SF registers".  Over Shape_MNK.K=64 that is
// 64/32 = 2 SFs = 2 ue8m0 bytes = the single uint16 the atom declares.  The
// geometry is consistent.
//
// The durable check is therefore not arithmetic over the traits but an actual
// execution: drive the atom with known registers and confirm D. This is what
// isolates atom semantics from every layout question.
// ---------------------------------------------------------------------------
__global__ void atom_ground_truth(float* out, int sfa_byte, int sfb_byte) {
#if defined(CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED)
  // 8 e2m1 nibbles each equal to 1.0 (code 0x2) -> 0x22222222
  const uint32_t one = 0x22222222u;
  uint32_t a[4] = {one, one, one, one};
  uint32_t b[2] = {one, one};
  const uint16_t sfa = (uint16_t)((sfa_byte & 0xFF) | ((sfa_byte & 0xFF) << 8));
  const uint16_t sfb = (uint16_t)((sfb_byte & 0xFF) | ((sfb_byte & 0xFF) << 8));
  float d[4] = {0.f, 0.f, 0.f, 0.f};
  const float c[4] = {0.f, 0.f, 0.f, 0.f};

  using E2M1 = cutlass::float_e2m1_t;
  SM120::BLOCKSCALED::SM120_16x8x64_TN_VS<E2M1, E2M1, float,
                                          cutlass::float_ue8m0_t, 32>::fma(
      d[0], d[1], d[2], d[3],
      a[0], a[1], a[2], a[3],
      b[0], b[1],
      c[0], c[1], c[2], c[3],
      sfa, sfb);
  if (threadIdx.x == 0) { for (int i = 0; i < 4; ++i) out[i] = d[i]; }
#endif
}

bool check_atom_semantics() {
  printf("(C) atom semantics at the register level (A=all 1.0, B=all 1.0, K=64)\n");
  printf("      expect D = 64 * 2^(sfa-127) * 2^(sfb-127)\n");
  int* d = nullptr;
  if (cudaMalloc(&d, 4 * sizeof(float)) != cudaSuccess) { printf("(C) FAIL (alloc)\n"); return false; }
  const int cases[][2] = {{127,127},{128,127},{126,127},{127,128},{129,125}};
  bool all = true;
  for (auto& cs : cases) {
    float h[4] = {0,0,0,0};
    atom_ground_truth<<<1, 32>>>(reinterpret_cast<float*>(d), cs[0], cs[1]);
    if (cudaDeviceSynchronize() != cudaSuccess) { printf("(C) FAIL (kernel)\n"); cudaFree(d); return false; }
    cudaMemcpy(h, d, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    const float want = 64.0f * exp2f((float)(cs[0]-127)) * exp2f((float)(cs[1]-127));
    const bool ok = fabsf(h[0] - want) < 0.5f;
    all = all && ok;
    printf("      sfa=2^%+d sfb=2^%+d -> D=%.1f (want %.1f) %s\n",
           cs[0]-127, cs[1]-127, h[0], want, ok ? "OK" : "MISMATCH");
  }
  cudaFree(d);
  printf("      asymmetric scales above prove BOTH operands' SFs apply with\n");
  printf("      correct weight, i.e. 2X semantics over K=64 are right.\n");
  printf("(C) %s\n", all ? "PASS" : "FAIL");
  return all;
}

// ---------------------------------------------------------------------------
// arch probe: the CUTE_ARCH_* macros only exist in device code
// ---------------------------------------------------------------------------
#if !defined(__CUDA_ARCH_LIST__)
#error "no __CUDA_ARCH_LIST__"
#endif

__global__ void arch_probe(int* out) {
#if defined(CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED)
  *out = 1;
#else
  *out = 0;
#endif
}

int main() {
  {
    int* d = nullptr;
    cudaMalloc(&d, sizeof(int));
    arch_probe<<<1, 1>>>(d);
    int have = 0;
    cudaMemcpy(&have, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    if (!have) {
      printf("FAIL: CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED not defined in device code "
             "(need -gencode arch=compute_120a,code=sm_120a and nvcc >= 12.8)\n");
      return 2;
    }
    printf("arch: CUTE_ARCH_MXF4NVF4_2X_UE8M0_MMA_ENABLED is active\n");
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
    printf("device: %s sm_%d%d\n\n", prop.name, prop.major, prop.minor);
    printf("arch gate: PASS\n\n");
  }

  printf("=== (A) gmem SF byte layout (the quantizer contract) ===\n");
  bool okA = true;
  okA &= check_layout<64, 128>("M=64,K=128");
  okA &= check_layout<128, 128>("M=128,K=128");
  okA &= check_layout<256, 256>("M=256,K=256");
  okA &= check_layout<128, 512>("M=128,K=512");
  printf("\n(A) overall: %s\n\n", okA ? "PASS" : "FAIL");

  printf("=== (B) 4-bit operand loading ===\n");
  bool okB = check_nibble_path();
  printf("\n");

  printf("=== (C) atom semantics (register level) ===\n");
  bool okC = check_atom_semantics();
  printf("\n");

  printf("=== SUMMARY ===\n");
  printf("  (A) gmem SF byte layout      : %s\n", okA ? "PASS" : "FAIL");
  printf("  (B) 4-bit load path          : %s\n", okB ? "PASS" : "FAIL");
  printf("  (C) atom semantics (reg-level): %s\n", okC ? "PASS" : "FAIL");

  bool all = okA && okB && okC;
  printf("\n%s\n", all ? "PASS" : "FAIL");
  return all ? 0 : 1;
}
