/*
 * Copyright (c) 2025 by SageAttention team.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cmath>
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "utils.h"

namespace flash {

using namespace cute;

template <int Rows, bool IsMXFP4 = false>
struct SoftmaxFused{

    using TensorT = decltype(make_fragment_like<float>(Shape<Int<Rows>>{}));
    TensorT row_sum, row_max, scores_scale;
    // Two constants fold the P-quantization scales into the exp2 above.
    //
    // The invariant is that the operand the PV MMA receives, `value = acc/AbsMaxP`,
    // must span [0, 6] to use the full e2m1 range (6 is the e2m1 max).
    //
    //   NVFP4 (SFVecSize 16, E4M3 SF): the SF is prob_max/(448*6) -- 448 is the
    //     E4M3 max and 6 the e2m1 max -- so acc is pre-scaled by 1/(448*6) and
    //     AbsMaxP by a further 1/6. value = 6*prob/prob_max in [0,6].
    //
    //   MXFP4 (SFVecSize 32, E8M0 SF): E8M0 carries no mantissa, so the 448 has no
    //     analogue and is dropped (fp8_scalexfp4_scale_log2 = 0). The 1/6 is still
    //     required -- e2m1 saturates at 6 regardless of SF format.
    //
    // Note: zeroing BOTH constants would still be numerically correct, but
    // value = prob/prob_max would span only [0,1], wasting ~2.5 bits of e2m1 and
    // visibly hurting quality. Keep the 1/6.
    static constexpr float fp8_scalexfp4_scale =
        IsMXFP4 ? 1.f : (1.f / (448 * 6));
    static constexpr float fp8_scalexfp4_scale_log2 =
        IsMXFP4 ? 0.f : -11.392317422778762f; // log2f(fp8_scalexfp4_scale)
    static constexpr float fp4_scale_log2 = -2.584962500721156f; // log2f(1/6)
    static constexpr int RowReductionThr = 4;

    // MXFP4 only: the E8M0 scale factor cannot represent `AbsMaxP` exactly (no
    // mantissa), so `quantize` emits ceil_pow2(AbsMaxP) into the SF register.
    // The PV gemm computes sum(P_q) * SF, so the P values MUST be divided by
    // that *same* rounded-up scale -- dividing by the raw AbsMaxP while the MMA
    // multiplies by the rounded-up SF leaves a scale mismatch of
    // AbsMaxP/ceil_pow2(AbsMaxP), which for prob_max near 1 is up to 2x. That
    // inflated every row of O by that factor (observed: O mean 1.75 instead of
    // 1.0 for v=ones, and cosine 0.18 on the bench).
    //
    // NVFP4 is immune because its E4M3 SF holds AbsMaxP/(448*6) with a
    // mantissa, so the divide and the multiply agree by construction.
    //
    // Rounding AbsMaxP up here (instead of at SF-emit time) makes divide ==
    // multiply exactly. It is exact and free: no exp2, no cvt, just an exponent
    // ceil, and it keeps e2m1's value range at [0,6] since AbsMaxP <= 1/6.
    CUTLASS_DEVICE static float ceil_pow2(float x) {
        if constexpr (IsMXFP4) {
            return __uint_as_float((__float_as_uint(x) + 0x007FFFFFu) & 0xFF800000u);
        } else {
            return x;
        }
    }

    CUTLASS_DEVICE SoftmaxFused(){};

    template<bool FirstTile, bool InfCheck = false, typename TensorAcc, typename TensorMax>
    CUTLASS_DEVICE auto online_softmax_with_quant(
        TensorAcc& acc, 
        TensorMax& AbsMaxP,
        const float softmax_scale_log2
    ) {
        Tensor acc_reduction_view = make_tensor(acc.data(), flash::convert_to_reduction_layout(acc.layout()));
        Tensor acc_conversion_view = make_tensor(acc.data(), flash::convert_to_conversion_layout(acc.layout()));
        Tensor acc_conversion_flatten = group_modes<1, 5>(group_modes<0, 2>(flatten(acc_conversion_view)));
        
        if constexpr (FirstTile) {
            fill(row_max, -INFINITY);
            clear(row_sum);
            fill(scores_scale, 1.f);

            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 1); // exchange max with neighbour thread of 8 elements
                    AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), max_recv);
                    // One 32-column K group spans all 4 lanes of the warp quad
                    // (2 lanes per 8-column tile-pair); xor(1) alone merged only
                    // half the group's columns, leaving the group max up to 2x
                    // too small. The second xor completes the quad butterfly
                    // {l, l^1, l^2, l^3} so every lane holds the full-group max.
                    AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 2));
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }
                
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2); // exchange max in a quad in a row
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                                        : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = ceil_pow2(flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2));
                }
            }
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
            }
        }
        else {
            Tensor scores_max_prev = make_fragment_like(row_max);
            cute::copy(row_max, scores_max_prev);
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < size<0>(acc_reduction_view); mi++) {
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1, 1>(acc_reduction_view); ni++) {
                    float local_max = -INFINITY;
                    CUTLASS_PRAGMA_UNROLL
                    for (int ei = 0; ei < size<1, 0>(acc_reduction_view); ei++) {
                        local_max = fmaxf(local_max, acc_reduction_view(mi, make_coord(ei, ni)));
                    }
                    float max_recv = __shfl_xor_sync(int32_t(-1), local_max, 1); // exchange max with neighbour thread of 8 elements
                    AbsMaxP(mi, ni) = fmaxf(local_max, max_recv);
                    // Complete the quad butterfly (see first-tile branch).
                    AbsMaxP(mi, ni) = fmaxf(AbsMaxP(mi, ni), __shfl_xor_sync(int32_t(-1), AbsMaxP(mi, ni), 2));
                    row_max(mi) = fmaxf(row_max(mi), AbsMaxP(mi, ni));
                }
                
                float max_recv = __shfl_xor_sync(int32_t(-1), row_max(mi), 2); // exchange max in a quad in a row
                row_max(mi) = fmaxf(row_max(mi), max_recv);

                float scores_max_cur = !InfCheck
                                        ? row_max(mi)
                                        : (row_max(mi) == -INFINITY ? 0.0f : row_max(mi));
                scores_scale(mi) = flash::ptx_exp2((scores_max_prev(mi) - scores_max_cur) * softmax_scale_log2);

                const float max_scaled = InfCheck
                                        ? (row_max(mi) == -INFINITY ? 0.f : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2))
                                        : (row_max(mi) * softmax_scale_log2 + fp8_scalexfp4_scale_log2);
                row_sum(mi) = row_sum(mi) * scores_scale(mi);
                CUTLASS_PRAGMA_UNROLL
                for (int ni = 0; ni < size<1>(acc_reduction_view); ni++) {
                    acc_reduction_view(mi, ni) = flash::ptx_exp2(acc_reduction_view(mi, ni) * softmax_scale_log2 - max_scaled);
                    row_sum(mi) += acc_reduction_view(mi, ni);
                }
                CUTLASS_PRAGMA_UNROLL
                for (int sfi = 0; sfi < size<1>(AbsMaxP); sfi++) {
                    AbsMaxP(mi, sfi) = ceil_pow2(flash::ptx_exp2(AbsMaxP(mi, sfi) * softmax_scale_log2 - max_scaled + fp4_scale_log2));
                }
                // scores_scale(mi) = max_scaled;
            }
        }
        // MXFP4: AbsMaxP is (mi,(n0,_1)) with mi stride 1 (probe-verified), so
        // the flat slot pairs (even i, i+1) are the TWO M ROWS of one
        // 32-column K group.  The PV atom's A-operand SF register supplies a
        // single uint16 -- one E8M0 byte per 32-column half -- for the pair of
        // rows the lane owns, so both rows must share one per-K-group scale.
        // Folding the pair maxima here, BEFORE the divide, keeps the divisor
        // identical to the byte `quantize` emits, so the softmax's divide and
        // the MMA's multiply agree exactly instead of differing by up to 2x.
        if constexpr (IsMXFP4) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i + 1 < size(AbsMaxP); i += 2) {
                float m = fmaxf(AbsMaxP(i), AbsMaxP(i + 1));
                AbsMaxP(i) = m;
                AbsMaxP(i + 1) = m;
            }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(AbsMaxP); ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < size<0>(acc_conversion_flatten); ++j)
                acc_conversion_flatten(j, i) /= AbsMaxP(i);
        }
    }

    template<typename TensorAcc>
    CUTLASS_DEVICE void finalize(TensorAcc& o_store) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 1; i < RowReductionThr; i <<= 1) {
                float sum_recv = __shfl_xor_sync(int32_t(-1), row_sum(mi), i);
                row_sum(mi) += sum_recv;
            }
            float sum = row_sum(mi);
            float inv_sum = (sum == 0.f || sum != sum) ? 0.f : 1 / sum;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) { 
                o_store_reduction_view(mi, ni) *= inv_sum;
             }
        }
    }

    template<typename TensorAcc>
    CUTLASS_DEVICE void rescale_o(TensorAcc& o_store, TensorAcc const& o_tmp) {
        Tensor o_store_reduction_view = make_tensor(o_store.data(), flash::convert_to_reduction_layout(o_store.layout()));
        Tensor o_tmp_reduction_view = make_tensor(o_tmp.data(), flash::convert_to_reduction_layout(o_tmp.layout()));
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < size(row_max); ++mi) {
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < size<1>(o_store_reduction_view); ++ni) { 
                o_store_reduction_view(mi, ni) = o_store_reduction_view(mi, ni) * scores_scale(mi) + o_tmp_reduction_view(mi, ni);
             }
        }

    }


};
} // namespace flash