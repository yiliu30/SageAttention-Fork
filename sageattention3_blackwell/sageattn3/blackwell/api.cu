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

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "params.h"
#include "launch.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen_q,
                      const size_t seqlen_k,
                      const size_t unpadded_seqlen_k,
                      const size_t seqlen_q_rounded,
                      const size_t seqlen_k_rounded,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor delta_s,
                      at::Tensor out,
                      const at::Tensor sfq,
                      const at::Tensor sfk,
                      const c10::optional<at::Tensor> &sfv,
                      const c10::optional<at::Tensor> &v_scale,
                      void *cu_seqlens_q_d,
                      void *cu_seqlens_k_d,
                      void *seqused_k,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int window_size_left,
                      int window_size_right,
                      bool per_block_mean,
                      bool is_bf16,
                      bool use_fp8_pv,
                      bool seqlenq_ngroups_swapped=false) {

    // Reset the parameters
    params = {};
    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.delta_s_ptr = delta_s.data_ptr();
    params.sfq_ptr = sfq.data_ptr();
    params.sfk_ptr = sfk.data_ptr();
    params.sfv_ptr = sfv.has_value() ? sfv->data_ptr() : nullptr;
    params.v_scale_ptr =
        v_scale.has_value() ? v_scale->data_ptr() : nullptr;
    
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-2) * 2;
    params.k_row_stride = k.stride(-2) * 2;
    params.v_row_stride = v.stride(-2) * (use_fp8_pv ? 1 : 2);
    params.q_head_stride = q.stride(-3) * 2;
    params.k_head_stride = k.stride(-3) * 2;
    params.v_head_stride = v.stride(-3) * (use_fp8_pv ? 1 : 2);

    params.ds_row_stride = delta_s.stride(-2);
    params.ds_head_stride = delta_s.stride(-3);
    
    params.sfq_row_stride = sfq.stride(-2);
    params.sfk_row_stride = sfk.stride(-2);
    params.sfq_head_stride = sfq.stride(-3);
    params.sfk_head_stride = sfk.stride(-3);
    if (sfv.has_value()) {
        params.sfv_row_stride = sfv->stride(-2);
        params.sfv_head_stride = sfv->stride(-3);
    }
    if (v_scale.has_value()) {
        params.v_scale_batch_stride = v_scale->stride(0);
        params.v_scale_head_stride = v_scale->stride(1);
        params.v_scale_row_stride = v_scale->stride(2);
    }
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-2);
    params.o_head_stride = out.stride(-3);

    if (cu_seqlens_q_d == nullptr) {
        params.q_batch_stride = q.stride(0) * 2;
        params.k_batch_stride = k.stride(0) * 2;
        params.v_batch_stride = v.stride(0) * (use_fp8_pv ? 1 : 2);
        params.ds_batch_stride = delta_s.stride(0);
        params.sfq_batch_stride = sfq.stride(0);
        params.sfk_batch_stride = sfk.stride(0);
        if (sfv.has_value()) {
            params.sfv_batch_stride = sfv->stride(0);
        }
        params.o_batch_stride = out.stride(0);
        if (seqlenq_ngroups_swapped) {
             params.q_batch_stride *= seqlen_q;
             params.o_batch_stride *= seqlen_q;
        }
    }

    params.cu_seqlens_q = static_cast<int *>(cu_seqlens_q_d);
    params.cu_seqlens_k = static_cast<int *>(cu_seqlens_k_d);
    params.seqused_k = static_cast<int *>(seqused_k);

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen_q = seqlen_q;
    params.seqlen_k = seqlen_k;
    params.unpadded_seqlen_k = unpadded_seqlen_k;
    params.seqlen_q_rounded = seqlen_q_rounded;
    params.seqlen_k_rounded = seqlen_k_rounded;
    params.d = d;
    params.d_rounded = d_rounded;

    params.head_divmod = cutlass::FastDivmod(int(h));

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
    __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
    params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    // params.p_dropout_in_uint = uint32_t(std::floor(params.p_dropout * 4294967295.0));
    // params.p_dropout_in_uint16_t = uint16_t(std::floor(params.p_dropout * 65535.0));
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This flash attention build does not support dropout.");
    #endif

    // Causal is the special case where window_size_right == 0 and window_size_left < 0.
    // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
    params.is_causal = window_size_left < 0 && window_size_right == 0;
    params.per_block_mean = per_block_mean;
    if (per_block_mean) {
        params.seqlen_s = seqlen_q;
    } else {
        params.seqlen_s = 128; // size of BLOCK_M
    }
    if (window_size_left < 0 && window_size_right >= 0) { window_size_left = seqlen_k; }
    if (window_size_left >= 0 && window_size_right < 0) { window_size_right = seqlen_k; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    #ifdef FLASHATTENTION_DISABLE_LOCAL
        TORCH_CHECK(params.is_causal || (window_size_left < 0 && window_size_right < 0),
            "This flash attention build does not support local attention.");
    #endif

    params.is_seqlens_k_cumulative = true;
    params.is_bf16 = is_bf16;
    params.is_e4m3 = use_fp8_pv;
    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This flash attention build does not support headdim not being a multiple of 32.");
    #endif
}

template<bool IsBF16>
void run_mha_fwd_dispatch_dtype(
    Flash_fwd_params &params,
    cudaStream_t stream,
    bool use_two_cta,
    bool bypass_p_packing,
    bool use_fp8_pv) {
    using OType = std::conditional_t<IsBF16, cutlass::bfloat16_t, cutlass::half_t>;
    if (params.d == 64) {
        run_mha_fwd_<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 64, OType>(
            params, stream, false, false, use_fp8_pv);
    } else if (params.d == 128) {
        run_mha_fwd_<cutlass::nv_float4_t<cutlass::float_e2m1_t>, 128, OType>(
            params, stream, use_two_cta, bypass_p_packing, use_fp8_pv);
    }
}

void run_mha_fwd(
    Flash_fwd_params &params,
    cudaStream_t stream,
    bool use_two_cta,
    bool bypass_p_packing,
    bool use_fp8_pv) {
    BOOL_SWITCH(params.is_bf16, IsBF16, ([&] {
        run_mha_fwd_dispatch_dtype<IsBF16>(
            params, stream, use_two_cta, bypass_p_packing, use_fp8_pv);
    }));
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen_q x num_heads x (head_size // 2)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x (head_size // 2)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x (head_size // 2)
        const at::Tensor &sfq,
        const at::Tensor &sfk,
        const c10::optional<at::Tensor> &sfv,
        const at::Tensor &delta_s,
        int unpadded_k,
        c10::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x head_size
        const float softmax_scale,
        bool is_causal, 
        bool per_block_mean,
        bool is_bf16,
        bool use_two_cta,
        bool bypass_p_packing,
        bool use_fp8_pv,
        const c10::optional<at::Tensor> &v_scale
    ) {

    at::cuda::CUDAGuard device_guard{(char)q.get_device()};
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm120 = dprops->major == 12 && dprops->minor == 0;
    bool is_sm121 = dprops->major == 12 && dprops->minor == 1;
    TORCH_CHECK(is_sm120 || is_sm121, "only supports Blackwell GPUs or newer.");

    auto q_dtype = q.scalar_type();
    auto sfq_dtype = sfq.scalar_type();
    TORCH_CHECK(q_dtype == torch::kUInt8, "q dtype must be uint8");
    TORCH_CHECK(k.scalar_type() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(
        v.scalar_type() ==
            (use_fp8_pv ? torch::kFloat8_e4m3fn : q_dtype),
        use_fp8_pv ? "value must have dtype float8_e4m3fn"
                   : "query and value must have the same dtype");
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    CHECK_DEVICE(delta_s);

    TORCH_CHECK(
        sfq_dtype == torch::kFloat8_e4m3fn,
        "Q scale dtype must be float8_e4m3fn");
    TORCH_CHECK(sfk.scalar_type() == sfq_dtype, "query and key must have the same dtype");
    CHECK_DEVICE(sfq); CHECK_DEVICE(sfk);
    if (use_fp8_pv) {
        TORCH_CHECK(
            !sfv.has_value(),
            "FP8 P x V does not use FP4 V block scales");
        TORCH_CHECK(v_scale.has_value(), "FP8 P x V requires v_scale");
        CHECK_DEVICE(v_scale.value());
        TORCH_CHECK(
            v_scale->scalar_type() == torch::kFloat32,
            "v_scale dtype must be float32");
    } else {
        TORCH_CHECK(sfv.has_value(), "NVFP4 P x V requires sfv");
        TORCH_CHECK(
            sfv->scalar_type() == sfq_dtype,
            "query and value scales must have the same dtype");
        CHECK_DEVICE(sfv.value());
        TORCH_CHECK(
            !v_scale.has_value(),
            "v_scale is only supported by FP8 P x V");
    }
    TORCH_CHECK(
        q.get_device() == k.get_device() &&
            q.get_device() == v.get_device() &&
            q.get_device() == sfq.get_device() &&
            q.get_device() == sfk.get_device() &&
            q.get_device() == delta_s.get_device(),
        "all attention inputs must be on the same CUDA device");
    if (sfv.has_value()) {
        TORCH_CHECK(
            q.get_device() == sfv->get_device(),
            "sfv must be on the same CUDA device as q");
    }
    if (v_scale.has_value()) {
        TORCH_CHECK(
            q.get_device() == v_scale->get_device(),
            "v_scale must be on the same CUDA device as q");
    }
    
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(delta_s.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    TORCH_CHECK(q.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "Input tensor must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "Input tensor must be contiguous");

    const auto sizes = q.sizes();
    auto opts = q.options();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[2];
    int num_heads = sizes[1];
    const int head_size_og = sizes[3];
    const int unpacked_head_size = head_size_og * 2;
    const int seqlen_k = k.size(2);
    const int num_heads_k = k.size(1);
    
    TORCH_CHECK(batch_size > 0, "batch size must be postive");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    TORCH_CHECK(unpacked_head_size == 64 || unpacked_head_size == 128, "Only support head size 64 and 128");
    TORCH_CHECK(!use_two_cta || unpacked_head_size == 128, "The two-CTA kernel requires head dimension 128");
    TORCH_CHECK(!bypass_p_packing || unpacked_head_size == 128, "The P-packing ablation requires head dimension 128");
    TORCH_CHECK(!bypass_p_packing || !use_two_cta, "The P-packing ablation only supports the baseline kernel");
    TORCH_CHECK(!use_fp8_pv || !use_two_cta, "FP8 P x V does not support the two-CTA kernel");
    TORCH_CHECK(!use_fp8_pv || !bypass_p_packing, "FP8 P x V is incompatible with bypass_p_packing");
    TORCH_CHECK(
        !use_fp8_pv || !is_causal || seqlen_q == seqlen_k,
        "FP8 P x V does not support causal attention with unequal query and key lengths");
    if (bypass_p_packing) {
        TORCH_WARN_ONCE(
            "bypass_p_packing is a performance diagnostic and returns "
            "numerically invalid attention output");
    }

    CHECK_SHAPE(q, batch_size, num_heads, seqlen_q, head_size_og);
    CHECK_SHAPE(k, batch_size, num_heads_k, seqlen_k, head_size_og);
    if (use_fp8_pv) {
        CHECK_SHAPE(v, batch_size, num_heads_k, unpacked_head_size, seqlen_k);
        CHECK_SHAPE(
            v_scale.value(),
            batch_size,
            num_heads_k,
            unpacked_head_size);
        TORCH_CHECK(
            v_scale->is_contiguous(),
            "v_scale must be contiguous");
    } else {
        CHECK_SHAPE(v, batch_size, num_heads_k, unpacked_head_size, seqlen_k/2);
    }
    const int query_block_size = use_two_cta ? 64 : 128;
    const int delta_s_rows = per_block_mean ? seqlen_q / query_block_size : 1;
    CHECK_SHAPE(delta_s, batch_size, num_heads, delta_s_rows, seqlen_k);
    // CHECK_SHAPE(sfq, batch_size, seqlen_q, num_heads, unpacked_head_size);
    // CHECK_SHAPE(sfk, batch_size, seqlen_k, num_heads_k, unpacked_head_size);
    // CHECK_SHAPE(sfv, batch_size, unpacked_head_size, num_heads_k, seqlen_k);
    TORCH_CHECK(unpacked_head_size % 8 == 0, "head_size must be a multiple of 8");
    
    auto dtype = is_bf16 ? at::ScalarType::BFloat16 : at::ScalarType::Half;
    at::Tensor out = torch::empty({batch_size, num_heads, seqlen_q, unpacked_head_size}, opts.dtype(dtype));

    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    // const int head_size = round_multiple(head_size_og, 8);
    // const int head_size_rounded = round_multiple(head_size, 32);
    const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
    const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    at::Tensor p;

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k, unpadded_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     unpacked_head_size, unpacked_head_size,
                     q, k, v, delta_s, out, 
                     sfq, sfk, sfv, v_scale,
                     /*cu_seqlens_q_d=*/nullptr,
                     /*cu_seqlens_k_d=*/nullptr,
                     /*seqused_k=*/nullptr,
                     nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     /*window_size_left=*/-1,
                     /*window_size_right=*/is_causal ? 0 : -1,
                     per_block_mean,
                     is_bf16,
                     use_fp8_pv
                    );
    // TODO: 132 sm count?
    auto tile_count_semaphore = is_causal ? torch::full({1}, 132, opts.dtype(torch::kInt32)) : torch::empty({1}, opts.dtype(torch::kInt32));
    params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();

    if (seqlen_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(
            params,
            stream,
            use_two_cta,
            bypass_p_packing,
            use_fp8_pv);
    } else {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    // at::Tensor out_padded = out;
    // if (head_size_og % 8 != 0) {
    //     out = out.index({"...", torch::indexing::Slice(torch::indexing::None, head_size_og)});
    //     if (out_.has_value()) { out_.value().copy_(out); }
    // }

    // return {out, q_padded, k_padded, v_padded, out_padded, softmax_lse, p};
    // cudaDeviceSynchronize();
    // auto err = cudaGetLastError();
    // printf("%s\n", cudaGetErrorString(err));
    return {out, softmax_lse};
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "FlashAttention";
    m.def(
        "fwd",
        &mha_fwd,
        "Forward pass",
        pybind11::arg("q"),
        pybind11::arg("k"),
        pybind11::arg("v"),
        pybind11::arg("sfq"),
        pybind11::arg("sfk"),
        pybind11::arg("sfv"),
        pybind11::arg("delta_s"),
        pybind11::arg("unpadded_k"),
        pybind11::arg("out"),
        pybind11::arg("softmax_scale"),
        pybind11::arg("is_causal"),
        pybind11::arg("per_block_mean"),
        pybind11::arg("is_bf16"),
        pybind11::arg("use_two_cta") = false,
        pybind11::arg("bypass_p_packing") = false,
        pybind11::arg("use_fp8_pv") = false,
        pybind11::arg("v_scale") = c10::nullopt);
}
