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

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

#include "blockscaled_layout.h"
#include "cute_extension.h"
#include "named_barrier.h"
using namespace cute;

template <
    int kStages,
    int EpiStages,
    typename Element,
    typename ElementSF,
    typename OutputType,
    typename SmemLayoutQ,
    typename SmemLayoutK,
    typename SmemLayoutV,
    typename SmemLayoutDS,
    typename SmemLayoutO,
    typename SmemLayoutSFQ,
    typename SmemLayoutSFK,
    typename SmemLayoutSFV
>
struct SharedStorageQKVOwithSF : cute::aligned_struct<128, _0>{
    
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>> smem_q;
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>> smem_k;
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> smem_SFQ;
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> smem_SFK;
    cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> smem_SFV;
    alignas(1024) cute::ArrayEngine<float, cute::cosize_v<SmemLayoutDS>> smem_ds;
    alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutV>> smem_v;
    alignas(1024) cute::ArrayEngine<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    
    struct {
        alignas(16) typename cutlass::PipelineTmaAsync<1>::SharedStorage pipeline_q;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_k;
        alignas(16) typename cutlass::PipelineTmaAsync<kStages>::SharedStorage pipeline_v;
        alignas(16) typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>::SharedStorage barrier_o;
        int tile_count_semaphore;
    };
  };

template <
    int kHeadDim_, 
    int kBlockM_, 
    int kBlockN_, 
    int kStages_,  
    int kClusterM_, 
    bool BlockMean_,
    typename ElementPairType_ = cutlass::nv_float4_t<cutlass::float_e2m1_t>, 
    typename ElementOut_ = cutlass::bfloat16_t
>
struct Flash_fwd_kernel_traits {
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr bool BlockMean = BlockMean_;
    static constexpr bool SmoothQ = true;
    static_assert(kHeadDim % 32 == 0);
    static_assert(kBlockM == 64 || kBlockM == 128);
    static constexpr int kNWarps = kBlockM == 128 ? 12 : 8;
    static constexpr int kNThreads = kNWarps * cutlass::NumThreadsPerWarp;
    static constexpr int kClusterM = kClusterM_;
    static constexpr int kStages = kStages_;
    static constexpr int EpiStages = 1;
    // ---------------------------------------------------------------------
    // Quantization format is selected by ElementPairType_:
    //   cutlass::nv_float4_t<e2m1> -> NVFP4, per-16 E4M3 scales (SFVecSize 16)
    //   cutlass::mx_float4_t<e2m1> -> MXFP4, per-32 E8M0 scales (SFVecSize 32)
    // ---------------------------------------------------------------------
    using Element = cutlass::float_e2m1_t;
    using ElementSF = typename ElementPairType_::ScaleFactorType;
    // Elements covered by one scale factor: 16 for E4M3, 32 for E8M0.
    static constexpr int SFVectorSize = cute::is_same_v<ElementSF, cutlass::float_ue8m0_t> ? 32 : 16;
    static_assert(SFVectorSize == 16 || SFVectorSize == 32, "unsupported scale-factor type");
    static constexpr int NumSFQK = kHeadDim / SFVectorSize;
    static constexpr int NumSFPV = kBlockN / SFVectorSize;
    using ElementAccum = float;
    using ElementOut = ElementOut_;
    using index_t = int64_t;
    using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using ClusterShape_MNK = Shape<_1, _1, _1>;
    using PermTileM = decltype(cute::min(size<0>(TileShape_MNK{}), _128{}));
    using PermTileN = _32;
    using PermTileK = Int<kHeadDim>;
    
    using ElementQMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());
    using ElementKMma = decltype(cutlass::gemm::collective::detail::sm1xx_kernel_input_element_to_mma_input_element<Element>());

    using AtomLayoutMNK = std::conditional_t<kBlockM == 128,
                                            Layout<Shape<_8, _1, _1>>,
                                            Layout<Shape<_4, _1, _1>>
                                            >;
    // Both formats use an N=32-per-atom 16x32x64 atom, so the (M16,N32) C
    // fragment layout that the online-softmax / quantization path depends on
    // (flash::convert_to_conversion_layout asserts MmaAtomN == 8) is identical
    // for NVFP4 and MXFP4.  They differ only in the scale-factor semantics:
    //   NVFP4 -> 4X scaling, ue4m3 (E4M3) scales, SFVecSize 16
    //   MXFP4 -> 2X scaling, ue8m0 (E8M0) scales, SFVecSize 32
    // NOTE: the MXFP4 atom is the N=32 counterpart (not upstream's N=8
    // SM120_16x8x64_TN_VS). Both *build*, but the upstream N=8 atom has a
    // 4-values-per-thread C layout ((_4,_8),(_2,_2)), whereas this kernel's
    // hand-written LayoutP/LayoutSFP and its online-softmax P-quantization
    // assume 16 values per thread -- the geometry shared by the NVFP4 atom and
    // by SM120_16x32x64_TN_VS_MXFP4 ((_4,_8),((_2,_4),_2)). Using the N=8 atom
    // writes P and its scale factors to the wrong registers.
    using MmaAtomQK = std::conditional_t<
        SFVectorSize == 32,
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_MXFP4,
        cute::SM120::BLOCKSCALED::SM120_16x32x64_TN_VS_NVFP4>;

    using TiledMmaQK = decltype(cute::make_tiled_mma(
        MmaAtomQK{},
        AtomLayoutMNK{},
        Tile<PermTileM, PermTileN, PermTileK>{}
      ));

    using TiledMmaPV = decltype(cute::make_tiled_mma(
        MmaAtomQK{},
        AtomLayoutMNK{},
        Tile<PermTileM, _32, PermTileK>{}
      ));
    
    static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVectorSize;

    using GmemTiledCopy = SM90_TMA_LOAD;
    using GmemTiledCopySF = SM90_TMA_LOAD;

    using SmemLayoutAtomQ = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomK = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomV = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomVt = decltype(cutlass::gemm::collective::detail::sm120_rr_smem_selector<Element, decltype(size<1>(TileShape_MNK{}))>());
    using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
    using SmemLayoutK =
        decltype(tile_to_shape(SmemLayoutAtomK{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutV =
        decltype(tile_to_shape(SmemLayoutAtomV{},
                 make_shape(shape<1>(TileShape_MNK{}), shape<2>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutVt =
        decltype(tile_to_shape(SmemLayoutAtomVt{},
                 make_shape(shape<2>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));
    using SmemLayoutAtomDS = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<_0, _1>>;
    using SmemLayoutDS = 
        decltype(tile_to_shape(SmemLayoutAtomDS{},
            make_shape(shape<0>(TileShape_MNK{}), shape<1>(TileShape_MNK{}), Int<kStages>{})));

    using SmemCopyAtomQ = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomKV = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;
    using SmemCopyAtomDS = Copy_Atom<UniversalCopy<float>, float>;

    using BlkScaledConfig = flash::BlockScaledConfig<SFVectorSize>;
    using LayoutSF = typename BlkScaledConfig::LayoutSF;
    using SfAtom = typename BlkScaledConfig::SfAtom;
    using SmemLayoutAtomSFQ = decltype(BlkScaledConfig::deduce_smem_layoutSFQ(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFK = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaQK{}, TileShape_MNK{}));
    using SmemLayoutAtomSFV = decltype(BlkScaledConfig::deduce_smem_layoutSFKV(TiledMmaPV{}, TileShape_MNK{}));
    using SmemLayoutAtomSFVt = decltype(BlkScaledConfig::deduce_smem_layoutSFVt(TiledMmaPV{}, Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>{}));
    // Register-fragment layout of the SF for the PV gemm's A operand (P).
    //
    // P is (kBlockM, kBlockN); the PV atom's K is 64, so each atom-K block
    // consumes MMA_NSF SF slots along K and there are kBlockN/64 such blocks.
    // The SF is broadcast across the rows of a 16-element block (stride 0, and
    // the row mode has stride 0 too), so a per-k_block slice is
    //   (64/MMA_NSF, MMA_NSF) : (0, 1)
    // which is what cute::SM120::BLOCKSCALED::mma_unpack requires:
    //   size(SFA)                 == size<2>(Shape_MNK) == 64
    //   cosize(layout(SFA))       == 64/SFVecSize
    // Both hold because 64/MMA_NSF * MMA_NSF == 64 and, since MMA_NSF is
    // 64/SFVecSize, cosize == MMA_NSF == 64/SFVecSize.
    //
    // MMA_NSF is 4 for NVFP4 (so this is bit-identical to the original
    // hardcoded (16,4):(0,1) / stride 4) and 2 for MXFP4 (giving (32,2):(0,1)).
    using LayoutSFP = decltype(
      make_layout(
          make_shape(make_shape(Int<64 / MMA_NSF>{}, Int<MMA_NSF>{}), _1{}, Int<kBlockN / 64>{}),
          make_stride(make_stride(_0{}, _1{}), _0{}, Int<MMA_NSF>{})
      )
    );
    using LayoutP = decltype(
      make_layout(
        make_shape(make_shape(_8{}, _2{}, _2{}), _1{}, Int<kBlockN / 64>{}),
        make_stride(make_stride(_1{}, _8{}, _16{}), _0{}, _32{})
      )
    );
    using SmemLayoutSFQ = decltype(make_layout(
        shape(SmemLayoutAtomSFQ{}),
        stride(SmemLayoutAtomSFQ{})
      ));
    using SmemLayoutSFK = decltype(make_layout(
        append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFK{}), size(filter_zeros(SmemLayoutAtomSFK{})))
      ));
    using SmemLayoutSFV = decltype(make_layout(
        append(shape(SmemLayoutAtomSFV{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFV{}), size(filter_zeros(SmemLayoutAtomSFV{})))
      ));
    using SmemLayoutSFVt = decltype(make_layout(
        append(shape(SmemLayoutAtomSFVt{}), Int<kStages>{}),
        append(stride(SmemLayoutAtomSFVt{}), size(filter_zeros(SmemLayoutAtomSFVt{})))
      ));

    using SmemLayoutAtomO = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GMMA::Major::K, ElementOut,
        decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutO = decltype(tile_to_shape(SmemLayoutAtomO{}, select<0, 2>(TileShape_MNK{}), Step<_1, _2>{}));
    using SharedStorage = SharedStorageQKVOwithSF<kStages, EpiStages, Element, ElementSF, ElementOut,
        SmemLayoutQ, SmemLayoutK, SmemLayoutV, SmemLayoutDS, 
        SmemLayoutO, SmemLayoutSFQ, SmemLayoutSFK, SmemLayoutSFVt>;
    using MainloopPipeline = typename cutlass::PipelineTmaAsync<kStages>;
    using PipelineState = typename cutlass::PipelineState<kStages>;
    using MainloopPipelineQ = cutlass::PipelineTmaAsync<1>;
    using PipelineParamsQ = typename MainloopPipelineQ::Params;
    using PipelineStateQ = typename cutlass::PipelineState<1>;
    using EpilogueBarrier = typename flash::OrderedSequenceBarrierVarGroupSize<EpiStages, 2>;
};

