/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Blocked Scale configs specific for SM100 BlockScaled MMA
*/

#pragma once

#include "cutlass/layout/matrix.h"

#include "cute/int_tuple.hpp"
#include "cute/atom/mma_traits_sm100.hpp"

namespace flash {

/////////////////////////////////////////////////////////////////////////////////////////////////
using namespace cute;

template<int SFVecSize, UMMA::Major major = UMMA::Major::K>
struct BlockScaledBasicChunk {

  using Blk_MN    = _64;
  using Blk_SF    =   _4;

  using SfAtom  = Layout< Shape< Shape<_16,_4>, Shape<Int<SFVecSize>, _4>>,
                               Stride<Stride<_16,_4>, Stride<           _0, _1>>>;
};

template<int SFVecSize_>
struct BlockScaledConfig {
  // We are creating the SFA and SFB tensors' layouts in the collective since they always have the same layout.
  // k-major order
  static constexpr int SFVecSize = SFVecSize_;
  // MMA_NSF scale factors per atom along K. The block-scaled atom has K=64, and
  // each SF covers SFVecSize consecutive K elements, so MMA_NSF = 64/SFVecSize
  // (4 for NVFP4 at SFVecSize=16, 2 for MXFP4 at SFVecSize=32).
  static constexpr int MMA_NSF = 64 / SFVecSize;
  using BlkScaledChunk = BlockScaledBasicChunk<SFVecSize>;
  using Blk_MN    = _64;
  using Blk_SF    =   _4; 
  using mnBasicBlockShape  =  Shape<_16,_4>;
  using mnBasicBlockStride = Stride<_16,_4>;
  // The gmem SF atom's K mode is FIXED at 4 scale-factor slots, independent of
  // SFVecSize.  The physical gmem layout stores one SF slot per (row, 64-K
  // block) pair, so a 4-wide K group always spans 4 * 64 = 256 K elements --
  // exactly the 4 MMAs of a 256-K row of the (seq, dim) tile.  Those 4 slots
  // are contiguous bytes (stride 1) belonging to the SAME row; the row sub-mode
  // then advances by 16 bytes.
  //
  // This is a property of the *tile* layout, not of the atom: it is why the
  // verified byte offset is a function of `kb` alone for MXFP4 and of
  // `(kb/4, kb%4)` for NVFP4 (for any fixed row block).  Only MMA_NSF
  // (= 64/SFVecSize, the number of SF slots one MMA actually consumes) depends
  // on SFVecSize; it is used for the SMEM layouts and for the per-MMA SF
  // register packing, never here.
  //
  // Making this mode Int<MMA_NSF> instead is what broke SFVecSize 32: the atom
  // then covered 4 K slabs of MMA_NSF slots each, so its K extent no longer
  // matched the tile's, and `construct_tma_gbasis` resolved to a degenerate
  // take<0,0> ("Could not find a common tile-gmem vectorization") while the
  // gmem offsets drifted away from the quantizer's layout.
  using kBasicBlockShape  = Shape<Int<SFVecSize>, _4>; // SFVecSize, 4
  using kBasicBlockStride = Stride<_0, _1>;
  using SfAtom  = Layout< Shape< mnBasicBlockShape, kBasicBlockShape>,
                          Stride<mnBasicBlockStride, kBasicBlockStride>>;

  using LayoutSF = decltype(blocked_product(SfAtom{},
                                make_layout(
                                    make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
                                    make_stride(int32_t(0), _1{}, int32_t(0), int32_t(0)))));
  // A single indivisible block will hold 4 scale factors of 64 rows/columns (A/B matrix).
  // 4 is chosen to make consecutive 32bits of data to have scale factors for only a single row (col). 32bits corresponds to the TMEM word size
  using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
  using sSF_strideMN = decltype(prepend(Blk_Elems{},  mnBasicBlockStride{}));

  // Number of INDEPENDENT 256-K slabs in the SF atom's K mode.
  //
  // A single SF atom spans exactly one 256-K slab: 4 k-blocks of 64 K
  // elements.  Hence this factor is always 1.  It used to be written
  // Blk_SF/MMA_NSF, which is 1 only for NVFP4 (MMA_NSF = 4) and 2 for
  // MXFP4 (MMA_NSF = 2) -- and that stray factor 2 was the MXFP4 bug:
  // it doubled the K extent of the atom (and thus size(SmemLayoutAtomSFQ),
  // 32768 instead of the required 16384) while the atom's own K mode still
  // only strided 4 slots, so the atom was not an affine tiling of the tile
  // and construct_tma_gbasis could not find a common tile-gmem vectorization.
  // Writing it as the literal 1 keeps the NVFP4 layout bit-identical
  // (1 == 4/4) and makes MXFP4 agree with the verified gmem byte layout.
  static constexpr int SFKSlabs = 1;
    

  // The following function is provided for user fill dynamic problem size to the layout_SFA.
  template < class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFQKV(ProblemShape problem_shape) {
    auto [Seqlen, Dim, HeadNum, Batch] = problem_shape;
    return tile_to_shape(SfAtom{}, make_shape(Seqlen, Dim, HeadNum, Batch), Step<_2,_1,_3,_4>{});
  }

  // The following function is provided for user fill dynamic problem size to the layout_SFB.
  template <class ProblemShape>
  CUTE_HOST_DEVICE
  static constexpr auto
  tile_atom_to_shape_SFVt(ProblemShape problem_shape) {
    auto [Dim, Seqlen, HeadNum, Batch] = problem_shape;
    return tile_to_shape(SfAtom{}, make_shape(Dim, Seqlen, HeadNum, Batch), Step<_2,_1,_3,_4>{});
  }

  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFQ(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {
    
    using sSFQ_shapeK = decltype(prepend(make_shape(Int<SFKSlabs>{}, size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
    using sSFQ_shapeM = decltype(prepend(size<0>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));
    using sSFQ_strideM = sSF_strideMN;
    using sSFQ_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, size<0>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
    using sSFQ_shape = decltype(make_shape(sSFQ_shapeM{}, sSFQ_shapeK{}));
    using sSFQ_stride = decltype(make_stride(sSFQ_strideM{}, sSFQ_strideK{}));
    using SmemLayoutAtomSFQ = decltype(make_layout(sSFQ_shape{},  sSFQ_stride{}));
    return SmemLayoutAtomSFQ{};
  }

  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFKV(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {
  
    using sSFK_shapeK = decltype(prepend(make_shape(Int<SFKSlabs>{}, size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
    using sSFK_shapeN = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));
    using sSFK_strideN = sSF_strideMN;
    using sSFK_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
    using sSFK_shape = decltype(make_shape(sSFK_shapeN{}, sSFK_shapeK{}));
    using sSFK_stride = decltype(make_stride(sSFK_strideN{}, sSFK_strideK{}));
    using SmemLayoutAtomSFK = decltype(make_layout(sSFK_shape{}, sSFK_stride{}));
    return SmemLayoutAtomSFK{};
  }

  template<class TiledMma, class TileShape_MNK>
  CUTE_HOST_DEVICE
  static constexpr auto
  deduce_smem_layoutSFVt(TiledMma tiled_mma, TileShape_MNK tileshape_mnk) {
  
    using sSFVt_shapeK = decltype(prepend(make_shape(Int<SFKSlabs>{}, size<2>(TileShape_MNK{}) / Int<SFVecSize>{} / Blk_SF{}), kBasicBlockShape{}));
    using sSFVt_shapeN = decltype(prepend(size<1>(TileShape_MNK{}) / Blk_MN{}, mnBasicBlockShape{}));
    using sSFVt_strideN = sSF_strideMN;
    using sSFVt_strideK = decltype(prepend(make_stride(Int<MMA_NSF>{}, size<1>(TileShape_MNK{}) / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
    using sSFVt_shape = decltype(make_shape(sSFVt_shapeN{}, sSFVt_shapeK{}));
    using sSFVt_stride = decltype(make_stride(sSFVt_strideN{}, sSFVt_strideK{}));
    using SmemLayoutAtomSFVt = decltype(make_layout(sSFVt_shape{}, sSFVt_stride{}));
    return SmemLayoutAtomSFVt{};
  }
};


} // namespace flash
