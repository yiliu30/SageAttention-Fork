# MXFP4 scale-factor byte layout (SFVecSize = 32)

Derived from `flash::BlockScaledConfig<SFVecSize>::tile_atom_to_shape_SFQKV` /
`tile_atom_to_shape_SFVt`, and verified exhaustively against the cute layout
evaluator (`g++` probe, 0 errors over all (row, k-block) pairs at L=128).

## Layout for Q and K (`tile_atom_to_shape_SFQKV`)

Tile of `L` rows x `D` columns, SF-blocked 32 columns at a time.
For row `r` and K-block `kb` (`kb = k / 32`), where `tok = r % 64`:

| SFVecSize | byte offset |
|---|---|
| 16 (NVFP4) | `(r/64)*512 + (kb/4)*256 + (kb%4) + (tok/16)*4 + (tok%16)*16` |
| **32 (MXFP4)** | `(r/64)*256 + kb + (tok/16)*4 + (tok%16)*16` |

The MXFP4 form differs in two ways:
- the 64-row tile stride halves (512 -> 256 bytes) because there are half as
  many K-blocks per row, and
- the four K-blocks of a row are **contiguous in the low two bits** (stride 1),
  rather than splitting into a `(kb/4)*256 + (kb%4)` pattern, because
  `MMA_NSF = 64/SFVecSize = 2` and `Blk_SF=4` gives only 2*4 = 8 SF slots per
  row block at SFVecSize 32 (vs 16 at SFVecSize 16).

## Layout for V (`tile_atom_to_shape_SFVt`)

V is stored transposed as `(D, N)`. For head-dim index `d` and seq-block `nb`:

```
offset = (d/64)*256 + nb + ((nb%64)/16)*4 + ((nb%64)%16)*16   ... with d and nb
```

More precisely, evaluated at L=128, D=128 (verified):
`L(d, nb*32) = (d/64)*256 + nb + ((nb*32%64)/16)*4 + ((nb*32%64)%16)*16`
which for nb in {0,1,2,3} gives the observed 0,1,2,3 / 16,17,18,19 / 4,5,6,7.

Because `((nb*32) % 64) % 16 == 0` for nb in {0,1,2,3}, this reduces to:

| nb | offset for `d < 64` |
|---|---|
| 0 | 0 |
| 1 | 1 |
| 2 | 4 |
| 3 | 5 |

and `+256` for `d >= 64`.

## Key consequence for the quantizer

`CVT_FP4_ELTS_PER_THREAD` becomes 32 (one E8M0 byte per thread, covering one
32-element block), so `localMax` is a plain per-thread max over 32 elements with
**no cross-lane shuffle** — this is the main simplification vs the NVFP4 path,
which needed 16 elements per thread plus a `__shfl_xor_sync(..., 1)` to build a
16-element block across two lanes.

The byte offset formula must switch on `SFVecSize`; at 32 the `kb` term is a
direct addend rather than a split `(kb/4, kb%4)` pair.

## P-side (A-operand) SFA register fragment map (verified on SM120)

Empirically verified with a standalone probe of
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::{2X,4X}.m16n8k64`
(data-anchored group-order resolution + 20-iteration full random stress,
`/tmp/mxfp4_probe/a_nibble_probe2`):

For the m16x64 A block (16 P rows x 64 K cols), per warp:

- Rows 0-7 of the A block: lane `4*m` (i.e. `lane % 4 == 0`) carries the
  scales for row `m` (rows = `l/4`).
- Rows 8-15: lane `4*(m-8)+1` (`lane % 4 == 1`) carries the scales for row `m`.
- Lanes with `lane % 4 in {2,3}` are never read for SFA.
- Scale-group `g` of the 64-K atom (2X: two 32-col groups; 4X: four 16-col
  groups) is read from **byte `g`** (low to high) of that lane's SF register.
  2X: 16-bit register (bytes 0,1); 4X: 32-bit register (bytes 0..3).
- B-side (SFB, `tidB = c`): lane `4*n + c` carries column `n` of sub-tile `c`,
  group `g` in byte `g`.

### Kernel design consequences (`mainloop_tma_ws.h` / `softmax_fused.h`)

The kernel's `AbsMaxP` fragment slot `(mi, ni)` (flat index `mi + 2*n0` per
64-col stage, `mi` = row half, `n0` = 32-col half, plus the 64-col stage
index) is the max over the 8 P values of row `l/4 + 8*mi` held by the lane in
that group. The full 32-col group max therefore needs a quad reduction:
intra-lane max over the 8 elements, then `__shfl_xor_sync(..., 1)` **and**
`__shfl_xor_sync(..., 2)` (butterfly over lanes `{l, l^1, l^2, l^3}`, which
cover all four 8-col sub-tiles). xor(1) alone reduced only half the group's
columns and left the scale up to 2x too small (e2m1 saturation).

- **MXFP4 (2X)**: before the P divide, the two flat slots of each group
  (`(row0, g)`, `(row8, g)`) are folded to their pair max, so both 8-row
  halves share one scale per 32-col group; the SFP pack is
  `(byte0 = slot(sf_row), byte1 = slot(sf_row + 2))` with `sf_row = lane & 1`,
  which puts the folded `(G_lo, G_hi)` pair in every lane's 16-bit register.
  The divide and the MMA multiply then agree exactly (both use the
  `ceil_pow2`-rounded value; E8M0 has no mantissa).
- **NVFP4 (4X)**: no fold; the 4 bytes per lane are built from slots
  `[0..3]` with a MASK/shuffle interleave (`0xFF00FF`, shifted by
  `(lane & 1) * 8`) that expands each 32-col group max into the two 16-col
  SF groups the 4X atom provides. After the quad reduction the result is
  per-row `(G0, G0, G1, G1)` in group order, matching the hardware map.
