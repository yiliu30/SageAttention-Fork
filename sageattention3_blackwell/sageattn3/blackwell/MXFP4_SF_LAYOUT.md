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
