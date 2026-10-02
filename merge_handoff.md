# Merge stage harness: implementation spec

## Context

This is the "MSAA render" stage of the compute-shader sparse-strip pipeline planned
in https://github.com/linebender/vello/issues/1953 (read that first). In that
pipeline, flattening produces a line soup, a tile stage emits one entry per
(line, tile) pair, a sort groups entries by (path, tile), and a strip-allocation
scan assigns output locations. The render stage then consumes the sorted entries:
for each (path, tile) run it accumulates every line's per-pixel contribution and
resolves the result (nonzero / even-odd) to one antialiased tile. Conceptually a
segmented reduce over tiles of per-pixel winding; runs that cross a workgroup
boundary leave partials that a later seam-joining pass (not in scope here) combines, as in Li et al.,
"Efficient GPU Path Rendering Using Scanline Rasterization" (SIGGRAPH Asia 2016).

"Merge" here means that reduce: lines are mapped to per-pixel values, values for
the same tile are summed, then resolved. The scheduling follows the sparse branch
prototype (https://github.com/linebender/vello/blob/sparse/shader/merge.wgsl,
which stubs the area computation) with two changes worked out in design:

1. Sparseness via a fixed partition of the 8x8 tile into 4 "cosets" (a 2x2
   Bayer pattern). Any line touches at most 4 cells of each coset, so per-line
   work per coset is a fixed-trip loop of 4 instead of a dense 64-pixel loop.
2. Two consecutive entries per thread with a case table that keeps every lane on
   one instruction stream; case bits only predicate memory operations.

This first version accumulates signed area (fill_path's formula from
research/vello_shaders/shader/fine.wgsl) in fixed point and outputs 8-bit alpha.
The eventual conflation-free version swaps the payload for MSAA sample masks; the
scheduling is unchanged (see "Later").

Goal of the harness: implement and benchmark this one shader in isolation. A CPU
program builds the inputs (tile, sort, allocate done naively) from real scenes
(tiger etc.); the GPU runs only the merge. Output is inspected visually, side by
side with current Vello GPU output.

## Geometry and tiles

- Tile: 8x8 pixels. Pixel (x, y) in tile, x, y in 0..8.
- Coset of a pixel: `coset = 2*(x & 1) + (y & 1)`. Position within coset:
  `pos = (y >> 1) * 4 + (x >> 1)`.
- Fact: any line meets at most 4 cells of each coset (proof: coset cells in a column
  are separated by gaps of width 1; a line with |slope| <= 1 spans a y-interval of
  length <= 1 over one column, so it meets at most one coset cell per column; for
  |slope| >= 1 swap x and y). This bounds per-coset per-line work at 4.

## Inputs (built on CPU)

Line soup: array of `vec4<f32>` (x0, y0, x1, y1), global coordinates. Output of
flattening (reuse vello_common flattening and SVG loading unchanged).

Entries: sorted array of 16-byte records, one per (line, tile it touches):

| field    | type | meaning                                                          |
|----------|------|------------------------------------------------------------------|
| key      | u32  | (path id, tile x, tile y) packed; equal key = same run           |
| line     | u32  | index into line soup                                             |
| backdrop | i32  | coarse winding at tile left edge; nonzero only on a run's first entry |
| out      | u32  | output index for the run's tile; read by heads only              |

Sorted by key. A run = maximal range of equal keys. Threads hold 2 consecutive
entries each. Runs longer than 8 entries are split into pieces of 8 (fake
boundaries at `(i - head) % 8 == 0`); each piece has its own output index and only
the first piece carries the backdrop. Pieces are folded on the CPU for the harness.

Per-workgroup partials (seam joining is out of scope for now; only the output
format is defined so the shader writes it). Fixed-size record per workgroup, no
allocation:

```wgsl
struct Partial {
    key: u32,              // run key, or sentinel if none
    winding: array<i32, 64>, // unresolved fixed-point per-pixel sums, coset-major order
}
struct WorkgroupPartials {
    incoming: Partial,     // run continuing in from the previous workgroup
    outgoing: Partial,     // run leaving through the last entry
}
```

The incoming partial is the pseudo-head's unresolved result (backdrop 0); the
outgoing is what the last run would have added to its head's slot. Pieces from
runs > 8 entries use the same idea with their own output index. For the harness,
fold on the CPU or ignore and note the affected tiles.

## Per-line setup

Implicit line `a*x + b*y = d` in tile-local coordinates, `a = y1 - y0`,
`b = x0 - x1`, `d = a*x0 + b*y0`. Half-width `h = (|a| + |b|) * (0.5 - 2^-16)`.
Per row y: `t_y = b*cy - d` with `cy = y + 0.5`; row constant `dy_y` = signed
clipped y-extent of the segment in row y (as in fill_path in
research/vello_shaders/shader/fine.wgsl: `clamp(y0 - y, 0, 1) - clamp(y1 - y, 0, 1)`).

## Cell classification (per coset, 16 cells)

`s = fma(a, cx, t_y)`, `cx = x + 0.5`.

- `|s| < h`: partial cell, needs area evaluation.
- else: side cell; contribution is `dy_y` or 0 depending on sign of `s` and the
  convention from fill_path (coverage accumulates to the right of the edge).
- The epsilon in `h` MUST bias toward "not partial"; this is what guarantees <= 4
  partial cells per coset. Cells are half-open.
- Rows outside the segment's y-range contribute 0 (`dy_y = 0`).

## Phase 2 (per coset, per line)

Classify 16 cells -> 16-bit hit mask + 16-bit sign mask. Fixed-trip loop of 4
over the hit mask (`ctz`, `m &= m - 1`), evaluating the fill_path per-pixel area
formula for each partial cell. Result per cell: i32 fixed point,
`i32(round(area * 65536.0))`. Side cells: `dy_y` or 0 in the same fixed point.

Results are produced in coset walk order and consumed in the same order (FIFO,
4 entries, no dynamic indexing). The merge walks pixels coset by coset, in the
same 16-cell sequence phase 2 used.

## Merge stage

Workgroup: 256 threads (tune). Per thread: two consecutive entries (line 0, line 1).
Boundaries from keys: `b0` = key differs from previous entry, `b1` = line 0 and
line 1 keys differ, `b2` = line 1 and next entry keys differ. Case `b = b0 b1 b2`.
Head index (first entry of the run, local to the workgroup) via one workgroup
prefix-max scan over boundary flags. Piece head: `head + 8 * ((i - head) / 8)`.
Head slot = slot of thread `head_index / 2`. The first entry of a workgroup is a
pseudo-head (backdrop 0, result is a partial).

Slot: per thread, `chunk` pixels x i32 in workgroup shared memory. Outer loop: for
each coset, for each chunk of that coset's 16-pixel walk. Barriers per line = 1/chunk.
Try chunk 4 (8 KB at 256 threads) and chunk 8 (16 KB). "store at ix" means write
the resolved output for the run headed at line ix.

Per chunk, all lanes execute the same instruction stream; case bits only predicate
memory operations:

```
ix = 0 if b == 110 else 1
m0 = compute (phase 2) for line ix, this chunk
if b in (011, 110, 111): resolve m0, store at ix
m1 = compute for line 1 - ix
s = m1 if b in (010, 011, 110, 111) else m0 + m1
if b in (010, 100, 110):
    s0 = m0 if b == 010 else s
    atomic store s0 into own slot
barrier
if b in (000, 001, 010, 011): atomic add s into head slot
barrier
if b in (010, 100, 101, 110, 111):
    if b in (010, 100, 110): s = load own slot
    resolve s, store at (0 if b in (100, 101, 111) else 1)
```

Resolve: add backdrop (heads only), then nonzero (`min(1, |w|)`) or even-odd
(triangle wave) as in fill_path, to 8-bit alpha.

No barrier is needed between resolve and the next chunk's slot store (same thread,
same slot index). Store-before-add ordering across threads is enforced by the
first barrier; add-before-load by the second.

## Output

Per run, 64 x u8 alpha, stored coset-major: index `coset * 16 + pos` within the
run's tile, so a run's results for one coset are 16 contiguous bytes written as
4 u32 (pack 4 consecutive walk positions; chunk >= 4 keeps packing within a chunk).
Consumer maps lane `l` -> `coset = l >> 4`, `pos = l & 15`,
`x = 2*(pos & 3) + (coset >> 1)`, `y = 2*(pos >> 2) + (coset & 1)`.

## Harness

1. Load SVG, flatten to line soup (vello_common).
2. CPU tiler at 8x8: for each line, enumerate tiles touched; emit entries; compute
   backdrop per (path, tile) by the usual coarse winding count.
3. Sort entries by key; split runs > 8; assign output indices.
4. Upload line soup and entries; dispatch merge; read back alpha tiles.
5. Composite on CPU for a side-by-side with current Vello GPU output.
6. Time with timestamp queries. Knobs as constants: chunk size, workgroup size,
   and a one-line-per-thread variant for comparison (one-bit case logic:
   head -> store slot (stage 1), load/resolve (stage 3); non-head -> atomic add).

Expected measurable differences two lines vs one: barriers per line halve, atomic
RMWs per line roughly halve, slot round trips drop for runs that fit in one thread.

## Later (not in scope now)

- msaa16 masks instead of area: payload becomes a u32 of 2-bit fields
  (sample `4h + l` at bit `8l + 2h`), accumulate in 4-bit lanes
  (`x & 0x33333333`, `(x >> 2) & 0x33333333`), 8-line batches bound lane values,
  output compaction `x = E | (E >> 7); x &= 0x00FF00FF; x |= x >> 8`.
- Seam joining: GPU fold pass consuming WorkgroupPartials and piece outputs.
- Texture output as an alternative to the coset-major buffer.
