# Merge harness: status notes

Companion to `merge_handoff.md` (the spec). Records decisions made while implementing and
the first measurements. Started 2026-10-02.

## Layout

- `src/scene.rs`: SVG -> line soup via `vello_common` (fills; strokes expanded to fills).
- `src/tiler.rs`: CPU tile/sort/allocate. Entry format per the spec plus two flag bits:
  `line` bit 31 = `LINE_LEFT_EDGE`, `out` bit 31 = `OUT_PARTIAL`.
- `src/reference.rs`: f64 dense oracle (`PathTruth`, independent of tiling) and a CPU mirror
  of the shader's per-entry math (`entry_contrib`, `run_winding`, `resolve`).
- `src/gpu.rs`: wgpu harness, timestamp queries, variant selection by shader header consts.
- `shader/phase2.wgsl`: per-line setup, coset classification, FIFO area evaluation, resolve,
  output store. `merge1.wgsl` / `merge2.wgsl`: one / two entries per thread. `merge0.wgsl`:
  diagnostic, phase 2 only.

Run: `cargo run --release -- <svg> <scale> <out_dir> <iterations> <replicate>`.
Default is the tiger at 4x; `replicate` duplicates the entry stream with distinct path ids
to get a throughput-sized workload.

## Conventions settled

- Winding sign follows fill_path: an upward line contributes +1 to pixels on its right.
- Backdrop of tile (tx, ty) = signed count of lines crossing the row's top edge
  `y = 8 ty` (half-open `ymin <= 8ty < ymax`) at an x in a tile strictly left of tx.
- A point at x belongs to tile `floor(x / 8)`. The row-clipped segment touches tiles
  `floor(xa/8) ..= floor(xb/8)`; every tile after the first gets `LINE_LEFT_EDGE`.
- In the shader, the segment is clipped to the tile's x-range for `dy` per row; the
  infinite line (direction from the unclipped segment, offset from a clipped endpoint)
  drives classification. Entries with `LINE_LEFT_EDGE` add fill_path's y_edge term,
  `sign(dx) * clamp(row - y_edge + 1, 0, 1)`, with `y_edge = d / b`.
- Area weight is fill_path's formula rewritten without the difference of squares
  (`area_weight` in both WGSL and Rust). The original form loses ~4% of a row's coverage
  for near-vertical segments in f32 and CPU/GPU disagreed on it by up to 10/255.
- Resolve is integer-only so CPU and GPU agree exactly.
- Pieces: a run is split at workgroup boundaries and every 8 entries after that; every
  piece head carries `OUT_PARTIAL | index` and writes unresolved i32 winding (plus the
  head's backdrop). The CPU folds. This replaces the spec's `WorkgroupPartials` record
  for now; the real pipeline needs an allocation stage that does the same alignment.
- Entry stream is padded to whole workgroups with sentinel key `u32::MAX`, a degenerate
  line, and a dummy alpha slot.

## Correctness

Tiger 800x800: all variants match the CPU mirror on all but 3-4 of 12296 runs (1 LSB,
float rounding), and match the f64 oracle to within 1 alpha LSB on every pixel.
The "at most 4 partial cells per coset" bound held on all 119K (entry, coset) pairs.

## First measurements (Apple M5 Max, Metal, 16x replicated tiger = 480K entries)

| variant                | median ms |
|------------------------|-----------|
| compute only (floor)   | 0.632     |
| one line, wg256, ch4   | 0.831 (0.660 after phase-2 fixes) |
| two lines, wg256, ch4  | 0.855 (0.665 after phase-2 fixes) |
| one line, wg256, ch8   | 0.910     |
| two lines, wg256, ch8  | 0.958     |
| one line, wg128, ch4   | 0.825     |
| two lines, wg128, ch4  | 0.845     |

Observations:
- Phase 2 is ~75% of the time; slot/atomic/barrier scheduling is the other ~25%.
- Two lines per thread is at parity with one line, not faster, on this GPU. Before making
  the phase-2 compute branch-free (both lines always computed, case bits only select), the
  two-line variant was 30% slower from warp divergence on `ix`.
- Chunk 8 is slower than chunk 4 in both variants.
- Workgroup size 128 vs 256 makes no difference.

## Phase 2 profile by ablation (same workload, `PROFILE=1` env var)

`merge0.wgsl` runs phase 2 only, with a compile-time `PROFILE` level enabling one stage at
a time and folding results into a register so nothing is eliminated. Timings were checked
for clock drift (early vs late quarter medians identical, 200 vs 1000 iterations
identical, untimed warm-up burst before the timed passes).

| stage (cumulative)                   | before | after  |
|--------------------------------------|--------|--------|
| read entry + line, line_setup        | 0.01   | 0.01   |
| + coset classification (64 cells)    | 0.13   | 0.13   |
| + FIFO fill (cell_area for hits)     | 0.17   | 0.17   |
| + compute_chunk consumption          | 0.45   | 0.31   |
| + store_result for heads             | 0.63   | 0.47   |
| full one-line merge                  | 0.83   | 0.66   |

"After" = compute_chunk hand-unrolled per coset row (cells written explicitly instead of
a dynamically indexed loop), even-odd resolve using a mask instead of `%`, and the path
flag loaded once per entry instead of per store. Consumption had cost 2.3x the
classification it consumes, which points at the returned `array<i32, CHUNK>` and nested
loops landing in stack memory; the remaining 0.14 ms there is still the largest phase-2
item and is mostly `row_consts` (two round-and-convert per row) and the FIFO shuffles.
Scheduling (slots, atomics, barriers) is 0.19 ms on top of phase 2 in both variants.

## What CHUNK controls

CHUNK is the number of coset-walk positions (pixels) each thread processes per barrier
pair. It sets the slot size per thread in workgroup memory (`WG_SIZE * CHUNK` i32), the
size of the `m0`, `m1`, `s` register arrays, the output store granularity (`CHUNK / 4`
u32 of packed alpha), and the barrier count: `2 * 4 * 16 / CHUNK` per thread (32 at
chunk 4, 16 at chunk 8). Chunk 4 is one coset row, so row constants are computed once
per chunk. Chunk 8 has been consistently slower, so barriers are not the bottleneck
here; register pressure from the larger arrays is the likelier cost.

## Chunk 1 and 2 (CHUNK below 4 supported)

Full merge, 480K entries, wg256, median ms:

| chunk | one line | two lines | phase-2 floor (compute only) | merge overhead |
|-------|----------|-----------|------------------------------|----------------|
| 1     | 0.503    | 0.492     | 0.430                        | 0.07           |
| 2     | 0.918    | 0.913     | 0.645                        | 0.27           |
| 4     | 0.661    | 0.664     | 0.468                        | 0.19           |
| 8     | 0.793    | 0.837     |                              |                |

Chunk 1 is fastest despite 128 barriers per thread, and chunk 2 is slowest. The ordering
tracks how the `array<i32, CHUNK>` values are handled, not the barrier count: at chunk 1
every array is a scalar, at chunk 4 the cells are written explicitly, and chunk 2 goes
through a dynamically indexed loop (stack). Barriers and workgroup atomics are cheap on
this GPU: the entire slot/atomic/barrier overhead at chunk 1 is 0.07 ms, 14% of the total.
Conclusion so far: on Apple silicon the design should minimize per-thread array state;
per-pixel processing with scalar registers beats batching pixels per barrier.

## Packed register stores instead of atomicOr

Sub-word chunks first used `atomicOr` into a zeroed alpha buffer. Since the same thread
stores every chunk of an output stream (case bits are fixed per thread), the resolved
bytes now accumulate in a register per stream (`pack`, or `pack_imm` / `pack_final` in
the two-line variant) and one plain store happens per word. No zero-initialization of
the output is needed. Chunk 1, 480K entries, median ms:

| variant   | atomicOr | packed register |
|-----------|----------|-----------------|
| one line  | 0.503    | 0.481           |
| two lines | 0.492    | 0.455           |

First configuration where two lines per thread beats one line (by 6%).

## Scale 8 (1600x1600) and 4x4 tiles

Tile size is a cargo feature (`--features tile4`, build with a separate target dir). At
4x4 the cosets are 4 cells each with at most 2 partial cells, chunks are 1 or 2, and a run's
output is 4 u32. Same scene, 16x replicated, median ms:

| tile | entries (natural) | runs   | one line ch1 | two lines ch1 | ns/entry (two) |
|------|-------------------|--------|--------------|---------------|----------------|
| 8x8  | 51K               | 26.5K  | 0.800        | 0.757         | 0.91           |
| 4x4  | 86K               | 57.5K  | 0.484        | 0.432         | 0.31           |

4x4 does 1.67x the entries but each costs a third as much, so the merge stage is 1.75x
faster for the same scene. Alpha output is also smaller (57.5K x 16 B vs 26.5K x 64 B).
Costs land elsewhere: 1.67x the tile/sort work, 2.2x the runs for strip allocation.
Mean run length at 4x4 is 1.5, at 8x8 1.9.

Natural (unreplicated) 1600x1600 tiger merge: 0.069 ms at 8x8 (latency bound, ~200
workgroups), 1.33 ns/entry vs 0.91 saturated.

## Phase 2 profile at 4x4 vs 8x8 (same scene, 16x replicated, chunk 1, stage deltas in ms)

| stage                         | 4x4 (1.37M entries) | 8x8 (823K entries) |
|-------------------------------|---------------------|--------------------|
| read entry + line, setup      | 0.024               | 0.017              |
| coset classification          | 0.117               | 0.198              |
| FIFO fill (cell_area on hits) | 0.079               | 0.074              |
| compute_chunk consumption     | 0.104               | 0.236              |
| store_result (heads)          | 0.090               | 0.207              |
| phase 2 total                 | 0.414               | 0.732              |
| full two-line merge           | 0.432               | 0.757              |

The FIFO fill is the only stage doing geometry-proportional work (area of cells the line
actually crosses), and it costs the same at both tile sizes, since the set of crossed
pixels does not depend on tiling. Everything else scales with pixels per entry
(classification, consumption, resolve) or with entry count (setup, store), i.e. it is
dense overhead: 82% of phase 2 at 4x4 and 90% at 8x8. The slot/atomic/barrier
scheduling is 0.02 ms in both. At chunk 2 (4x4) consumption and store are each ~0.1 ms
worse, the same array-codegen effect seen at 8x8.

## Dense vs sparse (coset) phase 2

`DENSE` (header const, `dense` on Variant, `DENSE=1` env for the profile run) evaluates
every cell with the full area formula, row span computed once per coset row, no
classification and no FIFO. Walk order and output layout are unchanged. Same scene
(1600x1600 tiger), 16x replicated, median ms:

| tile | variant            | sparse | dense | speedup |
|------|--------------------|--------|-------|---------|
| 8x8  | phase 2 only, ch1  | 0.732  | 0.434 | 1.69x   |
| 8x8  | one line, ch1      | 0.800  | 0.522 | 1.53x   |
| 8x8  | two lines, ch1     | 0.753  | 0.460 | 1.64x   |
| 8x8  | two lines, ch4     | 1.093  | 0.934 | 1.17x   |
| 4x4  | phase 2 only, ch1  | 0.414  | 0.211 | 1.96x   |
| 4x4  | one line, ch1      | 0.481  | 0.289 | 1.66x   |
| 4x4  | two lines, ch1     | 0.431  | 0.231 | 1.87x   |

Dense wins everywhere. The coset scheme saves at most 12 of 16 area evaluations per
coset but pays 16 FMA-and-compare classifications plus mask and FIFO bookkeeping, and
area_weight is only about a dozen operations including one division. Best configuration
so far: dense, 4x4, two lines per thread, chunk 1, at 0.231 ms for 1.37M entries
(0.17 ns/entry), 3.3x faster than the original sparse 8x8 chunk 4 two-line (0.757 ms,
or 1.09 ms before the codegen fixes). Output still within 1 LSB of the oracle.

## Row skipping in the dense path (tried and removed)

Branching around rows with `dy == 0` gave no gain (8x8 two-line ch1: 0.458 -> 0.464 ms;
ch4: 0.943 -> 1.059) although 64% of entry rows at 8x8 and 51% at 4x4 are empty. Rows
empty across an entire 32-lane SIMD group of consecutive entries are only 0.7% and
0.4%, so the group executes the covered path anyway. Per-lane sparseness inside the
lockstep walk is bounded by that group statistic unless lanes are grouped by coverage.

## Stripped-down shader (`shader/merge_simple.wgsl`, `Mode::Simple`)

Self-contained, one entry per thread, sparse cosets, one pixel per barrier pair, scalar
locals only, no includes or switches. Written to rule out compiler bad luck in the
modular version and to be read. Same scene, 16x replicated, median ms:

| tile | simple | modular one line ch1 sparse | one line ch1 dense |
|------|--------|-----------------------------|--------------------|
| 8x8  | 0.824  | 0.809                       | 0.521              |
| 4x4  | 0.481  | 0.485                       | 0.289              |

Identical within noise: the sparse scheme's cost is in its structure, not in how the
modular code compiled.

## Quadrant variant (`shader/merge_quadrant.wgsl`, `Mode::Quadrant`, 8x8 only)

8x8 entries in, merged as 4x4 quadrants with explicit load balancing. Each thread holds
one entry, computes the quadrants its line touches (the 4x4 tiler's rule inside the
tile) and the quadrant backdrop deltas relative to the tile's top-left corner: top-edge
crossings at x in [0,4) feed NE, tile left-edge crossings with y_edge <= 4 feed SW and
SE, midline crossings at x in [0,4) feed SE. Counts are packed 8 bits per quadrant and
exclusively scanned; items are quadrant-major so a run's items in one quadrant are
contiguous; each item finds its entry by binary search and its group head directly from
the scan at the run head. Up to 3 items per thread, dense 4x4 evaluation, two barriers
per cell; heads atomicExchange their slot to zero and resolve with backdrop + delta; run
heads write untouched quadrants as uniform values. Pieces split only at workgroup
boundaries. Output layout quadrant-major: q * 16 + ly * 4 + lx.

Items on the tiger: 85733, matching the 4x4 tiler's 85734 entries.

Two exactness lessons, both already present in the tiler: decisions at x = 0 must be
derived from the tiler's left-edge flag rather than recomputed (a line through the tile
corner computed to x = -1e-6 and lost its NE delta), and x on the line must be taken from
the endpoint when the clip y equals an endpoint y (a vertex at exactly x = 4.000
interpolated to 3.99999 and gained a spurious left-edge term).

Same scene, 16x replicated, median ms (correct within 1 LSB on all but 38 runs):

| configuration                         | total | setup only |
|---------------------------------------|-------|------------|
| quadrant, 256 entries, 3 slots/thread | 0.498 | 0.252      |
| quadrant, 128 entries, 2 slots/thread | 0.593 | 0.298      |
| dense 8x8 one line ch1 (reference)    | 0.521 |            |
| dense 4x4 one line ch1 (same items)   | 0.289 |            |

The cell loop itself costs about 0.25 ms, in line with the plain 4x4 merge, so the per-item
work is as efficient as hoped. The setup (two 8-round scans, geometry, delta atomics,
three binary searches and geometry rebuilds per thread, uniform-quadrant writes) costs as
much again and erases the gain. Candidates: scatter (entry, quadrant) into a shared item
table instead of binary searching, run both scans in one loop sharing barriers, cache the
tile-local line in shared memory, subgroup scans.

## MSAA 16 (`src/msaa.rs`, `shader/msaa1.wgsl`, run with `MSAA=1`)

Payload per (line, pixel): 16-bit mask of samples right of the line within the segment's
y-range (sign by direction) plus the y_edge term, so each sample gets {-1, 0, +1}. Samples
follow the D3D11 16x pattern as in Vello (sorted by y, so y-range tests are contiguous bit
ranges). Half-plane masks come from Vello's `mask_lut_16` (64 translations x 32 slopes per
sign, 8 KB), with fill_path_ms's walk parameters evaluated directly per cell via
`sub_ix = (y - y0i) + x_sign * (x - x0i)`. Accumulation: two slot words per pixel with
4-bit lanes biased at 8; adders atomicAdd the signed packed payload as an integer (lanes
stay in [1, 15] for up to 7 adders, so the multi-digit sum is exact); the head keeps its own
payload in registers and adds it at resolve, hence pieces of head + 7. Output: 16-bit mask
per pixel, two per u32, coset-major; partial pieces write 16 i32 per pixel.

Correctness: the CPU reference matches the exact f64 sample-point oracle on 99.7% of
pixels and 99.98% of samples (LUT quantization, max 4 samples off in a pixel). GPU
inline-LUT and FIFO modes match the CPU reference bit for bit; the direct 16-compare
mode matches the oracle instead. Rounding: WGSL round() is half-to-even, mirrored with
`round_ties_even` on the CPU.

Resolve: SWAR. Expected lane words from backdrop and own payload (saturating cases for
backdrops outside [-6, 7]: expected values outside [1, 15] never match, encoded as e = 0 or
a forced bit), xor against the slot words, or-reduce within nibbles, compress to the
16-bit mask; even-odd uses lane parity. Partial pieces write (w0, w1, own, backdrop) per
pixel and the CPU decodes. The first version had a 16-iteration per-sample loop: 78% of
the time, and even as a rarely taken slow path its presence cost 0.8 ms (register
pressure), so there is no loop anywhere in the resolve now.

Timing, 1600x1600 tiger, 16x replicated, chunk 1, median ms:

| variant                              | 8x8 (823K entries) | 4x4 (1.37M entries) |
|--------------------------------------|--------------------|---------------------|
| one line, inline LUT                 | 1.04               | 0.52                |
| one line, coset FIFO of LUT masks    | 1.20               | 0.67                |
| one line, direct 16 samples          | 3.49               | 1.50                |
| two lines, inline LUT                | 1.21               | 0.56                |
| two lines, direct 16 samples         | 3.70               | 1.59                |
| dense area version, same scene       | 0.52               | 0.23                |

Ablation, 8x8 one line inline LUT (`MPROF` env: 2 = payload compute only, 1 = + merge,
0 = full): 0.60 / 0.79 / 1.04 ms. Payload compute is now the largest part; merge
(slots, atomics, barriers) 0.19 as in the area version; resolve 0.25.

Branch vs always-gather (`alwayslut`: compute the LUT index and gather for every cell,
select by classification afterwards; same results, no divergence, 4-8x the gathers):
8x8 1.044 -> 1.012 ms, 4x4 0.522 -> 0.507. The unconditional gather is slightly faster, so
the sparse gathers were not buying anything: the 8 KB LUT is cache-resident and the branch
costs more than the extra loads. The 3x over direct evaluation comes from one FMA plus a
lookup replacing 16 FMA-and-compares per cell, not from skipping cells.

Payload-stage ablation (always-gather mode, 8x8, `MABL` env bits: 1 = no LUT gather,
2 = no classification arithmetic; `MPROF=4` = no lane spreads), median ms:

| removed              | full  | compute only (MPROF=2) |
|----------------------|-------|------------------------|
| nothing              | 1.018 | 0.578                  |
| LUT gather           | 0.968 | 0.517                  |
| classification       | 1.010 | 0.555                  |
| gather + classify    | 0.947 | 0.504                  |
| lane spreads         | 1.012 |                        |

Gather ~5%, classification 1-2%, spreads <1%. The remaining ~0.5 ms of payload is a broad
per-cell floor of small arithmetic (LUT index, row masks, selects), close to the dense
area version's compute-only floor (0.43). The MSAA-vs-area gap is in merge + resolve
(0.44 vs 0.05 ms): two atomics per cell instead of one, and the SWAR resolve on heads
(0.25). Passing the head's four payload spreads into the resolve instead of recomputing
them made one-line 3% slower (four more live registers across the barriers) and two-line
9% faster; reverted, two-line still behind.

Observations:
- MSAA costs 2.0x the dense area version at 8x8 and 2.3x at 4x4; the first version was 7x.
- The LUT beats direct 16-sample evaluation by 2.9-3.4x.
- Classifying each cell and looking the LUT up inline beats the FIFO precompute by 15-28%.
- Two lines per thread is 8-16% slower than one line here, as for the area version
  before chunk 1 + packed stores; the per-thread state doubles.
- 4x4 is 2x faster than 8x8 for the same scene, as for the area version.
- Even-odd (`EVENODD=1` marks every path even-odd): all area and MSAA variants stay
  within 1 LSB / bit-exact against their references.

## MSAA quadrant variant (`shader/msaa_quadrant.wgsl`, `Mode::MsaaQuadrant`)

The quadrant expansion with the MSAA payload. (run, quadrant) groups form pieces of 8
items (head + 7 adders, as the 4-bit lanes require); a run resolves in place only if it
lies in one workgroup and every group fits one piece, otherwise each workgroup segment
writes sum over quadrants of max(1, ceil(count_q / 8)) compact partial records, untouched
quadrants included. The CPU allocation mirrors the expansion (`allocate_quadrant_msaa`)
and folds records (`fold_quadrant_partials`). Item count per (run head, quadrant) is
accumulated in shared memory for the record layout.

Correctness: 35 of 26551 runs differ from the 8x8 reference by 47 single samples, all
ties where a y_edge or segment-end y lands exactly on a sample row boundary (y = k/16 +
1/32) and the two decompositions round differently; the 8x8 reference itself resolves
such ties inconsistently between its crossing and y_edge masks (one spurious sample).

Timing, 16x replicated, inline LUT, median ms:

| variant                       | total | setup only | cell loop |
|-------------------------------|-------|------------|-----------|
| MSAA one line 8x8             | 1.040 |            |           |
| MSAA quadrant 8x8             | 0.986 | 0.343      | 0.64      |
| MSAA one line 4x4 (same items)| 0.52  |            |           |

Only a 5% win over one-line 8x8. Setup is 0.34 (vs 0.25 for the area quadrant: the MSAA
Setup struct is ~17 fields and is built three times per thread), and the cell loop is
0.64 against roughly 0.45 for the 4x4 one-line loop: three items' worth of Setup, RowMasks
and payload registers per thread. Both halves are register-pressure costs, not barrier or
atomic costs.

## Ideas not yet tried

- Phase 2 is the target: `cell_area` per hit, 16 classification FMAs per coset, and
  `row_consts` per row. Profile with Metal's shader profiler.
- Subgroup prefix-max for the head scan instead of the shared-memory scan.
- Chunk 16 (whole coset per barrier pair) needs more than 16 KB of workgroup memory at
  256 threads; request a higher limit or use wg128.
- Side-by-side with vello_cpu / vello_gpu output (the oracle currently plays that role).
- Even-odd is implemented but untested (pico_svg does not parse fill-rule).
