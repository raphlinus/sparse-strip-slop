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

## Ideas not yet tried

- Phase 2 is the target: `cell_area` per hit, 16 classification FMAs per coset, and
  `row_consts` per row. Profile with Metal's shader profiler.
- Subgroup prefix-max for the head scan instead of the shared-memory scan.
- Chunk 16 (whole coset per barrier pair) needs more than 16 KB of workgroup memory at
  256 threads; request a higher limit or use wg128.
- Side-by-side with vello_cpu / vello_gpu output (the oracle currently plays that role).
- Even-odd is implemented but untested (pico_svg does not parse fill-rule).
