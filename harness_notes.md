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
| one line, wg256, ch4   | 0.831     |
| two lines, wg256, ch4  | 0.855     |
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

## Ideas not yet tried

- Phase 2 is the target: `cell_area` per hit, 16 classification FMAs per coset, and
  `row_consts` per row. Profile with Metal's shader profiler.
- Subgroup prefix-max for the head scan instead of the shared-memory scan.
- Chunk 16 (whole coset per barrier pair) needs more than 16 KB of workgroup memory at
  256 threads; request a higher limit or use wg128.
- Side-by-side with vello_cpu / vello_gpu output (the oracle currently plays that role).
- Even-odd is implemented but untested (pico_svg does not parse fill-rule).
