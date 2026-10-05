mod composite;
mod gpu;
mod msaa;
mod reference;
mod scene;
mod tiler;

use std::collections::HashMap;

use composite::{Image, color_f32};
use gpu::{Gpu, Mode, Variant};
use reference::{FIXED_ONE, PathTruth, classify_hits, quadrants_touched, resolve, run_winding};
use scene::Scene;
use tiler::{LINE_LEFT_EDGE, PIXELS, Run, RunOutput, TILE, Tiled};

/// Resolved alpha per run, row-major within the tile.
type RunAlphas = Vec<[u8; PIXELS]>;

fn reference_windings(scene: &Scene, tiled: &Tiled) -> Vec<[i32; PIXELS]> {
    tiled.runs.iter().map(|r| run_winding(scene, tiled, r)).collect()
}

fn resolve_all(scene: &Scene, tiled: &Tiled, windings: &[[i32; PIXELS]]) -> RunAlphas {
    tiled
        .runs
        .iter()
        .zip(windings)
        .map(|(r, w)| {
            let eo = scene.paths[r.path as usize].even_odd;
            std::array::from_fn(|i| resolve(w[i], eo))
        })
        .collect()
}

/// Convert GPU output to per-run windings (partials folded; `None` for resolved runs) and alphas.
fn gpu_run_results(
    scene: &Scene,
    tiled: &Tiled,
    result: &gpu::GpuResult,
    variant: &Variant,
) -> (Vec<Option<[i32; PIXELS]>>, RunAlphas) {
    let mut windings = Vec::with_capacity(tiled.runs.len());
    let mut alphas = Vec::with_capacity(tiled.runs.len());
    for r in &tiled.runs {
        let eo = scene.paths[r.path as usize].even_odd;
        match r.output {
            RunOutput::Alpha(o) => {
                let words = &result.alpha[o as usize * (PIXELS / 4)..(o as usize + 1) * (PIXELS / 4)];
                let mut a = [0u8; PIXELS];
                for y in 0..TILE {
                    for x in 0..TILE {
                        let ci = variant.layout_index(x, y);
                        a[(y * TILE + x) as usize] = (words[ci / 4] >> (8 * (ci % 4))) as u8;
                    }
                }
                windings.push(None);
                alphas.push(a);
            }
            RunOutput::Partials { first, count } => {
                let mut w = [0i32; PIXELS];
                for p in first..first + count {
                    let vals = &result.partials[p as usize * PIXELS..(p as usize + 1) * PIXELS];
                    for y in 0..TILE {
                        for x in 0..TILE {
                            w[(y * TILE + x) as usize] += vals[variant.layout_index(x, y)];
                        }
                    }
                }
                windings.push(Some(w));
                alphas.push(std::array::from_fn(|i| resolve(w[i], eo)));
            }
        }
    }
    (windings, alphas)
}

struct TruthStats {
    pixels: usize,
    max_diff: u8,
    over1: usize,
    over2: usize,
}

/// Composite one set of run alphas and compare against the dense truth.
fn composite_and_check(scene: &Scene, tiled: &Tiled, alphas: &RunAlphas, check: bool) -> (Image, TruthStats) {
    let mut img = Image::new(scene.width, scene.height);
    let mut stats = TruthStats {
        pixels: 0,
        max_diff: 0,
        over1: 0,
        over2: 0,
    };
    let mut run_ix = 0;
    for (path_ix, path) in scene.paths.iter().enumerate() {
        let color = color_f32(path.color);
        let bd = &tiled.backdrops[path_ix];
        let x0 = bd.tx0 * TILE;
        let y0 = bd.ty0 * TILE;
        let w = (bd.w * TILE).min(scene.width.saturating_sub(x0));
        let h = (bd.h * TILE).min(scene.height.saturating_sub(y0));
        let truth = check.then(|| {
            let lines = &scene.lines[path.line_start as usize..path.line_end as usize];
            PathTruth::compute(lines, x0, y0, w, h)
        });
        let mut by_tile: HashMap<(u32, u32), usize> = HashMap::new();
        while run_ix < tiled.runs.len() && tiled.runs[run_ix].path == path_ix as u32 {
            let run = &tiled.runs[run_ix];
            by_tile.insert((run.tx, run.ty), run_ix);
            run_ix += 1;
        }
        for ty in bd.ty0..bd.ty0 + bd.h {
            for tx in bd.tx0..bd.tx0 + bd.w {
                let alpha = match by_tile.get(&(tx, ty)) {
                    Some(&ri) => alphas[ri],
                    None => {
                        let a = resolve(bd.get(tx, ty) * FIXED_ONE, path.even_odd);
                        if a == 0 {
                            continue;
                        }
                        [a; PIXELS]
                    }
                };
                img.blend_tile(tx, ty, &alpha, color);
                if let Some(truth) = &truth {
                    for y in 0..TILE {
                        for x in 0..TILE {
                            let px = tx * TILE + x;
                            let py = ty * TILE + y;
                            if px >= scene.width || py >= scene.height {
                                continue;
                            }
                            let t = truth.alpha(px, py, path.even_odd);
                            let d = t.abs_diff(alpha[(y * TILE + x) as usize]);
                            stats.max_diff = stats.max_diff.max(d);
                            stats.over1 += (d > 1) as usize;
                            stats.over2 += (d > 2) as usize;
                            stats.pixels += 1;
                        }
                    }
                }
            }
        }
    }
    (img, stats)
}

fn print_tiling_stats(tiled: &Tiled, scene: &Scene) {
    let n_left = tiled.entries.iter().filter(|e| e.line & LINE_LEFT_EDGE != 0 && e.key != tiler::SENTINEL_KEY).count();
    let mut hist = [0usize; 10];
    let mut split_runs = 0usize;
    for r in &tiled.runs {
        let len = (r.entry_end - r.entry_start) as usize;
        hist[len.min(9)] += 1;
        if matches!(r.output, RunOutput::Partials { .. }) {
            split_runs += 1;
        }
    }
    println!(
        "  {} entries (+{} padding, {} with left-edge flag), {} runs, {} alpha outputs, {} partial pieces in {} split runs",
        tiled.runs.last().map_or(0, |r| r.entry_end),
        tiled.entries.len() - tiled.runs.last().map_or(0, |r| r.entry_end as usize),
        n_left,
        tiled.runs.len(),
        tiled.n_alpha - 1,
        tiled.n_partials,
        split_runs,
    );
    println!("  run length histogram (1..8, 9+): {:?}", &hist[1..]);
    let mut max_hits = 0;
    let mut hit_hist = [0usize; PIXELS / 4 + 1];
    for e in &tiled.entries {
        if e.key == tiler::SENTINEL_KEY {
            continue;
        }
        let tx = e.key & 0x3ff;
        let ty = (e.key >> 10) & 0x3ff;
        let line = &scene.lines[(e.line & !LINE_LEFT_EDGE) as usize];
        for m in classify_hits(line, tx, ty) {
            let n = m.count_ones() as usize;
            hit_hist[n] += 1;
            max_hits = max_hits.max(n);
        }
    }
    if TILE == 8 {
        let mut items = 0usize;
        let mut max_q = 0u32;
        let mut hist = [0usize; 5];
        for e in &tiled.entries {
            if e.key == tiler::SENTINEL_KEY {
                continue;
            }
            let line = &scene.lines[(e.line & !LINE_LEFT_EDGE) as usize];
            let n = quadrants_touched(line, e.key & 0x3ff, (e.key >> 10) & 0x3ff).count_ones();
            items += n as usize;
            max_q = max_q.max(n);
            hist[n as usize] += 1;
        }
        println!(
            "  quadrant items: {} (max {} per entry, histogram by count {:?})",
            items, max_q, hist
        );
    }
    println!(
        "  partial cells per (entry, coset): max {}, histogram {:?}",
        max_hits,
        &hit_hist[..=max_hits]
    );
}

fn main() {
    if std::env::var("MSAA").is_ok() {
        msaa_main();
        return;
    }
    let args: Vec<String> = std::env::args().collect();
    let svg = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/Users/raphl/gh/vello/assets/Ghostscript_Tiger.svg".into());
    let scale: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let out_dir = args.get(3).cloned().unwrap_or_else(|| "out".into());
    let iterations: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let replicate: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
    std::fs::create_dir_all(&out_dir).unwrap();

    let t0 = std::time::Instant::now();
    let scene = scene::load_svg(&svg, scale);
    println!(
        "scene {}x{}: {} paths, {} lines (flatten {:.1?})",
        scene.width,
        scene.height,
        scene.paths.len(),
        scene.lines.len(),
        t0.elapsed()
    );
    let path_flags: Vec<u32> = (0..replicate)
        .flat_map(|_| scene.paths.iter().map(|p| p.even_odd as u32))
        .collect();
    let mut lines_padded = scene.lines.clone();
    lines_padded.push([0.0; 4]);

    let gpu = Gpu::new();
    if !gpu.has_timestamps {
        println!("note: adapter lacks timestamp queries; no GPU timings");
    }

    let profile_only = std::env::var("PROFILE").is_ok();
    let profile_chunk: u32 = std::env::var("CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    let profile_dense = std::env::var("DENSE").is_ok();
    let v = |mode: Mode, chunk: u32, dense: bool| Variant { mode, profile: 5, dense, quad_slots: 3, cell_mode: 0, wg_size: 256, chunk };
    let variants: Vec<Variant> = if profile_only {
        (0..=5)
            .map(|profile| Variant { mode: Mode::ComputeOnly, profile, dense: profile_dense, quad_slots: 3, cell_mode: 0, wg_size: 256, chunk: profile_chunk })
            .collect()
    } else {
        vec![
            v(Mode::Quadrant, 1, true),
            Variant { quad_slots: 2, ..v(Mode::Quadrant, 1, true) },
            v(Mode::OneLine, 1, true),
            v(Mode::TwoLines, 1, true),
            v(Mode::Simple, 1, false),
        ]
    };
    // A chunk must be a sub-row (<= TILE/2 cells) or whole rows (multiple of 4 at TILE 8).
    let variants: Vec<Variant> = variants
        .into_iter()
        .filter(|v| (v.chunk <= TILE / 2 || (TILE == 8 && v.chunk % 4 == 0)) && (v.mode != Mode::Quadrant || TILE == 8))
        .collect();
    let mut truth_written = false;
    for (vi, variant) in variants.iter().enumerate() {
        println!("== {} ==", variant.name());
        let t1 = std::time::Instant::now();
        let mut tiled = tiler::tile(&scene, variant.entries_per_wg(), variant.piece_size());
        let t_tile = t1.elapsed();
        if vi == 0 {
            print_tiling_stats(&tiled, &scene);
        }
        if replicate > 1 {
            tiler::replicate(&mut tiled, replicate, scene.paths.len() as u32);
        }
        let t2 = std::time::Instant::now();
        let ref_w = reference_windings(&scene, &tiled);
        let ref_alpha = resolve_all(&scene, &tiled, &ref_w);
        let t_ref = t2.elapsed();
        println!("  cpu: tile {:.1?}, reference merge {:.1?}", t_tile, t_ref);

        let result = gpu.run(
            *variant,
            &lines_padded,
            &tiled.entries,
            &path_flags,
            tiled.n_alpha,
            tiled.n_partials,
            iterations,
            None,
            None,
        );
        if !result.times_ms.is_empty() {
            // Skip warm-up iterations; the first timestamp pair is also unreliable on Metal.
            let skip = 1;
            let n_zero = result.times_ms[skip..].iter().filter(|&&t| t == 0.0).count();
            let all: Vec<f64> = result.times_ms[skip..].iter().copied().filter(|&t| t > 0.0).collect();
            let median = |v: &[f64]| {
                let mut v = v.to_vec();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v[v.len() / 2]
            };
            let q = all.len() / 4;
            println!(
                "  clock drift check: early-quarter median {:.3} ms, late-quarter median {:.3} ms",
                median(&all[..q.max(1)]),
                median(&all[all.len() - q.max(1)..])
            );
            let mut t = all;
            if n_zero > 0 {
                println!("  ({} zero-length timestamp pairs dropped)", n_zero);
            }
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "  gpu: {} entries, {} workgroups, min {:.3} ms, median {:.3} ms, max {:.3} ms over {} iterations ({:.2} ns/entry)",
                tiled.entries.len(),
                tiled.entries.len() / variant.entries_per_wg(),
                t[0],
                t[t.len() / 2],
                t[t.len() - 1],
                t.len(),
                t[t.len() / 2] * 1e6 / tiled.entries.len() as f64
            );
        }

        if variant.mode == Mode::ComputeOnly {
            println!("  (diagnostic variant, output not checked)");
            continue;
        }
        // Compare GPU with the CPU reference, run by run.
        let (gpu_w, gpu_alpha) = gpu_run_results(&scene, &tiled, &result, variant);
        let mut alpha_mismatch_runs = 0usize;
        let mut alpha_max = 0u8;
        let mut alpha_pixels_off = 0usize;
        let mut winding_max = 0i32;
        let mut worst: Option<(usize, u8)> = None;
        for ri in 0..tiled.runs.len() {
            let mut bad = 0u8;
            for i in 0..PIXELS {
                let d = gpu_alpha[ri][i].abs_diff(ref_alpha[ri][i]);
                if d != 0 {
                    alpha_pixels_off += 1;
                    alpha_max = alpha_max.max(d);
                    bad = bad.max(d);
                }
                if let Some(w) = gpu_w[ri] {
                    winding_max = winding_max.max((w[i] - ref_w[ri][i]).abs());
                }
            }
            if bad > 0 {
                alpha_mismatch_runs += 1;
                if worst.is_none_or(|(_, b)| bad > b) {
                    worst = Some((ri, bad));
                }
            }
        }
        let first_bad = worst.map(|(ri, _)| ri);
        println!(
            "  gpu vs reference: {} of {} runs differ ({} pixels, max alpha diff {}); max partial winding diff {} (fixed units)",
            alpha_mismatch_runs,
            tiled.runs.len(),
            alpha_pixels_off,
            alpha_max,
            winding_max
        );
        if let Some(ri) = first_bad {
            let r = &tiled.runs[ri];
            println!(
                "  worst mismatch: run {} path {} tile ({}, {}) entries {}..{} output {:?}",
                ri, r.path, r.tx, r.ty, r.entry_start, r.entry_end, r.output
            );
            for e in &tiled.entries[r.entry_start as usize..r.entry_end as usize] {
                let l = scene.lines[(e.line & !LINE_LEFT_EDGE) as usize];
                let ox = r.tx as f32 * TILE as f32;
                let oy = r.ty as f32 * TILE as f32;
                println!(
                    "    entry: line {} local ({:.3}, {:.3}) -> ({:.3}, {:.3}) left_edge {} backdrop {} out {:#x}",
                    e.line & !LINE_LEFT_EDGE,
                    l[0] - ox, l[1] - oy, l[2] - ox, l[3] - oy,
                    e.line & LINE_LEFT_EDGE != 0,
                    e.backdrop,
                    e.out
                );
            }
            let t = TILE as usize;
            for y in 0..t {
                let row: Vec<String> = (0..t)
                    .map(|x| format!("{:3}/{:3}", gpu_alpha[ri][y * t + x], ref_alpha[ri][y * t + x]))
                    .collect();
                println!("    {}", row.join(" "));
            }
        }

        let (img_gpu, stats) = composite_and_check(&scene, &tiled, &gpu_alpha, true);
        println!(
            "  gpu vs truth over {} pixels: max alpha diff {}, {} pixels differ by >1, {} by >2",
            stats.pixels, stats.max_diff, stats.over1, stats.over2
        );
        img_gpu.write_png(&format!("{out_dir}/gpu_{}.png", variant.name()));
        if !truth_written {
            let (img_ref, stats) = composite_and_check(&scene, &tiled, &ref_alpha, true);
            println!(
                "  reference vs truth: max alpha diff {}, {} pixels differ by >1",
                stats.max_diff, stats.over1
            );
            img_ref.write_png(&format!("{out_dir}/reference.png"));
            truth_written = true;
        }
    }
}

/// Per workgroup segment of a run: item counts per quadrant (mirrors the shader).
fn segment_qcounts(scene: &Scene, tiled: &Tiled, start: usize, end: usize) -> [u32; 4] {
    let mut cnt = [0u32; 4];
    for e in &tiled.entries[start..end] {
        let line = &scene.lines[(e.line & !LINE_LEFT_EDGE) as usize];
        let m = quadrants_touched(line, e.key & 0x3ff, (e.key >> 10) & 0x3ff);
        for q in 0..4 {
            cnt[q] += (m >> q) & 1;
        }
    }
    cnt
}

/// Records of a segment: sum over quadrants of max(1, ceil(count / 8)).
fn segment_records(cnt: &[u32; 4]) -> u32 {
    cnt.iter().map(|&c| c.div_ceil(8).max(1)).sum()
}

/// Output allocation for the MSAA quadrant variant: a run is resolved in place only if it
/// lies in one workgroup and every (run, quadrant) group has at most 8 items; otherwise
/// each workgroup segment gets `segment_records` partial records.
fn allocate_quadrant_msaa(scene: &Scene, tiled: &mut Tiled, entries_per_wg: usize) {
    let mut n_alpha = 0u32;
    let mut n_partials = 0u32;
    let n_runs = tiled.runs.len();
    for ri in 0..n_runs {
        let (start, end) = (tiled.runs[ri].entry_start as usize, tiled.runs[ri].entry_end as usize);
        let mut seg_starts = vec![start];
        let mut k = start + 1;
        while k < end {
            if k % entries_per_wg == 0 {
                seg_starts.push(k);
            }
            k += 1;
        }
        let one_segment = seg_starts.len() == 1;
        let cnt0 = segment_qcounts(scene, tiled, start, end);
        for e in &mut tiled.entries[start..end] {
            e.out = 0;
        }
        if one_segment && cnt0.iter().all(|&c| c <= 8) {
            tiled.entries[start].out = n_alpha;
            tiled.runs[ri].output = RunOutput::Alpha(n_alpha);
            n_alpha += 1;
        } else {
            let first = n_partials;
            for (si, &ss) in seg_starts.iter().enumerate() {
                let se = seg_starts.get(si + 1).copied().unwrap_or(end);
                let cnt = segment_qcounts(scene, tiled, ss, se);
                tiled.entries[ss].out = tiler::OUT_PARTIAL | n_partials;
                n_partials += segment_records(&cnt);
            }
            tiled.runs[ri].output = RunOutput::Partials { first, count: n_partials - first };
        }
    }
    // Padding entries write to a dummy alpha slot.
    let real_end = tiled.runs.last().map_or(0, |r| r.entry_end as usize);
    for e in &mut tiled.entries[real_end..] {
        e.out = n_alpha;
    }
    tiled.n_alpha = n_alpha + 1;
    tiled.n_partials = n_partials;
}

/// Fold the quadrant variant's partial records of a run into per-sample windings.
fn fold_quadrant_partials(
    scene: &Scene,
    tiled: &Tiled,
    run: &Run,
    first: u32,
    partials: &[i32],
    entries_per_wg: usize,
) -> Vec<[i32; 16]> {
    let mut w = vec![[0i32; 16]; PIXELS];
    let (start, end) = (run.entry_start as usize, run.entry_end as usize);
    let mut rec = first as usize;
    let mut ss = start;
    while ss < end {
        let mut se = ss + 1;
        while se < end && se % entries_per_wg != 0 {
            se += 1;
        }
        let cnt = segment_qcounts(scene, tiled, ss, se);
        for q in 0..4u32 {
            let pieces = cnt[q as usize].div_ceil(8).max(1) as usize;
            for _ in 0..pieces {
                for p in 0..16u32 {
                    let base = (rec * 16 + p as usize) * 4;
                    let w0 = partials[base] as u32;
                    let w1 = partials[base + 1] as u32;
                    let own = partials[base + 2] as u32;
                    let bd = partials[base + 3];
                    let x = 4 * (q & 1) + (p & 3);
                    let y = 4 * (q >> 1) + (p >> 2);
                    let pix = (y * TILE + x) as usize;
                    for sidx in 0..16 {
                        let word = if sidx < 8 { w0 } else { w1 };
                        let lane = ((word >> (4 * (sidx & 7))) & 0xf) as i32 - 8;
                        let o = ((own >> sidx) & 1) as i32 - ((own >> (16 + sidx)) & 1) as i32;
                        w[pix][sidx] += lane + o + bd;
                    }
                }
                rec += 1;
            }
        }
        ss = se;
    }
    w
}

/// Flat item list for the MSAA flat variant: (entry, quadrant, run first entry, 0),
/// quadrant-major within each run, padded to whole workgroups with sentinel items.
fn build_flat_items(scene: &Scene, tiled: &Tiled, wg: usize) -> Vec<[u32; 4]> {
    let mut items = Vec::new();
    for r in &tiled.runs {
        let (start, end) = (r.entry_start as usize, r.entry_end as usize);
        let masks: Vec<u32> = tiled.entries[start..end]
            .iter()
            .map(|e| {
                let line = &scene.lines[(e.line & !LINE_LEFT_EDGE) as usize];
                quadrants_touched(line, e.key & 0x3ff, (e.key >> 10) & 0x3ff)
            })
            .collect();
        let before = items.len();
        for q in 0..4u32 {
            for (k, m) in masks.iter().enumerate() {
                if (m >> q) & 1 != 0 {
                    items.push([(start + k) as u32, q, start as u32, 0]);
                }
            }
        }
        if items.len() == before {
            // A run whose lines touch no quadrant (corner grazes) still needs a thread to
            // write its backdrop-only output: one item on quadrant 0 with a zero payload.
            items.push([start as u32, 0, start as u32, 0]);
        }
    }
    let real_end = tiled.runs.last().map_or(0, |r| r.entry_end);
    let padded = items.len().div_ceil(wg) * wg;
    items.resize(padded, [real_end, 0, real_end, 0]);
    items
}

/// Item counts per quadrant of a run within item positions [a, b).
fn flat_segment_qcounts(items: &[[u32; 4]], a: usize, b: usize) -> [u32; 4] {
    let mut cnt = [0u32; 4];
    for it in &items[a..b] {
        cnt[it[1] as usize] += 1;
    }
    cnt
}

/// Output allocation for the flat variant: segments are workgroup ranges in item space.
/// Returns the item range of each run.
fn allocate_flat_msaa(items: &mut [[u32; 4]], tiled: &mut Tiled, wg: usize) -> Vec<(usize, usize)> {
    let mut n_alpha = 0u32;
    let mut n_partials = 0u32;
    let mut ranges = Vec::with_capacity(tiled.runs.len());
    let mut pos = 0usize;
    let n_runs = tiled.runs.len();
    for ri in 0..n_runs {
        let start_e = tiled.runs[ri].entry_start as usize;
        let end_e = tiled.runs[ri].entry_end as usize;
        let a = pos;
        while pos < items.len() && (items[pos][0] as usize) < end_e && (items[pos][0] as usize) >= start_e {
            pos += 1;
        }
        let b = pos;
        ranges.push((a, b));
        for e in &mut tiled.entries[start_e..end_e] {
            e.out = 0;
        }
        let one_segment = b == a || (a / wg == (b - 1) / wg);
        let cnt = flat_segment_qcounts(items, a, b);
        if one_segment && cnt.iter().all(|&c| c <= 8) {
            tiled.entries[start_e].out = n_alpha;
            tiled.runs[ri].output = RunOutput::Alpha(n_alpha);
            n_alpha += 1;
        } else {
            // Each workgroup segment gets a contiguous block of records; its base is stored
            // in the 4th field of the segment's first item, which is the run head item of
            // that workgroup, where the shader reads it.
            let first = n_partials;
            let mut s = a;
            while s < b {
                let e = ((s / wg) + 1) * wg;
                let se = e.min(b);
                let c = flat_segment_qcounts(items, s, se);
                items[s][3] = n_partials;
                n_partials += segment_records(&c);
                s = se;
            }
            tiled.entries[start_e].out = tiler::OUT_PARTIAL | first;
            tiled.runs[ri].output = RunOutput::Partials { first, count: n_partials - first };
        }
    }
    let real_end = tiled.runs.last().map_or(0, |r| r.entry_end as usize);
    for e in &mut tiled.entries[real_end..] {
        e.out = n_alpha;
    }
    tiled.n_alpha = n_alpha + 1;
    tiled.n_partials = n_partials;
    ranges
}

/// Fold the flat variant's partial records of a run (item range [a, b)).
fn fold_flat_partials(items: &[[u32; 4]], a: usize, b: usize, first: u32, partials: &[i32], wg: usize) -> Vec<[i32; 16]> {
    let mut w = vec![[0i32; 16]; PIXELS];
    let mut rec = first as usize;
    let mut s = a;
    while s < b {
        let e = ((s / wg) + 1) * wg;
        let se = e.min(b);
        let cnt = flat_segment_qcounts(items, s, se);
        for q in 0..4u32 {
            let pieces = cnt[q as usize].div_ceil(8).max(1) as usize;
            for _ in 0..pieces {
                for p in 0..16u32 {
                    let base = (rec * 16 + p as usize) * 4;
                    let w0 = partials[base] as u32;
                    let w1 = partials[base + 1] as u32;
                    let own = partials[base + 2] as u32;
                    let bd = partials[base + 3];
                    let x = 4 * (q & 1) + (p & 3);
                    let y = 4 * (q >> 1) + (p >> 2);
                    let pix = (y * TILE + x) as usize;
                    for sidx in 0..16 {
                        let word = if sidx < 8 { w0 } else { w1 };
                        let lane = ((word >> (4 * (sidx & 7))) & 0xf) as i32 - 8;
                        let o = ((own >> sidx) & 1) as i32 - ((own >> (16 + sidx)) & 1) as i32;
                        w[pix][sidx] += lane + o + bd;
                    }
                }
                rec += 1;
            }
        }
        s = se;
    }
    w
}

/// Host mirror of the GPU expansion: item/run/record counts for buffer sizing, and per-run
/// layout (item base, quadrant counts, partial flag) for folding results.
struct RunLayout {
    base: usize,
    cnt: [u32; 4],
    partial: bool,
}

fn mirror_expansion(scene: &Scene, entries: &[tiler::Entry], wg: usize) -> (usize, usize, usize, Vec<RunLayout>) {
    let dummy_line = scene.lines.len() as u32;
    let mut layouts = Vec::new();
    let mut n_items = 0usize;
    let mut n_records = 0usize;
    let mut n_runs = 0usize;
    let mut i = 0;
    while i < entries.len() {
        let key = entries[i].key;
        let mut j = i;
        let mut cnt = [0u32; 4];
        while j < entries.len() && entries[j].key == key {
            let e = &entries[j];
            let li = e.line & 0x0fff_ffff;
            if li < dummy_line {
                let line = &scene.lines[li as usize];
                let m = quadrants_touched(line, e.key & 0x3ff, (e.key >> 10) & 0x3ff);
                for q in 0..4 {
                    cnt[q] += (m >> q) & 1;
                }
            }
            j += 1;
        }
        let base = n_items;
        let mut g = base;
        let mut partial = false;
        for q in 0..4 {
            let c = cnt[q] as usize;
            if c > 8 || (c > 0 && g / wg != (g + c - 1) / wg) {
                partial = true;
            }
            g += c;
        }
        if partial {
            let mut gq = base;
            for q in 0..4 {
                let c = cnt[q] as usize;
                if c == 0 {
                    n_records += 1;
                } else {
                    let mut p = gq;
                    while p < gq + c {
                        n_records += 1;
                        let nb = (p / wg + 1) * wg;
                        p = (p + 8).min(nb);
                    }
                }
                gq += c;
            }
        }
        n_items = g;
        n_runs += 1;
        layouts.push(RunLayout { base, cnt, partial });
        i = j;
    }
    (n_items, n_runs, n_records, layouts)
}

/// Fold the expanded pipeline's output for one run into per-pixel masks.
fn fold_expanded(res: &gpu::ExpandedResult, ri: usize, lay: &RunLayout, even_odd: bool) -> [u16; PIXELS] {
    let mut m = [0u16; PIXELS];
    if !lay.partial {
        let words = &res.masks[ri * 32..ri * 32 + 32];
        for y in 0..TILE {
            for x in 0..TILE {
                let ci = reference::quadrant_major_index(x, y);
                m[(y * TILE + x) as usize] = (words[ci / 2] >> (16 * (ci % 2))) as u16;
            }
        }
        return m;
    }
    let mut w = vec![[0i32; 16]; PIXELS];
    let mut decode = |rec: usize, q: u32| {
        for p in 0..16u32 {
            let base = (rec * 16 + p as usize) * 4;
            let w0 = res.partials[base] as u32;
            let w1 = res.partials[base + 1] as u32;
            let own = res.partials[base + 2] as u32;
            let bd = res.partials[base + 3];
            let x = 4 * (q & 1) + (p & 3);
            let y = 4 * (q >> 1) + (p >> 2);
            let pix = (y * TILE + x) as usize;
            for sidx in 0..16 {
                let word = if sidx < 8 { w0 } else { w1 };
                let lane = ((word >> (4 * (sidx & 7))) & 0xf) as i32 - 8;
                let o = ((own >> sidx) & 1) as i32 - ((own >> (16 + sidx)) & 1) as i32;
                w[pix][sidx] += lane + o + bd;
            }
        }
    };
    let mut g = lay.base;
    for q in 0..4u32 {
        let c = lay.cnt[q as usize] as usize;
        if c == 0 {
            let rec = res.run_info[ri * 4 + q as usize];
            assert!(rec < 0xffff_fffe, "missing uniform record for run {ri} q {q}");
            decode(rec as usize, q);
        } else {
            for p in g..g + c {
                let e4 = res.entries4[p];
                if e4.out & tiler::OUT_PARTIAL != 0 {
                    decode((e4.out & !tiler::OUT_PARTIAL) as usize, q);
                }
            }
        }
        g += c;
    }
    std::array::from_fn(|i| msaa::resolve_mask(&w[i], even_odd))
}

/// MSAA 16 path: CPU reference against the sample-point oracle, then GPU variants.
fn msaa_main() {
    let args: Vec<String> = std::env::args().collect();
    let svg = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "/Users/raphl/gh/vello/assets/Ghostscript_Tiger.svg".into());
    let scale: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(4.0);
    let out_dir = args.get(3).cloned().unwrap_or_else(|| "out".into());
    std::fs::create_dir_all(&out_dir).unwrap();

    let scene = scene::load_svg(&svg, scale);
    println!("scene {}x{}: {} paths, {} lines", scene.width, scene.height, scene.paths.len(), scene.lines.len());
    let lut = msaa::make_mask_lut_16();
    let tiled = tiler::tile(&scene, 256, tiler::PIECE);
    println!("tiles: {} entries, {} runs", tiled.entries.len(), tiled.runs.len());

    // CPU reference masks per run.
    let t0 = std::time::Instant::now();
    let ref_masks: Vec<[u16; PIXELS]> = tiled
        .runs
        .iter()
        .map(|r| {
            let eo = scene.paths[r.path as usize].even_odd;
            let w = msaa::run_sample_winding(&lut, &scene, &tiled, r);
            std::array::from_fn(|i| msaa::resolve_mask(&w[i], eo))
        })
        .collect();
    println!("cpu msaa reference: {:.1?}", t0.elapsed());

    // Compare with the sample oracle and composite.
    let t1 = std::time::Instant::now();
    let mut img = Image::new(scene.width, scene.height);
    let mut img_truth = Image::new(scene.width, scene.height);
    let mut n_pixels = 0usize;
    let mut n_pixels_diff = 0usize;
    let mut n_samples_diff = 0usize;
    let mut max_samples_diff = 0u32;
    let mut worst: Option<(u32, u32, u32)> = None;
    let mut run_ix = 0;
    for (path_ix, path) in scene.paths.iter().enumerate() {
        let color = color_f32(path.color);
        let bd = &tiled.backdrops[path_ix];
        let x0 = bd.tx0 * TILE;
        let y0 = bd.ty0 * TILE;
        let w = (bd.w * TILE).min(scene.width.saturating_sub(x0));
        let h = (bd.h * TILE).min(scene.height.saturating_sub(y0));
        let lines = &scene.lines[path.line_start as usize..path.line_end as usize];
        let truth = msaa::SampleTruth::compute(lines, x0, y0, w, h);
        let mut by_tile: HashMap<(u32, u32), usize> = HashMap::new();
        while run_ix < tiled.runs.len() && tiled.runs[run_ix].path == path_ix as u32 {
            let run = &tiled.runs[run_ix];
            by_tile.insert((run.tx, run.ty), run_ix);
            run_ix += 1;
        }
        for ty in bd.ty0..bd.ty0 + bd.h {
            for tx in bd.tx0..bd.tx0 + bd.w {
                let masks: [u16; PIXELS] = match by_tile.get(&(tx, ty)) {
                    Some(&ri) => ref_masks[ri],
                    None => {
                        let bdv = bd.get(tx, ty);
                        let covered = if path.even_odd { bdv & 1 != 0 } else { bdv != 0 };
                        if !covered {
                            continue;
                        }
                        [0xffff; PIXELS]
                    }
                };
                let alpha: [u8; PIXELS] = std::array::from_fn(|i| msaa::mask_alpha(masks[i]));
                img.blend_tile(tx, ty, &alpha, color);
                let mut tile_diff = 0u32;
                for y in 0..TILE {
                    for x in 0..TILE {
                        let px = tx * TILE + x;
                        let py = ty * TILE + y;
                        if px >= scene.width || py >= scene.height {
                            continue;
                        }
                        let t = truth.mask(px, py, path.even_odd);
                        img_truth.blend(px, py, msaa::mask_alpha(t), color);
                        let d = (t ^ masks[(y * TILE + x) as usize]).count_ones();
                        n_pixels += 1;
                        n_pixels_diff += (d > 0) as usize;
                        n_samples_diff += d as usize;
                        max_samples_diff = max_samples_diff.max(d);
                        tile_diff += d;
                    }
                }
                if tile_diff > 0 && worst.is_none_or(|(_, _, d)| tile_diff > d) {
                    worst = Some((tx, ty, tile_diff));
                }
            }
        }
    }
    println!(
        "reference vs sample oracle over {} pixels: {} pixels differ ({:.3}%), {} samples differ ({:.4}% of samples), max {} samples in one pixel ({:.1?})",
        n_pixels,
        n_pixels_diff,
        100.0 * n_pixels_diff as f64 / n_pixels as f64,
        n_samples_diff,
        100.0 * n_samples_diff as f64 / (16 * n_pixels) as f64,
        max_samples_diff,
        t1.elapsed()
    );
    if let Some((tx, ty, d)) = worst {
        println!("worst tile ({}, {}): {} samples differ", tx, ty, d);
    }
    img.write_png(&format!("{out_dir}/msaa_reference.png"));
    img_truth.write_png(&format!("{out_dir}/msaa_truth.png"));
    println!("wrote {out_dir}/msaa_reference.png and {out_dir}/msaa_truth.png");

    // GPU variants.
    let iterations: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
    let replicate: usize = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(1);
    let gpu = Gpu::new();
    let lut_packed = msaa::pack_lut(&lut);
    let path_flags: Vec<u32> = (0..replicate)
        .flat_map(|_| scene.paths.iter().map(|p| p.even_odd as u32))
        .collect();
    let mut lines_padded = scene.lines.clone();
    lines_padded.push([0.0; 4]);
    let msaa_variants = [
        (Mode::Msaa, 0u32),
        (Mode::MsaaQuadrant, 0),
        (Mode::MsaaFlat, 0),
        (Mode::MsaaFlat, 3),
    ];
    for (mode, cell_mode) in msaa_variants {
        let variant = Variant {
            mode,
            profile: 0,
            dense: false,
            quad_slots: 3,
            cell_mode,
            wg_size: 256,
            chunk: 1,
        };
        // Re-tile per variant: entries per workgroup differ, and so do piece boundaries.
        let mut tiled = tiler::tile(&scene, variant.entries_per_wg(), variant.piece_size());
        if mode == Mode::MsaaQuadrant {
            if TILE != 8 {
                continue;
            }
            allocate_quadrant_msaa(&scene, &mut tiled, variant.entries_per_wg());
        }
        let mut flat_items: Option<Vec<[u32; 4]>> = None;
        let mut flat_ranges: Vec<(usize, usize)> = Vec::new();
        if mode == Mode::MsaaFlat {
            if TILE != 8 {
                continue;
            }
            let mut items = build_flat_items(&scene, &tiled, variant.wg_size as usize);
            flat_ranges = allocate_flat_msaa(&mut items, &mut tiled, variant.wg_size as usize);
            flat_items = Some(items);
        }
        let ref_masks: Vec<[u16; PIXELS]> = tiled
            .runs
            .iter()
            .map(|r| {
                let eo = scene.paths[r.path as usize].even_odd;
                let w = msaa::run_sample_winding(&lut, &scene, &tiled, r);
                std::array::from_fn(|i| msaa::resolve_mask(&w[i], eo))
            })
            .collect();
        if replicate > 1 {
            let base_entries = tiled.entries.len();
            tiler::replicate(&mut tiled, replicate, scene.paths.len() as u32);
            if let Some(items) = &mut flat_items {
                // Replicate items alongside entries: entry indices shift by whole copies,
                // record bases by whole partial blocks.
                let base_items: Vec<[u32; 4]> = items.clone();
                let n_part = tiled.n_partials / replicate as u32;
                for c in 1..replicate as u32 {
                    for it in &base_items {
                        items.push([
                            it[0] + c * base_entries as u32,
                            it[1],
                            it[2] + c * base_entries as u32,
                            if it[3] != 0 { it[3] + c * n_part } else { 0 },
                        ]);
                    }
                }
            }
        }
        println!("== {} ==", variant.name());
        let result = gpu.run(
            variant,
            &lines_padded,
            &tiled.entries,
            &path_flags,
            tiled.n_alpha,
            tiled.n_partials,
            iterations,
            Some(&lut_packed),
            flat_items.as_deref(),
        );
        if !result.times_ms.is_empty() {
            let skip = 1;
            let mut t: Vec<f64> = result.times_ms[skip..].iter().copied().filter(|&t| t > 0.0).collect();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!(
                "  gpu: {} entries, min {:.3} ms, median {:.3} ms, max {:.3} ms ({:.2} ns/entry)",
                tiled.entries.len(),
                t[0],
                t[t.len() / 2],
                t[t.len() - 1],
                t[t.len() / 2] * 1e6 / tiled.entries.len() as f64
            );
        }
        // Compare GPU masks with the CPU reference (exact), folding partials.
        let mut runs_diff = 0usize;
        let mut pixels_diff = 0usize;
        let mut samples_diff = 0usize;
        let mut worst: Option<(usize, u32)> = None;
        let mut gpu_masks: Vec<[u16; PIXELS]> = Vec::with_capacity(tiled.runs.len());
        for (ri, r) in tiled.runs.iter().enumerate() {
            let eo = scene.paths[r.path as usize].even_odd;
            let mut m = [0u16; PIXELS];
            match r.output {
                RunOutput::Alpha(o) => {
                    let words = &result.alpha[o as usize * (PIXELS / 2)..(o as usize + 1) * (PIXELS / 2)];
                    for y in 0..TILE {
                        for x in 0..TILE {
                            let ci = variant.layout_index(x, y);
                            m[(y * TILE + x) as usize] = (words[ci / 2] >> (16 * (ci % 2))) as u16;
                        }
                    }
                }
                RunOutput::Partials { first, .. } if mode == Mode::MsaaFlat => {
                    let (a, b) = flat_ranges[ri];
                    let w = fold_flat_partials(flat_items.as_ref().unwrap(), a, b, first, &result.partials, variant.wg_size as usize);
                    for i in 0..PIXELS {
                        m[i] = msaa::resolve_mask(&w[i], eo);
                    }
                }
                RunOutput::Partials { first, .. } if mode == Mode::MsaaQuadrant => {
                    let w = fold_quadrant_partials(&scene, &tiled, r, first, &result.partials, variant.entries_per_wg());
                    for i in 0..PIXELS {
                        m[i] = msaa::resolve_mask(&w[i], eo);
                    }
                }
                RunOutput::Partials { first, count } => {
                    for y in 0..TILE {
                        for x in 0..TILE {
                            let ci = variant.layout_index(x, y);
                            let mut w = [0i32; 16];
                            for p in first..first + count {
                                let base = (p as usize * PIXELS + ci) * 4;
                                let w0 = result.partials[base] as u32;
                                let w1 = result.partials[base + 1] as u32;
                                let own = result.partials[base + 2] as u32;
                                let bd = result.partials[base + 3];
                                for sidx in 0..16 {
                                    let word = if sidx < 8 { w0 } else { w1 };
                                    let lane = ((word >> (4 * (sidx & 7))) & 0xf) as i32 - 8;
                                    let o = ((own >> sidx) & 1) as i32 - ((own >> (16 + sidx)) & 1) as i32;
                                    w[sidx] += lane + o + bd;
                                }
                            }
                            m[(y * TILE + x) as usize] = msaa::resolve_mask(&w, eo);
                        }
                    }
                }
            }
            let mut d_run = 0u32;
            for i in 0..PIXELS {
                let d = (m[i] ^ ref_masks[ri][i]).count_ones();
                d_run += d;
                pixels_diff += (d > 0) as usize;
                samples_diff += d as usize;
            }
            if d_run > 0 {
                runs_diff += 1;
                if worst.is_none_or(|(_, b)| d_run > b) {
                    worst = Some((ri, d_run));
                }
            }
            gpu_masks.push(m);
        }
        println!(
            "  gpu vs reference: {} of {} runs differ, {} pixels, {} samples",
            runs_diff,
            tiled.runs.len(),
            pixels_diff,
            samples_diff
        );
        if let Some((ri, d)) = worst {
            let r = &tiled.runs[ri];
            println!(
                "  worst: run {} path {} tile ({}, {}) entries {}..{} output {:?}: {} samples",
                ri, r.path, r.tx, r.ty, r.entry_start, r.entry_end, r.output, d
            );
            for e in &tiled.entries[r.entry_start as usize..r.entry_end as usize] {
                let l = scene.lines[(e.line & !LINE_LEFT_EDGE) as usize];
                let ox = r.tx as f32 * TILE as f32;
                let oy = r.ty as f32 * TILE as f32;
                println!(
                    "    entry: line {} local ({:.3}, {:.3}) -> ({:.3}, {:.3}) left_edge {} backdrop {}",
                    e.line & !LINE_LEFT_EDGE, l[0] - ox, l[1] - oy, l[2] - ox, l[3] - oy,
                    e.line & LINE_LEFT_EDGE != 0, e.backdrop
                );
            }
            for y in 0..TILE as usize {
                let row: Vec<String> = (0..TILE as usize)
                    .map(|x| format!("{:04x}/{:04x}", gpu_masks[ri][y * TILE as usize + x], ref_masks[ri][y * TILE as usize + x]))
                    .collect();
                println!("    {}", row.join(" "));
            }
        }
        if mode == Mode::Msaa && cell_mode == 0 {
            let alphas: Vec<[u8; PIXELS]> = gpu_masks
                .iter()
                .map(|m| std::array::from_fn(|i| msaa::mask_alpha(m[i])))
                .collect();
            let (img_gpu, _) = composite_and_check(&scene, &tiled, &alphas, false);
            img_gpu.write_png(&format!("{out_dir}/msaa_gpu.png"));
        }
    }

    // GPU expansion pipeline: 8x8 entries -> 4x4 quadrant entries -> merge.
    if TILE == 8 {
        for cell_mode in [0u32, 3] {
            let wg = 256usize;
            let mut tiled = tiler::tile(&scene, wg, usize::MAX);
            let ref_masks: Vec<[u16; PIXELS]> = tiled
                .runs
                .iter()
                .map(|r| {
                    let eo = scene.paths[r.path as usize].even_odd;
                    let w = msaa::run_sample_winding(&lut, &scene, &tiled, r);
                    std::array::from_fn(|i| msaa::resolve_mask(&w[i], eo))
                })
                .collect();
            if replicate > 1 {
                tiler::replicate(&mut tiled, replicate, scene.paths.len() as u32);
            }
            let (n_items, n_runs, n_records, layouts) = mirror_expansion(&scene, &tiled.entries, wg);
            let cm = ["inlinelut", "fifolut", "direct", "alwayslut"][cell_mode as usize];
            println!("== expanded_{cm}_wg{wg} ==");
            println!("  host mirror: {} items, {} runs, {} partial records", n_items, n_runs, n_records);
            let res = gpu.run_expanded(
                wg as u32,
                cell_mode,
                &lines_padded,
                &tiled.entries,
                &path_flags,
                &lut_packed,
                n_items,
                n_runs,
                n_records,
                scene.lines.len() as u32,
                iterations,
            );
            println!("  gpu counters: {} items, {} runs", res.n_items, res.n_runs);
            if std::env::var("EXPDET").is_ok() {
                // Determinism check: run again and diff per-run results.
                let res2 = gpu.run_expanded(wg as u32, cell_mode, &lines_padded, &tiled.entries, &path_flags,
                    &lut_packed, n_items, n_runs, n_records, scene.lines.len() as u32, 5);
                let mut nd = 0;
                for (ri, r) in tiled.runs.iter().enumerate() {
                    let eo = scene.paths[r.path as usize].even_odd;
                    let a = fold_expanded(&res, ri, &layouts[ri], eo);
                    let b = fold_expanded(&res2, ri, &layouts[ri], eo);
                    if a != b {
                        nd += 1;
                        if nd <= 5 {
                            let lay = &layouts[ri];
                            let d: u32 = (0..PIXELS).map(|i| (a[i] ^ b[i]).count_ones()).sum();
                            println!("  nondeterministic run {} partial {} cnt {:?} base {} ({} samples differ)", ri, lay.partial, lay.cnt, lay.base, d);
                            for e in &tiled.entries[r.entry_start as usize..r.entry_end as usize] {
                                let l = scene.lines[(e.line & 0x0fff_ffff) as usize];
                                let ox = r.tx as f32 * 8.0;
                                let oy = r.ty as f32 * 8.0;
                                println!("    entry: local ({:.4}, {:.4}) -> ({:.4}, {:.4}) left_edge {} backdrop {}",
                                    l[0] - ox, l[1] - oy, l[2] - ox, l[3] - oy, e.line & LINE_LEFT_EDGE != 0, e.backdrop);
                            }
                        }
                    }
                }
                println!("  determinism: {} runs differ between two executions", nd);
            }
            let mut cols = [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for row in &res.times_ms[1..] {
                for k in 0..6 {
                    if row[k] > 0.0 {
                        cols[k].push(row[k]);
                    }
                }
            }
            let med = |v: &mut Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                if v.is_empty() { 0.0 } else { v[v.len() / 2] }
            };
            let m: Vec<f64> = cols.iter_mut().map(med).collect();
            let expand: f64 = m[..5].iter().sum();
            println!(
                "  gpu: expansion {:.3} ms (map {:.3}, scan {:.3}, runs {:.3}, scatter {:.3}, fixup {:.3}), merge {:.3} ms, total {:.3} ms",
                expand, m[0], m[1], m[2], m[3], m[4], m[5], expand + m[5]
            );
            let mut runs_diff = 0usize;
            let mut pixels_diff = 0usize;
            let mut samples_diff = 0usize;
            let mut worst: Option<(usize, u32)> = None;
            for (ri, r) in tiled.runs.iter().enumerate() {
                let eo = scene.paths[r.path as usize].even_odd;
                let gm = fold_expanded(&res, ri, &layouts[ri], eo);
                let mut d_run = 0u32;
                for i in 0..PIXELS {
                    let d = (gm[i] ^ ref_masks[ri][i]).count_ones();
                    d_run += d;
                    pixels_diff += (d > 0) as usize;
                    samples_diff += d as usize;
                }
                if d_run > 0 {
                    runs_diff += 1;
                    if worst.is_none_or(|(_, b)| d_run > b) {
                        worst = Some((ri, d_run));
                    }
                }
            }
            println!(
                "  gpu vs reference: {} of {} runs differ, {} pixels, {} samples",
                runs_diff, tiled.runs.len(), pixels_diff, samples_diff
            );
            if let Some((ri, d)) = worst {
                let r = &tiled.runs[ri];
                println!(
                    "  worst: run {} path {} tile ({}, {}) entries {}..{} partial {}: {} samples",
                    ri, r.path, r.tx, r.ty, r.entry_start, r.entry_end, layouts[ri].partial, d
                );
            }
            if let Some(dump) = std::env::var("EXPDUMP").ok().and_then(|v| v.parse::<usize>().ok()) {
                let r = &tiled.runs[dump];
                let eo = scene.paths[r.path as usize].even_odd;
                let gm = fold_expanded(&res, dump, &layouts[dump], eo);
                println!("  dump run {} partial {} cnt {:?} base {}", dump, layouts[dump].partial, layouts[dump].cnt, layouts[dump].base);
                for e in &tiled.entries[r.entry_start as usize..r.entry_end as usize] {
                    let l = scene.lines[(e.line & 0x0fff_ffff) as usize];
                    let ox = r.tx as f32 * 8.0;
                    let oy = r.ty as f32 * 8.0;
                    println!("    entry: local ({:.5}, {:.5}) -> ({:.5}, {:.5}) left_edge {} backdrop {}",
                        l[0] - ox, l[1] - oy, l[2] - ox, l[3] - oy, e.line & LINE_LEFT_EDGE != 0, e.backdrop);
                }
                let lay = &layouts[dump];
                let mut g = lay.base;
                for q in 0..4 {
                    for p in g..g + lay.cnt[q] as usize {
                        let e4 = res.entries4[p];
                        println!("    item pos {} q {} line {} flag {} backdrop {} out {:#x}", p, q, e4.line & 0x0fff_ffff, e4.line >> 31, e4.backdrop, e4.out);
                    }
                    g += lay.cnt[q] as usize;
                }
                for y in 0..8 {
                    let row: Vec<String> = (0..8).map(|x| format!("{:04x}/{:04x}", gm[y * 8 + x], ref_masks[dump][y * 8 + x])).collect();
                    println!("    {}", row.join(" "));
                }
            }
            if std::env::var("EXPLIST").is_ok() {
                let mut ids = Vec::new();
                for (ri, r) in tiled.runs.iter().enumerate() {
                    let eo = scene.paths[r.path as usize].even_odd;
                    let gm = fold_expanded(&res, ri, &layouts[ri], eo);
                    let d: u32 = (0..PIXELS).map(|i| (gm[i] ^ ref_masks[ri][i]).count_ones()).sum();
                    if d > 0 {
                        let c = layouts[ri].cnt;
                        ids.push(format!("{ri}:{d}:p{}:c{}-{}-{}-{}", layouts[ri].partial as u8, c[0], c[1], c[2], c[3]));
                    }
                }
                println!("  difflist {}", ids.join(" "));
            }
        }
    }
}
