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
use tiler::{LINE_LEFT_EDGE, PIXELS, RunOutput, TILE, Tiled};

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
    let mut tiled = tiled;
    if replicate > 1 {
        tiler::replicate(&mut tiled, replicate, scene.paths.len() as u32);
    }
    for cell_mode in 0..3u32 {
        let variant = Variant {
            mode: Mode::Msaa,
            profile: 0,
            dense: false,
            quad_slots: 3,
            cell_mode,
            wg_size: 256,
            chunk: 1,
        };
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
                RunOutput::Partials { first, count } => {
                    for y in 0..TILE {
                        for x in 0..TILE {
                            let ci = variant.layout_index(x, y);
                            let mut w = [0i32; 16];
                            for p in first..first + count {
                                let base = (p as usize * PIXELS + ci) * 16;
                                for sidx in 0..16 {
                                    w[sidx] += result.partials[base + sidx];
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
                    .map(|x| format!("{:04x}/{:04x}", gpu_masks[ri][y * 8 + x], ref_masks[ri][y * 8 + x]))
                    .collect();
                println!("    {}", row.join(" "));
            }
        }
        if cell_mode == 0 {
            let alphas: Vec<[u8; PIXELS]> = gpu_masks
                .iter()
                .map(|m| std::array::from_fn(|i| msaa::mask_alpha(m[i])))
                .collect();
            let (img_gpu, _) = composite_and_check(&scene, &tiled, &alphas, false);
            img_gpu.write_png(&format!("{out_dir}/msaa_gpu.png"));
        }
    }
}
