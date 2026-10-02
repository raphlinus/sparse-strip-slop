//! Two CPU references.
//!
//! * `PathTruth`: dense per-pixel signed area over all of a path's lines, unclipped, in f32.
//!   Independent of the tiler; the oracle for correctness.
//! * `entry_contrib` / `run_winding`: the per-(line, tile) computation the merge shader is
//!   specified to perform (x-clipped segment, fill_path area, y_edge term, fixed point).

use crate::scene::Scene;
use crate::tiler::{Entry, LINE_LEFT_EDGE, Run, TILE, Tiled};

pub const FIXED_SHIFT: u32 = 16;
pub const FIXED_ONE: i32 = 1 << FIXED_SHIFT;

#[inline]
fn fixed(v: f32) -> i32 {
    (v * FIXED_ONE as f32).round_ties_even() as i32
}

/// Mean over the segment's x-interval `[xmin, xmax]` (relative to the pixel's left edge)
/// of the coverage to the right, `clamp(1 - x, 0, 1)`. Algebraically fill_path's formula,
/// written without the difference of squares so near-vertical segments stay accurate.
#[inline]
pub fn area_weight(xmin: f32, xmax: f32) -> f32 {
    let w = xmax - xmin;
    let mid = 1.0 - (0.5 * (xmin + xmax)).clamp(0.0, 1.0);
    let x0c = xmin.clamp(0.0, 1.0);
    let x1c = xmax.clamp(0.0, 1.0);
    let left = (-xmin).clamp(0.0, w);
    let slanted = (left + (x1c - x0c) * (1.0 - 0.5 * (x0c + x1c))) / w;
    if w < 1.0e-6 { mid } else { slanted }
}

/// f64 version for the oracle.
#[inline]
pub fn area_weight64(xmin: f64, xmax: f64) -> f64 {
    let w = xmax - xmin;
    let mid = 1.0 - (0.5 * (xmin + xmax)).clamp(0.0, 1.0);
    let x0c = xmin.clamp(0.0, 1.0);
    let x1c = xmax.clamp(0.0, 1.0);
    let left = (-xmin).clamp(0.0, w);
    let slanted = (left + (x1c - x0c) * (1.0 - 0.5 * (x0c + x1c))) / w;
    if w < 1.0e-12 { mid } else { slanted }
}

/// f64 version of `row_span` for the oracle.
#[inline]
pub fn row_span64(p0: [f64; 2], p1: [f64; 2], row: f64) -> Option<(f64, f64, f64)> {
    let y = p0[1] - row;
    let dyl = p1[1] - p0[1];
    let y0 = y.clamp(0.0, 1.0);
    let y1 = (y + dyl).clamp(0.0, 1.0);
    let dy = y0 - y1;
    if dy == 0.0 {
        return None;
    }
    let t0 = (y0 - y) / dyl;
    let t1 = (y1 - y) / dyl;
    let dx = p1[0] - p0[0];
    let x0 = p0[0] + t0 * dx;
    let x1 = p0[0] + t1 * dx;
    Some((x0.min(x1), x0.max(x1), dy))
}

/// Dense ground-truth coverage for one path over a pixel rectangle, computed in f64 from
/// the unclipped lines (independent of the tiler and of the shader's math).
pub struct PathTruth {
    pub x0: u32,
    pub y0: u32,
    pub w: u32,
    pub h: u32,
    /// Signed area (winding integral) per pixel, row-major.
    pub area: Vec<f64>,
}

impl PathTruth {
    pub fn compute(lines: &[[f32; 4]], x0: u32, y0: u32, w: u32, h: u32) -> Self {
        let mut area = vec![0f64; (w * h) as usize];
        // Difference array for the "entirely to the right" constant contribution.
        let mut diff = vec![0f64; ((w + 1) * h) as usize];
        for l in lines {
            let p0 = [l[0] as f64 - x0 as f64, l[1] as f64 - y0 as f64];
            let p1 = [l[2] as f64 - x0 as f64, l[3] as f64 - y0 as f64];
            let ymin = p0[1].min(p1[1]).floor().max(0.0) as u32;
            let ymax = (p0[1].max(p1[1]).ceil().max(0.0) as u32).min(h);
            for row in ymin..ymax {
                let Some((xmin, xmax, dy)) = row_span64(p0, p1, row as f64) else {
                    continue;
                };
                let ia = xmin.floor().max(0.0) as i64;
                let ib = (xmax.ceil() as i64).min(w as i64);
                let rowbase = (row * w) as usize;
                for i in ia.max(0)..ib.max(0) {
                    let wgt = area_weight64(xmin - i as f64, xmax - i as f64);
                    area[rowbase + i as usize] += wgt * dy;
                }
                let right = ib.max(0) as usize;
                if right < w as usize {
                    diff[(row * (w + 1)) as usize + right] += dy;
                }
            }
        }
        for row in 0..h as usize {
            let mut acc = 0.0;
            for i in 0..w as usize {
                acc += diff[row * (w as usize + 1) + i];
                area[row * w as usize + i] += acc;
            }
        }
        Self { x0, y0, w, h, area }
    }

    pub fn alpha(&self, px: u32, py: u32, even_odd: bool) -> u8 {
        if px < self.x0 || py < self.y0 || px >= self.x0 + self.w || py >= self.y0 + self.h {
            return 0;
        }
        let a = self.area[((py - self.y0) * self.w + (px - self.x0)) as usize];
        let c = if even_odd {
            (a - 2.0 * (0.5 * a).round()).abs()
        } else {
            a.abs().min(1.0)
        };
        (c * 255.0).round() as u8
    }
}

/// Mirror of the shader's per-line setup (see shader/phase2.wgsl).
#[derive(Clone, Copy, Debug)]
pub struct LineSetup {
    pub a: f32,
    pub b: f32,
    pub d: f32,
    pub h: f32,
    pub q0: [f32; 2],
    pub q1: [f32; 2],
    pub recip: f32,
    pub ye: f32,
    pub ye_sign: f32,
}

pub fn line_setup(line: &[f32; 4], tx: u32, ty: u32, left_edge: bool) -> LineSetup {
    let tf = TILE as f32;
    let p0 = [line[0] - tx as f32 * tf, line[1] - ty as f32 * tf];
    let p1 = [line[2] - tx as f32 * tf, line[3] - ty as f32 * tf];
    let delta = [p1[0] - p0[0], p1[1] - p0[1]];
    let mut q0 = p0;
    let mut q1 = p1;
    if delta[0] != 0.0 {
        let ta = (0.0 - p0[0]) / delta[0];
        let tb = (tf - p0[0]) / delta[0];
        let t_lo = ta.min(tb).max(0.0);
        let t_hi = ta.max(tb).min(1.0).max(t_lo);
        q0 = [p0[0] + t_lo * delta[0], p0[1] + t_lo * delta[1]];
        q1 = [p0[0] + t_hi * delta[0], p0[1] + t_hi * delta[1]];
    }
    let a = delta[1];
    let b = -delta[0];
    let d = a * q0[0] + b * q0[1];
    let h = (a.abs() + b.abs()) * (0.5 - 1.0 / 65536.0);
    let (ye, ye_sign) = if left_edge && b != 0.0 {
        (d / b, delta[0].signum())
    } else {
        (0.0, 0.0)
    };
    LineSetup {
        a,
        b,
        d,
        h,
        q0,
        q1,
        recip: 1.0 / (q1[1] - q0[1]),
        ye,
        ye_sign,
    }
}

#[inline]
pub fn row_dy(ls: &LineSetup, y: u32) -> f32 {
    let yrel = ls.q0[1] - y as f32;
    let y0c = yrel.clamp(0.0, 1.0);
    let y1c = (yrel + (ls.q1[1] - ls.q0[1])).clamp(0.0, 1.0);
    y0c - y1c
}

#[inline]
pub fn cell_area(ls: &LineSetup, x: u32, y: u32) -> i32 {
    let yrel = ls.q0[1] - y as f32;
    let y0c = yrel.clamp(0.0, 1.0);
    let y1c = (yrel + (ls.q1[1] - ls.q0[1])).clamp(0.0, 1.0);
    let dy = y0c - y1c;
    if dy == 0.0 {
        return 0;
    }
    let t0 = (y0c - yrel) * ls.recip;
    let t1 = (y1c - yrel) * ls.recip;
    let dxc = ls.q1[0] - ls.q0[0];
    let xs = ls.q0[0] - x as f32;
    let x0 = xs + t0 * dxc;
    let x1 = xs + t1 * dxc;
    fixed(area_weight(x0.min(x1), x0.max(x1)) * dy)
}

/// Classification test for one cell: partial (needs area evaluation) and side.
#[inline]
pub fn classify_cell(ls: &LineSetup, x: u32, y: u32) -> (bool, bool) {
    let t_y = ls.b * (y as f32 + 0.5) - ls.d;
    let s = ls.a.mul_add(x as f32 + 0.5, t_y);
    (s.abs() < ls.h, (s > 0.0) == (ls.a > 0.0))
}

/// Contribution of one (line, tile) entry: 64 fixed-point values, row-major (y * 8 + x),
/// computed exactly the way the shader does (classification, side shortcut, y_edge).
pub fn entry_contrib(line: &[f32; 4], tx: u32, ty: u32, left_edge: bool) -> [i32; 64] {
    let ls = line_setup(line, tx, ty, left_edge);
    let mut out = [0i32; 64];
    for y in 0..TILE {
        let dy_fixed = fixed(row_dy(&ls, y));
        let ye_fixed = fixed(ls.ye_sign * (y as f32 - ls.ye + 1.0).clamp(0.0, 1.0));
        for x in 0..TILE {
            let (partial, side) = classify_cell(&ls, x, y);
            let v = if partial {
                cell_area(&ls, x, y)
            } else if side {
                dy_fixed
            } else {
                0
            };
            out[(y * TILE + x) as usize] = v + ye_fixed;
        }
    }
    out
}

/// Unresolved per-pixel winding (fixed point) for a run, including backdrop.
pub fn run_winding(scene: &Scene, tiled: &Tiled, run: &Run) -> [i32; 64] {
    let mut acc = [0i32; 64];
    for e in &tiled.entries[run.entry_start as usize..run.entry_end as usize] {
        add_entry(scene, e, run, &mut acc);
    }
    acc
}

pub fn add_entry(scene: &Scene, e: &Entry, run: &Run, acc: &mut [i32; 64]) {
    let line = &scene.lines[(e.line & !LINE_LEFT_EDGE) as usize];
    let c = entry_contrib(line, run.tx, run.ty, e.line & LINE_LEFT_EDGE != 0);
    for i in 0..64 {
        acc[i] += c[i];
    }
    if e.backdrop != 0 {
        for v in acc.iter_mut() {
            *v += e.backdrop * FIXED_ONE;
        }
    }
}

/// Resolve fixed-point winding to 8-bit alpha. Integer-only so CPU and GPU agree exactly.
pub fn resolve(w: i32, even_odd: bool) -> u8 {
    let c = if even_odd {
        let m = w.rem_euclid(2 * FIXED_ONE);
        if m > FIXED_ONE { 2 * FIXED_ONE - m } else { m }
    } else {
        w.unsigned_abs().min(FIXED_ONE as u32) as i32
    };
    ((c as i64 * 255 + (FIXED_ONE as i64 / 2)) >> FIXED_SHIFT) as u8
}

/// Index of pixel (x, y) in the coset-major output layout.
#[inline]
pub fn coset_major_index(x: u32, y: u32) -> usize {
    let (c, p) = coset_of(x, y);
    (c * 16 + p) as usize
}

/// Coset partition helpers (see handoff doc).
#[inline]
pub fn coset_of(x: u32, y: u32) -> (u32, u32) {
    (2 * (x & 1) + (y & 1), (y >> 1) * 4 + (x >> 1))
}

/// Cells of each coset the (clipped, tile-local) line passes through, by the shader's
/// classification test. Returns 4 16-bit hit masks indexed by position within coset.
pub fn classify_hits(line: &[f32; 4], tx: u32, ty: u32) -> [u16; 4] {
    let ls = line_setup(line, tx, ty, false);
    let mut masks = [0u16; 4];
    for y in 0..TILE {
        for x in 0..TILE {
            if classify_cell(&ls, x, y).0 {
                let (c, p) = coset_of(x, y);
                masks[c as usize] |= 1 << p;
            }
        }
    }
    masks
}
