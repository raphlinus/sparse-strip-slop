//! MSAA 16 payload: per (line, pixel) sample contributions, the half-plane mask LUT ported
//! from Vello research (`vello_encoding/src/mask.rs`, `fine.wgsl` fill_path_ms), a CPU
//! mirror of the shader math, and an exact sample-point oracle.
//!
//! Sample i of a pixel sits at y = (i + 0.5) / 16 and x = (PATTERN_16[i] + 0.5) / 16, so
//! samples are sorted by y and "samples within a y-range" is a contiguous bit range.
//! Convention as for the area version: an upward line contributes +1 to samples on its
//! right; the tile backdrop and the y_edge term apply per sample.

use crate::reference::{LineSetup, classify_cell, line_setup};
use crate::scene::Scene;
use crate::tiler::{Entry, LINE_LEFT_EDGE, PIXELS, Run, TILE, Tiled};

/// D3D11 standard 16x sample pattern (x index per sample, samples ordered by y).
pub const PATTERN_16: [u8; 16] = [1, 8, 4, 11, 15, 7, 3, 12, 0, 9, 5, 13, 2, 10, 6, 14];

pub const MASK16_WIDTH: usize = 64;
pub const MASK16_HEIGHT: usize = 64;
const ONE_MINUS_ULP: f32 = 0.99999994;
const ROBUST_EPSILON: f32 = 2e-7;

fn one_mask_16(slope: f64, mut translation: f64, is_pos: bool) -> u16 {
    if is_pos {
        translation = 1. - translation;
    }
    let mut result = 0;
    for (i, item) in PATTERN_16.iter().enumerate() {
        let mut y = (i as f64 + 0.5) * 0.0625;
        let x = (*item as f64 + 0.5) * 0.0625;
        if !is_pos {
            y = 1. - y;
        }
        if (x - (1.0 - translation)) * (1. - slope) - (y - translation) * slope >= 0. {
            result |= 1 << i;
        }
    }
    result
}

/// Half-plane mask LUT: two blocks (negative then positive slopes) of 32 slopes x 64
/// translations, 4096 u16 entries.
pub fn make_mask_lut_16() -> Vec<u16> {
    (0..MASK16_WIDTH * MASK16_HEIGHT)
        .map(|i| {
            const HALF_HEIGHT: usize = MASK16_HEIGHT / 2;
            let u = i % MASK16_WIDTH;
            let v = i / MASK16_WIDTH;
            let is_pos = v >= HALF_HEIGHT;
            let y = ((v % HALF_HEIGHT) as f64 + 0.5) * (1.0 / HALF_HEIGHT as f64);
            let x = (u as f64 + 0.5) * (1.0 / MASK16_WIDTH as f64);
            one_mask_16(y, x, is_pos)
        })
        .collect()
}

/// LUT as packed u32 words (two u16 per word, low half first) for the GPU.
pub fn pack_lut(lut: &[u16]) -> Vec<u32> {
    lut.chunks(2).map(|c| c[0] as u32 | ((c[1] as u32) << 16)).collect()
}

#[inline]
fn span(a: f32, b: f32) -> u32 {
    (a.max(b).ceil() - a.min(b).floor()).max(1.0) as u32
}

/// Line rasterization parameters from fill_path_ms, for the segment clipped to the tile
/// box (tile-local). `zf = a * sub_ix + b` along the line's Manhattan walk.
#[derive(Clone, Copy, Debug)]
pub struct LutSetup {
    pub a: f32,
    pub b: f32,
    pub x_sign: f32,
    pub x0i: i32,
    pub y0i: i32,
    pub is_pos: bool,
}

pub fn lut_setup(s0: [f32; 2], s1: [f32; 2]) -> LutSetup {
    let is_down = s1[1] >= s0[1];
    let (xy0, xy1) = if is_down { (s0, s1) } else { (s1, s0) };
    let dx = (xy1[0] - xy0[0]).abs();
    let dy = xy1[1] - xy0[1];
    let idxdy = 1.0 / (dx + dy);
    let mut a = dx * idxdy;
    let is_pos = xy1[0] >= xy0[0];
    let x_sign = if is_pos { 1.0 } else { -1.0 };
    let xt0 = (xy0[0] * x_sign).floor();
    let c = xy0[0] * x_sign - xt0;
    let y0i = xy0[1].floor();
    let ytop = y0i + 1.0;
    let b = ((dy * c + dx * (ytop - xy0[1])) * idxdy).min(ONE_MINUS_ULP);
    let count_x = span(xy0[0], xy1[0]) - 1;
    let count = count_x + span(xy0[1], xy1[1]);
    let robust_err = (a * (count as f32 - 1.0) + b).floor() - count_x as f32;
    if robust_err != 0.0 {
        a -= ROBUST_EPSILON * robust_err.signum();
    }
    let x0i = (xt0 * x_sign + 0.5 * (x_sign - 1.0)) as i32;
    LutSetup {
        a,
        b,
        x_sign,
        x0i,
        y0i: y0i as i32,
        is_pos,
    }
}

/// Half-plane mask (samples right of the line) for pixel (x, y) the line passes through.
#[inline]
pub fn lut_mask(lut: &[u16], ls: &LutSetup, x: u32, y: u32) -> u16 {
    let sub_ix = (y as i32 - ls.y0i) as f32 + ls.x_sign * (x as i32 - ls.x0i) as f32;
    let zf = ls.a * sub_ix + ls.b;
    let z = zf.floor();
    let block = if ls.is_pos { (MASK16_WIDTH * MASK16_HEIGHT / 2) as u32 } else { 0 };
    let half_height = (MASK16_HEIGHT / 2) as f32;
    let row = (ls.a * half_height).min(half_height - 1.0).floor() as u32 * MASK16_WIDTH as u32;
    let col = ((zf - z) * MASK16_WIDTH as f32).floor() as u32;
    lut[(block + row + col) as usize]
}

/// Samples with y in [lo, hi) (row-relative, each clamped to [0, 1]).
#[inline]
pub fn range_mask(lo: f32, hi: f32) -> u16 {
    // WGSL round() is half-to-even.
    let s_lo = (16.0 * lo).round_ties_even() as u32;
    let s_hi = (16.0 * hi).round_ties_even() as u32;
    let m = (0xffffu32 << s_lo) & !(0xffffu32.checked_shl(s_hi).unwrap_or(0));
    (m & 0xffff) as u16
}

/// Per (line, tile) MSAA setup: area-version line setup (classification, clipped segment,
/// y_edge) plus the LUT walk parameters for the tile-box-clipped segment.
pub struct MsaaSetup {
    pub ls: LineSetup,
    pub lut: LutSetup,
    pub sign_pos: bool,
}

pub fn msaa_setup(line: &[f32; 4], tx: u32, ty: u32, left_edge: bool) -> MsaaSetup {
    let ls = line_setup(line, tx, ty, left_edge);
    // Clip the x-clipped segment to y in [0, TILE] as well.
    let (mut s0, mut s1) = (ls.q0, ls.q1);
    let dy = s1[1] - s0[1];
    if dy != 0.0 {
        let tf = TILE as f32;
        let ta = (0.0 - s0[1]) / dy;
        let tb = (tf - s0[1]) / dy;
        let t_lo = ta.min(tb).max(0.0);
        let t_hi = ta.max(tb).min(1.0).max(t_lo);
        let dx = s1[0] - s0[0];
        let q0 = [s0[0] + t_lo * dx, s0[1] + t_lo * dy];
        let q1 = [s0[0] + t_hi * dx, s0[1] + t_hi * dy];
        s0 = q0;
        s1 = q1;
    }
    MsaaSetup {
        lut: lut_setup(s0, s1),
        // Upward line (y1 < y0, a < 0) contributes +1.
        sign_pos: ls.a < 0.0,
        ls,
    }
}

/// Per-pixel sample contributions of one entry as (plus, minus) 16-bit masks, row-major
/// pixel order. A sample has at most one of the two bits set.
pub fn entry_masks(lut: &[u16], line: &[f32; 4], tx: u32, ty: u32, left_edge: bool) -> [(u16, u16); PIXELS] {
    let st = msaa_setup(line, tx, ty, left_edge);
    let ls = &st.ls;
    let mut out = [(0u16, 0u16); PIXELS];
    let ye_pos = ls.ye_sign > 0.0;
    for y in 0..TILE {
        let yrel = ls.q0[1] - y as f32;
        let y0c = yrel.clamp(0.0, 1.0);
        let y1c = (yrel + (ls.q1[1] - ls.q0[1])).clamp(0.0, 1.0);
        let range = range_mask(y0c.min(y1c), y0c.max(y1c));
        // y_edge term: samples at or below the left-edge crossing.
        let ye = if ls.ye_sign != 0.0 {
            range_mask((ls.ye - y as f32).clamp(0.0, 1.0), 1.0)
        } else {
            0
        };
        for x in 0..TILE {
            let (partial, side) = classify_cell(ls, x, y);
            let hp: u16 = if partial {
                lut_mask(lut, &st.lut, x, y)
            } else if side {
                0xffff
            } else {
                0
            };
            let cross = range & hp;
            let (mut plus, mut minus) = if st.sign_pos { (cross, 0) } else { (0, cross) };
            if ye_pos {
                plus |= ye;
            } else {
                minus |= ye;
            }
            let both = plus & minus;
            out[(y * TILE + x) as usize] = (plus & !both, minus & !both);
        }
    }
    out
}

/// Unresolved per-sample winding for a run, including backdrop: 16 i32 per pixel.
pub fn run_sample_winding(lut: &[u16], scene: &Scene, tiled: &Tiled, run: &Run) -> Vec<[i32; 16]> {
    let mut acc = vec![[0i32; 16]; PIXELS];
    for e in &tiled.entries[run.entry_start as usize..run.entry_end as usize] {
        add_entry(lut, scene, e, run, &mut acc);
    }
    acc
}

pub fn add_entry(lut: &[u16], scene: &Scene, e: &Entry, run: &Run, acc: &mut [[i32; 16]]) {
    let line = &scene.lines[(e.line & !LINE_LEFT_EDGE) as usize];
    let m = entry_masks(lut, line, run.tx, run.ty, e.line & LINE_LEFT_EDGE != 0);
    for p in 0..PIXELS {
        let (plus, minus) = m[p];
        for s in 0..16 {
            acc[p][s] += ((plus >> s) & 1) as i32 - ((minus >> s) & 1) as i32 + e.backdrop;
        }
    }
}

/// Resolve per-sample windings to a coverage mask.
#[inline]
pub fn resolve_mask(w: &[i32; 16], even_odd: bool) -> u16 {
    let mut m = 0u16;
    for s in 0..16 {
        let covered = if even_odd { w[s] & 1 != 0 } else { w[s] != 0 };
        m |= (covered as u16) << s;
    }
    m
}

#[inline]
pub fn mask_alpha(m: u16) -> u8 {
    ((m.count_ones() * 255 + 8) / 16) as u8
}

/// Exact sample-point winding for one path over a pixel rectangle (i32 per sample), from
/// the unclipped lines in f64. Independent of tiling and of the LUT.
pub struct SampleTruth {
    pub x0: u32,
    pub y0: u32,
    pub w: u32,
    pub h: u32,
    /// winding[(py * w + px) * 16 + s]
    pub winding: Vec<i16>,
}

impl SampleTruth {
    pub fn compute(lines: &[[f32; 4]], x0: u32, y0: u32, w: u32, h: u32) -> Self {
        let n = (w as usize) * (h as usize) * 16;
        // Difference array along x per (pixel row, sample row): diff[(py * 16 + s) * (w + 1) + px]
        let mut diff = vec![0i16; (h as usize) * 16 * (w as usize + 1)];
        for l in lines {
            let (lx0, ly0, lx1, ly1) = (l[0] as f64 - x0 as f64, l[1] as f64 - y0 as f64, l[2] as f64 - x0 as f64, l[3] as f64 - y0 as f64);
            if ly0 == ly1 {
                continue;
            }
            let sign: i16 = if ly1 < ly0 { 1 } else { -1 };
            let (ymin, ymax) = (ly0.min(ly1), ly0.max(ly1));
            // Sample rows k (global index, y = (k + 0.5) / 16) with ymin <= y < ymax.
            let k_lo = ((ymin * 16.0 - 0.5).ceil().max(0.0)) as i64;
            let k_hi = ((ymax * 16.0 - 0.5).ceil()).min((h as i64 * 16) as f64) as i64; // exclusive
            let slope = (lx1 - lx0) / (ly1 - ly0);
            for k in k_lo..k_hi {
                let ys = (k as f64 + 0.5) / 16.0;
                let xc = lx0 + (ys - ly0) * slope;
                let py = (k / 16) as usize;
                let s = (k % 16) as usize;
                let xs_off = (PATTERN_16[s] as f64 + 0.5) / 16.0;
                // First pixel whose sample in this sample row is right of the crossing.
                let px_first = (xc - xs_off).floor() as i64 + 1;
                let px_first = px_first.max(0);
                if px_first < w as i64 {
                    diff[(py * 16 + s) * (w as usize + 1) + px_first as usize] += sign;
                }
            }
        }
        let mut winding = vec![0i16; n];
        for py in 0..h as usize {
            for s in 0..16 {
                let row = &diff[(py * 16 + s) * (w as usize + 1)..];
                let mut acc = 0i16;
                for px in 0..w as usize {
                    acc += row[px];
                    winding[(py * w as usize + px) * 16 + s] = acc;
                }
            }
        }
        Self { x0, y0, w, h, winding }
    }

    pub fn mask(&self, px: u32, py: u32, even_odd: bool) -> u16 {
        if px < self.x0 || py < self.y0 || px >= self.x0 + self.w || py >= self.y0 + self.h {
            return 0;
        }
        let base = (((py - self.y0) * self.w + (px - self.x0)) * 16) as usize;
        let mut m = 0u16;
        for s in 0..16 {
            let w = self.winding[base + s];
            let covered = if even_odd { w & 1 != 0 } else { w != 0 };
            m |= (covered as u16) << s;
        }
        m
    }
}
