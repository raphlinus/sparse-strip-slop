//! CPU stand-in for the tile, sort, and strip-allocation stages.
//!
//! Conventions (shared with the merge shader and the CPU reference):
//! - Tiles are TILE x TILE (8 by default, 4 with the `tile4` feature). A point at x belongs to tile `floor(x / 8)` (half-open).
//! - Winding sign: an upward line (y1 < y0) contributes +1 to pixels on its right,
//!   matching `dy = clamp(y0) - clamp(y1)` in fill_path.
//! - Backdrop of tile (tx, ty) = sum over lines crossing the row's top edge `y = 8 ty`
//!   (half-open: `ymin <= 8 ty < ymax`) at an x strictly in a tile left of tx, of that sign.
//! - An entry carries `LINE_LEFT_EDGE` when the row-clipped segment has points in a tile
//!   left of tx, i.e. it crosses into tx through its left edge. The shader then adds the
//!   `y_edge` term of fill_path for that entry.

use bytemuck::{Pod, Zeroable};

use crate::scene::Scene;

pub const TILE: u32 = if cfg!(feature = "tile4") { 4 } else { 8 };
pub const PIXELS: usize = (TILE * TILE) as usize;
pub const PIECE: usize = 8;
pub const LINE_LEFT_EDGE: u32 = 1 << 31;
pub const OUT_PARTIAL: u32 = 1 << 31;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Entry {
    pub key: u32,
    pub line: u32,
    pub backdrop: i32,
    pub out: u32,
}

#[derive(Clone, Copy, Debug)]
pub enum RunOutput {
    /// Resolved 8-bit alpha tile at this index in the alpha buffer.
    Alpha(u32),
    /// `count` unresolved i32 partial tiles starting at this index in the partials buffer.
    Partials { first: u32, count: u32 },
}

#[derive(Clone, Debug)]
pub struct Run {
    pub path: u32,
    pub tx: u32,
    pub ty: u32,
    pub entry_start: u32,
    pub entry_end: u32,
    pub output: RunOutput,
}

/// Per-path coarse winding over the path's tile bounding box.
pub struct Backdrops {
    pub tx0: u32,
    pub ty0: u32,
    pub w: u32,
    pub h: u32,
    pub winding: Vec<i32>,
}

impl Backdrops {
    pub fn get(&self, tx: u32, ty: u32) -> i32 {
        if tx < self.tx0 || ty < self.ty0 || tx >= self.tx0 + self.w || ty >= self.ty0 + self.h {
            0
        } else {
            self.winding[((ty - self.ty0) * self.w + (tx - self.tx0)) as usize]
        }
    }
}

#[allow(dead_code)]
pub struct Tiled {
    pub width_tiles: u32,
    pub height_tiles: u32,
    pub entries: Vec<Entry>,
    pub runs: Vec<Run>,
    pub backdrops: Vec<Backdrops>,
    pub n_alpha: u32,
    pub n_partials: u32,
}

pub fn pack_key(path: u32, tx: u32, ty: u32) -> u32 {
    assert!(path < (1 << 12) && tx < (1 << 10) && ty < (1 << 10));
    (path << 20) | (ty << 10) | tx
}

/// x on the line at a given y. Used for both the backdrop crossing and the
/// row-clipped extent, so both see the identical value.
#[inline]
fn x_at(l: &[f32; 4], y: f32) -> f32 {
    l[0] + (y - l[1]) * ((l[2] - l[0]) / (l[3] - l[1]))
}

/// Sentinel key for padding entries past the end of the real data.
pub const SENTINEL_KEY: u32 = u32::MAX;

/// `entries_per_wg` is the number of entries one workgroup consumes; pieces are split at
/// those boundaries so the shader's local piece-boundary rule matches this allocation.
pub fn tile(scene: &Scene, entries_per_wg: usize, piece_size: usize) -> Tiled {
    let width_tiles = scene.width.div_ceil(TILE);
    let height_tiles = scene.height.div_ceil(TILE);
    let tf = TILE as f32;

    let mut raw: Vec<(u32, u32)> = Vec::new(); // (key, line | flag)
    let mut backdrops = Vec::with_capacity(scene.paths.len());

    for (path_ix, path) in scene.paths.iter().enumerate() {
        let lines = &scene.lines[path.line_start as usize..path.line_end as usize];
        // Path tile bbox, clamped to the viewport.
        let mut bx0 = f32::INFINITY;
        let mut by0 = f32::INFINITY;
        let mut bx1 = f32::NEG_INFINITY;
        let mut by1 = f32::NEG_INFINITY;
        for l in lines {
            bx0 = bx0.min(l[0]).min(l[2]);
            bx1 = bx1.max(l[0]).max(l[2]);
            by0 = by0.min(l[1]).min(l[3]);
            by1 = by1.max(l[1]).max(l[3]);
        }
        let tx0 = ((bx0 / tf).floor().max(0.0) as u32).min(width_tiles);
        let ty0 = ((by0 / tf).floor().max(0.0) as u32).min(height_tiles);
        // Everything right of the bbox can still be affected by backdrop, so the
        // backdrop grid spans to the right edge of the viewport.
        let tx1 = width_tiles;
        let ty1 = ((by1 / tf).ceil().max(0.0) as u32).min(height_tiles);
        let w = tx1.saturating_sub(tx0);
        let h = ty1.saturating_sub(ty0);
        // deltas[row][col] accumulates at col = tx_c + 1 (first tile that sees it).
        let mut deltas = vec![0i32; (w * h) as usize];

        for (li, l) in lines.iter().enumerate() {
            let line_ix = path.line_start + li as u32;
            let (y0, y1) = (l[1], l[3]);
            if y0 == y1 {
                continue;
            }
            let (ymin, ymax) = (y0.min(y1), y0.max(y1));
            let sign = if y1 < y0 { 1 } else { -1 };
            let row_start = (ymin / tf).floor().max(0.0) as u32;
            let row_end = ((ymax / tf).ceil().max(0.0) as u32).min(height_tiles); // exclusive
            for ty in row_start..row_end {
                let top = ty as f32 * tf;
                let bot = top + tf;
                // Backdrop: top-edge crossing.
                if ymin <= top && top < ymax {
                    let xc = x_at(l, top);
                    let txc = (xc / tf).floor();
                    let col = txc as i64 + 1;
                    let col = col.max(tx0 as i64);
                    if col < tx1 as i64 && ty >= ty0 && ty < ty1 {
                        deltas[((ty - ty0) * w + (col as u32 - tx0)) as usize] += sign;
                    }
                }
                // Row-clipped x extent.
                let ya = ymin.max(top);
                let yb = ymax.min(bot);
                let xa_ = if ya == y0 { l[0] } else if ya == y1 { l[2] } else { x_at(l, ya) };
                let xb_ = if yb == y0 { l[0] } else if yb == y1 { l[2] } else { x_at(l, yb) };
                let (xa, xb) = (xa_.min(xb_), xa_.max(xb_));
                let tx_first = (xa / tf).floor();
                let tx_last = (xb / tf).floor();
                let lo = tx_first.max(0.0) as u32;
                let hi = (tx_last.min((width_tiles - 1) as f32)) as i64;
                if tx_last < 0.0 {
                    continue;
                }
                for tx in lo as i64..=hi {
                    let tx = tx as u32;
                    let flag = if (tx as f32) > tx_first { LINE_LEFT_EDGE } else { 0 };
                    raw.push((pack_key(path_ix as u32, tx, ty), line_ix | flag));
                }
            }
        }
        // Prefix sum along rows.
        for row in 0..h as usize {
            let mut acc = 0;
            for col in 0..w as usize {
                acc += deltas[row * w as usize + col];
                deltas[row * w as usize + col] = acc;
            }
        }
        backdrops.push(Backdrops {
            tx0,
            ty0,
            w,
            h,
            winding: deltas,
        });
    }

    raw.sort_unstable();

    // Build entries, runs, pieces, and output allocation.
    let mut entries = Vec::with_capacity(raw.len());
    let mut runs = Vec::new();
    let mut n_alpha = 0u32;
    let mut n_partials = 0u32;
    let mut i = 0;
    while i < raw.len() {
        let key = raw[i].0;
        let mut j = i;
        while j < raw.len() && raw[j].0 == key {
            j += 1;
        }
        let path = key >> 20;
        let ty = (key >> 10) & 0x3ff;
        let tx = key & 0x3ff;
        let len = j - i;
        let backdrop = backdrops[path as usize].get(tx, ty);
        // Piece boundaries: run start, workgroup boundary, or 8 entries since the last
        // workgroup boundary / run start.
        let mut piece_starts = Vec::with_capacity(1);
        let mut seg_start = i;
        for k in i..j {
            if k == i || k % entries_per_wg == 0 {
                seg_start = k;
                piece_starts.push(k);
            } else if (k - seg_start) % piece_size == 0 {
                piece_starts.push(k);
            }
        }
        let n_pieces = piece_starts.len();
        let output = if n_pieces == 1 {
            let o = RunOutput::Alpha(n_alpha);
            n_alpha += 1;
            o
        } else {
            let o = RunOutput::Partials {
                first: n_partials,
                count: n_pieces as u32,
            };
            n_partials += n_pieces as u32;
            o
        };
        let mut piece_ix = 0;
        for (k, &(_, line)) in raw[i..j].iter().enumerate() {
            let out = if piece_ix < n_pieces && piece_starts[piece_ix] == i + k {
                let o = match output {
                    RunOutput::Alpha(o) => o,
                    RunOutput::Partials { first, .. } => OUT_PARTIAL | (first + piece_ix as u32),
                };
                piece_ix += 1;
                o
            } else {
                0
            };
            entries.push(Entry {
                key,
                line,
                backdrop: if k == 0 { backdrop } else { 0 },
                out,
            });
        }
        let _ = len;
        runs.push(Run {
            path,
            tx,
            ty,
            entry_start: i as u32,
            entry_end: j as u32,
            output,
        });
        i = j;
    }

    // Pad to a whole number of workgroups. The padded run writes to a dummy alpha slot
    // (index n_alpha) and references a degenerate dummy line (index scene.lines.len()).
    let dummy_line = scene.lines.len() as u32;
    let padded = entries.len().div_ceil(entries_per_wg) * entries_per_wg;
    entries.resize(
        padded,
        Entry {
            key: SENTINEL_KEY,
            line: dummy_line,
            backdrop: 0,
            out: n_alpha,
        },
    );
    n_alpha += 1;

    Tiled {
        width_tiles,
        height_tiles,
        entries,
        runs,
        backdrops,
        n_alpha,
        n_partials,
    }
}

/// Replicate the entry stream `n` times with distinct path ids so the GPU sees `n` times
/// the work with the same statistics. Runs and backdrops stay those of the first copy;
/// output indices of later copies are offset by whole output buffers.
pub fn replicate(tiled: &mut Tiled, n: usize, n_paths: u32) {
    assert!(n_paths as usize * n <= 1 << 12, "path ids must fit in 12 bits");
    let base: Vec<Entry> = tiled.entries.clone();
    for c in 1..n as u32 {
        for e in &base {
            let mut e = *e;
            if e.key != SENTINEL_KEY {
                e.key += c * n_paths << 20;
            }
            if e.out & OUT_PARTIAL != 0 {
                e.out += c * tiled.n_partials;
            } else if e.key == SENTINEL_KEY || e.out != 0 || is_piece_head(&base, &e) {
                e.out += c * tiled.n_alpha;
            }
            tiled.entries.push(e);
        }
    }
    tiled.n_alpha *= n as u32;
    tiled.n_partials *= n as u32;
}

/// Whether an entry carries an output index (only piece heads do; `out == 0` is ambiguous
/// between "no output" and "alpha slot 0", so resolve by position).
fn is_piece_head(base: &[Entry], e: &Entry) -> bool {
    // Alpha slot 0 belongs to the very first run, whose head is entry 0.
    std::ptr::eq(e, &base[0])
}
