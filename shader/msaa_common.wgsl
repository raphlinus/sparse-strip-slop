// MSAA 16 merge: one entry per thread, one pixel per barrier pair, coset walk order.
//
// Payload per (line, pixel): a 16-bit mask of samples right of the line within the
// segment's y-range (sign by line direction), plus the y_edge term, giving per-sample
// values in {-1, 0, +1}. Accumulation: two slot words per pixel with 4-bit lanes (samples
// 0..7 and 8..15), initialized by the head to 0x88888888; adders atomicAdd the signed
// packed payload as a plain integer. Lanes stay within [1, 15] for up to 7 adders, so the
// multi-digit sum is exact. The head keeps its own payload in registers and adds it at
// resolve, hence pieces of head + 7.
//
// CELL_MODE (header const): how a cell's half-plane mask is obtained.
//   0: classify each cell against the infinite line; LUT lookup for hit cells inline.
//   1: classify a coset at once; LUT masks of the (at most HALF) hit cells precomputed
//      into a FIFO with a fixed-trip loop, popped in walk order (the handoff design).
//   2: dense: every cell evaluates all 16 samples directly (no LUT, no classification).
// Prepended with `const TILE`, `const WG_SIZE`, `const CELL_MODE`.

const HALF: u32 = TILE / 2u;
const N_POS: u32 = HALF * HALF;
const PIXELS: u32 = TILE * TILE;
const LINE_LEFT_EDGE: u32 = 0x80000000u;
const OUT_PARTIAL: u32 = 0x80000000u;
const ONE_MINUS_ULP: f32 = 0.99999994;
const ROBUST_EPSILON: f32 = 2e-7;
const MASK16_WIDTH: u32 = 64u;
const MASK16_HALF_HEIGHT: u32 = 32u;

struct Entry {
    key: u32,
    line: u32,
    backdrop: i32,
    out: u32,
}

@group(0) @binding(0) var<storage, read> lines: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> entries: array<Entry>;
@group(0) @binding(2) var<storage, read_write> mask_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> partials_out: array<i32>;
@group(0) @binding(4) var<storage, read> path_flags: array<u32>;
@group(0) @binding(5) var<storage, read> mask_lut: array<u32>;


// Sample x offsets (D3D11 16x pattern), as (PATTERN_16[i] + 0.5) / 16, for CELL_MODE 2.
const SAMPLE_X: array<f32, 16> = array<f32, 16>(
    0.09375, 0.53125, 0.28125, 0.71875, 0.96875, 0.46875, 0.21875, 0.78125,
    0.03125, 0.59375, 0.34375, 0.84375, 0.15625, 0.65625, 0.40625, 0.90625);

fn span(a: f32, b: f32) -> u32 {
    return u32(max(ceil(max(a, b)) - floor(min(a, b)), 1.0));
}

// Samples with row-relative y in [lo, hi).
fn range_mask(lo: f32, hi: f32) -> u32 {
    let s_lo = u32(round(16.0 * lo));
    let s_hi = u32(round(16.0 * hi));
    let m = (0xffffu << s_lo) & ~select(0xffffu << s_hi, 0u, s_hi >= 16u);
    return m & 0xffffu;
}

// Compress the low bit of each nibble into 8 contiguous bits (inverse of spread8).
fn compress8(v: u32) -> u32 {
    var x = v & 0x11111111u;
    x = (x | (x >> 3u)) & 0x03030303u;
    x = (x | (x >> 6u)) & 0x000F000Fu;
    x = (x | (x >> 12u)) & 0xffu;
    return x;
}

// Spread 8 mask bits into the low bit of each nibble.
fn spread8(b: u32) -> u32 {
    var x = (b | (b << 12u)) & 0x000F000Fu;
    x = (x | (x << 6u)) & 0x03030303u;
    x = (x | (x << 3u)) & 0x11111111u;
    return x;
}

struct Setup {
    // Infinite line (unclipped direction, offset from a clipped point) for classification.
    a: f32,
    b: f32,
    d: f32,
    h: f32,
    // Segment clipped to the tile's x-range: y-range per row.
    q0y: f32,
    q1y: f32,
    // For CELL_MODE 2: x on the line as a function of y.
    x_at_y0: f32,
    dxdy: f32,
    // y_edge term.
    ye: f32,
    ye_pos: bool,
    has_ye: bool,
    sign_pos: bool,
    // LUT walk parameters (fill_path_ms) for the tile-box-clipped segment.
    la: f32,
    lb: f32,
    x_sign: f32,
    x0i: i32,
    y0i: i32,
    is_pos: bool,
}

fn setup(e: Entry) -> Setup {
    let l = lines[e.line & ~LINE_LEFT_EDGE];
    let origin = vec2(f32((e.key & 0x3ffu) * TILE), f32(((e.key >> 10u) & 0x3ffu) * TILE));
    let p0 = l.xy - origin;
    let p1 = l.zw - origin;
    let delta = p1 - p0;
    var q0 = p0;
    var q1 = p1;
    if delta.x != 0.0 {
        let ta = (0.0 - p0.x) / delta.x;
        let tb = (f32(TILE) - p0.x) / delta.x;
        let t_lo = max(min(ta, tb), 0.0);
        let t_hi = max(min(max(ta, tb), 1.0), t_lo);
        q0 = p0 + t_lo * delta;
        q1 = p0 + t_hi * delta;
    }
    var s: Setup;
    s.a = delta.y;
    s.b = -delta.x;
    s.d = s.a * q0.x + s.b * q0.y;
    s.h = (abs(s.a) + abs(s.b)) * (0.5 - 1.0 / 65536.0);
    s.q0y = q0.y;
    s.q1y = q1.y;
    s.dxdy = select(0.0, delta.x / delta.y, delta.y != 0.0);
    s.x_at_y0 = p0.x - p0.y * s.dxdy;
    s.ye = 0.0;
    s.ye_pos = false;
    s.has_ye = false;
    if (e.line & LINE_LEFT_EDGE) != 0u && s.b != 0.0 {
        s.ye = s.d / s.b;
        s.ye_pos = delta.x > 0.0;
        s.has_ye = true;
    }
    s.sign_pos = s.a < 0.0;

    // Clip to y in [0, TILE] as well for the LUT walk.
    var c0 = q0;
    var c1 = q1;
    let dy = q1.y - q0.y;
    if dy != 0.0 {
        let ta = (0.0 - q0.y) / dy;
        let tb = (f32(TILE) - q0.y) / dy;
        let t_lo = max(min(ta, tb), 0.0);
        let t_hi = max(min(max(ta, tb), 1.0), t_lo);
        let dq = q1 - q0;
        c0 = q0 + t_lo * dq;
        c1 = q0 + t_hi * dq;
    }
    let is_down = c1.y >= c0.y;
    let xy0 = select(c1, c0, is_down);
    let xy1 = select(c0, c1, is_down);
    let ldx = abs(xy1.x - xy0.x);
    let ldy = xy1.y - xy0.y;
    let idxdy = 1.0 / (ldx + ldy);
    var la = ldx * idxdy;
    let is_pos = xy1.x >= xy0.x;
    let x_sign = select(-1.0, 1.0, is_pos);
    let xt0 = floor(xy0.x * x_sign);
    let c = xy0.x * x_sign - xt0;
    let y0i = floor(xy0.y);
    let ytop = y0i + 1.0;
    let lb = min((ldy * c + ldx * (ytop - xy0.y)) * idxdy, ONE_MINUS_ULP);
    let count_x = span(xy0.x, xy1.x) - 1u;
    let count = count_x + span(xy0.y, xy1.y);
    let robust_err = floor(la * (f32(count) - 1.0) + lb) - f32(count_x);
    if robust_err != 0.0 {
        la -= ROBUST_EPSILON * sign(robust_err);
    }
    s.la = la;
    s.lb = lb;
    s.x_sign = x_sign;
    s.x0i = i32(xt0 * x_sign + 0.5 * (x_sign - 1.0));
    s.y0i = i32(y0i);
    s.is_pos = is_pos;
    return s;
}

fn lut_mask(s: Setup, x: u32, y: u32) -> u32 {
    let sub_ix = f32(i32(y) - s.y0i) + s.x_sign * f32(i32(x) - s.x0i);
    let zf = s.la * sub_ix + s.lb;
    let z = floor(zf);
    let block = select(0u, MASK16_WIDTH * MASK16_HALF_HEIGHT, s.is_pos);
    let hh = f32(MASK16_HALF_HEIGHT);
    let row = u32(floor(min(s.la * hh, hh - 1.0))) * MASK16_WIDTH;
    let col = u32(floor((zf - z) * f32(MASK16_WIDTH)));
    let ix = block + row + col;
    return (mask_lut[ix >> 1u] >> ((ix & 1u) * 16u)) & 0xffffu;
}

// Direct evaluation: samples right of the line (CELL_MODE 2).
fn direct_mask(s: Setup, x: u32, y: u32) -> u32 {
    var m = 0u;
    for (var i = 0u; i < 16u; i++) {
        let ys = f32(y) + (f32(i) + 0.5) / 16.0;
        let xl = fma(ys, s.dxdy, s.x_at_y0);
        m |= u32(xl < f32(x) + SAMPLE_X[i]) << i;
    }
    return m;
}

fn classify(s: Setup, x: u32, y: u32) -> vec2<bool> {
    let t_y = s.b * (f32(y) + 0.5) - s.d;
    let v = fma(s.a, f32(x) + 0.5, t_y);
    return vec2(abs(v) < s.h, (v > 0.0) == (s.a > 0.0));
}

// SWAR resolve of one pixel. `w0`, `w1`: slot words (4-bit lanes biased at 8) holding the
// adders' sum; `pm`: the head's own (plus, minus) sample masks; `lo`, `hi`: the same, packed
// signed. Sample s has winding lane - 8 + own + backdrop. Nonzero: covered iff
// lane != base - own with base = 8 - backdrop; lanes are in [1, 15], so expected values
// outside that range never match, encoded below with e = 0 or a forced bit. Even-odd:
// parity of lane + own + backdrop.
fn resolve_words(w0: u32, w1: u32, pm: vec2<u32>, lo: u32, hi: u32, backdrop: i32, even_odd: bool) -> u32 {
    let r = 0x11111111u;
    let plus_lo = spread8(pm.x & 0xffu);
    let plus_hi = spread8(pm.x >> 8u);
    let minus_lo = spread8(pm.y & 0xffu);
    let minus_hi = spread8(pm.y >> 8u);
    var x0: u32;
    var x1: u32;
    if even_odd {
        let pb = u32(backdrop & 1) * r;
        x0 = (w0 ^ plus_lo ^ minus_lo ^ pb) & r;
        x1 = (w1 ^ plus_hi ^ minus_hi ^ pb) & r;
    } else {
        let base = 8 - backdrop;
        var e0: u32;
        var e1: u32;
        var force0 = 0u;
        var force1 = 0u;
        if base >= 1 && base <= 14 {
            e0 = u32(base) * r - lo;
            e1 = u32(base) * r - hi;
        } else if base == 15 {
            e0 = 15u * r - plus_lo;
            e1 = 15u * r - plus_hi;
            force0 = minus_lo;
            force1 = minus_hi;
        } else if base == 16 {
            e0 = 15u * plus_lo;
            e1 = 15u * plus_hi;
        } else if base == 0 {
            e0 = minus_lo;
            e1 = minus_hi;
        } else {
            e0 = 0u;
            e1 = 0u;
        }
        x0 = (w0 ^ e0) | force0;
        x1 = (w1 ^ e1) | force1;
        x0 |= x0 >> 2u;
        x0 |= x0 >> 1u;
        x1 |= x1 >> 2u;
        x1 |= x1 >> 1u;
    }
    return compress8(x0) | (compress8(x1) << 8u);
}

// Store a resolved pixel (or the compact partial record) for the piece headed at `e`.
fn store_pixel(e: Entry, even_odd: bool, cell: u32, w0: u32, w1: u32, pm: vec2<u32>, lo: u32, hi: u32, pack: ptr<function, u32>) {
    let out_ix = e.out & ~OUT_PARTIAL;
    if (e.out & OUT_PARTIAL) != 0u {
        let base_ix = (out_ix * PIXELS + cell) * 4u;
        partials_out[base_ix] = i32(w0);
        partials_out[base_ix + 1u] = i32(w1);
        partials_out[base_ix + 2u] = i32(pm.x | (pm.y << 16u));
        partials_out[base_ix + 3u] = e.backdrop;
    } else {
        let mask = resolve_words(w0, w1, pm, lo, hi, e.backdrop, even_odd);
        *pack |= mask << (16u * (cell & 1u));
        if (cell & 1u) == 1u {
            mask_out[out_ix * (PIXELS / 2u) + (cell >> 1u)] = *pack;
            *pack = 0u;
        }
    }
}

// Half-plane mask for one cell without FIFO state (CELL_MODE 0, 2 or 3).
fn cell_hp(s: Setup, x: u32, y: u32) -> u32 {
    if CELL_MODE == 2u {
        return direct_mask(s, x, y);
    }
    let cl = classify(s, x, y);
    if CELL_MODE == 3u {
        // Gather for every cell, select afterwards: no divergence, dense gathers.
        let m = lut_mask(s, x, y);
        return select(select(0u, 0xffffu, cl.y), m, cl.x);
    }
    if cl.x {
        return lut_mask(s, x, y);
    }
    return select(0u, 0xffffu, cl.y);
}

fn select_entry(f: Entry, t: Entry, cond: bool) -> Entry {
    return Entry(select(f.key, t.key, cond), select(f.line, t.line, cond),
                 select(f.backdrop, t.backdrop, cond), select(f.out, t.out, cond));
}

struct RowMasks {
    range: u32,
    ye: u32,
}

fn row_masks(s: Setup, y: u32) -> RowMasks {
    var rm: RowMasks;
    let yrel = s.q0y - f32(y);
    let y0c = clamp(yrel, 0.0, 1.0);
    let y1c = clamp(yrel + (s.q1y - s.q0y), 0.0, 1.0);
    rm.range = range_mask(min(y0c, y1c), max(y0c, y1c));
    rm.ye = 0u;
    if s.has_ye {
        rm.ye = range_mask(clamp(s.ye - f32(y), 0.0, 1.0), 1.0);
    }
    return rm;
}

// Per-sample values for the cell as (plus, minus) 16-bit masks.
fn cell_payload(s: Setup, hp: u32, rm: RowMasks) -> vec2<u32> {
    let cross = rm.range & hp;
    var plus = select(0u, cross, s.sign_pos);
    var minus = select(cross, 0u, s.sign_pos);
    plus |= select(0u, rm.ye, s.ye_pos);
    minus |= select(rm.ye, 0u, s.ye_pos);
    let both = plus & minus;
    return vec2(plus & ~both, minus & ~both);
}

