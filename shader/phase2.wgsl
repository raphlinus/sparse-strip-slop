// Per-line setup and per-cell evaluation shared by the merge variants.
// Appended after a header that defines WG_SIZE and CHUNK.

const TILE: u32 = 8u;
const FIXED_ONE: f32 = 65536.0;
const LINE_LEFT_EDGE: u32 = 0x80000000u;
const OUT_PARTIAL: u32 = 0x80000000u;
const N_CHUNKS: u32 = 16u / CHUNK;

struct Entry {
    key: u32,
    line: u32,
    backdrop: i32,
    out: u32,
}

@group(0) @binding(0) var<storage, read> lines: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> entries: array<Entry>;
@group(0) @binding(2) var<storage, read_write> alpha_out: array<u32>;
@group(0) @binding(3) var<storage, read_write> partials_out: array<i32>;
@group(0) @binding(4) var<storage, read> path_flags: array<u32>;

struct LineSetup {
    // Infinite line a*x + b*y = d (tile-local), direction from the unclipped segment.
    a: f32,
    b: f32,
    d: f32,
    h: f32,
    // Segment clipped to x in [0, TILE], tile-local.
    q0: vec2<f32>,
    q1: vec2<f32>,
    recip: f32,
    // y_edge term: ye_sign * clamp(row - ye + 1, 0, 1) per row; ye_sign = 0 if none.
    ye: f32,
    ye_sign: f32,
}

fn line_setup(e: Entry) -> LineSetup {
    let l = lines[e.line & ~LINE_LEFT_EDGE];
    let tx = e.key & 0x3ffu;
    let ty = (e.key >> 10u) & 0x3ffu;
    let origin = vec2(f32(tx * TILE), f32(ty * TILE));
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
    var ls: LineSetup;
    ls.a = delta.y;
    ls.b = -delta.x;
    ls.d = ls.a * q0.x + ls.b * q0.y;
    ls.h = (abs(ls.a) + abs(ls.b)) * (0.5 - 1.0 / 65536.0);
    ls.q0 = q0;
    ls.q1 = q1;
    ls.recip = 1.0 / (q1.y - q0.y);
    ls.ye = 0.0;
    ls.ye_sign = 0.0;
    if (e.line & LINE_LEFT_EDGE) != 0u && ls.b != 0.0 {
        ls.ye = ls.d / ls.b;
        ls.ye_sign = sign(delta.x);
    }
    return ls;
}

// Coset walk: pos in 0..16 within coset c maps to pixel (x, y).
fn coset_x(c: u32, pos: u32) -> u32 {
    return 2u * (pos & 3u) + (c >> 1u);
}
fn coset_y(c: u32, pos: u32) -> u32 {
    return 2u * (pos >> 2u) + (c & 1u);
}

struct RowConsts {
    dy_fixed: i32,
    ye_fixed: i32,
}

fn row_dy(ls: LineSetup, y: u32) -> f32 {
    let yrel = ls.q0.y - f32(y);
    let y0c = clamp(yrel, 0.0, 1.0);
    let y1c = clamp(yrel + (ls.q1.y - ls.q0.y), 0.0, 1.0);
    return y0c - y1c;
}

fn row_consts(ls: LineSetup, y: u32) -> RowConsts {
    var rc: RowConsts;
    rc.dy_fixed = i32(round(row_dy(ls, y) * FIXED_ONE));
    rc.ye_fixed = i32(round(ls.ye_sign * clamp(f32(y) - ls.ye + 1.0, 0.0, 1.0) * FIXED_ONE));
    return rc;
}

// Mean over the segment's x-interval [xmin, xmax] (relative to the pixel's left edge) of
// the coverage to the right, clamp(1 - x, 0, 1). Algebraically fill_path's formula, but
// written without the difference of squares so near-vertical segments stay accurate.
fn area_weight(xmin: f32, xmax: f32) -> f32 {
    let w = xmax - xmin;
    let mid = 1.0 - clamp(0.5 * (xmin + xmax), 0.0, 1.0);
    let x0c = clamp(xmin, 0.0, 1.0);
    let x1c = clamp(xmax, 0.0, 1.0);
    // Length of the interval left of the pixel (full weight), then the part inside.
    let left = clamp(-xmin, 0.0, w);
    let slanted = (left + (x1c - x0c) * (1.0 - 0.5 * (x0c + x1c))) / w;
    return select(slanted, mid, w < 1.0e-6);
}

// Per-pixel area for a partial cell, in fixed point (without y_edge).
fn cell_area(ls: LineSetup, x: u32, y: u32) -> i32 {
    let yrel = ls.q0.y - f32(y);
    let y0c = clamp(yrel, 0.0, 1.0);
    let y1c = clamp(yrel + (ls.q1.y - ls.q0.y), 0.0, 1.0);
    let dy = y0c - y1c;
    if dy == 0.0 {
        return 0;
    }
    let t0 = (y0c - yrel) * ls.recip;
    let t1 = (y1c - yrel) * ls.recip;
    let dxc = ls.q1.x - ls.q0.x;
    let xs = ls.q0.x - f32(x);
    let x0 = xs + t0 * dxc;
    let x1 = xs + t1 * dxc;
    let xmin = min(x0, x1);
    let xmax = max(x0, x1);
    let area = area_weight(xmin, xmax);
    return i32(round(area * dy * FIXED_ONE));
}

// Per (line, coset) state: hit mask, side mask, and a 4-deep FIFO of partial-cell areas
// in walk order.
struct CosetState {
    hits: u32,
    sides: u32,
    f0: i32,
    f1: i32,
    f2: i32,
    f3: i32,
}

fn coset_setup(ls: LineSetup, c: u32) -> CosetState {
    var cs: CosetState;
    var hits = 0u;
    var sides = 0u;
    let a_pos = ls.a > 0.0;
    for (var r = 0u; r < 4u; r++) {
        let y = 2u * r + (c & 1u);
        let t_y = ls.b * (f32(y) + 0.5) - ls.d;
        for (var k = 0u; k < 4u; k++) {
            let x = 2u * k + (c >> 1u);
            let s = fma(ls.a, f32(x) + 0.5, t_y);
            let pos = r * 4u + k;
            hits |= u32(abs(s) < ls.h) << pos;
            sides |= u32((s > 0.0) == a_pos) << pos;
        }
    }
    cs.hits = hits;
    cs.sides = sides;
    // Fixed-trip fill of the FIFO.
    var m = hits;
    var vals = array<i32, 4>(0, 0, 0, 0);
    for (var i = 0u; i < 4u; i++) {
        let pos = firstTrailingBit(m);
        var v = 0;
        if m != 0u {
            v = cell_area(ls, coset_x(c, pos), coset_y(c, pos));
        }
        vals[i] = v;
        m &= m - 1u;
    }
    cs.f0 = vals[0];
    cs.f1 = vals[1];
    cs.f2 = vals[2];
    cs.f3 = vals[3];
    return cs;
}

// Produce the CHUNK values for chunk `ch` of coset `c`, consuming the FIFO.
fn compute_chunk(ls: LineSetup, cs: ptr<function, CosetState>, c: u32, ch: u32) -> array<i32, CHUNK> {
    var out: array<i32, CHUNK>;
    var rc: RowConsts;
    for (var i = 0u; i < CHUNK; i++) {
        let pos = ch * CHUNK + i;
        if (pos & 3u) == 0u {
            // Row constants are shared by the 4 cells of a coset row.
            rc = row_consts(ls, coset_y(c, pos));
        }
        var v: i32;
        if (((*cs).hits >> pos) & 1u) != 0u {
            v = (*cs).f0;
            (*cs).f0 = (*cs).f1;
            (*cs).f1 = (*cs).f2;
            (*cs).f2 = (*cs).f3;
        } else {
            v = select(0, rc.dy_fixed, (((*cs).sides >> pos) & 1u) != 0u);
        }
        out[i] = v + rc.ye_fixed;
    }
    return out;
}

fn resolve(w: i32, even_odd: bool) -> u32 {
    var cv: i32;
    if even_odd {
        var m = w % 131072;
        if m < 0 {
            m += 131072;
        }
        cv = select(m, 131072 - m, m > 65536);
    } else {
        cv = min(abs(w), 65536);
    }
    return u32((cv * 255 + 32768) >> 16u);
}

// Store the result for the run/piece headed at entry `e`: resolved alpha, or an
// unresolved partial if the output index carries OUT_PARTIAL. `vals` is for chunk
// `ch` of coset `c`.
fn store_result(e: Entry, vals: array<i32, CHUNK>, c: u32, ch: u32) {
    let bd = e.backdrop * 65536;
    if (e.out & OUT_PARTIAL) != 0u {
        let base = (e.out & ~OUT_PARTIAL) * 64u + c * 16u + ch * CHUNK;
        for (var i = 0u; i < CHUNK; i++) {
            partials_out[base + i] = vals[i] + bd;
        }
    } else {
        let even_odd = (path_flags[e.key >> 20u] & 1u) != 0u;
        let base = e.out * 16u + c * 4u + (ch * CHUNK) / 4u;
        for (var g = 0u; g < CHUNK / 4u; g++) {
            var packed = 0u;
            for (var i = 0u; i < 4u; i++) {
                packed |= resolve(vals[g * 4u + i] + bd, even_odd) << (8u * i);
            }
            alpha_out[base + g] = packed;
        }
    }
}
