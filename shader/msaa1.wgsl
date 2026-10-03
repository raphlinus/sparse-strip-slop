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

var<workgroup> sh_slot: array<atomic<u32>, 2u * WG_SIZE>;
var<workgroup> sh_scan: array<u32, WG_SIZE>;

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

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let i0 = wg_id.x * WG_SIZE + t;
    let e = entries[i0];
    let boundary = t == 0u || entries[i0 - 1u].key != e.key;
    var v = select(0u, t, boundary);
    sh_scan[t] = v;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i++) {
        workgroupBarrier();
        var other = 0u;
        if t >= (1u << i) {
            other = sh_scan[t - (1u << i)];
        }
        workgroupBarrier();
        v = max(v, other);
        sh_scan[t] = v;
    }
    let head = v + ((t - v) & ~7u);
    let is_head = t == head;
    let even_odd = (path_flags[e.key >> 20u] & 1u) != 0u;
    let backdrop = e.backdrop;
    let is_partial = (e.out & OUT_PARTIAL) != 0u;
    let out_ix = e.out & ~OUT_PARTIAL;

    let s = setup(e);
    var pack = 0u;
    for (var c = 0u; c < 4u; c++) {
        let cx0 = c >> 1u;
        let cy0 = c & 1u;
        // CELL_MODE 1: classify the coset and fill a FIFO of LUT masks.
        var hits = 0u;
        var sides = 0u;
        var f0 = 0u;
        var f1 = 0u;
        var f2 = 0u;
        var f3 = 0u;
        if CELL_MODE == 1u {
            for (var r = 0u; r < HALF; r++) {
                for (var k = 0u; k < HALF; k++) {
                    let cl = classify(s, 2u * k + cx0, 2u * r + cy0);
                    let pos = r * HALF + k;
                    hits |= u32(cl.x) << pos;
                    sides |= u32(cl.y) << pos;
                }
            }
            var m = hits;
            for (var i = 0u; i < HALF; i++) {
                let pos = firstTrailingBit(m);
                var hp = 0u;
                if m != 0u {
                    hp = lut_mask(s, 2u * (pos % HALF) + cx0, 2u * (pos / HALF) + cy0);
                }
                f3 = select(f3, hp, i == 3u);
                f2 = select(f2, hp, i == 2u);
                f1 = select(f1, hp, i == 1u);
                f0 = select(f0, hp, i == 0u);
                m &= m - 1u;
            }
        }

        var rm: RowMasks;
        for (var pos = 0u; pos < N_POS; pos++) {
            let x = 2u * (pos % HALF) + cx0;
            let y = 2u * (pos / HALF) + cy0;
            if pos % HALF == 0u {
                rm = row_masks(s, y);
            }
            var hp: u32;
            if CELL_MODE == 0u {
                let cl = classify(s, x, y);
                if cl.x {
                    hp = lut_mask(s, x, y);
                } else {
                    hp = select(0u, 0xffffu, cl.y);
                }
            } else if CELL_MODE == 1u {
                if ((hits >> pos) & 1u) != 0u {
                    hp = f0;
                    f0 = f1;
                    f1 = f2;
                    f2 = f3;
                } else {
                    hp = select(0u, 0xffffu, ((sides >> pos) & 1u) != 0u);
                }
            } else {
                hp = direct_mask(s, x, y);
            }
            let pm = cell_payload(s, hp, rm);
            let lo = spread8(pm.x & 0xffu) - spread8(pm.y & 0xffu);
            let hi = spread8(pm.x >> 8u) - spread8(pm.y >> 8u);

            if is_head {
                atomicStore(&sh_slot[2u * t], 0x88888888u);
                atomicStore(&sh_slot[2u * t + 1u], 0x88888888u);
            }
            workgroupBarrier();
            if !is_head {
                atomicAdd(&sh_slot[2u * head], lo);
                atomicAdd(&sh_slot[2u * head + 1u], hi);
            }
            workgroupBarrier();
            if is_head {
                let w0 = atomicLoad(&sh_slot[2u * t]);
                let w1 = atomicLoad(&sh_slot[2u * t + 1u]);
                let cell = c * N_POS + pos;
                var mask = 0u;
                for (var i = 0u; i < 16u; i++) {
                    let word = select(w1, w0, i < 8u);
                    let lane = i32((word >> (4u * (i & 7u))) & 0xfu) - 8;
                    let own = i32((pm.x >> i) & 1u) - i32((pm.y >> i) & 1u);
                    let w = lane + own + backdrop;
                    if is_partial {
                        partials_out[(out_ix * PIXELS + cell) * 16u + i] = w;
                    }
                    let covered = select(w != 0, (w & 1) != 0, even_odd);
                    mask |= u32(covered) << i;
                }
                if !is_partial {
                    pack |= mask << (16u * (cell & 1u));
                    if (cell & 1u) == 1u {
                        mask_out[out_ix * (PIXELS / 2u) + (cell >> 1u)] = pack;
                        pack = 0u;
                    }
                }
            }
        }
    }
}
