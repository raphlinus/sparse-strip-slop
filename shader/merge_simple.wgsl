// Stripped-down merge: one entry per thread, sparse coset walk, one pixel per barrier
// pair, no arrays. Prepended with `const TILE` and `const WG_SIZE`.
//
// Per entry: clip the line to the tile's x-range, then for each of the 4 cosets classify
// its cells against the infinite line (hit = line passes through the cell), evaluate the
// area of the (at most TILE/2) hit cells into a small FIFO, and walk the coset's cells in
// order: hit cells pop the FIFO, other cells get the row's dy or 0 by side, plus the
// y_edge term. Each cell value is summed into the run head's slot in workgroup memory;
// heads resolve and write.

const HALF: u32 = TILE / 2u;          // cells per coset row, also the FIFO depth
const N_POS: u32 = HALF * HALF;       // cells per coset
const PIXELS: u32 = TILE * TILE;
const FIXED_ONE: f32 = 65536.0;
const LINE_LEFT_EDGE: u32 = 0x80000000u;
const OUT_PARTIAL: u32 = 0x80000000u;

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

var<workgroup> sh_slot: array<atomic<i32>, WG_SIZE>;
var<workgroup> sh_scan: array<u32, WG_SIZE>;

// Mean of clamp(1 - x, 0, 1) over [xmin, xmax] (pixel-relative): coverage to the right.
fn area_weight(xmin: f32, xmax: f32) -> f32 {
    let w = xmax - xmin;
    let mid = 1.0 - clamp(0.5 * (xmin + xmax), 0.0, 1.0);
    let x0c = clamp(xmin, 0.0, 1.0);
    let x1c = clamp(xmax, 0.0, 1.0);
    let left = clamp(-xmin, 0.0, w);
    let slanted = (left + (x1c - x0c) * (1.0 - 0.5 * (x0c + x1c))) / w;
    return select(slanted, mid, w < 1.0e-6);
}

fn resolve(w: i32, even_odd: bool) -> u32 {
    let m = w & 131071;
    let eo = select(m, 131072 - m, m > 65536);
    let cv = select(min(abs(w), 65536), eo, even_odd);
    return u32((cv * 255 + 32768) >> 16u);
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let i0 = wg_id.x * WG_SIZE + t;
    let e = entries[i0];

    // Run head index within the workgroup: inclusive prefix max of boundary positions.
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
    // Pieces of 8 entries from the head; the piece head owns the slot.
    let head = v + ((t - v) & ~7u);
    let is_head = t == head;
    let even_odd = (path_flags[e.key >> 20u] & 1u) != 0u;
    let backdrop = e.backdrop * 65536;
    let is_partial = (e.out & OUT_PARTIAL) != 0u;
    let out_ix = e.out & ~OUT_PARTIAL;

    // Line setup, tile-local. Direction (a, b) from the unclipped segment; the segment
    // itself clipped to x in [0, TILE] for the per-row extents.
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
    let a = delta.y;
    let b = -delta.x;
    let d = a * q0.x + b * q0.y;
    let h = (abs(a) + abs(b)) * (0.5 - 1.0 / 65536.0);
    let a_pos = a > 0.0;
    let dyc = q1.y - q0.y;
    let dxc = q1.x - q0.x;
    let recip = 1.0 / dyc;
    var ye = 0.0;
    var ye_sign = 0.0;
    if (e.line & LINE_LEFT_EDGE) != 0u && b != 0.0 {
        ye = d / b;
        ye_sign = sign(delta.x);
    }

    var pack = 0u;
    for (var c = 0u; c < 4u; c++) {
        let cx0 = c >> 1u;  // x parity of this coset
        let cy0 = c & 1u;   // y parity

        // Classify the N_POS cells: hit if the infinite line passes through the cell.
        var hits = 0u;
        var sides = 0u;
        for (var r = 0u; r < HALF; r++) {
            let t_y = b * (f32(2u * r + cy0) + 0.5) - d;
            for (var k = 0u; k < HALF; k++) {
                let s = fma(a, f32(2u * k + cx0) + 0.5, t_y);
                let pos = r * HALF + k;
                hits |= u32(abs(s) < h) << pos;
                sides |= u32((s > 0.0) == a_pos) << pos;
            }
        }

        // Area of each hit cell, in walk order, into a FIFO of depth HALF (<= 4).
        var f0 = 0;
        var f1 = 0;
        var f2 = 0;
        var f3 = 0;
        var m = hits;
        for (var i = 0u; i < HALF; i++) {
            let pos = firstTrailingBit(m);
            var area = 0;
            if m != 0u {
                let x = 2u * (pos % HALF) + cx0;
                let y = 2u * (pos / HALF) + cy0;
                let yrel = q0.y - f32(y);
                let y0c = clamp(yrel, 0.0, 1.0);
                let y1c = clamp(yrel + dyc, 0.0, 1.0);
                let dy = y0c - y1c;
                if dy != 0.0 {
                    let xs = q0.x - f32(x);
                    let xa = xs + (y0c - yrel) * recip * dxc;
                    let xb = xs + (y1c - yrel) * recip * dxc;
                    area = i32(round(area_weight(min(xa, xb), max(xa, xb)) * dy * FIXED_ONE));
                }
            }
            // Shift in at the tail (fills f0 first).
            f3 = select(f3, area, i == 3u);
            f2 = select(f2, area, i == 2u);
            f1 = select(f1, area, i == 1u);
            f0 = select(f0, area, i == 0u);
            m &= m - 1u;
        }

        // Walk the coset's cells.
        var dy_fixed = 0;
        var ye_fixed = 0;
        for (var pos = 0u; pos < N_POS; pos++) {
            if pos % HALF == 0u {
                let y = 2u * (pos / HALF) + cy0;
                let yrel = q0.y - f32(y);
                let dy = clamp(yrel, 0.0, 1.0) - clamp(yrel + dyc, 0.0, 1.0);
                dy_fixed = i32(round(dy * FIXED_ONE));
                ye_fixed = i32(round(ye_sign * clamp(f32(y) - ye + 1.0, 0.0, 1.0) * FIXED_ONE));
            }
            var val: i32;
            if ((hits >> pos) & 1u) != 0u {
                val = f0;
                f0 = f1;
                f1 = f2;
                f2 = f3;
            } else {
                val = select(0, dy_fixed, ((sides >> pos) & 1u) != 0u);
            }
            val += ye_fixed;

            // Merge: head stores, others add, head reads back.
            if is_head {
                atomicStore(&sh_slot[t], val);
            }
            workgroupBarrier();
            if !is_head {
                atomicAdd(&sh_slot[head], val);
            }
            workgroupBarrier();
            if is_head {
                let w = atomicLoad(&sh_slot[t]) + backdrop;
                let cell = c * N_POS + pos;  // coset-major output index
                if is_partial {
                    partials_out[out_ix * PIXELS + cell] = w;
                } else {
                    pack |= resolve(w, even_odd) << (8u * (cell & 3u));
                    if (cell & 3u) == 3u {
                        alpha_out[out_ix * (PIXELS / 4u) + (cell >> 2u)] = pack;
                        pack = 0u;
                    }
                }
            }
        }
    }
}
