// MSAA 16 with 8x8 entries merged as load-balanced 4x4 quadrants (see merge_quadrant.wgsl
// for the expansion, scans, item mapping and quadrant backdrop deltas, and msaa_common.wgsl
// for the payload and resolve). Items of one (run, quadrant) group form pieces of 8 (head +
// 7 adders, as the 4-bit lanes require); a run is resolved in place only if every group fits
// in one piece and the run lies in one workgroup, otherwise each piece writes a compact
// partial record (w0, w1, own, backdrop), laid out per workgroup segment as
// sum over quadrants of max(1, ceil(count_q / 8)) records, untouched quadrants included.
// Output per resolved run: 32 u32, quadrant-major (q * 16 + ly * 4 + lx), two pixels each.
// Prepended with msaa_common.wgsl; needs `const TILE` = 8, `const WG_SIZE`, `const CELL_MODE`.

const ENTRIES: u32 = WG_SIZE;
const MAX_ITEMS: u32 = 3u * WG_SIZE;
const BIAS: u32 = 0x88888888u;

var<workgroup> sh_head: array<u32, ENTRIES>;
var<workgroup> sh_cnt: array<u32, ENTRIES>;
var<workgroup> sh_scan: array<u32, ENTRIES>;
var<workgroup> sh_delta: array<atomic<i32>, 4u * ENTRIES>;
var<workgroup> sh_qcount: array<atomic<u32>, 4u * ENTRIES>;   // items per (run head, quadrant)
var<workgroup> sh_touched: array<atomic<u32>, ENTRIES>;
var<workgroup> sh_slot: array<atomic<u32>, 2u * MAX_ITEMS>;
var<workgroup> sh_base: u32;
var<workgroup> sh_totals: vec4<u32>;

fn byte(v: u32, q: u32) -> u32 {
    return (v >> (8u * q)) & 0xffu;
}

fn x_at(p0: vec2<f32>, p1: vec2<f32>, y: f32) -> f32 {
    if y == p0.y {
        return p0.x;
    }
    if y == p1.y {
        return p1.x;
    }
    let d = p1 - p0;
    return p0.x + (y - p0.y) * (d.x / d.y);
}

// Partial record index of (quadrant q, piece k) within a run segment whose counts are in
// sh_qcount at run head h.
fn record_index(h: u32, q: u32, k: u32) -> u32 {
    var ix = 0u;
    for (var qq = 0u; qq < q; qq++) {
        let n = atomicLoad(&sh_qcount[h * 4u + qq]);
        ix += max(1u, (n + 7u) / 8u);
    }
    return ix + k;
}

struct Item {
    valid: bool,
    is_head: bool,
    q: u32,
    slot: u32,          // item position of the piece head
    s: Setup,
    out: u32,           // run's out field (with OUT_PARTIAL) for heads
    rec: u32,           // partial record index for heads
    bd: i32,            // backdrop + quadrant delta (piece 0 only), for heads
    even_odd: bool,
}

// Setup of (entry, quadrant): like msaa_common::setup but clipped to the quadrant's box,
// with the quadrant's left-edge rule.
fn setup_quad(ent: Entry, q: u32) -> Setup {
    let l = lines[ent.line & ~LINE_LEFT_EDGE];
    let origin = vec2(f32((ent.key & 0x3ffu) * TILE), f32(((ent.key >> 10u) & 0x3ffu) * TILE));
    let p0 = l.xy - origin;
    let p1 = l.zw - origin;
    let d = p1 - p0;
    let qx = q & 1u;
    let qy = q >> 1u;
    let x_lo = f32(4u * qx);
    let y_lo = f32(4u * qy);
    // Clip x to the quadrant's range.
    var q0 = p0;
    var q1 = p1;
    if d.x != 0.0 {
        let ta = (x_lo - p0.x) / d.x;
        let tb = (x_lo + 4.0 - p0.x) / d.x;
        let t_lo = max(min(ta, tb), 0.0);
        let t_hi = max(min(max(ta, tb), 1.0), t_lo);
        q0 = p0 + t_lo * d;
        q1 = p0 + t_hi * d;
    }
    var s: Setup;
    s.a = d.y;
    s.b = -d.x;
    s.d = s.a * q0.x + s.b * q0.y;
    s.h = (abs(s.a) + abs(s.b)) * (0.5 - 1.0 / 65536.0);
    s.q0y = q0.y;
    s.q1y = q1.y;
    s.dxdy = select(0.0, d.x / d.y, d.y != 0.0);
    s.x_at_y0 = p0.x - p0.y * s.dxdy;
    s.sign_pos = s.a < 0.0;
    s.ye = 0.0;
    s.ye_pos = false;
    s.has_ye = false;
    if qx == 1u {
        // Left edge is the tile's vertical midline. Flag if the segment (clipped to the
        // tile's x-range, then to this quadrant row) has points left of it.
        var c0 = p0;
        var c1 = p1;
        if d.x != 0.0 {
            let ta = (0.0 - p0.x) / d.x;
            let tb = (8.0 - p0.x) / d.x;
            let t_lo = max(min(ta, tb), 0.0);
            let t_hi = max(min(max(ta, tb), 1.0), t_lo);
            c0 = p0 + t_lo * d;
            c1 = p0 + t_hi * d;
        }
        let ymin = min(c0.y, c1.y);
        let ymax = max(c0.y, c1.y);
        let ya = max(ymin, y_lo);
        let yb = min(ymax, y_lo + 4.0);
        var xa = min(c0.x, c1.x);
        if d.y != 0.0 {
            xa = min(x_at(p0, p1, ya), x_at(p0, p1, yb));
        }
        if xa < 4.0 && d.x != 0.0 {
            s.ye = p0.y + (4.0 - p0.x) * (d.y / d.x);
            s.ye_pos = d.x > 0.0;
            s.has_ye = true;
        }
    } else if (ent.line & LINE_LEFT_EDGE) != 0u && d.x != 0.0 {
        // Same expression as msaa_common::setup (d / b from the tile-x-clipped start), so
        // the y_edge sample boundary rounds identically to the 8x8 reference.
        let ta = (0.0 - p0.x) / d.x;
        let tb = (8.0 - p0.x) / d.x;
        let t_lo = max(min(ta, tb), 0.0);
        let qt = p0 + t_lo * d;
        let dt = s.a * qt.x + s.b * qt.y;
        let ye_tile = dt / s.b;
        if (qy == 0u && ye_tile < 4.0) || (qy == 1u && ye_tile > 4.0) {
            s.ye = ye_tile;
            s.ye_pos = d.x > 0.0;
            s.has_ye = true;
        }
    }
    // LUT walk for the segment clipped to the quadrant box (y as well).
    var c0 = q0;
    var c1 = q1;
    let dy = q1.y - q0.y;
    if dy != 0.0 {
        let ta = (y_lo - q0.y) / dy;
        let tb = (y_lo + 4.0 - q0.y) / dy;
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

fn make_item(p: u32) -> Item {
    var it: Item;
    it.valid = false;
    it.is_head = false;
    let totals = sh_totals;
    let b1 = totals.x;
    let b2 = b1 + totals.y;
    let b3 = b2 + totals.z;
    let n_items = b3 + totals.w;
    if p >= n_items {
        return it;
    }
    var q = 0u;
    var block = 0u;
    if p >= b3 { q = 3u; block = b3; } else if p >= b2 { q = 2u; block = b2; } else if p >= b1 { q = 1u; block = b1; }
    let r = p - block;
    var lo = 0u;
    var hi = ENTRIES;
    while hi > lo + 1u {
        let mid = (lo + hi) >> 1u;
        if byte(sh_scan[mid], q) <= r {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let e = lo;
    let h = sh_head[e];
    let group_head = block + byte(sh_scan[h], q);
    let rank = p - group_head;
    it.valid = true;
    it.q = q;
    it.slot = group_head + (rank & ~7u);
    it.is_head = (rank & 7u) == 0u;
    let ent = entries[sh_base + e];
    it.s = setup_quad(ent, q);
    if it.is_head {
        let he = entries[sh_base + h];
        it.out = he.out;
        it.rec = (he.out & ~OUT_PARTIAL) + record_index(h, q, rank >> 3u);
        it.bd = select(0, he.backdrop + atomicLoad(&sh_delta[h * 4u + q]), rank < 8u);
        it.even_odd = (path_flags[he.key >> 20u] & 1u) != 0u;
    }
    return it;
}

// Resolve or record one pixel of a piece head.
fn emit(it: Item, cell: u32, w0: u32, w1: u32, pm: vec2<u32>, lo: u32, hi: u32, pack: ptr<function, u32>) {
    if (it.out & OUT_PARTIAL) != 0u {
        let base_ix = (it.rec * 16u + (cell & 15u)) * 4u;
        partials_out[base_ix] = i32(w0);
        partials_out[base_ix + 1u] = i32(w1);
        partials_out[base_ix + 2u] = i32(pm.x | (pm.y << 16u));
        partials_out[base_ix + 3u] = it.bd;
    } else {
        let mask = resolve_words(w0, w1, pm, lo, hi, it.bd, it.even_odd);
        *pack |= mask << (16u * (cell & 1u));
        if (cell & 1u) == 1u {
            mask_out[it.out * (PIXELS / 2u) + (cell >> 1u)] = *pack;
            *pack = 0u;
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let base = wg_id.x * ENTRIES;
    let e = entries[base + t];
    if t == 0u {
        sh_base = base;
    }
    for (var i = t; i < 2u * MAX_ITEMS; i += WG_SIZE) {
        atomicStore(&sh_slot[i], BIAS);
    }
    for (var q = 0u; q < 4u; q++) {
        atomicStore(&sh_delta[t * 4u + q], 0);
        atomicStore(&sh_qcount[t * 4u + q], 0u);
    }
    atomicStore(&sh_touched[t], 0u);

    let boundary = t == 0u || entries[base + t - 1u].key != e.key;
    var v = select(0u, t, boundary);
    sh_head[t] = v;
    for (var i = 0u; i < firstTrailingBit(ENTRIES); i++) {
        workgroupBarrier();
        var other = 0u;
        if t >= (1u << i) {
            other = sh_head[t - (1u << i)];
        }
        workgroupBarrier();
        v = max(v, other);
        sh_head[t] = v;
    }
    let h = v;

    // Touched quadrants and deltas (identical to merge_quadrant.wgsl).
    let l = lines[e.line & ~LINE_LEFT_EDGE];
    let origin = vec2(f32((e.key & 0x3ffu) * TILE), f32(((e.key >> 10u) & 0x3ffu) * TILE));
    let p0 = l.xy - origin;
    let p1 = l.zw - origin;
    let d = p1 - p0;
    var c0 = p0;
    var c1 = p1;
    if d.x != 0.0 {
        let ta = (0.0 - p0.x) / d.x;
        let tb = (8.0 - p0.x) / d.x;
        let t_lo = max(min(ta, tb), 0.0);
        let t_hi = max(min(max(ta, tb), 1.0), t_lo);
        c0 = p0 + t_lo * d;
        c1 = p0 + t_hi * d;
    }
    let ymin = min(c0.y, c1.y);
    let ymax = max(c0.y, c1.y);
    let ymin_u = min(p0.y, p1.y);
    let ymax_u = max(p0.y, p1.y);
    let sign_y = select(-1, 1, p1.y < p0.y);
    let tile_flag = (e.line & LINE_LEFT_EDGE) != 0u;
    var mask = 0u;
    for (var qy = 0u; qy < 2u; qy++) {
        let top = f32(4u * qy);
        let bot = top + 4.0;
        if ymin < bot && ymax > top {
            let ya = max(ymin, top);
            let yb = min(ymax, bot);
            var xa = min(c0.x, c1.x);
            var xb = max(c0.x, c1.x);
            if d.y != 0.0 {
                let x1 = x_at(p0, p1, ya);
                let x2 = x_at(p0, p1, yb);
                xa = min(x1, x2);
                xb = max(x1, x2);
            }
            let qa = u32(clamp(floor(xa * 0.25), 0.0, 1.0));
            let qb = u32(clamp(floor(xb * 0.25), 0.0, 1.0));
            for (var qx = qa; qx <= qb; qx++) {
                mask |= 1u << (2u * qy + qx);
            }
        }
    }
    var d_ne = 0;
    var d_sw = 0;
    var d_se = 0;
    let top_min_x = d.x * d.y > 0.0;
    var ye_tile = 0.0;
    if d.x != 0.0 {
        ye_tile = p0.y + (0.0 - p0.x) * (d.y / d.x);
    }
    if d.y != 0.0 {
        if ymin_u <= 0.0 && 0.0 < ymax_u {
            let inside = !(tile_flag && top_min_x);
            if inside && x_at(p0, p1, 0.0) < 4.0 {
                d_ne += sign_y;
            }
        }
        if ymin_u <= 4.0 && 4.0 < ymax_u {
            var inside = true;
            if tile_flag && d.x != 0.0 {
                inside = select(ye_tile >= 4.0, ye_tile <= 4.0, top_min_x);
            }
            if inside && x_at(p0, p1, 4.0) < 4.0 {
                d_se += sign_y;
            }
        }
    }
    if tile_flag && d.x != 0.0 && ye_tile <= 4.0 {
        let sgn = select(-1, 1, d.x > 0.0);
        d_sw += sgn;
        d_se += sgn;
    }
    let cnt = ((mask >> 0u) & 1u) | (((mask >> 1u) & 1u) << 8u) | (((mask >> 2u) & 1u) << 16u) | (((mask >> 3u) & 1u) << 24u);
    sh_cnt[t] = cnt;
    workgroupBarrier();
    if d_ne != 0 {
        atomicAdd(&sh_delta[h * 4u + 1u], d_ne);
    }
    if d_sw != 0 {
        atomicAdd(&sh_delta[h * 4u + 2u], d_sw);
    }
    if d_se != 0 {
        atomicAdd(&sh_delta[h * 4u + 3u], d_se);
    }
    atomicOr(&sh_touched[h], mask);
    for (var q = 0u; q < 4u; q++) {
        if (mask & (1u << q)) != 0u {
            atomicAdd(&sh_qcount[h * 4u + q], 1u);
        }
    }

    var acc = cnt;
    sh_scan[t] = acc;
    for (var i = 0u; i < firstTrailingBit(ENTRIES); i++) {
        workgroupBarrier();
        var other = 0u;
        if t >= (1u << i) {
            other = sh_scan[t - (1u << i)];
        }
        workgroupBarrier();
        acc += other;
        sh_scan[t] = acc;
    }
    workgroupBarrier();
    if t == ENTRIES - 1u {
        let excl = acc - cnt;
        sh_totals = vec4(byte(excl, 0u) + byte(cnt, 0u), byte(excl, 1u) + byte(cnt, 1u),
                         byte(excl, 2u) + byte(cnt, 2u), byte(excl, 3u) + byte(cnt, 3u));
    }
    sh_scan[t] = acc - cnt;
    workgroupBarrier();

    // Untouched quadrants: uniform mask or one uniform record (run heads).
    if boundary {
        let touched = atomicLoad(&sh_touched[t]);
        let even_odd = (path_flags[e.key >> 20u] & 1u) != 0u;
        for (var q = 0u; q < 4u; q++) {
            if (touched & (1u << q)) == 0u {
                let w = e.backdrop + atomicLoad(&sh_delta[t * 4u + q]);
                if (e.out & OUT_PARTIAL) != 0u {
                    let rec = (e.out & ~OUT_PARTIAL) + record_index(t, q, 0u);
                    for (var i = 0u; i < 16u; i++) {
                        partials_out[(rec * 16u + i) * 4u] = i32(BIAS);
                        partials_out[(rec * 16u + i) * 4u + 1u] = i32(BIAS);
                        partials_out[(rec * 16u + i) * 4u + 2u] = 0;
                        partials_out[(rec * 16u + i) * 4u + 3u] = w;
                    }
                } else {
                    let covered = select(w != 0, (w & 1) != 0, even_odd);
                    let word = select(0u, 0xffffffffu, covered);
                    for (var i = 0u; i < 8u; i++) {
                        mask_out[e.out * (PIXELS / 2u) + q * 8u + i] = word;
                    }
                }
            }
        }
    }

    let it0 = make_item(t);
    let it1 = make_item(t + WG_SIZE);
    let it2 = make_item(t + 2u * WG_SIZE);
    var pack0 = 0u;
    var pack1 = 0u;
    var pack2 = 0u;
    var rm0: RowMasks;
    var rm1: RowMasks;
    var rm2: RowMasks;
    if MPROF == 1u {
        // Ablation: setup only.
        if it0.valid && it1.valid && it0.slot == 0xfffffffu {
            mask_out[0] = it0.q + it1.q + it2.q;
        }
        return;
    }

    for (var cell = 0u; cell < 16u; cell++) {
        let ly = cell >> 2u;
        let lx = cell & 3u;
        if lx == 0u {
            rm0 = row_masks(it0.s, 4u * (it0.q >> 1u) + ly);
            rm1 = row_masks(it1.s, 4u * (it1.q >> 1u) + ly);
            rm2 = row_masks(it2.s, 4u * (it2.q >> 1u) + ly);
        }
        let x0 = 4u * (it0.q & 1u) + lx;
        let x1 = 4u * (it1.q & 1u) + lx;
        let x2 = 4u * (it2.q & 1u) + lx;
        let y0 = 4u * (it0.q >> 1u) + ly;
        let y1 = 4u * (it1.q >> 1u) + ly;
        let y2 = 4u * (it2.q >> 1u) + ly;
        let pm0 = cell_payload(it0.s, cell_hp(it0.s, x0, y0), rm0);
        let pm1 = cell_payload(it1.s, cell_hp(it1.s, x1, y1), rm1);
        let pm2 = cell_payload(it2.s, cell_hp(it2.s, x2, y2), rm2);
        let lo0 = spread8(pm0.x & 0xffu) - spread8(pm0.y & 0xffu);
        let hi0 = spread8(pm0.x >> 8u) - spread8(pm0.y >> 8u);
        let lo1 = spread8(pm1.x & 0xffu) - spread8(pm1.y & 0xffu);
        let hi1 = spread8(pm1.x >> 8u) - spread8(pm1.y >> 8u);
        let lo2 = spread8(pm2.x & 0xffu) - spread8(pm2.y & 0xffu);
        let hi2 = spread8(pm2.x >> 8u) - spread8(pm2.y >> 8u);

        // Slots were initialized to BIAS; heads re-bias after reading (below).
        if it0.valid && !it0.is_head {
            atomicAdd(&sh_slot[2u * it0.slot], lo0);
            atomicAdd(&sh_slot[2u * it0.slot + 1u], hi0);
        }
        if it1.valid && !it1.is_head {
            atomicAdd(&sh_slot[2u * it1.slot], lo1);
            atomicAdd(&sh_slot[2u * it1.slot + 1u], hi1);
        }
        if it2.valid && !it2.is_head {
            atomicAdd(&sh_slot[2u * it2.slot], lo2);
            atomicAdd(&sh_slot[2u * it2.slot + 1u], hi2);
        }
        workgroupBarrier();
        if it0.is_head {
            let w0 = atomicExchange(&sh_slot[2u * it0.slot], BIAS);
            let w1 = atomicExchange(&sh_slot[2u * it0.slot + 1u], BIAS);
            emit(it0, it0.q * 16u + cell, w0, w1, pm0, lo0, hi0, &pack0);
        }
        if it1.is_head {
            let w0 = atomicExchange(&sh_slot[2u * it1.slot], BIAS);
            let w1 = atomicExchange(&sh_slot[2u * it1.slot + 1u], BIAS);
            emit(it1, it1.q * 16u + cell, w0, w1, pm1, lo1, hi1, &pack1);
        }
        if it2.is_head {
            let w0 = atomicExchange(&sh_slot[2u * it2.slot], BIAS);
            let w1 = atomicExchange(&sh_slot[2u * it2.slot + 1u], BIAS);
            emit(it2, it2.q * 16u + cell, w0, w1, pm2, lo2, hi2, &pack2);
        }
        workgroupBarrier();
    }
}
