// 8x8 entries, merged as 4x4 quadrants with explicit load balancing.
//
// Each thread holds one 8x8 entry and computes which quadrants its line touches (the 4x4
// tiler's rule applied inside the tile) plus the coarse-winding deltas of the quadrants
// relative to the tile's top-left corner. Per-quadrant counts are packed 8 bits each and
// scanned; work items are laid out quadrant-major so a run's items in one quadrant are
// contiguous. Each thread then takes up to 3 items (positions t, t + WG, t + 2 WG), and the
// 16 cells of a quadrant are merged in lockstep: every item adds into its group head's slot,
// heads exchange the slot back to zero and resolve with backdrop + delta. Run heads write
// untouched quadrants as uniform values. Output per run is quadrant-major:
// index = q * 16 + ly * 4 + lx with q = 2 * (y / 4) + x / 4.
//
// Prepended with `const TILE` (must be 8), `const WG_SIZE` (threads) and `const ENTRIES`
// (entries per workgroup; threads >= ENTRIES idle in the per-entry phase). Item slots per
// thread: 3 * ENTRIES / WG_SIZE, which must be 2 or 3.

const FIXED_ONE: f32 = 65536.0;
const LINE_LEFT_EDGE: u32 = 0x80000000u;
const OUT_PARTIAL: u32 = 0x80000000u;
const MAX_ITEMS: u32 = 3u * ENTRIES;
const N_SLOTS: u32 = MAX_ITEMS / WG_SIZE;

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

var<workgroup> sh_head: array<u32, ENTRIES>;       // run head (local entry index) scan
var<workgroup> sh_cnt: array<u32, ENTRIES>;        // packed per-quadrant touch counts (0/1)
var<workgroup> sh_scan: array<u32, ENTRIES>;       // packed exclusive prefix of sh_cnt
var<workgroup> sh_delta: array<atomic<i32>, 4u * ENTRIES>;   // [run head * 4 + q]
var<workgroup> sh_touched: array<atomic<u32>, ENTRIES>;      // [run head] quadrant bits
var<workgroup> sh_slot: array<atomic<i32>, MAX_ITEMS>;
var<workgroup> sh_base: u32;                        // workgroup entry base
var<workgroup> sh_totals: vec4<u32>;                // items per quadrant

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

fn byte(v: u32, q: u32) -> u32 {
    return (v >> (8u * q)) & 0xffu;
}

// x on the (tile-local) line at height y. Exact at the endpoints, as in the tiler, so a
// vertex on a quadrant boundary is not pushed across it by rounding.
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

struct Item {
    valid: bool,
    is_head: bool,
    q: u32,
    slot: u32,
    // Segment clipped to the quadrant's x-range, tile-local.
    s0: vec2<f32>,
    s1: vec2<f32>,
    recip: f32,
    ye: f32,
    ye_sign: f32,
    out: u32,
    bd: i32,
    even_odd: bool,
}

struct RowSpan {
    xmin: f32,
    xmax: f32,
    dy: f32,
    ye_fixed: i32,
}

fn item_row(it: Item, y: u32) -> RowSpan {
    var rs: RowSpan;
    let yrel = it.s0.y - f32(y);
    let y0c = clamp(yrel, 0.0, 1.0);
    let y1c = clamp(yrel + (it.s1.y - it.s0.y), 0.0, 1.0);
    rs.dy = y0c - y1c;
    let t0 = (y0c - yrel) * it.recip;
    let t1 = (y1c - yrel) * it.recip;
    let dxc = it.s1.x - it.s0.x;
    let x0 = it.s0.x + t0 * dxc;
    let x1 = it.s0.x + t1 * dxc;
    rs.xmin = min(x0, x1);
    rs.xmax = max(x0, x1);
    rs.ye_fixed = i32(round(it.ye_sign * clamp(f32(y) - it.ye + 1.0, 0.0, 1.0) * FIXED_ONE));
    return rs;
}

fn cell_value(rs: RowSpan, x: u32) -> i32 {
    var v = 0;
    if rs.dy != 0.0 {
        let xf = f32(x);
        v = i32(round(area_weight(rs.xmin - xf, rs.xmax - xf) * rs.dy * FIXED_ONE));
    }
    return v + rs.ye_fixed;
}

// Build the item at global item position p (quadrant-major).
fn make_item(p: u32) -> Item {
    var it: Item;
    it.valid = false;
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
    // Last entry e with exclusive prefix (byte q) <= r.
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
    it.valid = true;
    it.q = q;
    it.slot = group_head;
    it.is_head = p == group_head;

    // Geometry for (entry e, quadrant q).
    let ent = entries[sh_base + e];
    let l = lines[ent.line & ~LINE_LEFT_EDGE];
    let origin = vec2(f32((ent.key & 0x3ffu) * TILE), f32(((ent.key >> 10u) & 0x3ffu) * TILE));
    let p0 = l.xy - origin;
    let p1 = l.zw - origin;
    let d = p1 - p0;
    let qx = q & 1u;
    let qy = q >> 1u;
    let x_lo = f32(4u * qx);
    var s0 = p0;
    var s1 = p1;
    if d.x != 0.0 {
        let ta = (x_lo - p0.x) / d.x;
        let tb = (x_lo + 4.0 - p0.x) / d.x;
        let t_lo = max(min(ta, tb), 0.0);
        let t_hi = max(min(max(ta, tb), 1.0), t_lo);
        s0 = p0 + t_lo * d;
        s1 = p0 + t_hi * d;
    }
    it.s0 = s0;
    it.s1 = s1;
    it.recip = 1.0 / (s1.y - s0.y);
    it.ye = 0.0;
    it.ye_sign = 0.0;
    if qx == 1u {
        // Left edge of the right quadrants is the tile's vertical midline x = 4. Flag if the
        // segment, clipped to this quadrant row's y-range, has points left of it.
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
        let ya = max(ymin, f32(4u * qy));
        let yb = min(ymax, f32(4u * qy + 4u));
        var xa = min(c0.x, c1.x);
        if d.y != 0.0 {
            xa = min(x_at(p0, p1, ya), x_at(p0, p1, yb));
        }
        if xa < 4.0 && d.x != 0.0 {
            it.ye = p0.y + (4.0 - p0.x) * (d.y / d.x);
            it.ye_sign = sign(d.x);
        }
    } else if (ent.line & LINE_LEFT_EDGE) != 0u && d.x != 0.0 {
        let ye_tile = p0.y + (0.0 - p0.x) * (d.y / d.x);
        // NW takes the row-varying term when the crossing is above the midline, SW when it
        // is below; a crossing exactly on the midline is a pure delta for SW and SE.
        if (qy == 0u && ye_tile < 4.0) || (qy == 1u && ye_tile > 4.0) {
            it.ye = ye_tile;
            it.ye_sign = sign(d.x);
        }
    }
    if it.is_head {
        let he = entries[sh_base + h];
        it.out = he.out;
        it.bd = (he.backdrop + atomicLoad(&sh_delta[h * 4u + q])) * 65536;
        it.even_odd = (path_flags[he.key >> 20u] & 1u) != 0u;
    }
    return it;
}

fn store_cell(it: Item, cell: u32, w: i32, pack: ptr<function, u32>) {
    let idx = it.q * 16u + cell;
    if (it.out & OUT_PARTIAL) != 0u {
        partials_out[(it.out & ~OUT_PARTIAL) * 64u + idx] = w;
    } else {
        *pack |= resolve(w, it.even_odd) << (8u * (cell & 3u));
        if (cell & 3u) == 3u {
            alpha_out[it.out * 16u + (idx >> 2u)] = *pack;
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
    let is_entry = t < ENTRIES;
    let te = min(t, ENTRIES - 1u);
    let e = entries[base + te];
    if t == 0u {
        sh_base = base;
    }
    for (var i = t; i < MAX_ITEMS; i += WG_SIZE) {
        atomicStore(&sh_slot[i], 0);
    }
    if is_entry {
        for (var q = 0u; q < 4u; q++) {
            atomicStore(&sh_delta[t * 4u + q], 0);
        }
        atomicStore(&sh_touched[t], 0u);
    }

    // Run head scan (inclusive prefix max of boundary positions).
    let boundary = is_entry && (t == 0u || entries[base + te - 1u].key != e.key);
    var v = select(0u, t, boundary);
    if is_entry {
        sh_head[t] = v;
    }
    for (var i = 0u; i < firstTrailingBit(ENTRIES); i++) {
        workgroupBarrier();
        var other = 0u;
        if is_entry && t >= (1u << i) {
            other = sh_head[t - (1u << i)];
        }
        workgroupBarrier();
        v = max(v, other);
        if is_entry {
            sh_head[t] = v;
        }
    }
    let h = v;

    // Which quadrants does this line touch, and what deltas does it contribute? The touch
    // test uses the segment clipped to the tile's x-range, as the tiler would see it.
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
    // Crossing tests use the unclipped y-range: a line through the tile's corner clips to
    // a start just inside the tile and would otherwise miss the edge it crosses.
    let ymin_u = min(p0.y, p1.y);
    let ymax_u = max(p0.y, p1.y);
    let sign_y = select(-1, 1, p1.y < p0.y);
    let tile_flag = (e.line & LINE_LEFT_EDGE) != 0u;
    var mask = 0u;
    if is_entry {
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
                // Point at x belongs to quadrant floor(x / 4); the tile clip gives x in [0, 8].
                let qa = u32(clamp(floor(xa * 0.25), 0.0, 1.0));
                let qb = u32(clamp(floor(xb * 0.25), 0.0, 1.0));
                for (var qx = qa; qx <= qb; qx++) {
                    mask |= 1u << (2u * qy + qx);
                }
            }
        }
    }
    // Deltas. Decisions at x = 0 must agree with the tiler, which encoded them in the
    // left-edge flag and the tile backdrop, so they are derived from the flag rather than
    // from a recomputed sign. `top_min_x`: the line's minimum x in the row is at its top end.
    var d_ne = 0;
    var d_sw = 0;
    var d_se = 0;
    let top_min_x = d.x * d.y > 0.0;
    var ye_tile = 0.0;
    if d.x != 0.0 {
        ye_tile = p0.y + (0.0 - p0.x) * (d.y / d.x);
    }
    if is_entry && d.y != 0.0 {
        if ymin_u <= 0.0 && 0.0 < ymax_u {
            // Top-edge crossing inside the tile unless the tiler counted it in the backdrop.
            let inside = !(tile_flag && top_min_x);
            if inside && x_at(p0, p1, 0.0) < 4.0 {
                d_ne += sign_y;
            }
        }
        if ymin_u <= 4.0 && 4.0 < ymax_u {
            // Midline crossing at x >= 0: guaranteed if the line never crosses x = 0 in
            // this tile row; otherwise equivalent to the y_edge side of the midline.
            var inside = true;
            if tile_flag && d.x != 0.0 {
                inside = select(ye_tile >= 4.0, ye_tile <= 4.0, top_min_x);
            }
            if inside && x_at(p0, p1, 4.0) < 4.0 {
                d_se += sign_y;
            }
        }
    }
    if is_entry && tile_flag && d.x != 0.0 && ye_tile <= 4.0 {
        let s = select(-1, 1, d.x > 0.0);
        d_sw += s;
        d_se += s;
    }
    let cnt = ((mask >> 0u) & 1u) | (((mask >> 1u) & 1u) << 8u) | (((mask >> 2u) & 1u) << 16u) | (((mask >> 3u) & 1u) << 24u);
    if is_entry {
        sh_cnt[t] = cnt;
    }
    workgroupBarrier();
    if is_entry {
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
    }

    // Packed exclusive scan of counts.
    var acc = cnt;
    if is_entry {
        sh_scan[t] = acc;
    }
    for (var i = 0u; i < firstTrailingBit(ENTRIES); i++) {
        workgroupBarrier();
        var other = 0u;
        if is_entry && t >= (1u << i) {
            other = sh_scan[t - (1u << i)];
        }
        workgroupBarrier();
        acc += other;
        if is_entry {
            sh_scan[t] = acc;
        }
    }
    workgroupBarrier();
    if t == ENTRIES - 1u {
        // Inclusive total per quadrant; the exclusive values below are all <= 255.
        let excl = acc - cnt;
        sh_totals = vec4(byte(excl, 0u) + byte(cnt, 0u), byte(excl, 1u) + byte(cnt, 1u),
                         byte(excl, 2u) + byte(cnt, 2u), byte(excl, 3u) + byte(cnt, 3u));
    }
    if is_entry {
        sh_scan[t] = acc - cnt;
    }
    workgroupBarrier();

    // Untouched quadrants of each run: uniform value from backdrop + delta (run heads).
    if boundary {
        let touched = atomicLoad(&sh_touched[t]);
        let even_odd = (path_flags[e.key >> 20u] & 1u) != 0u;
        for (var q = 0u; q < 4u; q++) {
            if (touched & (1u << q)) == 0u {
                let w = (e.backdrop + atomicLoad(&sh_delta[t * 4u + q])) * 65536;
                if (e.out & OUT_PARTIAL) != 0u {
                    for (var i = 0u; i < 16u; i++) {
                        partials_out[(e.out & ~OUT_PARTIAL) * 64u + q * 16u + i] = w;
                    }
                } else {
                    let a = resolve(w, even_odd);
                    let word = a | (a << 8u) | (a << 16u) | (a << 24u);
                    for (var i = 0u; i < 4u; i++) {
                        alpha_out[e.out * 16u + q * 4u + i] = word;
                    }
                }
            }
        }
    }

    // Items for this thread.
    let it0 = make_item(t);
    let it1 = make_item(t + WG_SIZE);
    var it2: Item;
    it2.valid = false;
    it2.is_head = false;
    if N_SLOTS > 2u {
        it2 = make_item(t + 2u * WG_SIZE);
    }
    var pack0 = 0u;
    var pack1 = 0u;
    var pack2 = 0u;
    var rs0: RowSpan;
    var rs1: RowSpan;
    var rs2: RowSpan;

    // QPROFILE 1: setup only (fold item state into a store nobody reads).
    if QPROFILE == 1u {
        if it0.valid && it1.valid && it0.slot == 0xfffffffu {
            partials_out[0] = i32(it0.q + it1.q + it2.q);
        }
        return;
    }
    for (var cell = 0u; cell < 16u; cell++) {
        let ly = cell >> 2u;
        let lx = cell & 3u;
        if lx == 0u {
            rs0 = item_row(it0, 4u * (it0.q >> 1u) + ly);
            rs1 = item_row(it1, 4u * (it1.q >> 1u) + ly);
            if N_SLOTS > 2u {
                rs2 = item_row(it2, 4u * (it2.q >> 1u) + ly);
            }
        }
        if it0.valid {
            atomicAdd(&sh_slot[it0.slot], cell_value(rs0, 4u * (it0.q & 1u) + lx));
        }
        if it1.valid {
            atomicAdd(&sh_slot[it1.slot], cell_value(rs1, 4u * (it1.q & 1u) + lx));
        }
        if N_SLOTS > 2u && it2.valid {
            atomicAdd(&sh_slot[it2.slot], cell_value(rs2, 4u * (it2.q & 1u) + lx));
        }
        workgroupBarrier();
        if it0.is_head {
            store_cell(it0, cell, atomicExchange(&sh_slot[it0.slot], 0) + it0.bd, &pack0);
        }
        if it1.is_head {
            store_cell(it1, cell, atomicExchange(&sh_slot[it1.slot], 0) + it1.bd, &pack1);
        }
        if N_SLOTS > 2u && it2.is_head {
            store_cell(it2, cell, atomicExchange(&sh_slot[it2.slot], 0) + it2.bd, &pack2);
        }
        workgroupBarrier();
    }
}
