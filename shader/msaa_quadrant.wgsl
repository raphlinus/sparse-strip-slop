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
    flags: u32,     // bit 0 valid, bit 1 head, bit 2 even_odd, bits 4-5 quadrant
    slot: u32,      // item position of the piece head
    out: u32,       // run's out field (with OUT_PARTIAL) for heads
    rec: u32,       // partial record index for heads
    bd: i32,        // backdrop + quadrant delta (piece 0 only), for heads
    s: QSetup,
}

fn item_valid(it: Item) -> bool { return (it.flags & 1u) != 0u; }
fn item_head(it: Item) -> bool { return (it.flags & 2u) != 0u; }
fn item_eo(it: Item) -> bool { return (it.flags & 4u) != 0u; }
fn item_q(it: Item) -> u32 { return (it.flags >> 4u) & 3u; }

fn make_item(p: u32) -> Item {
    var it: Item;
    it.flags = 0u;
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
    let is_head = (rank & 7u) == 0u;
    it.slot = group_head + (rank & ~7u);
    let ent = entries[sh_base + e];
    it.s = setup_quad(ent, q);
    var eo = false;
    if is_head {
        let he = entries[sh_base + h];
        it.out = he.out;
        it.rec = (he.out & ~OUT_PARTIAL) + record_index(h, q, rank >> 3u);
        it.bd = select(0, he.backdrop + atomicLoad(&sh_delta[h * 4u + q]), rank < 8u);
        eo = (path_flags[he.key >> 20u] & 1u) != 0u;
    }
    it.flags = 1u | (u32(is_head) << 1u) | (u32(eo) << 2u) | (q << 4u);
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
        let mask = resolve_words(w0, w1, pm, lo, hi, it.bd, item_eo(it));
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
        if item_valid(it0) && item_valid(it1) && it0.slot == 0xfffffffu {
            mask_out[0] = it0.flags + it1.flags + it2.flags;
        }
        return;
    }

    let q0 = item_q(it0);
    let q1 = item_q(it1);
    let q2 = item_q(it2);
    let v0 = item_valid(it0);
    let v1 = item_valid(it1);
    let v2 = item_valid(it2);
    for (var cell = 0u; cell < 16u; cell++) {
        let ly = cell >> 2u;
        let lx = cell & 3u;
        // Payloads only for valid items; valid positions are a prefix of the item range, so
        // this is uniform across all but one SIMD group per batch.
        var pm0 = vec2(0u, 0u);
        var pm1 = vec2(0u, 0u);
        var pm2 = vec2(0u, 0u);
        if v0 {
            if lx == 0u {
                rm0 = q_row_masks(it0.s, 4u * (q0 >> 1u) + ly);
            }
            pm0 = q_payload(it0.s, q_cell_hp(it0.s, 4u * (q0 & 1u) + lx, 4u * (q0 >> 1u) + ly), rm0);
        }
        if v1 {
            if lx == 0u {
                rm1 = q_row_masks(it1.s, 4u * (q1 >> 1u) + ly);
            }
            pm1 = q_payload(it1.s, q_cell_hp(it1.s, 4u * (q1 & 1u) + lx, 4u * (q1 >> 1u) + ly), rm1);
        }
        if v2 {
            if lx == 0u {
                rm2 = q_row_masks(it2.s, 4u * (q2 >> 1u) + ly);
            }
            pm2 = q_payload(it2.s, q_cell_hp(it2.s, 4u * (q2 & 1u) + lx, 4u * (q2 >> 1u) + ly), rm2);
        }
        let lo0 = spread8(pm0.x & 0xffu) - spread8(pm0.y & 0xffu);
        let hi0 = spread8(pm0.x >> 8u) - spread8(pm0.y >> 8u);
        let lo1 = spread8(pm1.x & 0xffu) - spread8(pm1.y & 0xffu);
        let hi1 = spread8(pm1.x >> 8u) - spread8(pm1.y >> 8u);
        let lo2 = spread8(pm2.x & 0xffu) - spread8(pm2.y & 0xffu);
        let hi2 = spread8(pm2.x >> 8u) - spread8(pm2.y >> 8u);

        if v0 && !item_head(it0) {
            atomicAdd(&sh_slot[2u * it0.slot], lo0);
            atomicAdd(&sh_slot[2u * it0.slot + 1u], hi0);
        }
        if v1 && !item_head(it1) {
            atomicAdd(&sh_slot[2u * it1.slot], lo1);
            atomicAdd(&sh_slot[2u * it1.slot + 1u], hi1);
        }
        if v2 && !item_head(it2) {
            atomicAdd(&sh_slot[2u * it2.slot], lo2);
            atomicAdd(&sh_slot[2u * it2.slot + 1u], hi2);
        }
        workgroupBarrier();
        if item_head(it0) {
            let w0 = atomicExchange(&sh_slot[2u * it0.slot], BIAS);
            let w1 = atomicExchange(&sh_slot[2u * it0.slot + 1u], BIAS);
            emit(it0, q0 * 16u + cell, w0, w1, pm0, lo0, hi0, &pack0);
        }
        if item_head(it1) {
            let w0 = atomicExchange(&sh_slot[2u * it1.slot], BIAS);
            let w1 = atomicExchange(&sh_slot[2u * it1.slot + 1u], BIAS);
            emit(it1, q1 * 16u + cell, w0, w1, pm1, lo1, hi1, &pack1);
        }
        if item_head(it2) {
            let w0 = atomicExchange(&sh_slot[2u * it2.slot], BIAS);
            let w1 = atomicExchange(&sh_slot[2u * it2.slot + 1u], BIAS);
            emit(it2, q2 * 16u + cell, w0, w1, pm2, lo2, hi2, &pack2);
        }
        workgroupBarrier();
    }
}
