// GPU expansion of sorted 8x8 entries into 4x4 quadrant entries for the MSAA merge.
//
// Passes (all share one bind group; header consts N (padded 8x8 entry count, multiple of
// WG_SIZE), N_WG = N / WG_SIZE, and the scratch region offsets):
//   e1_map:     per entry: touched quadrant mask, backdrop deltas, run boundary; per
//               workgroup inclusive scans of item count, boundary count and segmented delta
//               sum (stored per entry in `info`), and the workgroup aggregate.
//   e2_scan:    one workgroup: exclusive scan of the aggregates (monoid: sums, and the
//               segmented delta carry), totals into the counters.
//   e3_runs:    per entry: run id, item base, per-(run, quadrant) counts; the first entry
//               of a run records its meta, the last the run's delta totals.
//   e5_scatter: per entry: write its 4x4 entries at run base + quadrant group base + rank
//               (rank by atomic fill, order within a group is irrelevant to the merge).
//   e4_fixup:   per run: backdrop + delta on group heads, partial decision and record
//               allocation (pieces of 8, restarting at workgroup boundaries like the merge),
//               uniform output or record for untouched quadrants.
// 4x4 entry: key = 8x8 key; line = line index | LINE_LEFT_EDGE (tile flag) | q << 28;
// backdrop on group heads; out = run id, or OUT_PARTIAL | record for partial piece heads.

struct Entry {
    key: u32,
    line: u32,
    backdrop: i32,
    out: u32,
}

const LINE_LEFT_EDGE: u32 = 0x80000000u;
const OUT_PARTIAL: u32 = 0x80000000u;
const NONE: u32 = 0xffffffffu;
const BIAS: u32 = 0x88888888u;
const DBIAS: i32 = 512;

@group(0) @binding(0) var<storage, read_write> lines: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> entries8: array<Entry>;
// Per entry: x = mask | (dne+8) << 4 | (dsw+8) << 8 | (dse+8) << 12 | boundary << 16;
// y = inclusive item count within workgroup; z = inclusive boundary count within workgroup;
// w = segmented inclusive delta sum within workgroup, 3 x 10-bit biased by 512.
@group(0) @binding(2) var<storage, read_write> info: array<vec4<u32>>;
// Workgroup aggregates then exclusive prefixes: 2 vec4 per workgroup each:
// [count, nruns, has_boundary, 0], [dne, dsw, dse, 0]; prefixes at offset 2 * N_WG.
@group(0) @binding(3) var<storage, read_write> agg: array<vec4<i32>>;
// Scratch regions (see OFF_* consts): run_base[N], cnt[4N], fill[4N], delta[4N] (i32 bits),
// meta[4N] (first entry index, out8, backdrop8 bits, 0), run_info[4N], counters[4], flags[].
@group(0) @binding(4) var<storage, read_write> scratch: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> entries4: array<Entry>;
@group(0) @binding(6) var<storage, read_write> mask_out: array<u32>;
@group(0) @binding(7) var<storage, read_write> partials_out: array<i32>;

var<workgroup> sh_a: array<vec4<u32>, WG_SIZE>;   // (count, nruns, flag, 0)
var<workgroup> sh_d: array<vec4<i32>, WG_SIZE>;   // segmented deltas

fn x_at(p0: vec2<f32>, p1: vec2<f32>, y: f32) -> f32 {
    if y == p0.y { return p0.x; }
    if y == p1.y { return p1.x; }
    let d = p1 - p0;
    return p0.x + (y - p0.y) * (d.x / d.y);
}

fn unpack_d(w: u32) -> vec3<i32> {
    return vec3(i32(w & 0x3ffu) - DBIAS, i32((w >> 10u) & 0x3ffu) - DBIAS, i32((w >> 20u) & 0x3ffu) - DBIAS);
}
fn pack_d(d: vec3<i32>) -> u32 {
    return u32(d.x + DBIAS) | (u32(d.y + DBIAS) << 10u) | (u32(d.z + DBIAS) << 20u);
}

// Touched quadrant mask and deltas of an 8x8 entry (as in msaa_flat.wgsl).
fn entry_geometry(e: Entry) -> vec4<i32> {
    let l = lines[e.line & 0x0fffffffu];
    let origin = vec2(f32((e.key & 0x3ffu) * 8u), f32(((e.key >> 10u) & 0x3ffu) * 8u));
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
    return vec4(i32(mask), d_ne, d_sw, d_se);
}

@compute @workgroup_size(WG_SIZE)
fn e1_map(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wg: vec3<u32>) {
    let t = lid.x;
    let i = wg.x * WG_SIZE + t;
    let e = entries8[i];
    let g = entry_geometry(e);
    let mask = u32(g.x);
    let boundary = i == 0u || entries8[i - 1u].key != e.key;
    let packed = mask | (u32(g.y + 8) << 4u) | (u32(g.z + 8) << 8u) | (u32(g.w + 8) << 12u) | (u32(boundary) << 16u);
    var a = vec4(countOneBits(mask), u32(boundary), u32(boundary), 0u);
    var dd = vec4(g.y, g.z, g.w, 0);
    sh_a[t] = a;
    sh_d[t] = dd;
    for (var s = 0u; s < firstTrailingBit(WG_SIZE); s++) {
        workgroupBarrier();
        var oa = vec4(0u, 0u, 0u, 0u);
        var od = vec4(0, 0, 0, 0);
        if t >= (1u << s) {
            oa = sh_a[t - (1u << s)];
            od = sh_d[t - (1u << s)];
        }
        workgroupBarrier();
        // Segmented sum: if this element's segment already started (flag), keep; else add.
        let keep = a.z != 0u;
        dd = select(dd + od, dd, keep);
        a = vec4(a.x + oa.x, a.y + oa.y, a.z | oa.z, 0u);
        sh_a[t] = a;
        sh_d[t] = dd;
    }
    info[i] = vec4(packed, a.x, a.y, pack_d(dd.xyz));
    if t == WG_SIZE - 1u {
        agg[2u * wg.x] = vec4<i32>(a);
        agg[2u * wg.x + 1u] = dd;
    }
}

var<workgroup> sh_carry_a: vec4<i32>;
var<workgroup> sh_carry_d: vec4<i32>;

@compute @workgroup_size(WG_SIZE)
fn e2_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    if t == 0u {
        sh_carry_a = vec4(0, 0, 0, 0);
        sh_carry_d = vec4(0, 0, 0, 0);
    }
    for (var base = 0u; base < N_WG; base += WG_SIZE) {
        workgroupBarrier();
        let w = base + t;
        var a = vec4(0, 0, 0, 0);
        var dd = vec4(0, 0, 0, 0);
        if w < N_WG {
            a = agg[2u * w];
            dd = agg[2u * w + 1u];
        }
        let my_a = a;
        let my_d = dd;
        sh_a[t] = vec4<u32>(a);
        sh_d[t] = dd;
        for (var s = 0u; s < firstTrailingBit(WG_SIZE); s++) {
            workgroupBarrier();
            var oa = vec4(0u, 0u, 0u, 0u);
            var od = vec4(0, 0, 0, 0);
            if t >= (1u << s) {
                oa = sh_a[t - (1u << s)];
                od = sh_d[t - (1u << s)];
            }
            workgroupBarrier();
            let keep = a.z != 0;
            dd = select(dd + od, dd, keep);
            a = vec4(a.x + i32(oa.x), a.y + i32(oa.y), a.z | i32(oa.z), 0);
            sh_a[t] = vec4<u32>(a);
            sh_d[t] = dd;
        }
        workgroupBarrier();
        // Exclusive values: the inclusive value of the previous element, combined with the
        // carry from previous chunks unless that element's segment started in this chunk.
        let carry_a = sh_carry_a;
        let carry_d = sh_carry_d;
        var excl_a = vec4(carry_a.x, carry_a.y, 0, 0);
        var excl_d = carry_d;
        if t > 0u {
            let pa = sh_a[t - 1u];
            let pd = sh_d[t - 1u];
            excl_a = vec4(i32(pa.x) + carry_a.x, i32(pa.y) + carry_a.y, 0, 0);
            excl_d = select(pd + carry_d, pd, pa.z != 0u);
        }
        if w < N_WG {
            agg[2u * N_WG + 2u * w] = excl_a;
            agg[2u * N_WG + 2u * w + 1u] = excl_d;
        }
        workgroupBarrier();
        if t == WG_SIZE - 1u {
            sh_carry_a = vec4(a.x + carry_a.x, a.y + carry_a.y, 0, 0);
            sh_carry_d = select(dd + carry_d, dd, a.z != 0);
        }
    }
    workgroupBarrier();
    if t == 0u {
        atomicStore(&scratch[OFF_CTR + 1u], u32(sh_carry_a.y));   // n_runs
        atomicStore(&scratch[OFF_CTR + 2u], u32(sh_carry_a.x));   // n_items
    }
}

// Run id and global exclusive item count of entry i, from info and the workgroup prefix.
fn entry_run(i: u32, inf: vec4<u32>) -> vec2<u32> {
    let w = i / WG_SIZE;
    let pa = agg[2u * N_WG + 2u * w];
    let mask = inf.x & 0xfu;
    let boundary = (inf.x >> 16u) & 1u;
    let item_base = u32(pa.x) + inf.y - countOneBits(mask);
    let run = u32(pa.y) + inf.z - 1u;   // inclusive boundary count - 1
    return vec2(run, item_base);
}

@compute @workgroup_size(WG_SIZE)
fn e3_runs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let inf = info[i];
    let e = entries8[i];
    let rb = entry_run(i, inf);
    let run = rb.x;
    let mask = inf.x & 0xfu;
    let boundary = ((inf.x >> 16u) & 1u) != 0u;
    if boundary {
        atomicStore(&scratch[OFF_BASE + run], rb.y);
        atomicStore(&scratch[OFF_META + 4u * run], i);
        atomicStore(&scratch[OFF_META + 4u * run + 1u], e.out);
        atomicStore(&scratch[OFF_META + 4u * run + 2u], bitcast<u32>(e.backdrop));
    }
    let last = i + 1u == N || ((info[i + 1u].x >> 16u) & 1u) != 0u;
    if last {
        // Inclusive segmented delta: local value, plus the carry if no boundary in this
        // workgroup at or before this entry.
        let w = i / WG_SIZE;
        var dd = unpack_d(inf.w);
        if inf.z == 0u {
            dd += agg[2u * N_WG + 2u * w + 1u].xyz;
        }
        atomicStore(&scratch[OFF_DELTA + 4u * run + 1u], bitcast<u32>(dd.x));
        atomicStore(&scratch[OFF_DELTA + 4u * run + 2u], bitcast<u32>(dd.y));
        atomicStore(&scratch[OFF_DELTA + 4u * run + 3u], bitcast<u32>(dd.z));
    }
    for (var q = 0u; q < 4u; q++) {
        if (mask & (1u << q)) != 0u {
            atomicAdd(&scratch[OFF_CNT + 4u * run + q], 1u);
        }
    }
}

@compute @workgroup_size(WG_SIZE)
fn e5_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let inf = info[i];
    let e = entries8[i];
    let run = entry_run(i, inf).x;
    let mask = inf.x & 0xfu;
    let base = atomicLoad(&scratch[OFF_BASE + run]);
    var gb = base;
    for (var q = 0u; q < 4u; q++) {
        let cnt = atomicLoad(&scratch[OFF_CNT + 4u * run + q]);
        if (mask & (1u << q)) != 0u {
            let r = atomicAdd(&scratch[OFF_FILL + 4u * run + q], 1u);
            entries4[gb + r] = Entry(e.key, (e.line & 0x8fffffffu) | (q << 28u), 0, 0u);
        }
        gb += cnt;
    }
}

@compute @workgroup_size(WG_SIZE)
fn e4_fixup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    let n_runs = atomicLoad(&scratch[OFF_CTR + 1u]);
    if r >= n_runs {
        return;
    }
    let base = atomicLoad(&scratch[OFF_BASE + r]);
    let first8 = atomicLoad(&scratch[OFF_META + 4u * r]);
    let backdrop8 = bitcast<i32>(atomicLoad(&scratch[OFF_META + 4u * r + 2u]));
    let key = entries8[first8].key;
    let even_odd = (atomicLoad(&scratch[OFF_FLAGS + (key >> 20u)]) & 1u) != 0u;
    var cnt: array<u32, 4>;
    var gb: array<u32, 4>;
    var g = base;
    var partial = false;
    for (var q = 0u; q < 4u; q++) {
        cnt[q] = atomicLoad(&scratch[OFF_CNT + 4u * r + q]);
        gb[q] = g;
        if cnt[q] > 8u || (cnt[q] > 0u && (g / WG_SIZE) != ((g + cnt[q] - 1u) / WG_SIZE)) {
            partial = true;
        }
        g += cnt[q];
    }
    var delta: array<i32, 4>;
    delta[0] = 0;
    delta[1] = bitcast<i32>(atomicLoad(&scratch[OFF_DELTA + 4u * r + 1u]));
    delta[2] = bitcast<i32>(atomicLoad(&scratch[OFF_DELTA + 4u * r + 2u]));
    delta[3] = bitcast<i32>(atomicLoad(&scratch[OFF_DELTA + 4u * r + 3u]));
    if !partial {
        for (var q = 0u; q < 4u; q++) {
            let w = backdrop8 + delta[q];
            if cnt[q] == 0u {
                let covered = select(w != 0, (w & 1) != 0, even_odd);
                let word = select(0u, 0xffffffffu, covered);
                for (var k = 0u; k < 8u; k++) {
                    mask_out[r * 32u + q * 8u + k] = word;
                }
                atomicStore(&scratch[OFF_RINFO + 4u * r + q], NONE - 1u);
            } else {
                entries4[gb[q]].backdrop = w;
                entries4[gb[q]].out = r;
                atomicStore(&scratch[OFF_RINFO + 4u * r + q], NONE);
            }
        }
        return;
    }
    // Partial: count records (pieces of 8 restarting at workgroup boundaries, as the merge
    // forms them), allocate, assign.
    var n_rec = 0u;
    for (var q = 0u; q < 4u; q++) {
        if cnt[q] == 0u {
            n_rec += 1u;
        } else {
            var p = gb[q];
            let end = gb[q] + cnt[q];
            while p < end {
                n_rec += 1u;
                let nb = (p / WG_SIZE + 1u) * WG_SIZE;
                p = min(p + 8u, nb);
            }
        }
    }
    var rec = atomicAdd(&scratch[OFF_CTR], n_rec);
    for (var q = 0u; q < 4u; q++) {
        let w = backdrop8 + delta[q];
        if cnt[q] == 0u {
            for (var k = 0u; k < 16u; k++) {
                partials_out[(rec * 16u + k) * 4u] = i32(BIAS);
                partials_out[(rec * 16u + k) * 4u + 1u] = i32(BIAS);
                partials_out[(rec * 16u + k) * 4u + 2u] = 0;
                partials_out[(rec * 16u + k) * 4u + 3u] = w;
            }
            atomicStore(&scratch[OFF_RINFO + 4u * r + q], rec);
            rec += 1u;
        } else {
            atomicStore(&scratch[OFF_RINFO + 4u * r + q], NONE);
            var p = gb[q];
            let end = gb[q] + cnt[q];
            var first = true;
            while p < end {
                entries4[p].out = OUT_PARTIAL | rec;
                entries4[p].backdrop = select(0, w, first);
                first = false;
                rec += 1u;
                let nb = (p / WG_SIZE + 1u) * WG_SIZE;
                p = min(p + 8u, nb);
            }
        }
    }
}
