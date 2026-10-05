// MSAA 16, one 4x4 quadrant item per thread, items expanded from 8x8 entries by the host
// (stand-in for a GPU map-and-scan pass). Items are ordered quadrant-major within each run,
// so (run, quadrant) groups and their pieces of 8 are contiguous. Item record:
// (entry index, quadrant, run's first entry index, 0). Deltas are contributed by the item
// whose quadrant is the lowest touched by its entry. Output as msaa_quadrant.wgsl.
// Prepended with msaa_common.wgsl and msaa_quadrant_setup (QSetup + setup_quad + helpers).

var<workgroup> sh_run: array<u32, WG_SIZE>;       // run head item position scan
var<workgroup> sh_grp: array<u32, WG_SIZE>;       // group head item position scan
var<workgroup> sh_delta: array<atomic<i32>, 4u * WG_SIZE>;
var<workgroup> sh_qcount: array<atomic<u32>, 4u * WG_SIZE>;
var<workgroup> sh_touched: array<atomic<u32>, WG_SIZE>;
var<workgroup> sh_slot: array<atomic<u32>, 2u * WG_SIZE>;
var<workgroup> sh_start: array<u32, WG_SIZE>;     // 1 at run head items that start their run

@group(0) @binding(6) var<storage, read> items: array<vec4<u32>>;

const BIAS: u32 = 0x88888888u;

fn record_index(h: u32, q: u32, k: u32) -> u32 {
    var ix = 0u;
    for (var qq = 0u; qq < q; qq++) {
        let n = atomicLoad(&sh_qcount[h * 4u + qq]);
        ix += max(1u, (n + 7u) / 8u);
    }
    return ix + k;
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let p = wg_id.x * WG_SIZE + t;
    let item = items[p];
    let e = entries[item.x];
    let q = item.y;
    let run_first = entries[item.z];
    atomicStore(&sh_slot[2u * t], BIAS);
    atomicStore(&sh_slot[2u * t + 1u], BIAS);
    for (var qq = 0u; qq < 4u; qq++) {
        atomicStore(&sh_delta[t * 4u + qq], 0);
        atomicStore(&sh_qcount[t * 4u + qq], 0u);
    }
    atomicStore(&sh_touched[t], 0u);
    sh_start[t] = 0u;

    // Run and group boundaries, scanned together.
    var run_b = t == 0u;
    var grp_b = t == 0u;
    // Whether this item starts its run globally (not a continuation from the previous
    // workgroup): only that segment applies the run's backdrop.
    var run_start = p == 0u;
    if p > 0u {
        let prev = items[p - 1u];
        let pe = entries[prev.x];
        run_start = pe.key != e.key;
        if t > 0u {
            run_b = run_start;
            grp_b = run_b || prev.y != q;
        }
    }
    var vr = select(0u, t, run_b);
    var vg = select(0u, t, grp_b);
    sh_run[t] = vr;
    sh_grp[t] = vg;
    for (var i = 0u; i < firstTrailingBit(WG_SIZE); i++) {
        workgroupBarrier();
        var orr = 0u;
        var org = 0u;
        if t >= (1u << i) {
            orr = sh_run[t - (1u << i)];
            org = sh_grp[t - (1u << i)];
        }
        workgroupBarrier();
        vr = max(vr, orr);
        vg = max(vg, org);
        sh_run[t] = vr;
        sh_grp[t] = vg;
    }
    let run_head = vr;
    let grp_head = vg;
    if run_b {
        sh_start[t] = u32(run_start);
    }
    let rank = t - grp_head;
    let is_head = (rank & 7u) == 0u;
    let slot = grp_head + (rank & ~7u);

    // Entry geometry: touched mask and deltas, contributed once per entry (lowest quadrant).
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
    if (MABL & 4u) != 0u {
        // Ablation: no touch/delta geometry (wrong deltas, timing only).
        mask = 1u << q;
    }
    for (var qy = 0u; qy < 2u && (MABL & 4u) == 0u; qy++) {
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
    atomicAdd(&sh_qcount[run_head * 4u + q], 1u);
    atomicOr(&sh_touched[run_head], 1u << q);
    if (MABL & 4u) == 0u && q == firstTrailingBit(mask) {
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
        if d_ne != 0 {
            atomicAdd(&sh_delta[run_head * 4u + 1u], d_ne);
        }
        if d_sw != 0 {
            atomicAdd(&sh_delta[run_head * 4u + 2u], d_sw);
        }
        if d_se != 0 {
            atomicAdd(&sh_delta[run_head * 4u + 3u], d_se);
        }
    }
    let s = setup_quad(e, q);
    workgroupBarrier();

    // Head bookkeeping.
    let even_odd = (path_flags[run_first.key >> 20u] & 1u) != 0u;
    // Partial record base of this run's segment in this workgroup: stored in the run head
    // item's 4th field by the host.
    let seg_base = items[wg_id.x * WG_SIZE + run_head].w;
    var out = 0u;
    var rec = 0u;
    var bd = 0;
    // Backdrop applies once per run: in the segment that contains the run's first item.
    let seg_backdrop = select(0, run_first.backdrop, sh_start[run_head] != 0u);
    if is_head {
        out = run_first.out;
        rec = seg_base + record_index(run_head, q, rank >> 3u);
        bd = select(0, seg_backdrop + atomicLoad(&sh_delta[run_head * 4u + q]), rank < 8u);
    }
    // Untouched quadrants (run heads).
    if run_b {
        let touched = atomicLoad(&sh_touched[t]);
        for (var qq = 0u; qq < 4u; qq++) {
            if (touched & (1u << qq)) == 0u {
                let w = seg_backdrop + atomicLoad(&sh_delta[t * 4u + qq]);
                if (run_first.out & OUT_PARTIAL) != 0u {
                    let r = seg_base + record_index(t, qq, 0u);
                    for (var i = 0u; i < 16u; i++) {
                        partials_out[(r * 16u + i) * 4u] = i32(BIAS);
                        partials_out[(r * 16u + i) * 4u + 1u] = i32(BIAS);
                        partials_out[(r * 16u + i) * 4u + 2u] = 0;
                        partials_out[(r * 16u + i) * 4u + 3u] = w;
                    }
                } else {
                    let covered = select(w != 0, (w & 1) != 0, even_odd);
                    let word = select(0u, 0xffffffffu, covered);
                    for (var i = 0u; i < 8u; i++) {
                        mask_out[run_first.out * (PIXELS / 2u) + qq * 8u + i] = word;
                    }
                }
            }
        }
    }

    var pack = 0u;
    var rm: RowMasks;
    let qx = q & 1u;
    let qy = q >> 1u;
    if MPROF == 1u {
        // Ablation: setup only.
        if slot == 0xfffffffu && s.la == 7.0 {
            mask_out[0] = out + rec + u32(bd);
        }
        return;
    }
    for (var cell = 0u; cell < 16u; cell++) {
        let ly = cell >> 2u;
        let lx = cell & 3u;
        let y = 4u * qy + ly;
        if lx == 0u {
            rm = q_row_masks(s, y);
        }
        let pm = q_payload(s, q_cell_hp(s, 4u * qx + lx, y), rm);
        let lo = spread8(pm.x & 0xffu) - spread8(pm.y & 0xffu);
        let hi = spread8(pm.x >> 8u) - spread8(pm.y >> 8u);
        if !is_head {
            atomicAdd(&sh_slot[2u * slot], lo);
            atomicAdd(&sh_slot[2u * slot + 1u], hi);
        }
        workgroupBarrier();
        if is_head {
            let w0 = atomicExchange(&sh_slot[2u * slot], BIAS);
            let w1 = atomicExchange(&sh_slot[2u * slot + 1u], BIAS);
            let cellq = q * 16u + cell;
            if (out & OUT_PARTIAL) != 0u {
                let base_ix = (rec * 16u + cell) * 4u;
                partials_out[base_ix] = i32(w0);
                partials_out[base_ix + 1u] = i32(w1);
                partials_out[base_ix + 2u] = i32(pm.x | (pm.y << 16u));
                partials_out[base_ix + 3u] = bd;
            } else {
                let m = resolve_words(w0, w1, pm, lo, hi, bd, even_odd);
                pack |= m << (16u * (cellq & 1u));
                if (cellq & 1u) == 1u {
                    mask_out[out * (PIXELS / 2u) + (cellq >> 1u)] = pack;
                    pack = 0u;
                }
            }
        }
        workgroupBarrier();
    }
}
