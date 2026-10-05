const BIAS: u32 = 0x88888888u;
var<workgroup> sh_slot: array<atomic<u32>, 2u * WG_SIZE>;
var<workgroup> sh_scan: array<u32, WG_SIZE>;

// 128-bit ballot helpers (SGRED).
fn ballot_below(lane: u32) -> vec4<u32> {
    var m = vec4(0u, 0u, 0u, 0u);
    for (var w = 0u; w < 4u; w++) {
        let lo = w * 32u;
        if lane >= lo + 31u {
            m[w] = 0xffffffffu;
        } else if lane >= lo {
            m[w] = (2u << (lane - lo)) - 1u;
        }
    }
    return m;
}
fn ballot_any(b: vec4<u32>) -> bool {
    return (b.x | b.y | b.z | b.w) != 0u;
}
fn ballot_high(b: vec4<u32>) -> u32 {
    if b.w != 0u { return 96u + firstLeadingBit(b.w); }
    if b.z != 0u { return 64u + firstLeadingBit(b.z); }
    if b.y != 0u { return 32u + firstLeadingBit(b.y); }
    return firstLeadingBit(b.x);
}
fn ballot_low(b: vec4<u32>) -> u32 {
    if b.x != 0u { return firstTrailingBit(b.x); }
    if b.y != 0u { return 32u + firstTrailingBit(b.y); }
    if b.z != 0u { return 64u + firstTrailingBit(b.z); }
    return 96u + firstTrailingBit(b.w);
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_invocation_id) sg_lane: u32,
) {
    let t = local_id.x;
    let i0 = wg_id.x * WG_SIZE + t;
    let e = entries[i0];
    if SGRED != 0u {
        // Slots hold only cross-subgroup contributions; start at zero.
        atomicStore(&sh_slot[2u * t], 0u);
        atomicStore(&sh_slot[2u * t + 1u], 0u);
    }
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

    let s = setup(e);
    var pack = 0u;
    var acc = 0u;
    // SGRED: segment structure within the subgroup is fixed for the whole kernel.
    var sg_b = 0u;        // my segment's start lane
    var sg_last = 0u;     // my segment's last lane
    var sg_cont = false;  // I start a continuation segment (piece headed in an earlier subgroup)
    if SGRED != 0u {
        let ballot = subgroupBallot(is_head || sg_lane == 0u);
        let below = ballot_below(sg_lane);
        sg_b = ballot_high(ballot & below);
        let above = ballot & ~below;
        sg_last = select(sg_size - 1u, ballot_low(above) - 1u, ballot_any(above));
        sg_cont = sg_lane == sg_b && !is_head;
    }
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
            if CELL_MODE == 0u || CELL_MODE == 3u {
                hp = cell_hp(s, x, y);
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
            var lo: u32;
            var hi: u32;
            if MPROF == 4u {
                // Ablation: skip the lane spreads (wrong results, same traffic).
                lo = pm.x - pm.y;
                hi = pm.x ^ pm.y;
            } else {
                lo = spread8(pm.x & 0xffu) - spread8(pm.y & 0xffu);
                hi = spread8(pm.x >> 8u) - spread8(pm.y >> 8u);
            }

            if MPROF == 2u {
                // Payload compute only.
                acc ^= lo ^ hi;
                continue;
            }
            var w0 = 0u;
            var w1 = 0u;
            if SGRED != 0u {
                // Segmented sum of the adders' payloads within the subgroup (heads add 0).
                // Segment = lanes sharing a piece head; a piece continuing from an earlier
                // subgroup starts a segment at lane 0 and ships its sum through the slot.
                let add_lo = select(lo, 0u, is_head);
                let add_hi = select(hi, 0u, is_head);
                let incl_lo = subgroupInclusiveAdd(add_lo);
                let incl_hi = subgroupInclusiveAdd(add_hi);
                // The segment start lane (head, or lane 0 of a continuation) consumes the
                // total: its own exclusive prefix is the value before the segment.
                let end_lo = subgroupShuffle(incl_lo, sg_last);
                let end_hi = subgroupShuffle(incl_hi, sg_last);
                let tot_lo = end_lo - (incl_lo - add_lo);
                let tot_hi = end_hi - (incl_hi - add_hi);
                if sg_cont {
                    // Continuation of a piece headed in an earlier subgroup.
                    atomicAdd(&sh_slot[2u * head], tot_lo);
                    atomicAdd(&sh_slot[2u * head + 1u], tot_hi);
                }
                workgroupBarrier();
                if is_head {
                    w0 = BIAS + tot_lo + atomicExchange(&sh_slot[2u * t], 0u);
                    w1 = BIAS + tot_hi + atomicExchange(&sh_slot[2u * t + 1u], 0u);
                }
                workgroupBarrier();
            } else {
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
                    w0 = atomicLoad(&sh_slot[2u * t]);
                    w1 = atomicLoad(&sh_slot[2u * t + 1u]);
                }
            }
            if is_head {
                let cell = c * N_POS + pos;
                if MPROF == 1u {
                    pack |= ((w0 ^ w1) & 0xffffu) << (16u * (cell & 1u));
                    if (cell & 1u) == 1u {
                        mask_out[(e.out & ~OUT_PARTIAL) * (PIXELS / 2u) + (cell >> 1u)] = pack;
                        pack = 0u;
                    }
                    continue;
                }
                store_pixel(e, even_odd, cell, w0, w1, pm, lo, hi, &pack);
            }
        }
    }
    if MPROF == 2u && acc == 0x7eadbeefu {
        mask_out[0] = acc;
    }
}
