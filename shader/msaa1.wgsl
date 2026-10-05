var<workgroup> sh_slot: array<atomic<u32>, 2u * WG_SIZE>;
var<workgroup> sh_scan: array<u32, WG_SIZE>;

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

    let s = setup(e);
    var pack = 0u;
    var acc = 0u;
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
