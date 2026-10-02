// Two consecutive entries per thread, case-table scheduling (see merge_handoff.md).

fn select_entry(f: Entry, t: Entry, cond: bool) -> Entry {
    return Entry(select(f.key, t.key, cond), select(f.line, t.line, cond),
                 select(f.backdrop, t.backdrop, cond), select(f.out, t.out, cond));
}

var<workgroup> sh_slot: array<atomic<i32>, WG_SIZE * CHUNK>;
var<workgroup> sh_scan: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let base = wg_id.x * WG_SIZE * 2u;
    let i0 = base + 2u * t;
    let e0 = entries[i0];
    let e1 = entries[i0 + 1u];
    // Real run boundaries. The first entry of the workgroup is always a (pseudo-)head and
    // the last always ends a piece.
    let b0 = t == 0u || entries[i0 - 1u].key != e0.key;
    let b1 = e0.key != e1.key;
    let b2 = t == WG_SIZE - 1u || e1.key != entries[i0 + 2u].key;

    // Workgroup inclusive prefix-max of head indices (local entry index).
    var v = select(select(0u, 2u * t, b0), 2u * t + 1u, b1);
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
    workgroupBarrier();
    let head1 = v;
    var head0 = 2u * t;
    if !b0 {
        head0 = sh_scan[t - 1u];
    }
    // Piece boundaries every 8 entries from the head.
    let li0 = 2u * t;
    let li1 = li0 + 1u;
    let ph0 = head0 + ((li0 - head0) & ~7u);
    let ph1 = head1 + ((li1 - head1) & ~7u);
    let pb0 = li0 == ph0;
    let pb1 = li1 == ph1;
    let pb2 = b2 || (((li1 + 1u - head1) & 7u) == 0u);
    let b = (u32(pb0) << 2u) | (u32(pb1) << 1u) | u32(pb2);
    let head_slot = (ph0 >> 1u) * CHUNK;
    let own_slot = t * CHUNK;

    let ls0 = line_setup(e0);
    let ls1 = line_setup(e1);
    // ix: which line is computed first.
    let ix = select(1u, 0u, b == 6u);
    let imm_store = b == 3u || b == 6u || b == 7u;
    let s_is_m1 = b == 2u || b == 3u || b == 6u || b == 7u;
    let slot_store = b == 2u || b == 4u || b == 6u;
    let do_add = b < 4u;
    let final_store = b == 2u || b >= 4u;
    let final_at0 = b == 4u || b == 5u || b == 7u;
    let e_imm = select_entry(e1, e0, ix == 0u);
    let e_final = select_entry(e1, e0, final_at0);
    let eo_imm = path_even_odd(e_imm);
    let eo_final = path_even_odd(e_final);

    for (var c = 0u; c < 4u; c++) {
        var cs0 = coset_setup(ls0, c);
        var cs1 = coset_setup(ls1, c);
        for (var ch = 0u; ch < N_CHUNKS; ch++) {
            // Both lines are always computed; the case bits only choose which result is
            // treated as "first" (m0) and where results go.
            let ma = compute_chunk(ls0, &cs0, c, ch);
            let mb = compute_chunk(ls1, &cs1, c, ch);
            var m0: array<i32, CHUNK>;
            var m1: array<i32, CHUNK>;
            for (var i = 0u; i < CHUNK; i++) {
                m0[i] = select(mb[i], ma[i], ix == 0u);
                m1[i] = select(ma[i], mb[i], ix == 0u);
            }
            if imm_store {
                store_result(e_imm, eo_imm, m0, c, ch);
            }
            var s: array<i32, CHUNK>;
            for (var i = 0u; i < CHUNK; i++) {
                s[i] = select(m0[i] + m1[i], m1[i], s_is_m1);
            }
            if slot_store {
                for (var i = 0u; i < CHUNK; i++) {
                    let s0 = select(s[i], m0[i], b == 2u);
                    atomicStore(&sh_slot[own_slot + i], s0);
                }
            }
            workgroupBarrier();
            if do_add {
                for (var i = 0u; i < CHUNK; i++) {
                    atomicAdd(&sh_slot[head_slot + i], s[i]);
                }
            }
            workgroupBarrier();
            if final_store {
                if slot_store {
                    for (var i = 0u; i < CHUNK; i++) {
                        s[i] = atomicLoad(&sh_slot[own_slot + i]);
                    }
                }
                store_result(e_final, eo_final, s, c, ch);
            }
        }
    }
}
