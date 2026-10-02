// Diagnostic: phase 2 only, no slots, atomics, or barriers. PROFILE (header const)
// selects how much of phase 2 runs; every stage folds its results into `acc` so the
// compiler cannot eliminate it:
//   0: read entry and line
//   1: + line_setup
//   2: + coset classification (hit/side masks)
//   3: + FIFO fill (cell_area for hit cells)
//   4: + compute_chunk (side values, row constants, FIFO consumption)
//   5: + store_result for heads (= full phase 2 with output traffic)

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let i0 = wg_id.x * WG_SIZE + t;
    let e = entries[i0];
    let is_head = t == 0u || entries[i0 - 1u].key != e.key;
    var acc = 0;
    if PROFILE == 0u {
        let l = lines[e.line & ~LINE_LEFT_EDGE];
        acc = bitcast<i32>(bitcast<u32>(l.x) ^ bitcast<u32>(l.y) ^ bitcast<u32>(l.z) ^ bitcast<u32>(l.w));
    } else {
        let ls = line_setup(e);
    let eo = path_even_odd(e);
        if PROFILE == 1u {
            acc = bitcast<i32>(bitcast<u32>(ls.a) ^ bitcast<u32>(ls.b) ^ bitcast<u32>(ls.d) ^ bitcast<u32>(ls.h)
                ^ bitcast<u32>(ls.q0.x) ^ bitcast<u32>(ls.q1.y) ^ bitcast<u32>(ls.recip) ^ bitcast<u32>(ls.ye));
        } else {
            for (var c = 0u; c < 4u; c++) {
                var cs = coset_classify(ls, c);
                if PROFILE == 2u {
                    acc ^= i32(cs.hits ^ (cs.sides << 16u));
                    continue;
                }
                coset_fill(ls, c, &cs);
                if PROFILE == 3u {
                    acc ^= cs.f0 ^ cs.f1 ^ cs.f2 ^ cs.f3 ^ i32(cs.hits);
                    continue;
                }
                for (var ch = 0u; ch < N_CHUNKS; ch++) {
                    let m = compute_chunk(ls, &cs, c, ch);
                    if PROFILE >= 5u && is_head {
                        store_result(e, eo, m, c, ch);
                    } else {
                        for (var i = 0u; i < CHUNK; i++) {
                            acc ^= m[i];
                        }
                    }
                }
            }
        }
    }
    if acc == 0x7eadbeef {
        partials_out[0] = acc;
    }
}
