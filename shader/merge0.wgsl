// Diagnostic: phase 2 only. One entry per thread, no slots, atomics, or barriers.
// Heads write their own (unsummed) result so output traffic matches merge1; non-head
// results are folded into a register and written only on a never-true condition so the
// compiler cannot eliminate the work.

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let i0 = wg_id.x * WG_SIZE + t;
    let e = entries[i0];
    let is_head = t == 0u || entries[i0 - 1u].key != e.key;
    let ls = line_setup(e);
    var acc = 0;
    for (var c = 0u; c < 4u; c++) {
        var cs = coset_setup(ls, c);
        for (var ch = 0u; ch < N_CHUNKS; ch++) {
            let m = compute_chunk(ls, &cs, c, ch);
            if is_head {
                store_result(e, m, c, ch);
            } else {
                for (var i = 0u; i < CHUNK; i++) {
                    acc ^= m[i];
                }
            }
        }
    }
    if acc == 0x7eadbeef {
        partials_out[0] = acc;
    }
}
