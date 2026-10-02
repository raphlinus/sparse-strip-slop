// One entry per thread baseline.

var<workgroup> sh_slot: array<atomic<i32>, WG_SIZE * CHUNK>;
var<workgroup> sh_scan: array<u32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let i0 = wg_id.x * WG_SIZE + t;
    let e = entries[i0];
    let b0 = t == 0u || entries[i0 - 1u].key != e.key;

    var v = select(0u, t, b0);
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
    let head = v;
    let ph = head + ((t - head) & ~7u);
    let is_head = t == ph;
    let head_slot = ph * CHUNK;

    let ls = line_setup(e);
    let eo = path_even_odd(e);
    for (var c = 0u; c < 4u; c++) {
        var cs = coset_setup(ls, c);
        for (var ch = 0u; ch < N_CHUNKS; ch++) {
            var m = compute_chunk(ls, &cs, c, ch);
            if is_head {
                for (var i = 0u; i < CHUNK; i++) {
                    atomicStore(&sh_slot[head_slot + i], m[i]);
                }
            }
            workgroupBarrier();
            if !is_head {
                for (var i = 0u; i < CHUNK; i++) {
                    atomicAdd(&sh_slot[head_slot + i], m[i]);
                }
            }
            workgroupBarrier();
            if is_head {
                for (var i = 0u; i < CHUNK; i++) {
                    m[i] = atomicLoad(&sh_slot[head_slot + i]);
                }
                store_result(e, eo, m, c, ch);
            }
        }
    }
}
