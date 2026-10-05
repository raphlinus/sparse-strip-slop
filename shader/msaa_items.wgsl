// MSAA 16 merge over GPU-expanded 4x4 quadrant entries (see expand.wgsl): one entry per
// thread, the one-line cell loop over the quadrant's 16 cells. Entry.line bits 28-29 hold
// the quadrant. Resolved output: mask_out[out * 32 + q * 8 + cell / 2]; partial piece heads
// write compact records at (out & ~OUT_PARTIAL) * 16 + cell.
// Prepended with msaa_common.wgsl and msaa_quadrant_setup.wgsl.

var<workgroup> sh_scan: array<u32, WG_SIZE>;
var<workgroup> sh_slot: array<atomic<u32>, 2u * WG_SIZE>;
const BIAS: u32 = 0x88888888u;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let t = local_id.x;
    let i0 = wg_id.x * WG_SIZE + t;
    let e = entries[i0];
    let q = (e.line >> 28u) & 3u;
    // Slot init must precede the scan's barriers: adders may reach the first cell's
    // atomicAdd before the head thread would otherwise have stored the bias.
    atomicStore(&sh_slot[2u * t], BIAS);
    atomicStore(&sh_slot[2u * t + 1u], BIAS);
    var boundary = t == 0u;
    if t > 0u {
        let pe = entries[i0 - 1u];
        boundary = pe.key != e.key || ((pe.line >> 28u) & 3u) != q;
    }
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
    let he = entries[wg_id.x * WG_SIZE + head];
    let even_odd = (path_flags[e.key >> 20u] & 1u) != 0u;
    let s = setup_quad(e, q);

    var pack = 0u;
    var rm: RowMasks;
    let qx = q & 1u;
    let qy = q >> 1u;
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
            atomicAdd(&sh_slot[2u * head], lo);
            atomicAdd(&sh_slot[2u * head + 1u], hi);
        }
        workgroupBarrier();
        if is_head {
            let w0 = atomicExchange(&sh_slot[2u * t], BIAS);
            let w1 = atomicExchange(&sh_slot[2u * t + 1u], BIAS);
            if (he.out & OUT_PARTIAL) != 0u {
                let base_ix = ((he.out & ~OUT_PARTIAL) * 16u + cell) * 4u;
                partials_out[base_ix] = i32(w0);
                partials_out[base_ix + 1u] = i32(w1);
                partials_out[base_ix + 2u] = i32(pm.x | (pm.y << 16u));
                partials_out[base_ix + 3u] = he.backdrop;
            } else {
                let m = resolve_words(w0, w1, pm, lo, hi, he.backdrop, even_odd);
                let cellq = q * 16u + cell;
                pack |= m << (16u * (cellq & 1u));
                if (cellq & 1u) == 1u {
                    mask_out[he.out * 32u + (cellq >> 1u)] = pack;
                    pack = 0u;
                }
            }
        }
        workgroupBarrier();
    }
}
