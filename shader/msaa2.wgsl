// MSAA 16 merge, two consecutive entries per thread with the case table (see
// merge_handoff.md and merge2.wgsl). Payloads P0, P1 are packed signed lane words; a head
// keeps its own payload in registers, so slot stores are BIAS (+ the same-piece partner's
// payload in case 100). CELL_MODE 0 (inline LUT) or 2 (direct).

var<workgroup> sh_slot: array<atomic<u32>, 2u * WG_SIZE>;
var<workgroup> sh_scan: array<u32, WG_SIZE>;

const BIAS: u32 = 0x88888888u;

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
    let b0 = t == 0u || entries[i0 - 1u].key != e0.key;
    let b1 = e0.key != e1.key;
    let b2 = t == WG_SIZE - 1u || e1.key != entries[i0 + 2u].key;

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
    let li0 = 2u * t;
    let li1 = li0 + 1u;
    let ph0 = head0 + ((li0 - head0) & ~7u);
    let ph1 = head1 + ((li1 - head1) & ~7u);
    let pb0 = li0 == ph0;
    let pb1 = li1 == ph1;
    let pb2 = b2 || (((li1 + 1u - head1) & 7u) == 0u);
    let b = (u32(pb0) << 2u) | (u32(pb1) << 1u) | u32(pb2);
    let head_slot = 2u * (ph0 >> 1u);
    let own_slot = 2u * t;

    // Case predicates (b as 3-bit: b0 b1 b2).
    let imm_a = b == 6u || b == 7u || b == 3u;        // immediate resolve: e0 for 110/111, e1 for 011
    let imm_a_e1 = b == 3u;
    let slot_store = b == 2u || b == 4u || b == 6u;
    let slot_with_p1 = b == 4u;
    let do_add = b < 4u;
    let add_both = b < 2u;
    let fin = b == 2u || b == 4u || b == 5u || b == 6u || b == 7u;   // 111: second immediate (e1)
    let fin_load = b == 2u || b == 4u || b == 6u;
    let fin_e0 = b == 4u || b == 5u;
    let ea = select_entry(e0, e1, imm_a_e1);
    let ef = select_entry(e1, e0, fin_e0);
    let eo_a = (path_flags[ea.key >> 20u] & 1u) != 0u;
    let eo_f = (path_flags[ef.key >> 20u] & 1u) != 0u;

    let s0 = setup(e0);
    let s1 = setup(e1);
    var pack_a = 0u;
    var pack_f = 0u;
    var rm0: RowMasks;
    var rm1: RowMasks;
    for (var c = 0u; c < 4u; c++) {
        let cx0 = c >> 1u;
        let cy0 = c & 1u;
        for (var pos = 0u; pos < N_POS; pos++) {
            let x = 2u * (pos % HALF) + cx0;
            let y = 2u * (pos / HALF) + cy0;
            if pos % HALF == 0u {
                rm0 = row_masks(s0, y);
                rm1 = row_masks(s1, y);
            }
            let pm0 = cell_payload(s0, cell_hp(s0, x, y), rm0);
            let pm1 = cell_payload(s1, cell_hp(s1, x, y), rm1);
            let lo0 = spread8(pm0.x & 0xffu) - spread8(pm0.y & 0xffu);
            let hi0 = spread8(pm0.x >> 8u) - spread8(pm0.y >> 8u);
            let lo1 = spread8(pm1.x & 0xffu) - spread8(pm1.y & 0xffu);
            let hi1 = spread8(pm1.x >> 8u) - spread8(pm1.y >> 8u);
            let cell = c * N_POS + pos;

            if imm_a {
                // Single-entry piece: no adders, slot words are the bias.
                let pm = select(pm0, pm1, imm_a_e1);
                let lo = select(lo0, lo1, imm_a_e1);
                let hi = select(hi0, hi1, imm_a_e1);
                store_pixel(ea, eo_a, cell, BIAS, BIAS, pm, lo, hi, &pack_a);
            }
            if slot_store {
                atomicStore(&sh_slot[own_slot], BIAS + select(0u, lo1, slot_with_p1));
                atomicStore(&sh_slot[own_slot + 1u], BIAS + select(0u, hi1, slot_with_p1));
            }
            workgroupBarrier();
            if do_add {
                atomicAdd(&sh_slot[head_slot], lo0 + select(0u, lo1, add_both));
                atomicAdd(&sh_slot[head_slot + 1u], hi0 + select(0u, hi1, add_both));
            }
            workgroupBarrier();
            if fin {
                var w0 = BIAS;
                var w1 = BIAS;
                if fin_load {
                    w0 = atomicLoad(&sh_slot[own_slot]);
                    w1 = atomicLoad(&sh_slot[own_slot + 1u]);
                } else if b == 5u {
                    // 101: e0 head, e1 the only adder, nothing went through the slot.
                    w0 = BIAS + lo1;
                    w1 = BIAS + hi1;
                }
                let pm = select(pm1, pm0, fin_e0);
                let lo = select(lo1, lo0, fin_e0);
                let hi = select(hi1, hi0, fin_e0);
                store_pixel(ef, eo_f, cell, w0, w1, pm, lo, hi, &pack_f);
            }
        }
    }
}
