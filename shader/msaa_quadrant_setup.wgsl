// Compact quadrant item setup shared by msaa_quadrant.wgsl and msaa_flat.wgsl.

fn x_at(p0: vec2<f32>, p1: vec2<f32>, y: f32) -> f32 {
    if y == p0.y {
        return p0.x;
    }
    if y == p1.y {
        return p1.x;
    }
    let d = p1 - p0;
    return p0.x + (y - p0.y) * (d.x / d.y);
}

// Compact per-item state: only what the cell loop needs. h, sign and is_pos are derived.
struct QSetup {
    a: f32,
    b: f32,
    d: f32,
    q0y: f32,
    q1y: f32,
    ye: f32,
    ye_sign: f32,   // 0: no y_edge term
    la: f32,
    lb: f32,
    x_sign: f32,
    x0i: i32,
    y0i: i32,
}

fn q_classify(s: QSetup, x: u32, y: u32) -> vec2<bool> {
    let h = (abs(s.a) + abs(s.b)) * (0.5 - 1.0 / 65536.0);
    let t_y = s.b * (f32(y) + 0.5) - s.d;
    let v = fma(s.a, f32(x) + 0.5, t_y);
    return vec2(abs(v) < h, (v > 0.0) == (s.a > 0.0));
}

fn q_lut_mask(s: QSetup, x: u32, y: u32) -> u32 {
    let sub_ix = f32(i32(y) - s.y0i) + s.x_sign * f32(i32(x) - s.x0i);
    let zf = s.la * sub_ix + s.lb;
    let z = floor(zf);
    let block = select(0u, MASK16_WIDTH * MASK16_HALF_HEIGHT, s.x_sign > 0.0);
    let hh = f32(MASK16_HALF_HEIGHT);
    let row = u32(floor(min(s.la * hh, hh - 1.0))) * MASK16_WIDTH;
    let col = u32(floor((zf - z) * f32(MASK16_WIDTH)));
    let ix = block + row + col;
    return (mask_lut[ix >> 1u] >> ((ix & 1u) * 16u)) & 0xffffu;
}

fn q_cell_hp(s: QSetup, x: u32, y: u32) -> u32 {
    let cl = q_classify(s, x, y);
    if CELL_MODE == 3u {
        let m = q_lut_mask(s, x, y);
        return select(select(0u, 0xffffu, cl.y), m, cl.x);
    }
    if cl.x {
        return q_lut_mask(s, x, y);
    }
    return select(0u, 0xffffu, cl.y);
}

fn q_row_masks(s: QSetup, y: u32) -> RowMasks {
    var rm: RowMasks;
    let yrel = s.q0y - f32(y);
    let y0c = clamp(yrel, 0.0, 1.0);
    let y1c = clamp(yrel + (s.q1y - s.q0y), 0.0, 1.0);
    rm.range = range_mask(min(y0c, y1c), max(y0c, y1c));
    rm.ye = 0u;
    if s.ye_sign != 0.0 {
        rm.ye = range_mask(clamp(s.ye - f32(y), 0.0, 1.0), 1.0);
    }
    return rm;
}

fn q_payload(s: QSetup, hp: u32, rm: RowMasks) -> vec2<u32> {
    let cross = rm.range & hp;
    let sign_pos = s.a < 0.0;
    var plus = select(0u, cross, sign_pos);
    var minus = select(cross, 0u, sign_pos);
    plus |= select(0u, rm.ye, s.ye_sign > 0.0);
    minus |= select(rm.ye, 0u, s.ye_sign > 0.0);
    let both = plus & minus;
    return vec2(plus & ~both, minus & ~both);
}

// Setup of (entry, quadrant): clipped to the quadrant's box, with the quadrant's left-edge
// rule. The y_edge for the left quadrants uses the same expression as msaa_common::setup so
// its sample-row boundary rounds identically to the 8x8 reference.
fn setup_quad(ent: Entry, q: u32) -> QSetup {
    let l = lines[ent.line & 0x0fffffffu];
    let origin = vec2(f32((ent.key & 0x3ffu) * TILE), f32(((ent.key >> 10u) & 0x3ffu) * TILE));
    let p0 = l.xy - origin;
    let p1 = l.zw - origin;
    let d = p1 - p0;
    let qx = q & 1u;
    let qy = q >> 1u;
    let x_lo = f32(4u * qx);
    let y_lo = f32(4u * qy);
    var q0 = p0;
    var q1 = p1;
    if d.x != 0.0 {
        let ta = (x_lo - p0.x) / d.x;
        let tb = (x_lo + 4.0 - p0.x) / d.x;
        let t_lo = max(min(ta, tb), 0.0);
        let t_hi = max(min(max(ta, tb), 1.0), t_lo);
        q0 = p0 + t_lo * d;
        q1 = p0 + t_hi * d;
    }
    var s: QSetup;
    s.a = d.y;
    s.b = -d.x;
    s.d = s.a * q0.x + s.b * q0.y;
    s.q0y = q0.y;
    s.q1y = q1.y;
    s.ye = 0.0;
    s.ye_sign = 0.0;
    if qx == 1u {
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
        let ya = max(ymin, y_lo);
        let yb = min(ymax, y_lo + 4.0);
        var xa = min(c0.x, c1.x);
        if d.y != 0.0 {
            xa = min(x_at(p0, p1, ya), x_at(p0, p1, yb));
        }
        if xa < 4.0 && d.x != 0.0 {
            s.ye = p0.y + (4.0 - p0.x) * (d.y / d.x);
            s.ye_sign = sign(d.x);
        }
    } else if (ent.line & LINE_LEFT_EDGE) != 0u && d.x != 0.0 {
        let ta = (0.0 - p0.x) / d.x;
        let tb = (8.0 - p0.x) / d.x;
        let t_lo = max(min(ta, tb), 0.0);
        let qt = p0 + t_lo * d;
        let dt = s.a * qt.x + s.b * qt.y;
        let ye_tile = dt / s.b;
        if (qy == 0u && ye_tile < 4.0) || (qy == 1u && ye_tile > 4.0) {
            s.ye = ye_tile;
            s.ye_sign = sign(d.x);
        }
    }
    var c0 = q0;
    var c1 = q1;
    let dy = q1.y - q0.y;
    if dy != 0.0 {
        let ta = (y_lo - q0.y) / dy;
        let tb = (y_lo + 4.0 - q0.y) / dy;
        let t_lo = max(min(ta, tb), 0.0);
        let t_hi = max(min(max(ta, tb), 1.0), t_lo);
        let dq = q1 - q0;
        c0 = q0 + t_lo * dq;
        c1 = q0 + t_hi * dq;
    }
    let is_down = c1.y >= c0.y;
    let xy0 = select(c1, c0, is_down);
    let xy1 = select(c0, c1, is_down);
    let ldx = abs(xy1.x - xy0.x);
    let ldy = xy1.y - xy0.y;
    let idxdy = 1.0 / (ldx + ldy);
    var la = ldx * idxdy;
    let is_pos = xy1.x >= xy0.x;
    let x_sign = select(-1.0, 1.0, is_pos);
    let xt0 = floor(xy0.x * x_sign);
    let c = xy0.x * x_sign - xt0;
    let y0i = floor(xy0.y);
    let ytop = y0i + 1.0;
    let lb = min((ldy * c + ldx * (ytop - xy0.y)) * idxdy, ONE_MINUS_ULP);
    let count_x = span(xy0.x, xy1.x) - 1u;
    let count = count_x + span(xy0.y, xy1.y);
    let robust_err = floor(la * (f32(count) - 1.0) + lb) - f32(count_x);
    if robust_err != 0.0 {
        la -= ROBUST_EPSILON * sign(robust_err);
    }
    s.la = la;
    s.lb = lb;
    s.x_sign = x_sign;
    s.x0i = i32(xt0 * x_sign + 0.5 * (x_sign - 1.0));
    s.y0i = i32(y0i);
    return s;
}

