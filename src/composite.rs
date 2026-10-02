//! Minimal CPU compositor: source-over of solid colors with 8-bit coverage, sRGB space.

use vello_common::peniko::color::{AlphaColor, Srgb};

use crate::tiler::{PIXELS, TILE};

pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<[f32; 4]>,
}

impl Image {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![[1.0, 1.0, 1.0, 1.0]; (width * height) as usize],
        }
    }

    #[inline]
    pub fn blend(&mut self, x: u32, y: u32, alpha: u8, color: [f32; 4]) {
        if x >= self.width || y >= self.height || alpha == 0 {
            return;
        }
        let a = alpha as f32 / 255.0 * color[3];
        let px = &mut self.data[(y * self.width + x) as usize];
        for i in 0..3 {
            px[i] = color[i] * a + px[i] * (1.0 - a);
        }
        px[3] = a + px[3] * (1.0 - a);
    }

    /// Blend a tile given alpha in row-major (y * 8 + x) order.
    pub fn blend_tile(&mut self, tx: u32, ty: u32, alpha: &[u8; PIXELS], color: [f32; 4]) {
        for y in 0..TILE {
            for x in 0..TILE {
                self.blend(tx * TILE + x, ty * TILE + y, alpha[(y * TILE + x) as usize], color);
            }
        }
    }

    pub fn write_png(&self, path: &str) {
        let file = std::fs::File::create(path).expect("create png");
        let mut enc = png::Encoder::new(std::io::BufWriter::new(file), self.width, self.height);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc.write_header().expect("png header");
        let bytes: Vec<u8> = self
            .data
            .iter()
            .flat_map(|p| p.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8))
            .collect();
        writer.write_image_data(&bytes).expect("png data");
    }
}

pub fn color_f32(c: AlphaColor<Srgb>) -> [f32; 4] {
    c.components
}
