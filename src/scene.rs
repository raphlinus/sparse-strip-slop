//! SVG loading and flattening into a line soup tagged by path.

use vello_common::fearless_simd::Level;
use vello_common::flatten::{self, FlattenCtx, Line};
use vello_common::geometry::RectU16;
use vello_common::kurbo::{Affine, Stroke, StrokeCtx};
use vello_common::peniko::color::{AlphaColor, Srgb};
use vello_common::pico_svg::{Item, PicoSvg};

/// One filled path (strokes are expanded to fills).
#[derive(Clone, Debug)]
pub struct PathData {
    pub color: AlphaColor<Srgb>,
    pub even_odd: bool,
    /// Range into `Scene::lines`.
    pub line_start: u32,
    pub line_end: u32,
}

pub struct Scene {
    pub width: u32,
    pub height: u32,
    /// Global-coordinate line soup, `[x0, y0, x1, y1]`.
    pub lines: Vec<[f32; 4]>,
    pub paths: Vec<PathData>,
}

struct Flattener {
    level: Level,
    flatten_ctx: FlattenCtx,
    stroke_ctx: StrokeCtx,
    line_buf: Vec<Line>,
    cull: RectU16,
    scene: Scene,
}

impl Flattener {
    fn push(&mut self, color: AlphaColor<Srgb>) {
        let start = self.scene.lines.len() as u32;
        self.scene.lines.extend(
            self.line_buf
                .iter()
                .map(|l| [l.p0.x, l.p0.y, l.p1.x, l.p1.y]),
        );
        let end = self.scene.lines.len() as u32;
        if end > start {
            self.scene.paths.push(PathData {
                color,
                even_odd: false,
                line_start: start,
                line_end: end,
            });
        }
    }

    fn walk(&mut self, items: &[Item], transform: Affine) {
        for item in items {
            match item {
                Item::Fill(f) => {
                    flatten::fill(
                        self.level,
                        f.path.iter(),
                        transform,
                        &mut self.line_buf,
                        &mut self.flatten_ctx,
                        self.cull,
                    );
                    self.push(f.color);
                }
                Item::Stroke(s) => {
                    let style = Stroke::new(s.width);
                    flatten::stroke(
                        self.level,
                        s.path.iter(),
                        &style,
                        transform,
                        &mut self.line_buf,
                        &mut self.flatten_ctx,
                        &mut self.stroke_ctx,
                        self.cull,
                    );
                    self.push(s.color);
                }
                Item::Group(g) => {
                    self.walk(&g.children, transform * g.affine);
                }
            }
        }
    }
}

pub fn load_svg(path: &str, scale: f64) -> Scene {
    let svg = std::fs::read_to_string(path).expect("error reading svg");
    let parsed = PicoSvg::load(&svg, 1.0).expect("error parsing svg");
    let width = (parsed.size.width * scale).ceil() as u32;
    let height = (parsed.size.height * scale).ceil() as u32;
    let mut fl = Flattener {
        level: Level::try_detect().unwrap_or(Level::baseline()),
        flatten_ctx: FlattenCtx::default(),
        stroke_ctx: StrokeCtx::default(),
        line_buf: Vec::new(),
        cull: RectU16::new(0, 0, width as u16, height as u16),
        scene: Scene {
            width,
            height,
            lines: Vec::new(),
            paths: Vec::new(),
        },
    };
    fl.walk(&parsed.items, Affine::scale(scale));
    fl.scene
}
