/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch_builder::{BorderSideHelpers, BoxShadowMetrics};
use device::{TextureId, TextureFilter};
use euclid::{Point2D, Rect, Matrix4D, Size2D, Point4D};
use fnv::FnvHasher;
use frame::FrameId;
use internal_types::{AxisDirection, BoxShadowRasterOp, Glyph, GlyphKey, RasterItem};
use renderer::{BLUR_INFLATION_FACTOR, TEXT_TARGET_SIZE};
use resource_cache::ResourceCache;
use resource_list::ResourceList;
use std::cmp;
use std::collections::HashMap;
use std::mem;
use std::hash::{Hash, BuildHasherDefault};
use texture_cache::{TextureCacheItem, TexturePage};
use util::{self, RectHelpers};
use webrender_traits::{ColorF, FontKey, GlyphInstance, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderRadius};
use webrender_traits::{BoxShadowClipMode, GradientStop};

const MAX_PRIMITIVES_PER_PASS: usize = 4;
const INVALID_PRIM_INDEX: u32 = 0xffffffff;
const INVALID_CLIP_INDEX: u32 = 0xffffffff;

fn clip_for_box_shadow_clip_mode(box_bounds: &Rect<f32>,
                                 border_radius: f32,
                                 clip_mode: BoxShadowClipMode) -> (Option<Clip>, Option<Clip>) {
    match clip_mode {
        BoxShadowClipMode::None => {
            (None, None)
        }
        BoxShadowClipMode::Inset => {
            (Some(Clip::from_rect(box_bounds)), None)
        }
        BoxShadowClipMode::Outset => {
            (None, Some(Clip::from_rect(box_bounds)))
        }
    }
}

fn compute_box_shadow_rect(box_bounds: &Rect<f32>,
                               box_offset: &Point2D<f32>,
                               spread_radius: f32)
                               -> Rect<f32> {
    let mut rect = (*box_bounds).clone();
    rect.origin.x += box_offset.x;
    rect.origin.y += box_offset.y;
    rect.inflate(spread_radius, spread_radius)
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RectangleKind {
    Solid,
    HorizontalGradient,
    VerticalGradient,
    BorderCorner,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum RotationKind {
    Angle0,
    Angle90,
    Angle180,
    Angle270,
}

#[derive(Clone, Debug)]
pub struct PackedGlyph {
    pub p0: Point2D<f32>,
    pub p1: Point2D<f32>,
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
}

#[derive(Debug, Clone)]
pub struct TextRun {
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
    pub rect: Rect<f32>,
}

pub struct TextBuffer {
    pub texture_size: f32,
    pub page_allocator: TexturePage,
    pub glyphs: Vec<PackedGlyph>,
}

impl TextBuffer {
    fn new(size: u32) -> TextBuffer {
        TextBuffer {
            texture_size: size as f32,
            page_allocator: TexturePage::new(TextureId(0), size),
            glyphs: Vec::new(),
        }
    }

    fn push_text(&mut self, glyphs: Vec<PackedGlyph>) -> TextRun {
        let mut rect = Rect::zero();
        for glyph in &glyphs {
            rect = rect.union(&Rect::new(glyph.p0, Size2D::new(glyph.p1.x - glyph.p0.x, glyph.p1.y - glyph.p0.y)));
        }

        let size = Size2D::new(rect.size.width.ceil() as u32, rect.size.height.ceil() as u32);

        let origin = self.page_allocator
                         .allocate(&size, TextureFilter::Linear)
                         .expect("handle no texture space!");

        let text = TextRun {
            st0: Point2D::new(origin.x as f32 / self.texture_size,
                              origin.y as f32 / self.texture_size),
            st1: Point2D::new((origin.x + size.width) as f32 / self.texture_size,
                              (origin.y + size.height) as f32 / self.texture_size),
            rect: rect,
        };

        let d = Point2D::new(origin.x as f32, origin.y as f32) - rect.origin;
        for glyph in &glyphs {
            self.glyphs.push(PackedGlyph {
                st0: glyph.st0,
                st1: glyph.st1,
                p0: glyph.p0 + d,
                p1: glyph.p1 + d,
            });
        }

        text
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum PrimitiveShader {
    // Rect primary
    Rect,
    Rect_Clip,

    // Image primary
    Image,

    // Text primary
    Text,

    // Border corner primary
    BorderCorner,
    BorderCorner_Clip,

    // Box shadow primary
    BoxShadow,

    // Special fast paths
    Text_Rect,
    Image_Rect,

    // Generic paths
    // TODO(gw): Investigate how to handle these - perhaps still have a primary element?
    Generic2,
    Generic4,

    // Error
    Error,
}

#[derive(Debug)]
struct CollisionPair {
    k0: PrimitiveIndex,
    k1: PrimitiveIndex,
}

#[derive(Debug)]
pub struct Ubo<KEY: Eq + Hash, TYPE> {
    pub items: Vec<TYPE>,
    map: HashMap<KEY, usize, BuildHasherDefault<FnvHasher>>,
}

impl<KEY: Eq + Hash + Copy, TYPE: Clone> Ubo<KEY, TYPE> {
    fn new() -> Ubo<KEY, TYPE> {
        Ubo {
            items: Vec::new(),
            map: HashMap::with_hasher(Default::default()),
        }
    }

/*
    fn can_fit(&self, keys: &Vec<KEY>, kind: UboBindLocation, max_ubo_size: usize) -> bool {
        let max_item_count = kind.get_array_len(max_ubo_size);
        let new_item_count = keys.iter().filter(|key| !self.map.contains_key(key)).count();
        let item_count = self.items.len() + new_item_count;
        item_count < max_item_count
    }

    fn get_index(&self, key: KEY) -> u32 {
        self.map[&key] as u32
    }
*/

    fn maybe_insert_and_get_index(&mut self, key: KEY, data: &TYPE) -> u32 {
        let map = &mut self.map;
        let items = &mut self.items;

        *map.entry(key).or_insert_with(|| {
            let index = items.len();
            items.push(data.clone());
            index
        }) as u32
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrimitiveKind {
    Rectangle = 0,
    Image,
    Text,
    BorderCorner,
    BoxShadow,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PrimitiveIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct ClipIndex(u32);

#[derive(Debug, Copy, Clone)]
pub struct PrimitiveKey {
    kind: PrimitiveKind,
    index: PrimitiveIndex,
}

impl PrimitiveKey {
    fn new(kind: PrimitiveKind, index: usize) -> PrimitiveKey {
        PrimitiveKey {
            kind: kind,
            index: PrimitiveIndex(index as u32),
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct LayerTemplateIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct LayerInstanceIndex(u32);

#[derive(Clone, Debug)]
pub struct PackedLayer {
    transform: Matrix4D<f32>,
    inv_transform: Matrix4D<f32>,
    screen_vertices: [Point4D<f32>; 4],
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PackedPrimitive {
    pub p0: Point2D<f32>,
    pub p1: Point2D<f32>,
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
    pub color0: ColorF,
    pub color1: ColorF,
    pub kind: PrimitiveKind,
    pub rect_kind: RectangleKind,
    pub rotation: RotationKind,
    pub padding: u32,
}

#[derive(Debug)]
struct Primitive {
    layer: LayerTemplateIndex,
    xf_rect: TransformedRect,
    is_opaque: bool,
    packed: PackedPrimitive,
    intersecting_prims_behind: Vec<PrimitiveIndex>,
    intersecting_prims_in_front: Vec<PrimitiveIndex>,
    clip_in: Option<Clip>,
    clip_out: Option<Clip>,
}

// TODO (gw): Profile and create a smaller layout for simple passes if worthwhile...
#[derive(Debug)]
pub struct PackedDrawCommand {
    pub prim_indices: [u32; MAX_PRIMITIVES_PER_PASS],
    pub layer_indices: [u32; MAX_PRIMITIVES_PER_PASS],
    pub clip_info: [u32; 4],
}

impl PackedDrawCommand {
    fn set_primitive(&mut self, cmd_index: usize, prim_index: u32, layer_index: u32) {
        self.prim_indices[cmd_index] = prim_index;
        self.layer_indices[cmd_index] = layer_index;
    }
}

impl PackedDrawCommand {
    fn empty() -> PackedDrawCommand {
        PackedDrawCommand {
            prim_indices: [INVALID_PRIM_INDEX; MAX_PRIMITIVES_PER_PASS],
            layer_indices: [0; MAX_PRIMITIVES_PER_PASS],
            clip_info: [INVALID_CLIP_INDEX; 4],
        }
    }
}

#[derive(Debug)]
struct RenderPrimitive {
    layer_index_in_ubo: u32,
    clip_in_index_in_ubo: Option<u32>,
    clip_out_index_in_ubo: Option<u32>,
    key: PrimitiveKey,
    shader: PrimitiveShader,
    other_primitives: [(PrimitiveIndex, LayerTemplateIndex); MAX_PRIMITIVES_PER_PASS-1],
}

#[derive(Debug)]
struct TransformedRect {
    local_rect: Rect<f32>,
    vertices: [Point4D<f32>; 4],
    screen_rect: Rect<i32>,
}

impl TransformedRect {
    fn new(rect: &Rect<f32>, transform: &Matrix4D<f32>) -> TransformedRect {
        let vertices = [
            transform.transform_point4d(&Point4D::new(rect.origin.x,
                                                      rect.origin.y,
                                                      0.0,
                                                      1.0)),
            transform.transform_point4d(&Point4D::new(rect.bottom_left().x,
                                                      rect.bottom_left().y,
                                                      0.0,
                                                      1.0)),
            transform.transform_point4d(&Point4D::new(rect.bottom_right().x,
                                                      rect.bottom_right().y,
                                                      0.0,
                                                      1.0)),
            transform.transform_point4d(&Point4D::new(rect.top_right().x,
                                                      rect.top_right().y,
                                                      0.0,
                                                      1.0)),
        ];

        let mut screen_min = Point2D::new( 1000000,  1000000);
        let mut screen_max = Point2D::new(-1000000, -1000000);

        for vertex in &vertices {
            let inv_w = 1.0 / vertex.w;
            let vx = vertex.x * inv_w;
            let vy = vertex.y * inv_w;
            screen_min.x = cmp::min(screen_min.x, vx.floor() as i32);
            screen_min.y = cmp::min(screen_min.y, vy.floor() as i32);
            screen_max.x = cmp::max(screen_max.x, vx.ceil() as i32);
            screen_max.y = cmp::max(screen_max.y, vy.ceil() as i32);
        }

        TransformedRect {
            local_rect: *rect,
            vertices: vertices,
            screen_rect: Rect::new(screen_min, Size2D::new(screen_max.x - screen_min.x, screen_max.y - screen_min.y)),
        }
    }
}

struct LayerTemplate {
    packed: PackedLayer,
}

struct LayerInstance {
    layer_index: LayerTemplateIndex,
    primitives: Vec<PrimitiveKey>,
}

#[derive(Debug)]
pub struct Batch {
    pub shader: PrimitiveShader,
    pub layer_ubo_index: usize,
    pub prim_ubo_index: usize,
    pub clip_ubo_index: usize,
    pub commands: Vec<PackedDrawCommand>,
}

impl Batch {
    fn new(shader: PrimitiveShader,
           layer_ubo_index: usize,
           prim_ubo_index: usize,
           clip_ubo_index: usize) -> Batch {
        Batch {
            shader: shader,
            commands: Vec::new(),
            prim_ubo_index: prim_ubo_index,
            clip_ubo_index: clip_ubo_index,
            layer_ubo_index: layer_ubo_index,
        }
    }
}

pub struct Pass {
    pub viewport_size: Size2D<u32>,
    pub batches: Vec<Batch>,
}

pub struct Frame {
    pub layer_ubos: Vec<Ubo<LayerTemplateIndex, PackedLayer>>,
    pub primitive_ubos: Vec<Ubo<PrimitiveIndex, PackedPrimitive>>,
    pub clip_ubos: Vec<Ubo<ClipIndex, Clip>>,
    pub passes: Vec<Pass>,
    pub color_texture_id: TextureId,
    pub mask_texture_id: TextureId,
    pub text_buffer: TextBuffer,
}

pub struct FrameBuilderConfig {

}

impl FrameBuilderConfig {
    pub fn new() -> FrameBuilderConfig {
        FrameBuilderConfig {

        }
    }
}

pub struct FrameBuilder {
    screen_rect: Rect<i32>,
    layer_templates: Vec<LayerTemplate>,
    layer_instances: Vec<LayerInstance>,
    layer_stack: Vec<LayerTemplateIndex>,
    primitives: Vec<Primitive>,
    color_texture_id: TextureId,
    mask_texture_id: TextureId,
    text_buffer: TextBuffer,
    scroll_offset: Point2D<f32>,
    device_pixel_ratio: f32,
}

impl FrameBuilder {
    pub fn new(viewport_size: Size2D<f32>,
               scroll_offset: Point2D<f32>,
               device_pixel_ratio: f32) -> FrameBuilder {
        FrameBuilder {
            screen_rect: Rect::new(Point2D::zero(),
                                   Size2D::new(viewport_size.width as i32, viewport_size.height as i32)),
            layer_templates: Vec::new(),
            layer_instances: Vec::new(),
            layer_stack: Vec::new(),
            primitives: Vec::new(),
            color_texture_id: TextureId(0),
            mask_texture_id: TextureId(0),
            text_buffer: TextBuffer::new(TEXT_TARGET_SIZE),
            scroll_offset: scroll_offset,
            device_pixel_ratio: device_pixel_ratio,
        }
    }

    pub fn push_layer(&mut self,
                      rect: Rect<f32>,
                      transform: Matrix4D<f32>,
                      _: f32) {
        // TODO(gw): Not 3d transform correct!
        let scroll_transform = transform.translate(self.scroll_offset.x,
                                                   self.scroll_offset.y,
                                                   0.0);

        let layer_rect = TransformedRect::new(&rect, &transform);

        let template = LayerTemplate {
            packed: PackedLayer {
                inv_transform: scroll_transform.invert(),
                transform: scroll_transform,
                screen_vertices: layer_rect.vertices,
            },
        };

        self.layer_stack.push(LayerTemplateIndex(self.layer_templates.len() as u32));
        self.layer_templates.push(template);
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
    }

    fn add_axis_aligned_gradient_with_stops(&mut self,
                                            rect: &Rect<f32>,
                                            direction: AxisDirection,
                                            stops: &[GradientStop]) {
        for i in 0..(stops.len() - 1) {
            let (prev_stop, next_stop) = (&stops[i], &stops[i + 1]);
            let piece_rect;
            let rect_kind;
            match direction {
                AxisDirection::Horizontal => {
                    let prev_x = util::lerp(rect.origin.x, rect.max_x(), prev_stop.offset);
                    let next_x = util::lerp(rect.origin.x, rect.max_x(), next_stop.offset);
                    piece_rect = Rect::new(Point2D::new(prev_x, rect.origin.y),
                                           Size2D::new(next_x - prev_x, rect.size.height));
                    rect_kind = RectangleKind::HorizontalGradient;
                }
                AxisDirection::Vertical => {
                    let prev_y = util::lerp(rect.origin.y, rect.max_y(), prev_stop.offset);
                    let next_y = util::lerp(rect.origin.y, rect.max_y(), next_stop.offset);
                    piece_rect = Rect::new(Point2D::new(rect.origin.x, prev_y),
                                           Size2D::new(rect.size.width, next_y - prev_y));
                    rect_kind = RectangleKind::VerticalGradient;
                }
            }

            self.add_complex_rectangle(piece_rect,
                                       prev_stop.color,
                                       next_stop.color,
                                       rect_kind);
        }
    }

    pub fn add_gradient(&mut self,
                        rect: Rect<f32>,
                        start_point: &Point2D<f32>,
                        end_point: &Point2D<f32>,
                        stops: &ItemRange,
                        auxiliary_lists: &AuxiliaryLists) {
        if let Some(..) = self.should_add_prim(&rect) {
            let stops = auxiliary_lists.gradient_stops(stops);

            // Fast paths for axis-aligned gradients:
            if start_point.x == end_point.x {
                let rect = Rect::new(Point2D::new(rect.origin.x, start_point.y),
                                     Size2D::new(rect.size.width, end_point.y - start_point.y));
                self.add_axis_aligned_gradient_with_stops(&rect,
                                                          AxisDirection::Vertical,
                                                          stops);
            } else if start_point.y == end_point.y {
                let rect = Rect::new(Point2D::new(start_point.x, rect.origin.y),
                                     Size2D::new(end_point.x - start_point.x, rect.size.height));
                self.add_axis_aligned_gradient_with_stops(&rect,
                                                          AxisDirection::Horizontal,
                                                          stops);
                return
            } else {
                println!("TODO: Angle gradients!");
            }
        }
    }

    pub fn add_text(&mut self,
                    rect: Rect<f32>,
                    font_key: FontKey,
                    size: Au,
                    blur_radius: Au,
                    color: &ColorF,
                    src_glyphs: &[GlyphInstance],
                    resource_cache: &mut ResourceCache,
                    frame_id: FrameId,
                    device_pixel_ratio: f32) {
        if color.a == 0.0 {
            return
        }

        if let Some(..) = self.should_add_prim(&rect) {
            // Logic below to pick the primary render item depends on len > 0!
            assert!(src_glyphs.len() > 0);
            let mut glyph_key = GlyphKey::new(font_key, size, blur_radius, src_glyphs[0].index);
            let blur_offset = blur_radius.to_f32_px() * (BLUR_INFLATION_FACTOR as f32) / 2.0;
            let mut glyphs = Vec::new();

            // HACK HACK HACK - this is not great rasterizing glyphs here!
            let mut resource_list = ResourceList::new(device_pixel_ratio);
            for glyph in src_glyphs {
                let glyph = Glyph::new(size, blur_radius, glyph.index);
                resource_list.add_glyph(font_key, glyph);
            }
            resource_cache.add_resource_list(&resource_list, frame_id);
            resource_cache.raster_pending_glyphs(frame_id);

            for glyph in src_glyphs {
                glyph_key.index = glyph.index;
                let image_info = resource_cache.get_glyph(&glyph_key, frame_id);
                if let Some(image_info) = image_info {
                    // TODO(gw): Need a general solution to handle multiple texture pages per tile in WR2!
                    assert!(self.color_texture_id == TextureId(0) || self.color_texture_id == image_info.texture_id);
                    self.color_texture_id = image_info.texture_id;

                    let x = glyph.x + image_info.user_data.x0 as f32 / device_pixel_ratio - blur_offset;
                    let y = glyph.y - image_info.user_data.y0 as f32 / device_pixel_ratio - blur_offset;

                    let width = image_info.requested_rect.size.width as f32 / device_pixel_ratio;
                    let height = image_info.requested_rect.size.height as f32 / device_pixel_ratio;

                    let uv_rect = image_info.uv_rect();

                    glyphs.push(PackedGlyph {
                        p0: Point2D::new(x, y),
                        p1: Point2D::new(x + width, y + height),
                        st0: uv_rect.top_left,
                        st1: uv_rect.bottom_right,
                    });
                }
            }

            if glyphs.is_empty() {
                return;
            }

            let run = self.text_buffer.push_text(glyphs);

            // Extra cull check based on the tighter bounding rect
            // of the rasterized glyphs - this can often reduce the
            // number of tiles that a text element hits significantly!
            if let Some(xf_rect) = self.should_add_prim(&run.rect) {
                self.add_primitive(PrimitiveKind::Text,
                                   xf_rect,
                                   PackedPrimitive {
                                       p0: run.rect.origin,
                                       p1: run.rect.bottom_right(),
                                       st0: run.st0,
                                       st1: run.st1,
                                       color0: *color,
                                       color1: *color,
                                       kind: PrimitiveKind::Text,
                                       rect_kind: RectangleKind::Solid,
                                       rotation: RotationKind::Angle0,
                                       padding: 0,
                                    },
                                    None,
                                    None,
                                    false);
            }
        }
    }

    // FIXME(pcwalton): Assumes rectangles are well-formed with origin in TL
    fn add_box_shadow_corner(&mut self,
                             top_left: &Point2D<f32>,
                             bottom_right: &Point2D<f32>,
                             corner_area_top_left: &Point2D<f32>,
                             corner_area_bottom_right: &Point2D<f32>,
                             box_rect: &Rect<f32>,
                             color: &ColorF,
                             blur_radius: f32,
                             border_radius: f32,
                             clip_mode: BoxShadowClipMode,
                             resource_cache: &ResourceCache,
                             frame_id: FrameId,
                             rotation_angle: RotationKind,
                             clip_in: Option<Clip>,
                             clip_out: Option<Clip>) {
        let corner_area_rect =
            Rect::new(*corner_area_top_left,
                      Size2D::new(corner_area_bottom_right.x - corner_area_top_left.x,
                                  corner_area_bottom_right.y - corner_area_top_left.y));

        let rect = Rect::new(*top_left, Size2D::new(bottom_right.x - top_left.x,
                                                    bottom_right.y - top_left.y));

        if rect.size.width > 0.0 && rect.size.height > 0.0 {
            let clip_in = Some(Clip::from_rect(&corner_area_rect));

            let inverted = match clip_mode {
                BoxShadowClipMode::Outset | BoxShadowClipMode::None => false,
                BoxShadowClipMode::Inset => true,
            };

            let color_image = match BoxShadowRasterOp::create_corner(blur_radius,
                                                                     border_radius,
                                                                     box_rect,
                                                                     inverted,
                                                                     self.device_pixel_ratio) {
                Some(raster_item) => {
                    let raster_item = RasterItem::BoxShadow(raster_item);
                    resource_cache.get_raster(&raster_item, frame_id)
                }
                None => resource_cache.get_dummy_color_image(),
            };

            self.add_box_shadow_rect(&rect,
                                     color,
                                     &color_image,
                                     rotation_angle,
                                     clip_in,
                                     clip_out);
        }
    }

    fn add_box_shadow_edge(&mut self,
                           top_left: &Point2D<f32>,
                           bottom_right: &Point2D<f32>,
                           box_rect: &Rect<f32>,
                           color: &ColorF,
                           blur_radius: f32,
                           border_radius: f32,
                           clip_mode: BoxShadowClipMode,
                           resource_cache: &ResourceCache,
                           frame_id: FrameId,
                           rotation_angle: RotationKind,
                           clip_in: Option<Clip>,
                           clip_out: Option<Clip>) {
        if top_left.x >= bottom_right.x || top_left.y >= bottom_right.y {
            return
        }

        let rect = Rect::new(*top_left, Size2D::new(bottom_right.x - top_left.x,
                                                    bottom_right.y - top_left.y));

        if rect.size.width > 0.0 && rect.size.height > 0.0 {
            let inverted = match clip_mode {
                BoxShadowClipMode::Outset | BoxShadowClipMode::None => false,
                BoxShadowClipMode::Inset => true,
            };

            let color_image = match BoxShadowRasterOp::create_edge(blur_radius,
                                                                   border_radius,
                                                                   box_rect,
                                                                   inverted,
                                                                   self.device_pixel_ratio) {
                Some(raster_item) => {
                    let raster_item = RasterItem::BoxShadow(raster_item);
                    resource_cache.get_raster(&raster_item, frame_id)
                }
                None => resource_cache.get_dummy_color_image(),
            };

            self.add_box_shadow_rect(&rect,
                                     color,
                                     &color_image,
                                     rotation_angle,
                                     clip_in,
                                     clip_out)
        }
    }

    fn add_box_shadow_sides(&mut self,
                            box_bounds: &Rect<f32>,
                            box_offset: &Point2D<f32>,
                            color: &ColorF,
                            blur_radius: f32,
                            spread_radius: f32,
                            border_radius: f32,
                            clip_mode: BoxShadowClipMode,
                            resource_cache: &ResourceCache,
                            frame_id: FrameId) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let (clip_in, clip_out) = clip_for_box_shadow_clip_mode(box_bounds,
                                                                border_radius,
                                                                clip_mode);

        // Draw the sides.
        //
        //      +--+------------------+--+
        //      |  |##################|  |
        //      +--+------------------+--+
        //      |##|                  |##|
        //      |##|                  |##|
        //      |##|                  |##|
        //      +--+------------------+--+
        //      |  |##################|  |
        //      +--+------------------+--+

        let horizontal_size = Size2D::new(metrics.br_inner.x - metrics.tl_inner.x,
                                          metrics.edge_size);
        let vertical_size = Size2D::new(metrics.edge_size,
                                        metrics.br_inner.y - metrics.tl_inner.y);
        let top_rect = Rect::new(metrics.tl_outer + Point2D::new(metrics.edge_size, 0.0),
                                 horizontal_size);
        let right_rect =
            Rect::new(metrics.tr_outer + Point2D::new(-metrics.edge_size, metrics.edge_size),
                      vertical_size);
        let bottom_rect =
            Rect::new(metrics.bl_outer + Point2D::new(metrics.edge_size, -metrics.edge_size),
                      horizontal_size);
        let left_rect = Rect::new(metrics.tl_outer + Point2D::new(0.0, metrics.edge_size),
                                  vertical_size);

        self.add_box_shadow_edge(&top_rect.origin,
                                 &top_rect.bottom_right(),
                                 &rect,
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 resource_cache,
                                 frame_id,
                                 RotationKind::Angle90,
                                 clip_in.clone(),
                                 clip_out.clone());
        self.add_box_shadow_edge(&right_rect.origin,
                                 &right_rect.bottom_right(),
                                 &rect,
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 resource_cache,
                                 frame_id,
                                 RotationKind::Angle180,
                                 clip_in.clone(),
                                 clip_out.clone());
        self.add_box_shadow_edge(&bottom_rect.origin,
                                 &bottom_rect.bottom_right(),
                                 &rect,
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 resource_cache,
                                 frame_id,
                                 RotationKind::Angle270,
                                 clip_in.clone(),
                                 clip_out.clone());
        self.add_box_shadow_edge(&left_rect.origin,
                                 &left_rect.bottom_right(),
                                 &rect,
                                 color,
                                 blur_radius,
                                 border_radius,
                                 clip_mode,
                                 resource_cache,
                                 frame_id,
                                 RotationKind::Angle0,
                                 clip_in,
                                 clip_out);
    }

    fn add_box_shadow_corners(&mut self,
                              box_bounds: &Rect<f32>,
                              box_offset: &Point2D<f32>,
                              color: &ColorF,
                              blur_radius: f32,
                              spread_radius: f32,
                              border_radius: f32,
                              clip_mode: BoxShadowClipMode,
                              resource_cache: &ResourceCache,
                              frame_id: FrameId) {
        // Draw the corners.
        //
        //      +--+------------------+--+
        //      |##|                  |##|
        //      +--+------------------+--+
        //      |  |                  |  |
        //      |  |                  |  |
        //      |  |                  |  |
        //      +--+------------------+--+
        //      |##|                  |##|
        //      +--+------------------+--+

        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let (clip_in, clip_out) = clip_for_box_shadow_clip_mode(box_bounds,
                                                                border_radius,
                                                                clip_mode);

        // Prevent overlap of the box shadow corners when the size of the blur is larger than the
        // size of the box.
        let center = Point2D::new(box_bounds.origin.x + box_bounds.size.width / 2.0,
                                  box_bounds.origin.y + box_bounds.size.height / 2.0);

        self.add_box_shadow_corner(&metrics.tl_outer,
                                   &Point2D::new(metrics.tl_outer.x + metrics.edge_size,
                                                 metrics.tl_outer.y + metrics.edge_size),
                                   &metrics.tl_outer,
                                   &center,
                                   &rect,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   resource_cache,
                                   frame_id,
                                   RotationKind::Angle0,
                                   clip_in.clone(),
                                   clip_out.clone());
        self.add_box_shadow_corner(&Point2D::new(metrics.tr_outer.x - metrics.edge_size,
                                                 metrics.tr_outer.y),
                                   &Point2D::new(metrics.tr_outer.x,
                                                 metrics.tr_outer.y + metrics.edge_size),
                                   &Point2D::new(center.x, metrics.tr_outer.y),
                                   &Point2D::new(metrics.tr_outer.x, center.y),
                                   &rect,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   resource_cache,
                                   frame_id,
                                   RotationKind::Angle90,
                                   clip_in.clone(),
                                   clip_out.clone());
        self.add_box_shadow_corner(&Point2D::new(metrics.br_outer.x - metrics.edge_size,
                                                 metrics.br_outer.y - metrics.edge_size),
                                   &Point2D::new(metrics.br_outer.x, metrics.br_outer.y),
                                   &center,
                                   &metrics.br_outer,
                                   &rect,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   resource_cache,
                                   frame_id,
                                   RotationKind::Angle180,
                                   clip_in.clone(),
                                   clip_out.clone());
        self.add_box_shadow_corner(&Point2D::new(metrics.bl_outer.x,
                                                 metrics.bl_outer.y - metrics.edge_size),
                                   &Point2D::new(metrics.bl_outer.x + metrics.edge_size,
                                                 metrics.bl_outer.y),
                                   &Point2D::new(metrics.bl_outer.x, center.y),
                                   &Point2D::new(center.x, metrics.bl_outer.y),
                                   &rect,
                                   &color,
                                   blur_radius,
                                   border_radius,
                                   clip_mode,
                                   resource_cache,
                                   frame_id,
                                   RotationKind::Angle270,
                                   clip_in,
                                   clip_out);
    }

    fn fill_outside_area_of_inset_box_shadow(&mut self,
                                             box_bounds: &Rect<f32>,
                                             box_offset: &Point2D<f32>,
                                             color: &ColorF,
                                             blur_radius: f32,
                                             spread_radius: f32,
                                             border_radius: f32) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);
        let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);

        let (clip_in, clip_out) = clip_for_box_shadow_clip_mode(box_bounds,
                                                                border_radius,
                                                                BoxShadowClipMode::Inset);

        // Fill in the outside area of the box.
        //
        //            +------------------------------+
        //      A --> |##############################|
        //            +--+--+------------------+--+--+
        //            |##|  |                  |  |##|
        //            |##+--+------------------+--+##|
        //            |##|  |                  |  |##|
        //      D --> |##|  |                  |  |##| <-- B
        //            |##|  |                  |  |##|
        //            |##+--+------------------+--+##|
        //            |##|  |                  |  |##|
        //            +--+--+------------------+--+--+
        //      C --> |##############################|
        //            +------------------------------+

        // A:
        self.add_solid_rectangle(&Rect::new(box_bounds.origin,
                                            Size2D::new(box_bounds.size.width,
                                                        metrics.tl_outer.y - box_bounds.origin.y)),
                                 color,
                                 clip_in.clone(),
                                 clip_out.clone());

        // B:
        self.add_solid_rectangle(&Rect::new(metrics.tr_outer,
                                            Size2D::new(box_bounds.max_x() - metrics.tr_outer.x,
                                                        metrics.br_outer.y - metrics.tr_outer.y)),
                                 color,
                                 clip_in.clone(),
                                 clip_out.clone());

        // C:
        self.add_solid_rectangle(&Rect::new(Point2D::new(box_bounds.origin.x, metrics.bl_outer.y),
                                            Size2D::new(box_bounds.size.width,
                                                        box_bounds.max_y() - metrics.br_outer.y)),
                                 color,
                                 clip_in.clone(),
                                 clip_out.clone());

        // D:
        self.add_solid_rectangle(&Rect::new(Point2D::new(box_bounds.origin.x, metrics.tl_outer.y),
                                            Size2D::new(metrics.tl_outer.x - box_bounds.origin.x,
                                                        metrics.bl_outer.y - metrics.tl_outer.y)),
                                 color,
                                 clip_in,
                                 clip_out);
    }

    pub fn add_box_shadow(&mut self,
                          box_bounds: &Rect<f32>,
                          box_offset: &Point2D<f32>,
                          color: &ColorF,
                          blur_radius: f32,
                          spread_radius: f32,
                          border_radius: f32,
                          clip_mode: BoxShadowClipMode,
                          resource_cache: &mut ResourceCache,
                          frame_id: FrameId) {
        let rect = compute_box_shadow_rect(box_bounds, box_offset, spread_radius);

        // Fast path.
        if blur_radius == 0.0 && spread_radius == 0.0 && clip_mode == BoxShadowClipMode::None {
            self.add_solid_rectangle(&rect, color, None, None);
            return;
        }

        // TODO(gw): hack hack - do this elsewhere!
        let mut resource_list = ResourceList::new(self.device_pixel_ratio);
        resource_list.add_box_shadow_corner(blur_radius,
                                            border_radius,
                                            &rect,
                                            false);
        resource_list.add_box_shadow_edge(blur_radius,
                                          border_radius,
                                          &rect,
                                          false);
        if clip_mode == BoxShadowClipMode::Inset {
            resource_list.add_box_shadow_corner(blur_radius,
                                                border_radius,
                                                &rect,
                                                true);
            resource_list.add_box_shadow_edge(blur_radius,
                                              border_radius,
                                              &rect,
                                              true);
        }
        resource_cache.add_resource_list(&resource_list, frame_id);

        let white_image = resource_cache.get_dummy_color_image();

        // Draw the corners.
        self.add_box_shadow_corners(box_bounds,
                                    box_offset,
                                    color,
                                    blur_radius,
                                    spread_radius,
                                    border_radius,
                                    clip_mode,
                                    resource_cache,
                                    frame_id);

        // Draw the sides.
        self.add_box_shadow_sides(box_bounds,
                                  box_offset,
                                  color,
                                  blur_radius,
                                  spread_radius,
                                  border_radius,
                                  clip_mode,
                                  resource_cache,
                                  frame_id);

        match clip_mode {
            BoxShadowClipMode::None => {
                // Fill the center area.
                self.add_solid_rectangle(box_bounds, color, None, None);
            }
            BoxShadowClipMode::Outset => {
                // Fill the center area.
                let metrics = BoxShadowMetrics::new(&rect, border_radius, blur_radius);
                if metrics.br_inner.x > metrics.tl_inner.x &&
                        metrics.br_inner.y > metrics.tl_inner.y {
                    let center_rect =
                        Rect::new(metrics.tl_inner,
                                  Size2D::new(metrics.br_inner.x - metrics.tl_inner.x,
                                              metrics.br_inner.y - metrics.tl_inner.y));

                    let clip_out_rect = Clip::from_rect(box_bounds);

                    // FIXME(pcwalton): This assumes the border radius is zero. That is not always
                    // the case!
                    self.add_box_shadow_rect(&center_rect,
                                             color,
                                             white_image,
                                             RotationKind::Angle0,
                                             None,
                                             Some(clip_out_rect));
                }
            }
            BoxShadowClipMode::Inset => {
                // Fill in the outsides.
                self.fill_outside_area_of_inset_box_shadow(box_bounds,
                                                           box_offset,
                                                           color,
                                                           blur_radius,
                                                           spread_radius,
                                                           border_radius);
            }
        }
    }

    pub fn add_image(&mut self,
                     rect: Rect<f32>,
                     _stretch_size: &Size2D<f32>,
                     image_key: ImageKey,
                     image_rendering: ImageRendering,
                     resource_cache: &mut ResourceCache,
                     frame_id: FrameId,
                     device_pixel_ratio: f32) {
        if let Some(xf_rect) = self.should_add_prim(&rect) {
            let mut resource_list = ResourceList::new(device_pixel_ratio);
            resource_list.add_image(image_key, image_rendering);
            resource_cache.add_resource_list(&resource_list, frame_id);

            let image_info = resource_cache.get_image(image_key, image_rendering, frame_id);
            let uv_rect = image_info.uv_rect();

            assert!(self.color_texture_id == TextureId(0) || self.color_texture_id == image_info.texture_id);
            self.color_texture_id = image_info.texture_id;

            self.add_primitive(PrimitiveKind::Image,
                               xf_rect,
                               PackedPrimitive {
                                   p0: rect.origin,
                                   p1: rect.bottom_right(),
                                   st0: uv_rect.top_left,
                                   st1: uv_rect.bottom_right,
                                   color0: ColorF::new(1.0, 1.0, 1.0, 1.0),
                                   color1: ColorF::new(1.0, 1.0, 1.0, 1.0),
                                   kind: PrimitiveKind::Image,
                                   rect_kind: RectangleKind::Solid,
                                   rotation: RotationKind::Angle0,
                                   padding: 0,
                               },
                               None,
                               None,
                               image_info.is_opaque);
        }
    }

    pub fn add_solid_rectangle(&mut self,
                               rect: &Rect<f32>,
                               color: &ColorF,
                               clip_in: Option<Clip>,
                               clip_out: Option<Clip>) {
        if color.a == 0.0 {
            return;
        }

        if let Some(xf_rect) = self.should_add_prim(&rect) {
            self.add_primitive(PrimitiveKind::Rectangle,
                               xf_rect,
                               PackedPrimitive {
                                   p0: rect.origin,
                                   p1: rect.bottom_right(),
                                   st0: Point2D::zero(),
                                   st1: Point2D::zero(),
                                   color0: *color,
                                   color1: *color,
                                   kind: PrimitiveKind::Rectangle,
                                   rect_kind: RectangleKind::Solid,
                                   rotation: RotationKind::Angle0,
                                   padding: 0,
                               },
                               clip_in,
                               clip_out,
                               color.a == 1.0);
        }
    }

    fn add_box_shadow_rect(&mut self,
                           rect: &Rect<f32>,
                           color: &ColorF,
                           texture_item: &TextureCacheItem,
                           rotation: RotationKind,
                           clip_in: Option<Clip>,
                           clip_out: Option<Clip>) {
        if color.a == 0.0 {
            return;
        }

        if let Some(xf_rect) = self.should_add_prim(&rect) {
            let uv_rect = texture_item.uv_rect();

            assert!(self.color_texture_id == TextureId(0) || self.color_texture_id == texture_item.texture_id);
            self.color_texture_id = texture_item.texture_id;

            self.add_primitive(PrimitiveKind::BoxShadow,
                               xf_rect,
                               PackedPrimitive {
                                   p0: rect.origin,
                                   p1: rect.bottom_right(),
                                   st0: uv_rect.top_left,
                                   st1: uv_rect.bottom_right,
                                   color0: *color,
                                   color1: *color,
                                   kind: PrimitiveKind::BoxShadow,
                                   rect_kind: RectangleKind::Solid,
                                   rotation: rotation,
                                   padding: 0,
                               },
                               clip_in,
                               clip_out,
                               texture_item.is_opaque);
        }
    }

    pub fn add_border_corner(&mut self,
                             rect: Rect<f32>,
                             color0: ColorF,
                             color1: ColorF,
                             rotation: RotationKind,
                             clip_in: Option<Clip>) {
        if color0.a == 0.0 && color1.a == 0.0 {
            return;
        }

        if let Some(xf_rect) = self.should_add_prim(&rect) {
            self.add_primitive(PrimitiveKind::BorderCorner,
                               xf_rect,
                               PackedPrimitive {
                                   p0: rect.origin,
                                   p1: rect.bottom_right(),
                                   st0: Point2D::zero(),
                                   st1: Point2D::zero(),
                                   color0: color0,
                                   color1: color1,
                                   kind: PrimitiveKind::BorderCorner,
                                   rect_kind: RectangleKind::BorderCorner,
                                   rotation: rotation,
                                   padding: 0,
                               },
                               clip_in,
                               None,
                               color0.a == 1.0 && color1.a == 1.0);
        }
    }

    fn add_complex_rectangle(&mut self,
                             rect: Rect<f32>,
                             color0: ColorF,
                             color1: ColorF,
                             rect_kind: RectangleKind) {
        if color0.a == 0.0 && color1.a == 0.0 {
            return;
        }

        if let Some(xf_rect) = self.should_add_prim(&rect) {
            self.add_primitive(PrimitiveKind::Rectangle,
                               xf_rect,
                               PackedPrimitive {
                                   p0: rect.origin,
                                   p1: rect.bottom_right(),
                                   st0: Point2D::zero(),
                                   st1: Point2D::zero(),
                                   color0: color0,
                                   color1: color1,
                                   kind: PrimitiveKind::Rectangle,
                                   rect_kind: rect_kind,
                                   rotation: RotationKind::Angle0,
                                   padding: 0,
                               },
                               None,
                               None,
                               color0.a == 1.0 && color1.a == 1.0);
        }
    }

    pub fn add_border(&mut self,
                      rect: Rect<f32>,
                      border: &BorderDisplayItem) {
        let radius = &border.radius;
        let left = &border.left;
        let right = &border.right;
        let top = &border.top;
        let bottom = &border.bottom;

        if (left.style != BorderStyle::Solid && left.style != BorderStyle::None) ||
           (top.style != BorderStyle::Solid && top.style != BorderStyle::None) ||
           (bottom.style != BorderStyle::Solid && bottom.style != BorderStyle::None) ||
           (right.style != BorderStyle::Solid && right.style != BorderStyle::None) {
            println!("TODO: Other border styles {:?} {:?} {:?} {:?}", left.style, top.style, bottom.style, right.style);
            return;
        }

        let tl_outer = Point2D::new(rect.origin.x, rect.origin.y);
        let tl_inner = tl_outer + Point2D::new(radius.top_left.width.max(left.width),
                                               radius.top_left.height.max(top.width));

        let tr_outer = Point2D::new(rect.origin.x + rect.size.width, rect.origin.y);
        let tr_inner = tr_outer + Point2D::new(-radius.top_right.width.max(right.width),
                                               radius.top_right.height.max(top.width));

        let bl_outer = Point2D::new(rect.origin.x, rect.origin.y + rect.size.height);
        let bl_inner = bl_outer + Point2D::new(radius.bottom_left.width.max(left.width),
                                               -radius.bottom_left.height.max(bottom.width));

        let br_outer = Point2D::new(rect.origin.x + rect.size.width,
                                    rect.origin.y + rect.size.height);
        let br_inner = br_outer - Point2D::new(radius.bottom_right.width.max(right.width),
                                               radius.bottom_right.height.max(bottom.width));

        let left_color = left.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let top_color = top.border_color(1.0, 2.0/3.0, 0.3, 0.7);
        let right_color = right.border_color(2.0/3.0, 1.0, 0.7, 0.3);
        let bottom_color = bottom.border_color(2.0/3.0, 1.0, 0.7, 0.3);

        // Edges
        self.add_solid_rectangle(&Rect::new(Point2D::new(tl_outer.x, tl_inner.y),
                                           Size2D::new(left.width, bl_inner.y - tl_inner.y)),
                                 &left_color,
                                 None,
                                 None);

        self.add_solid_rectangle(&Rect::new(Point2D::new(tl_inner.x, tl_outer.y),
                                           Size2D::new(tr_inner.x - tl_inner.x, tr_outer.y + top.width - tl_outer.y)),
                                 &top_color,
                                 None,
                                 None);

        self.add_solid_rectangle(&Rect::new(Point2D::new(br_outer.x - right.width, tr_inner.y),
                                           Size2D::new(right.width, br_inner.y - tr_inner.y)),
                                 &right_color,
                                 None,
                                 None);

        self.add_solid_rectangle(&Rect::new(Point2D::new(bl_inner.x, bl_outer.y - bottom.width),
                                           Size2D::new(br_inner.x - bl_inner.x, br_outer.y - bl_outer.y + bottom.width)),
                                 &bottom_color,
                                 None,
                                 None);

        // Corners
        let need_clip = radius.top_left != Size2D::zero() ||
                        radius.top_right != Size2D::zero() ||
                        radius.bottom_left != Size2D::zero() ||
                        radius.bottom_right != Size2D::zero();

        let clip = if need_clip {
            // TODO(gw): This is all wrong for non-uniform borders!
            let inner_radius = BorderRadius {
                top_left: Size2D::new(radius.top_left.width - left.width,
                                      radius.top_left.width - left.width),
                top_right: Size2D::new(radius.top_right.width - right.width,
                                       radius.top_right.width - right.width),
                bottom_left: Size2D::new(radius.bottom_left.width - left.width,
                                         radius.bottom_left.width - left.width),
                bottom_right: Size2D::new(radius.bottom_right.width - right.width,
                                          radius.bottom_right.width - right.width),
            };

            Some(Clip::from_border_radius(&rect,
                                          radius,
                                          &inner_radius))
        } else {
            None
        };

        self.add_border_corner(Rect::new(tl_outer,
                                         Size2D::new(tl_inner.x - tl_outer.x,
                                                     tl_inner.y - tl_outer.y)),
                           left_color,
                           top_color,
                           RotationKind::Angle0,
                           clip.clone());

        self.add_border_corner(Rect::new(Point2D::new(tr_inner.x, tr_outer.y),
                                         Size2D::new(tr_outer.x - tr_inner.x,
                                                     tr_inner.y - tr_outer.y)),
                           top_color,
                           right_color,
                           RotationKind::Angle90,
                           clip.clone());

        self.add_border_corner(Rect::new(br_inner,
                                         Size2D::new(br_outer.x - br_inner.x,
                                                     br_outer.y - br_inner.y)),
                           right_color,
                           bottom_color,
                           RotationKind::Angle180,
                           clip.clone());

        self.add_border_corner(Rect::new(Point2D::new(bl_outer.x, bl_inner.y),
                                         Size2D::new(bl_inner.x - bl_outer.x,
                                                     bl_outer.y - bl_inner.y)),
                           bottom_color,
                           left_color,
                           RotationKind::Angle270,
                           clip.clone());
    }

    // TODO(gw): This is brute force currently for testing GPU
    //           performance. A proper implementation should use
    //           a DBVT or (more likely) a 2d SAP.
    // TODO(gw): This only works on non-transformed items, for now.
    //           To expand to 3d, make this a broadphase, followed
    //           by a narrowphase that tests potential collisions
    //           between pairs that have transforms.
    fn build_collision_pairs(&self) -> Vec<CollisionPair> {
        let mut pairs = Vec::new();
        let prim_count = self.primitives.len();

        for i in 0..prim_count {
            let p0 = &self.primitives[i];

            for j in i+1..prim_count {
                let p1 = &self.primitives[j];

                if p0.xf_rect.screen_rect.intersects(&p1.xf_rect.screen_rect) {
                    pairs.push(CollisionPair {
                        k0: PrimitiveIndex(i as u32),
                        k1: PrimitiveIndex(j as u32),
                    });
                }
            }
        }

        pairs
    }

    fn assign_primitive_ordering(&mut self, pairs: &Vec<CollisionPair>) {
        for pair in pairs {
            let PrimitiveIndex(i0) = pair.k0;
            let PrimitiveIndex(i1) = pair.k1;

            debug_assert!(i0 < i1);
            self.primitives[i0 as usize].intersecting_prims_in_front.push(pair.k1);
            self.primitives[i1 as usize].intersecting_prims_behind.push(pair.k0);
        }

        // TODO: One or both of these sorts might not be required,
        //       depending on the eventual solution for collision
        //       pair generation.
        for prim_index in 0..self.primitives.len() {
            // TODO(gw): Nastiness here to work around the borrow checker.
            //           Might be able to shuffle these structs around to make this simpler.
            let mut intersecting_prims_in_front = mem::replace(&mut self.primitives[prim_index].intersecting_prims_in_front, Vec::new());
            let mut intersecting_prims_behind = mem::replace(&mut self.primitives[prim_index].intersecting_prims_behind, Vec::new());

            intersecting_prims_in_front.sort();
            intersecting_prims_behind.sort();

            // Look for early primitives that can be dropped.
            // The simple case is that a primitive is opaque that encompasses the primary primitive.
            // Again, this is not transform safe (yet)!
            let mut last_opaque_index = None;
            let prim_screen_rect = self.primitives[prim_index].xf_rect.screen_rect;
            for (array_index, prim_index) in intersecting_prims_behind.iter().enumerate() {
                let prim_behind = self.get_prim(*prim_index);
                if prim_behind.is_opaque &&
                   prim_behind.xf_rect.screen_rect.contains_rect(&prim_screen_rect) {
                    last_opaque_index = Some(array_index);
                }
            }
            if let Some(last_opaque_index) = last_opaque_index {
                if last_opaque_index != 0 {
                    intersecting_prims_behind = intersecting_prims_behind.split_off(last_opaque_index);
                }
            }

            self.primitives[prim_index].intersecting_prims_in_front = intersecting_prims_in_front;
            self.primitives[prim_index].intersecting_prims_behind = intersecting_prims_behind;
        }
    }

    fn get_prim(&self, index: PrimitiveIndex) -> &Primitive {
        let PrimitiveIndex(prim_index) = index;
        &self.primitives[prim_index as usize]
    }

    fn build_primitive(&self,
                       key: PrimitiveKey,
                       layer_index_in_ubo: u32,
                       clip_in_index_in_ubo: Option<u32>,
                       clip_out_index_in_ubo: Option<u32>) -> Option<RenderPrimitive> {
        let prim = self.get_prim(key.index);

        // If this primitive is occluded, there is no
        // need to submit it as a separate primitive,
        // as it will be handled by the blending logic
        // for that primitive!
        if self.primitive_is_occluded(prim) {
            return None;
        }

        let mut others = [(PrimitiveIndex(0), LayerTemplateIndex(0)); MAX_PRIMITIVES_PER_PASS-1];
        if !prim.is_opaque {
            if prim.intersecting_prims_behind.len() > MAX_PRIMITIVES_PER_PASS-1 {
                println!("TODO: Found a prim with too many intersecting blends - multiple passes not handled yet!");
                println!("\tPrimitive {:?} {:?} {}", prim.xf_rect.screen_rect, prim.packed.kind, prim.is_opaque);
                for other_prim_index in &prim.intersecting_prims_behind {
                    let other_prim_index = *other_prim_index;
                    let other_prim = self.get_prim(other_prim_index);
                    println!("\t\t{:?} {:?} {:?}", other_prim.xf_rect.screen_rect, other_prim.packed.kind, other_prim.is_opaque);
                }

                return None;
            }

            for (array_index, other_prim_index) in prim.intersecting_prims_behind.iter().enumerate() {
                let other_prim_index = *other_prim_index;
                let other_prim = self.get_prim(other_prim_index);
                others[array_index] = (other_prim_index, other_prim.layer);
            }
        }

        let mut shader = None;

        // If this primitive is opaque, there's no need for secondary passes.
        if prim.is_opaque || prim.intersecting_prims_behind.is_empty() {
            shader = Some(match (key.kind, prim.clip_in.is_some()) {
                (PrimitiveKind::Rectangle, false) => PrimitiveShader::Rect,
                (PrimitiveKind::Rectangle, true) => PrimitiveShader::Rect_Clip,
                (PrimitiveKind::Image, _) => PrimitiveShader::Image,
                (PrimitiveKind::Text, _) => PrimitiveShader::Text,
                (PrimitiveKind::BorderCorner, false) => PrimitiveShader::BorderCorner,
                (PrimitiveKind::BorderCorner, true) => PrimitiveShader::BorderCorner_Clip,
                (PrimitiveKind::BoxShadow, _) => PrimitiveShader::BoxShadow,
            });
        } else {
            // Check for special fast paths
            if prim.intersecting_prims_behind.len() == 1 {
                let other_prim_index = prim.intersecting_prims_behind.last().unwrap();
                let other_prim = self.get_prim(*other_prim_index);

                shader = match (key.kind, other_prim.packed.kind, prim.clip_in.is_some()) {
                    (PrimitiveKind::Text, PrimitiveKind::Rectangle, false) => Some(PrimitiveShader::Text_Rect),
                    (PrimitiveKind::Image, PrimitiveKind::Rectangle, false) => Some(PrimitiveShader::Image_Rect),
                    (k0, k1, c) => {
                        //println!("Missing fast path: {:?} {:?} {:?}", k0, k1, c);
                        None
                    }
                }
            }

/*
            if shader.is_none() && prim.intersecting_prims_behind.len() == 2 {
                let pi1 = prim.intersecting_prims_behind[0];
                let p1 = self.get_prim(pi1);

                let pi2 = prim.intersecting_prims_behind[1];
                let p2 = self.get_prim(pi2);

                shader = match (key.kind, p1.packed.kind, p2.packed.kind, prim.clip.is_some()) {
                    (PrimitiveKind::Text, PrimitiveKind::Text, PrimitiveKind::Rectangle, false) => Some(PrimitiveShader::Text_Rect),
                    (k0, k1, c) => { println!("TODO2: {:?} {:?} {:?}", k0, k1, c); None }
                }
            }
            */

            if shader.is_none() {
                // Generic shader technique paths
                shader = if prim.intersecting_prims_behind.len() == 1 {
                    Some(PrimitiveShader::Generic2)
                } else {
                    Some(PrimitiveShader::Generic4)
                };
            }
        }

        shader.map(|shader| {
            RenderPrimitive {
                shader: shader,
                key: key,
                layer_index_in_ubo: layer_index_in_ubo,
                clip_in_index_in_ubo: clip_in_index_in_ubo,
                clip_out_index_in_ubo: clip_out_index_in_ubo,
                other_primitives: others,
            }
        })
    }

    // TODO(gw): Not transform safe!
    fn primitive_is_occluded(&self, prim: &Primitive) -> bool {
        for prim_in_front_index in &prim.intersecting_prims_in_front {
            let prim_in_front = self.get_prim(*prim_in_front_index);

            // Don't need to be opaque due to batch ordering...
            if //prim_in_front.is_opaque &&
               prim_in_front.xf_rect.screen_rect.contains_rect(&prim.xf_rect.screen_rect) {
                //println!("prim {:?} is OCCLUDED by {:?}", prim.xf_rect.screen_rect, prim_in_front.xf_rect.screen_rect);
                return true;
            }
        }

        false
    }

    pub fn build(mut self) -> Frame {
        let collision_pairs = self.build_collision_pairs();

        self.assign_primitive_ordering(&collision_pairs);

        let mut layer_ubo = Ubo::new();
        let mut clip_ubo = Ubo::new();
        let mut render_prims = Vec::new();

        for layer_instance in &self.layer_instances {
            let LayerTemplateIndex(layer_index) = layer_instance.layer_index;
            let layer_template = &self.layer_templates[layer_index as usize].packed;
            let layer_index_in_ubo = layer_ubo.maybe_insert_and_get_index(layer_instance.layer_index,
                                                                          layer_template);

            for primitive_key in &layer_instance.primitives {
                let primitive_key = *primitive_key;
                let prim = self.get_prim(primitive_key.index);

                let clip_in_index_in_ubo = prim.clip_in.as_ref().map(|clip| {
                    // hack hack hack to ensure it always gets added - refcount these?
                    let clip_index = ClipIndex(clip_ubo.items.len() as u32);
                    clip_ubo.maybe_insert_and_get_index(clip_index, clip)
                });

                let clip_out_index_in_ubo = prim.clip_out.as_ref().map(|clip| {
                    // hack hack hack to ensure it always gets added - refcount these?
                    let clip_index = ClipIndex(clip_ubo.items.len() as u32);
                    clip_ubo.maybe_insert_and_get_index(clip_index, clip)
                });

                let render_prim = self.build_primitive(primitive_key,
                                                       layer_index_in_ubo,
                                                       clip_in_index_in_ubo,
                                                       clip_out_index_in_ubo)
                                      .unwrap_or_else(|| {
                                        RenderPrimitive {
                                            shader: PrimitiveShader::Error,
                                            key: primitive_key,
                                            layer_index_in_ubo: layer_index_in_ubo,
                                            clip_in_index_in_ubo: clip_in_index_in_ubo,
                                            clip_out_index_in_ubo: clip_out_index_in_ubo,
                                            other_primitives: [(PrimitiveIndex(0), LayerTemplateIndex(0)); MAX_PRIMITIVES_PER_PASS-1],
                                        }
                                      });
                render_prims.push(render_prim);
            }
        }

        // Build batches for each shader
        // TODO: Add support for multiple render passes here!
        let mut prim_ubo = Ubo::new();
        let mut batches: Vec<Batch> = Vec::new();

        for render_prim in render_prims.iter().rev() {
            let mut draw_cmd = PackedDrawCommand::empty();

            let main_prim = self.get_prim(render_prim.key.index);
            let main_prim_index_in_ubo = prim_ubo.maybe_insert_and_get_index(render_prim.key.index,
                                                                             &main_prim.packed);
            draw_cmd.set_primitive(0, main_prim_index_in_ubo, render_prim.layer_index_in_ubo);
            draw_cmd.clip_info[0] = render_prim.clip_in_index_in_ubo.unwrap_or(INVALID_CLIP_INDEX);
            draw_cmd.clip_info[1] = render_prim.clip_out_index_in_ubo.unwrap_or(INVALID_CLIP_INDEX);

            for (other_prim_index, other_prim_info) in render_prim.other_primitives.iter().enumerate() {
                let other_prim = self.get_prim(other_prim_info.0);
                let other_prim_index_in_ubo = prim_ubo.maybe_insert_and_get_index(other_prim_info.0,
                                                                                  &other_prim.packed);

                let LayerTemplateIndex(other_layer_index) = other_prim_info.1;
                let other_layer_template = &self.layer_templates[other_layer_index as usize].packed;
                let other_layer_index_in_ubo = layer_ubo.maybe_insert_and_get_index(other_prim_info.1,
                                                                                    other_layer_template);

                draw_cmd.set_primitive(1 + other_prim_index,
                                       other_prim_index_in_ubo,
                                       other_layer_index_in_ubo);
            }

            // TODO(gw): Search other batches to add to, based on shader and overlap detection.
            let need_new_batch = match batches.last() {
                Some(ref batch) => {
                    batch.shader != render_prim.shader
                }
                None => true,
            };

            if need_new_batch {
                let batch = Batch::new(render_prim.shader, 0, 0, 0);
                batches.push(batch);
            }

            let batch = batches.last_mut().unwrap();
            batch.commands.push(draw_cmd);
        }

        let layer_ubos = vec![layer_ubo];
        let primitive_ubos = vec![prim_ubo];
        let clip_ubos = vec![clip_ubo];
        let passes = vec![Pass {
            viewport_size: Size2D::new(self.screen_rect.size.width as u32,
                                       self.screen_rect.size.height as u32),
            batches: batches,
        }];

        Frame {
            layer_ubos: layer_ubos,
            primitive_ubos: primitive_ubos,
            clip_ubos: clip_ubos,
            passes: passes,
            color_texture_id: self.color_texture_id,
            mask_texture_id: self.mask_texture_id,
            text_buffer: self.text_buffer,
        }
    }

    fn should_add_prim(&self, rect: &Rect<f32>) -> Option<TransformedRect> {
        let current_layer = *self.layer_stack.last().unwrap();
        let LayerTemplateIndex(current_layer_index) = current_layer;
        let layer = &self.layer_templates[current_layer_index as usize];

        let xf_rect = TransformedRect::new(rect, &layer.packed.transform);
        if self.screen_rect.intersects(&xf_rect.screen_rect) {
            Some(xf_rect)
        } else {
            None
        }
    }

    fn add_primitive(&mut self,
                     kind: PrimitiveKind,
                     xf_rect: TransformedRect,
                     packed: PackedPrimitive,
                     clip_in: Option<Clip>,
                     clip_out: Option<Clip>,
                     is_opaque: bool) {
        let current_layer = *self.layer_stack.last().unwrap();

        if self.layer_instances.is_empty() ||
           self.layer_instances.last().unwrap().layer_index != current_layer {
            let instance = LayerInstance {
                layer_index: current_layer,
                primitives: Vec::new(),
            };
            self.layer_instances.push(instance);
        }

        let prim_key = PrimitiveKey::new(kind, self.primitives.len());
        self.primitives.push(Primitive {
            layer: current_layer,
            xf_rect: xf_rect,
            is_opaque: is_opaque,
            packed: packed,
            intersecting_prims_behind: Vec::new(),
            intersecting_prims_in_front: Vec::new(),
            clip_in: clip_in,
            clip_out: clip_out,
        });
        self.layer_instances.last_mut().unwrap().primitives.push(prim_key);
    }
}

#[derive(Debug, Clone)]
pub struct ClipCorner {
    position: Point2D<f32>,
    padding: Point2D<f32>,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

impl ClipCorner {
    fn invalid() -> ClipCorner {
        ClipCorner {
            position: Point2D::zero(),
            padding: Point2D::zero(),
            outer_radius_x: 0.0,
            outer_radius_y: 0.0,
            inner_radius_x: 0.0,
            inner_radius_y: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Clip {
    p0: Point2D<f32>,
    p1: Point2D<f32>,
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
}

impl Clip {
    pub fn from_clip_region(clip: &ComplexClipRegion) -> Clip {
        Clip {
            p0: clip.rect.origin,
            p1: clip.rect.bottom_right(),
            top_left: ClipCorner {
                position: Point2D::new(clip.rect.origin.x + clip.radii.top_left.width,
                                       clip.rect.origin.y + clip.radii.top_left.height),
                outer_radius_x: clip.radii.top_left.width,
                outer_radius_y: clip.radii.top_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
                padding: Point2D::zero(),
            },
            top_right: ClipCorner {
                position: Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.top_right.width,
                                       clip.rect.origin.y + clip.radii.top_right.height),
                outer_radius_x: clip.radii.top_right.width,
                outer_radius_y: clip.radii.top_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
                padding: Point2D::zero(),
            },
            bottom_left: ClipCorner {
                position: Point2D::new(clip.rect.origin.x + clip.radii.bottom_left.width,
                                       clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_left.height),
                outer_radius_x: clip.radii.bottom_left.width,
                outer_radius_y: clip.radii.bottom_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
                padding: Point2D::zero(),
            },
            bottom_right: ClipCorner {
                position: Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.bottom_right.width,
                                       clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_right.height),
                outer_radius_x: clip.radii.bottom_right.width,
                outer_radius_y: clip.radii.bottom_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
                padding: Point2D::zero(),
            },
        }
    }

    pub fn from_rect(rect: &Rect<f32>) -> Clip {
        Clip {
            p0: rect.origin,
            p1: rect.bottom_right(),
            top_left: ClipCorner::invalid(),
            top_right: ClipCorner::invalid(),
            bottom_left: ClipCorner::invalid(),
            bottom_right: ClipCorner::invalid(),
        }
    }

    pub fn from_border_radius(rect: &Rect<f32>,
                              outer_radius: &BorderRadius,
                              inner_radius: &BorderRadius) -> Clip {
        Clip {
            p0: rect.origin,
            p1: rect.bottom_right(),
            top_left: ClipCorner {
                position: Point2D::new(rect.origin.x + outer_radius.top_left.width,
                                       rect.origin.y + outer_radius.top_left.height),
                outer_radius_x: outer_radius.top_left.width,
                outer_radius_y: outer_radius.top_left.height,
                inner_radius_x: inner_radius.top_left.width,
                inner_radius_y: inner_radius.top_left.height,
                padding: Point2D::zero(),
            },
            top_right: ClipCorner {
                position: Point2D::new(rect.origin.x + rect.size.width - outer_radius.top_right.width,
                                       rect.origin.y + outer_radius.top_right.height),
                outer_radius_x: outer_radius.top_right.width,
                outer_radius_y: outer_radius.top_right.height,
                inner_radius_x: inner_radius.top_right.width,
                inner_radius_y: inner_radius.top_right.height,
                padding: Point2D::zero(),
            },
            bottom_left: ClipCorner {
                position: Point2D::new(rect.origin.x + outer_radius.bottom_left.width,
                                       rect.origin.y + rect.size.height - outer_radius.bottom_left.height),
                outer_radius_x: outer_radius.bottom_left.width,
                outer_radius_y: outer_radius.bottom_left.height,
                inner_radius_x: inner_radius.bottom_left.width,
                inner_radius_y: inner_radius.bottom_left.height,
                padding: Point2D::zero(),
            },
            bottom_right: ClipCorner {
                position: Point2D::new(rect.origin.x + rect.size.width - outer_radius.bottom_right.width,
                                       rect.origin.y + rect.size.height - outer_radius.bottom_right.height),
                outer_radius_x: outer_radius.bottom_right.width,
                outer_radius_y: outer_radius.bottom_right.height,
                inner_radius_x: inner_radius.bottom_right.width,
                inner_radius_y: inner_radius.bottom_right.height,
                padding: Point2D::zero(),
            },
        }
    }
}
