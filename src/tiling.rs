/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch_builder::BorderSideHelpers;
use device::{TextureId, TextureFilter};
use euclid::{Point2D, Rect, Matrix4, Size2D, Point4D};
use fnv::FnvHasher;
use frame::FrameId;
use internal_types::{Glyph, GlyphKey};
use renderer::{BLUR_INFLATION_FACTOR, TEXT_TARGET_SIZE};
use resource_cache::ResourceCache;
use resource_list::ResourceList;
use std::cmp;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry::{Occupied, Vacant};
//use std::f32;
use std::mem;
use std::hash::{Hash, BuildHasherDefault};
use texture_cache::TexturePage;
use util::RectHelpers;
use webrender_traits::{ColorF, FontKey, GlyphInstance, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderRadius};

const MAX_PRIMITIVES_PER_PASS: usize = 8;
const INVALID_PRIM_INDEX: u32 = 0xffffffff;

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

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum PrimitiveShader {
    OpaqueRectangle,
    OpaqueImage,
    OpaqueRectangleText,
    Generic2,
    Generic4,
    Generic6,
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
    Gradient,
    Text,

    Invalid,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PrimitiveIndex(u32);

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
    transform: Matrix4,
    inv_transform: Matrix4,
    screen_vertices: [Point4D<f32>; 4],
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct PackedPrimitive {
    pub p0: Point2D<f32>,
    pub p1: Point2D<f32>,
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
    pub color: ColorF,
    pub kind: PrimitiveKind,
    pub padding: [u32; 3],
}

#[derive(Debug)]
struct Primitive {
    layer: LayerTemplateIndex,
    xf_rect: TransformedRect,
    is_opaque: bool,
    packed: PackedPrimitive,
    intersecting_prims_behind: Vec<PrimitiveIndex>,
    intersecting_prims_in_front: Vec<PrimitiveIndex>,
}

// TODO (gw): Profile and create a smaller layout for simple passes if worthwhile...
#[derive(Debug)]
pub struct PackedDrawCommand {
    pub prim_indices: [u32; MAX_PRIMITIVES_PER_PASS],
    pub layer_indices: [u32; MAX_PRIMITIVES_PER_PASS],
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
        }
    }
}

#[derive(Debug)]
struct RenderPrimitive {
    layer_index_in_ubo: u32,
    key: PrimitiveKey,
    shader: PrimitiveShader,
    other_primitives: Vec<(PrimitiveIndex, LayerTemplateIndex)>,
}

#[derive(Debug)]
struct TransformedRect {
    local_rect: Rect<f32>,
    vertices: [Point4D<f32>; 4],
    screen_rect: Rect<i32>,
}

impl TransformedRect {
    fn new(rect: &Rect<f32>, transform: &Matrix4) -> TransformedRect {
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
    pub commands: Vec<PackedDrawCommand>,
}

impl Batch {
    fn new(shader: PrimitiveShader,
           layer_ubo_index: usize,
           prim_ubo_index: usize) -> Batch {
        Batch {
            shader: shader,
            commands: Vec::new(),
            prim_ubo_index: prim_ubo_index,
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
}

impl FrameBuilder {
    pub fn new(viewport_size: Size2D<f32>,
               scroll_offset: Point2D<f32>,) -> FrameBuilder {
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
        }
    }

    pub fn push_layer(&mut self,
                      rect: Rect<f32>,
                      transform: Matrix4,
                      _: f32) {
        // TODO(gw): Not 3d transform correct!
        let scroll_transform = transform.translate(self.scroll_offset.x,
                                                   self.scroll_offset.y,
                                                   0.0);

        let layer_rect = TransformedRect::new(&rect, &transform);

        let template = LayerTemplate {
            packed: PackedLayer {
                inv_transform: transform.invert(),
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

    pub fn add_gradient(&mut self,
                        _rect: Rect<f32>,
                        _start_point: &Point2D<f32>,
                        _end_point: &Point2D<f32>,
                        _stops: &ItemRange,
                        _auxiliary_lists: &AuxiliaryLists) {
        println!("TODO: gradient");
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

        if let Some(xf_rect) = self.should_add_prim(&rect) {
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
                                       color: *color,
                                       kind: PrimitiveKind::Text,
                                       padding: [0, 0, 0],
                                    },
                                    false);
            }
        }
    }

    pub fn add_image(&mut self,
                     rect: Rect<f32>,
                     stretch_size: &Size2D<f32>,
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
                                   color: ColorF::new(1.0, 1.0, 1.0, 1.0),
                                   kind: PrimitiveKind::Image,
                                   padding: [0, 0, 0],
                               },
                               image_info.is_opaque);
        }
    }

    pub fn add_rectangle(&mut self,
                         rect: Rect<f32>,
                         color: ColorF) {
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
                                   color: color,
                                   kind: PrimitiveKind::Rectangle,
                                   padding: [0, 0, 0],
                               },
                               color.a == 1.0);
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

        if left.style != BorderStyle::Solid ||
           top.style != BorderStyle::Solid ||
           bottom.style != BorderStyle::Solid ||
           right.style != BorderStyle::Solid {
            //println!("TODO: Other border styles");
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
        self.add_rectangle(Rect::new(Point2D::new(tl_outer.x, tl_inner.y),
                                     Size2D::new(left.width, bl_inner.y - tl_inner.y)),
                           left_color);

        self.add_rectangle(Rect::new(Point2D::new(tl_inner.x, tl_outer.y),
                                     Size2D::new(tr_inner.x - tl_inner.x, tr_outer.y + top.width - tl_outer.y)),
                           top_color);

        self.add_rectangle(Rect::new(Point2D::new(br_outer.x - right.width, tr_inner.y),
                                     Size2D::new(right.width, br_inner.y - tr_inner.y)),
                           right_color);

        self.add_rectangle(Rect::new(Point2D::new(bl_inner.x, bl_outer.y - bottom.width),
                                     Size2D::new(br_inner.x - bl_inner.x, br_outer.y - bl_outer.y + bottom.width)),
                           bottom_color);

        // Corners
        let need_clip = radius.top_left != Size2D::zero() ||
                        radius.top_right != Size2D::zero() ||
                        radius.bottom_left != Size2D::zero() ||
                        radius.bottom_right != Size2D::zero();

        if need_clip {
            // TODO(gw): This is all wrong for non-uniform borders!
            /*
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

            let clip = Clip::from_border_radius(&rect,
                                                radius,
                                                &inner_radius);

            //self.set_clip(clip);
            */
        }

        self.add_rectangle(Rect::new(tl_outer,
                                     Size2D::new(tl_inner.x - tl_outer.x,
                                                 tl_inner.y - tl_outer.y)),
                           left_color);

        self.add_rectangle(Rect::new(Point2D::new(tr_inner.x, tr_outer.y),
                                     Size2D::new(tr_outer.x - tr_inner.x,
                                                 tr_inner.y - tr_outer.y)),
                           top_color);

        self.add_rectangle(Rect::new(br_inner,
                                     Size2D::new(br_outer.x - br_inner.x,
                                                 br_outer.y - br_inner.y)),
                           right_color);

        self.add_rectangle(Rect::new(Point2D::new(bl_outer.x, bl_inner.y),
                                     Size2D::new(bl_inner.x - bl_outer.x,
                                                 bl_outer.y - bl_inner.y)),
                           bottom_color);

        if need_clip {
            //self.clear_clip();
        }
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
                       layer_index_in_ubo: u32) -> Option<RenderPrimitive> {
        let prim = self.get_prim(key.index);

        // If this primitive is occluded, there is no
        // need to submit it as a separate primitive,
        // as it will be handled by the blending logic
        // for that primitive!
        if self.primitive_is_occluded(prim) {
            return None;
        }

        // If this primitive is opaque, there's no need for secondary passes.
        if prim.is_opaque || prim.intersecting_prims_behind.is_empty() {
            match key.kind {
                PrimitiveKind::Rectangle => {
                    Some(RenderPrimitive {
                        shader: PrimitiveShader::OpaqueRectangle,
                        key: key,
                        layer_index_in_ubo: layer_index_in_ubo,
                        other_primitives: Vec::new(),
                    })
                }
                PrimitiveKind::Image => {
                    Some(RenderPrimitive {
                        shader: PrimitiveShader::OpaqueImage,
                        key: key,
                        layer_index_in_ubo: layer_index_in_ubo,
                        other_primitives: Vec::new(),
                    })
                }
                _ => {
                    println!("TODO: Opaque non-rect/image!");
                    None
                }
            }
        } else if prim.intersecting_prims_behind.len() == 1 {
            let back_prim_index = prim.intersecting_prims_behind[0];
            let back_prim = self.get_prim(back_prim_index);
            match (key.kind, back_prim.packed.kind) {
                (PrimitiveKind::Text, PrimitiveKind::Rectangle) => {
                    Some(RenderPrimitive {
                        shader: PrimitiveShader::OpaqueRectangleText,
                        key: key,
                        layer_index_in_ubo: layer_index_in_ubo,
                        other_primitives: vec![(back_prim_index, back_prim.layer)],
                    })
                }
                _ => {
                    Some(RenderPrimitive {
                        shader: PrimitiveShader::Generic2,
                        key: key,
                        layer_index_in_ubo: layer_index_in_ubo,
                        other_primitives: vec![(back_prim_index, back_prim.layer)],
                    })
                }
            }
        } else if prim.intersecting_prims_behind.len() < 5 {
            let mut others = Vec::new();
            for other_prim_index in &prim.intersecting_prims_behind {
                let other_prim_index = *other_prim_index;
                let other_prim = self.get_prim(other_prim_index);
                others.push((other_prim_index, other_prim.layer));
            }
            Some(RenderPrimitive {
                shader: PrimitiveShader::Generic4,
                key: key,
                layer_index_in_ubo: layer_index_in_ubo,
                other_primitives: others,
            })
        } else if prim.intersecting_prims_behind.len() < 7 {
            let mut others = Vec::new();
            for other_prim_index in &prim.intersecting_prims_behind {
                let other_prim_index = *other_prim_index;
                let other_prim = self.get_prim(other_prim_index);
                others.push((other_prim_index, other_prim.layer));
            }
            Some(RenderPrimitive {
                shader: PrimitiveShader::Generic6,
                key: key,
                layer_index_in_ubo: layer_index_in_ubo,
                other_primitives: others,
            })
        } else {
            println!("TODO: blending with multiple layers {:?}", prim);
            None
        }
    }

    // TODO(gw): Not transform safe!
    fn primitive_is_occluded(&self, prim: &Primitive) -> bool {
        for prim_in_front_index in &prim.intersecting_prims_in_front {
            let prim_in_front = self.get_prim(*prim_in_front_index);

            // Don't need to be opaque due to batch ordering...
            if //prim_in_front.is_opaque &&
               prim_in_front.xf_rect.screen_rect.contains_rect(&prim.xf_rect.screen_rect) {
                println!("prim {:?} is OCCLUDED by {:?}", prim.xf_rect.screen_rect, prim_in_front.xf_rect.screen_rect);
                return true;
            }
        }

        false
    }

    pub fn build(mut self) -> Frame {
        let collision_pairs = self.build_collision_pairs();

        self.assign_primitive_ordering(&collision_pairs);

        let mut layer_ubo = Ubo::new();
        let mut render_prims = Vec::new();

        for layer_instance in &self.layer_instances {
            let LayerTemplateIndex(layer_index) = layer_instance.layer_index;
            let layer_template = &self.layer_templates[layer_index as usize].packed;
            let layer_index_in_ubo = layer_ubo.maybe_insert_and_get_index(layer_instance.layer_index,
                                                                          layer_template);

            for primitive_key in &layer_instance.primitives {
                let render_prim = self.build_primitive(*primitive_key,
                                                       layer_index_in_ubo)
                                      .unwrap_or_else(|| {
                                        RenderPrimitive {
                                            shader: PrimitiveShader::Error,
                                            key: *primitive_key,
                                            layer_index_in_ubo: layer_index_in_ubo,
                                            other_primitives: Vec::new(),
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
                let batch = Batch::new(render_prim.shader, 0, 0);
                batches.push(batch);
            }

            let batch = batches.last_mut().unwrap();
            batch.commands.push(draw_cmd);
        }

        //println!("{:?}", layer_ubo.items);
        //println!("{:?}", prim_ubo.items);
        //println!("{:?}", opaque_rect_batch);

        let layer_ubos = vec![layer_ubo];
        let primitive_ubos = vec![prim_ubo];
        let passes = vec![Pass {
            viewport_size: Size2D::new(self.screen_rect.size.width as u32,
                                       self.screen_rect.size.height as u32),
            batches: batches,
        }];

        Frame {
            layer_ubos: layer_ubos,
            primitive_ubos: primitive_ubos,
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
        });
        self.layer_instances.last_mut().unwrap().primitives.push(prim_key);
    }
}

/*
#[derive(Debug, Clone)]
pub struct ClipCorner {
    position: Point2D<f32>,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
    padding: Point2D<f32>,
}

impl ClipCorner {
    pub fn invalid() -> ClipCorner {
        ClipCorner {
            position: Point2D::zero(),
            outer_radius_x: 0.0,
            outer_radius_y: 0.0,
            inner_radius_x: 0.0,
            inner_radius_y: 0.0,
            padding: Point2D::zero(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Clip {
    rect: Rect<f32>,
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
}

impl Clip {
    /*
    pub fn invalid() -> Clip {
        Clip {
            rect: Rect::zero(),
            top_left: ClipCorner::invalid(),
            top_right: ClipCorner::invalid(),
            bottom_left: ClipCorner::invalid(),
            bottom_right: ClipCorner::invalid(),
        }
    }
*/

    pub fn from_clip_region(clip: &ComplexClipRegion) -> Clip {
        Clip {
            rect: clip.rect,
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

    pub fn from_border_radius(rect: &Rect<f32>,
                              outer_radius: &BorderRadius,
                              inner_radius: &BorderRadius) -> Clip {
        Clip {
            rect: *rect,
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


/*

#[derive(Clone, Debug)]
pub struct PackedGradientStop {
    color: ColorF,
    offset: [f32; 4],
}

pub struct GradientStopPrimitive {
    stops: Vec<PackedGradientStop>,
}

#[derive(Clone, Debug)]
pub struct PackedGradient {
    p0: Point2D<f32>,
    p1: Point2D<f32>,
    constants: [f32; 4],
    stop_start_index: u32,
    stop_count: u32,
    pad0: u32,
    pad1: u32,
}

pub struct GradientPrimitive {
    rect: Rect<f32>,
    stops_index: GradientStopPrimitiveIndex,
    packed: PackedGradient,
}

*/

*/
