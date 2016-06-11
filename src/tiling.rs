/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::{Au};
use batch_builder::{BorderSideHelpers, BoxShadowMetrics};
use bsptree::BspTree;
use device::{TextureId, TextureFilter};
use euclid::{Point2D, Rect, Matrix4D, Size2D, Point4D};
use fnv::FnvHasher;
use frame::FrameId;
use internal_types::{AxisDirection, Glyph, GlyphKey, DevicePixel};
use layer::Layer;
use renderer::{BLUR_INFLATION_FACTOR};
use resource_cache::ResourceCache;
use resource_list::ResourceList;
use std::cmp;
use std::collections::{HashMap};
use std::f32;
use std::mem;
use std::hash::{BuildHasherDefault};
use texture_cache::{TexturePage, TextureCacheItem};
use util::{self, rect_from_points, rect_from_points_f};
use webrender_traits::{ColorF, FontKey, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderRadius};
use webrender_traits::{BoxShadowClipMode, PipelineId, ScrollLayerId};

const INVALID_LAYER_INDEX: u32 = 0xffffffff;

/*
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum BlendInput {
    None,
    InitialColor,
    PreviousOutput,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum BlendKind {
    None,
    Opacity,
}

#[derive(Debug)]
struct LayerInfo {
    blend_input: BlendInput,
    blend_kind: BlendKind,
    layer_index: RenderLayerIndex,
    instances: Vec<RenderableInstanceId>,
    value: f32,
    last_prim_is_opaque: bool,
}

impl LayerInfo {
    fn new(layer_index: RenderLayerIndex,
           value: f32) -> LayerInfo {
        LayerInfo {
            blend_input: BlendInput::None,
            blend_kind: BlendKind::None,
            layer_index: layer_index,
            value: value,
            instances: Vec::new(),
            last_prim_is_opaque: false,
        }
    }

    fn add_prim(&mut self,
                id: RenderableInstanceId,
                layer_index: RenderLayerIndex,
                prim_is_opaque: bool) -> bool {
        debug_assert!(layer_index == self.layer_index);

        if self.last_prim_is_opaque {
            false
        } else {
            self.instances.push(id);
            self.last_prim_is_opaque = prim_is_opaque;
            true
        }
    }

    fn is_opaque(&self) -> bool {
        self.value == 1.0 &&
        self.last_prim_is_opaque
    }
}
*/

#[derive(Debug)]
struct TilingInfo {
    x_tile_count: i32,
    y_tile_count: i32,
    x_tile_size: DevicePixel,
    y_tile_size: DevicePixel,
}

impl TilingInfo {
    #[inline(always)]
    fn get_tile_range(&self, rect: &Rect<DevicePixel>) -> TileRange {
        let px0 = rect.origin.x;
        let py0 = rect.origin.y;
        let px1 = rect.origin.x + rect.size.width;
        let py1 = rect.origin.y + rect.size.height;

        let tx0 = px0.0 / self.x_tile_size.0;
        let ty0 = py0.0 / self.y_tile_size.0;
        let tx1 = (px1.0 + self.x_tile_size.0 - 1) / self.x_tile_size.0;
        let ty1 = (py1.0 + self.y_tile_size.0 - 1) / self.y_tile_size.0;

        let tx0 = cmp::max(0, cmp::min(tx0, self.x_tile_count));
        let ty0 = cmp::max(0, cmp::min(ty0, self.y_tile_count));
        let tx1 = cmp::max(0, cmp::min(tx1, self.x_tile_count));
        let ty1 = cmp::max(0, cmp::min(ty1, self.y_tile_count));

        TileRange::new(tx0, ty0, tx1, ty1)
    }
}

#[derive(Debug)]
struct TileRange {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
}

impl TileRange {
    fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> TileRange {
        TileRange {
            x0: x0,
            y0: y0,
            x1: x1,
            y1: y1,
        }
    }
}

#[derive(Debug, Copy, Clone)]
enum ClipPart {
    All,
    //TopLeft,
    //TopRight,
    //BottomLeft,
    //BottomRight,
    None,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ClipChannel(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct ClipIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct PrimitiveId(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct RenderableId(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct RenderableInstanceId(u32);

#[derive(Clone, Debug)]
pub struct PackedLayer {
    transform: Matrix4D<f32>,
    inv_transform: Matrix4D<f32>,
    screen_vertices: [Point4D<f32>; 4],
    blend_info: [f32; 4],
}

#[derive(Debug)]
pub struct Ubo<TYPE> {
    pub items: Vec<TYPE>,
}

impl<TYPE: Clone> Ubo<TYPE> {
    fn new() -> Ubo<TYPE> {
        Ubo {
            items: Vec::new(),
        }
    }

    #[inline]
    fn push(&mut self, new_item: &TYPE) -> u32 {
        let index = self.items.len() as u32;
        self.items.push(new_item.clone());
        index
    }
}

const MAX_LAYERS_PER_COMPOSITE: usize = 4;
const MAX_PRIMS_PER_COMPOSITE: usize = 4;

#[allow(non_camel_case_types)]
#[derive(Debug, Eq, PartialEq, Copy, Clone, Hash)]
pub enum CompositeShader {
    Prim1_Opaque,
    Prim1,
    Prim2_Opaque,
    Prim2,

    Layer2_Prim2_Opaque
}

/*
#[derive(Debug, Clone, Copy)]
pub struct CompositeLayer {
    pub first_prim_index: u32,
    pub prim_count: u32,
    pub pad0: [u32; 2],
    pub value: f32,
    pub pad1: [u32; 3],
}

impl CompositeLayer {
    fn invalid() -> CompositeLayer {
        CompositeLayer {
            first_prim_index: 0,
            prim_count: 0,
            pad0: [0, 0],
            value: 0.0,
            pad1: [0, 0, 0],
        }
    }
}*/

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum BlendMode {
    None = 0,
    Accumulate,
    LayerOpacity,
}

#[derive(Debug, Clone)]
pub struct CompositeTile {
    pub rect: Rect<DevicePixel>,
    pub prim_indices: [RenderableInstanceId; MAX_PRIMS_PER_COMPOSITE],
    pub blend_modes: [BlendMode; MAX_PRIMS_PER_COMPOSITE],
    pub layer_values: [f32; MAX_PRIMS_PER_COMPOSITE],
}

impl CompositeTile {
    fn new(rect: &Rect<DevicePixel>) -> CompositeTile {
        CompositeTile {
            rect: *rect,
            //layers: [CompositeLayer::invalid(); MAX_LAYERS_PER_COMPOSITE],
            prim_indices: unsafe { mem::zeroed() },
            blend_modes: unsafe { mem::zeroed() },
            layer_values: unsafe { mem::zeroed() },
        }
    }

/*
    fn set_layer(&mut self,
                 layer_index: usize,
                 first_prim_index: u32,
                 prim_count: u32,
                 value: f32) {
        self.layers[layer_index] = CompositeLayer {
            first_prim_index: first_prim_index,
            prim_count: prim_count,
            pad0: [0, 0],
            value: value,
            pad1: [0, 0, 0],
        };
    }*/

    fn set_primitive(&mut self,
                     cmd_index: usize,
                     prim_index: RenderableInstanceId,
                     blend_mode: BlendMode,
                     layer_value: f32) {
        self.prim_indices[cmd_index] = prim_index;
        self.blend_modes[cmd_index] = blend_mode;
        self.layer_values[cmd_index] = layer_value;
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RenderLayerIndex(usize);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct PrimitiveIndex(usize);

#[derive(Debug)]
struct RectanglePrimitive {
    color: ColorF,
}

#[derive(Debug)]
struct TextPrimitive {
    color: ColorF,
    font_key: FontKey,
    size: Au,
    blur_radius: Au,
    glyph_range: ItemRange,
    //run_key: TextRunKey,
}

#[derive(Debug)]
struct ImagePrimitive {
    image_key: ImageKey,
    image_rendering: ImageRendering,
}

#[derive(Debug)]
struct GradientPrimitive {
    stops_range: ItemRange,
    dir: AxisDirection,
}

#[derive(Debug)]
struct BorderPrimitive {
    tl_outer: Point2D<f32>,
    tl_inner: Point2D<f32>,
    tr_outer: Point2D<f32>,
    tr_inner: Point2D<f32>,
    bl_outer: Point2D<f32>,
    bl_inner: Point2D<f32>,
    br_outer: Point2D<f32>,
    br_inner: Point2D<f32>,
    left_width: f32,
    top_width: f32,
    right_width: f32,
    bottom_width: f32,
    radius: BorderRadius,
    left_color: ColorF,
    top_color: ColorF,
    right_color: ColorF,
    bottom_color: ColorF,
}

#[derive(Debug)]
struct BoxShadowPrimitive {
    src_rect: Rect<f32>,
    bs_rect: Rect<f32>,
    color: ColorF,
    blur_radius: f32,
    spread_radius: f32,
    border_radius: f32,
    clip_mode: BoxShadowClipMode,
    metrics: BoxShadowMetrics,
}

#[derive(Debug)]
enum PrimitiveDetails {
    Rectangle(RectanglePrimitive),
    Text(TextPrimitive),
    Image(ImagePrimitive),
    Border(BorderPrimitive),
    Gradient(GradientPrimitive),
    BoxShadow(BoxShadowPrimitive),
}

#[derive(Debug)]
struct Primitive {
    rect: Rect<f32>,
    clip_index: Option<ClipIndex>,
    xf_rect: Option<TransformedRect>,
    details: PrimitiveDetails,
}

impl Primitive {
    fn build_renderables(&self,
                         layer_index: RenderLayerIndex,
                         prim_index: PrimitiveIndex,
                         tile_info: &TilingInfo,
                         device_pixel_ratio: f32,
                         renderables: &mut RenderableList,
                         auxiliary_lists: &AuxiliaryLists,
                         resource_cache: &ResourceCache,
                         frame_id: FrameId,
                         transform: &Matrix4D<f32>,
                         clips: &Vec<Clip>) {
        // Work out which BSP tree/tile this primitive belongs to.
        let xf_rect = self.xf_rect.as_ref().unwrap();

        match self.details {
            PrimitiveDetails::Rectangle(ref details) => {
                if let Some(clip_index) = self.clip_index {
                    if xf_rect.should_split() {
                        // TODO(gw): This doesn't handle 3d transformed rectangles
                        //           or irregular clip sizes at all!. Fix!
                        let rect = &xf_rect.local_rect;
                        let ClipIndex(clip_index) = clip_index;
                        let clip = &clips[clip_index as usize];
                        let radius = clip.top_left.outer_radius_x;

                        let tl_outer = Point2D::new(rect.origin.x, rect.origin.y);
                        let tl_inner = tl_outer + Point2D::new(radius, radius);

                        let tr_outer = Point2D::new(rect.origin.x + rect.size.width, rect.origin.y);
                        let tr_inner = tr_outer + Point2D::new(-radius, radius);

                        let bl_outer = Point2D::new(rect.origin.x, rect.origin.y + rect.size.height);
                        let bl_inner = bl_outer + Point2D::new(radius, -radius);

                        let br_outer = Point2D::new(rect.origin.x + rect.size.width,
                                                    rect.origin.y + rect.size.height);
                        let br_inner = br_outer - Point2D::new(radius, radius);

                        // Clipped corners
                        let r0 = rect_from_points_f(tl_outer.x, tl_outer.y, tl_inner.x, tl_inner.y);
                        let xf0 = TransformedRect::new(&r0, transform, device_pixel_ratio);
                        renderables.push_clipped_rect(&xf0, tile_info, layer_index, prim_index, &details.color);

                        let r1 = rect_from_points_f(tr_inner.x, tr_outer.y, tr_outer.x, tr_inner.y);
                        let xf1 = TransformedRect::new(&r1, transform, device_pixel_ratio);
                        renderables.push_clipped_rect(&xf1, tile_info, layer_index, prim_index, &details.color);

                        let r2 = rect_from_points_f(bl_outer.x, bl_inner.y, bl_inner.x, bl_outer.y);
                        let xf2 = TransformedRect::new(&r2, transform, device_pixel_ratio);
                        renderables.push_clipped_rect(&xf2, tile_info, layer_index, prim_index, &details.color);

                        let r3 = rect_from_points_f(br_inner.x, br_inner.y, br_outer.x, br_outer.y);
                        let xf3 = TransformedRect::new(&r3, transform, device_pixel_ratio);
                        renderables.push_clipped_rect(&xf3, tile_info, layer_index, prim_index, &details.color);

                        // Non-clipped regions
                        let r4 = rect_from_points_f(tl_outer.x, tl_inner.y, bl_inner.x, bl_inner.y);
                        let xf4 = TransformedRect::new(&r4, transform, device_pixel_ratio);
                        renderables.push_rect(&xf4, tile_info, layer_index, prim_index, &details.color);

                        let r5 = rect_from_points_f(tr_inner.x, tr_inner.y, br_outer.x, br_inner.y);
                        let xf5 = TransformedRect::new(&r5, transform, device_pixel_ratio);
                        renderables.push_rect(&xf5, tile_info, layer_index, prim_index, &details.color);

                        let r6 = rect_from_points_f(tl_inner.x, tl_inner.y, tr_inner.x, tr_outer.y);
                        let xf6 = TransformedRect::new(&r6, transform, device_pixel_ratio);
                        renderables.push_rect(&xf6, tile_info, layer_index, prim_index, &details.color);

                        let r7 = rect_from_points_f(bl_inner.x, bl_inner.y, br_inner.x, br_outer.y);
                        let xf7 = TransformedRect::new(&r7, transform, device_pixel_ratio);
                        renderables.push_rect(&xf7, tile_info, layer_index, prim_index, &details.color);

                        // Center
                        let r8 = rect_from_points_f(tl_inner.x, tl_inner.y, br_inner.x, br_inner.y);
                        let xf8 = TransformedRect::new(&r8, transform, device_pixel_ratio);
                        renderables.push_rect(&xf8, tile_info, layer_index, prim_index, &details.color);
                    } else {
                        renderables.push_clipped_rect(&xf_rect,
                                                      tile_info,
                                                      layer_index,
                                                      prim_index,
                                                      &details.color);
                    }
                } else {
                    renderables.push_rect(&xf_rect,
                                          tile_info,
                                          layer_index,
                                          prim_index,
                                          &details.color);
                }
            }
            PrimitiveDetails::Border(ref details) => {
                if xf_rect.should_split() {
                    // Some simple fast paths where the border can be
                    // drawn as rectangles.
                    // TODO(gw): Expand this to pick up more fast paths!
                    let same_color = details.left_color == details.top_color &&
                                     details.top_color == details.right_color &&
                                     details.right_color == details.bottom_color;
                    let zero_radius = details.radius.top_left.width == 0.0 &&
                                      details.radius.top_left.height == 0.0 &&
                                      details.radius.top_right.width == 0.0 &&
                                      details.radius.top_right.height == 0.0 &&
                                      details.radius.bottom_right.width == 0.0 &&
                                      details.radius.bottom_right.height == 0.0 &&
                                      details.radius.bottom_left.width == 0.0 &&
                                      details.radius.bottom_left.height == 0.0;

                    if same_color && zero_radius {
                        let top_rect = rect_from_points_f(details.tl_outer.x,
                                                          details.tl_outer.y,
                                                          details.tr_outer.x,
                                                          details.tr_inner.y);

                        let top_xf_rect = TransformedRect::new(&top_rect,
                                                               transform,
                                                               device_pixel_ratio);

                        renderables.push_rect(&top_xf_rect,
                                              tile_info,
                                              layer_index,
                                              prim_index,
                                              &details.top_color);

                        let left_rect = rect_from_points_f(details.tl_outer.x,
                                                           details.tl_inner.y,
                                                           details.bl_inner.x,
                                                           details.bl_inner.y);

                        let left_xf_rect = TransformedRect::new(&left_rect,
                                                                transform,
                                                                device_pixel_ratio);

                        renderables.push_rect(&left_xf_rect,
                                              tile_info,
                                              layer_index,
                                              prim_index,
                                              &details.left_color);

                        let right_rect = rect_from_points_f(details.tr_inner.x,
                                                            details.tr_inner.y,
                                                            details.br_outer.x,
                                                            details.br_inner.y);

                        let right_xf_rect = TransformedRect::new(&right_rect,
                                                                 transform,
                                                                 device_pixel_ratio);

                        renderables.push_rect(&right_xf_rect,
                                              tile_info,
                                              layer_index,
                                              prim_index,
                                              &details.right_color);

                        let bottom_rect = rect_from_points_f(details.bl_outer.x,
                                                             details.bl_inner.y,
                                                             details.br_outer.x,
                                                             details.br_outer.y);

                        let bottom_xf_rect = TransformedRect::new(&bottom_rect,
                                                                  transform,
                                                                  device_pixel_ratio);

                        renderables.push_rect(&bottom_xf_rect,
                                              tile_info,
                                              layer_index,
                                              prim_index,
                                              &details.bottom_color);

                    } else {
                        let c0 = rect_from_points_f(details.tl_outer.x,
                                                    details.tl_outer.y,
                                                    details.tl_inner.x,
                                                    details.tl_inner.y);
                        renderables.push_border(&TransformedRect::new(&c0, transform, device_pixel_ratio),
                                                tile_info,
                                                layer_index,
                                                prim_index);

                        let c1 = rect_from_points_f(details.tr_inner.x,
                                                    details.tr_outer.y,
                                                    details.tr_outer.x,
                                                    details.tr_inner.y);
                        renderables.push_border(&TransformedRect::new(&c1, transform, device_pixel_ratio),
                                                tile_info,
                                                layer_index,
                                                prim_index);

                        let c2 = rect_from_points_f(details.bl_outer.x,
                                                    details.bl_inner.y,
                                                    details.bl_inner.x,
                                                    details.bl_outer.y);
                        renderables.push_border(&TransformedRect::new(&c2, transform, device_pixel_ratio),
                                                tile_info,
                                                layer_index,
                                                prim_index);

                        let c3 = rect_from_points_f(details.br_inner.x,
                                                    details.br_inner.y,
                                                    details.br_outer.x,
                                                    details.br_outer.y);
                        renderables.push_border(&TransformedRect::new(&c3, transform, device_pixel_ratio),
                                                tile_info,
                                                layer_index,
                                                prim_index);

                        let top_rect = rect_from_points_f(details.tl_inner.x,
                                                          details.tl_outer.y,
                                                          details.tr_inner.x,
                                                          details.tr_outer.y + details.top_width);

                        let top_xf_rect = TransformedRect::new(&top_rect,
                                                               transform,
                                                               device_pixel_ratio);

                        renderables.push_rect(&top_xf_rect,
                                              tile_info,
                                              layer_index,
                                              prim_index,
                                              &details.top_color);

                        let left_rect = rect_from_points_f(details.tl_outer.x,
                                                           details.tl_inner.y,
                                                           details.tl_outer.x + details.left_width,
                                                           details.bl_inner.y);

                        let left_xf_rect = TransformedRect::new(&left_rect,
                                                                transform,
                                                                device_pixel_ratio);

                        renderables.push_rect(&left_xf_rect,
                                              tile_info,
                                              layer_index,
                                              prim_index,
                                              &details.left_color);

                        let right_rect = rect_from_points_f(details.tr_outer.x - details.right_width,
                                                            details.tr_inner.y,
                                                            details.br_outer.x,
                                                            details.br_inner.y);

                        let right_xf_rect = TransformedRect::new(&right_rect,
                                                                 transform,
                                                                 device_pixel_ratio);

                        renderables.push_rect(&right_xf_rect,
                                              tile_info,
                                              layer_index,
                                              prim_index,
                                              &details.right_color);

                        let bottom_rect = rect_from_points_f(details.bl_inner.x,
                                                             details.bl_outer.y - details.top_width,
                                                             details.br_inner.x,
                                                             details.br_outer.y);

                        let bottom_xf_rect = TransformedRect::new(&bottom_rect,
                                                                  transform,
                                                                  device_pixel_ratio);

                        renderables.push_rect(&bottom_xf_rect,
                                              tile_info,
                                              layer_index,
                                              prim_index,
                                              &details.bottom_color);

                    }
                } else {
                    renderables.push_border(&xf_rect,
                                            tile_info,
                                            layer_index,
                                            prim_index)
                }
            }
            PrimitiveDetails::Text(ref details) => {
                let mut color_texture_id = TextureId(0);

                let src_glyphs = auxiliary_lists.glyph_instances(&details.glyph_range);
                let mut glyph_key = GlyphKey::new(details.font_key,
                                                  details.size,
                                                  details.blur_radius,
                                                  src_glyphs[0].index);
                let blur_offset = details.blur_radius.to_f32_px() * (BLUR_INFLATION_FACTOR as f32) / 2.0;
                let mut glyphs = Vec::new();

                for glyph in src_glyphs {
                    glyph_key.index = glyph.index;
                    let image_info = resource_cache.get_glyph(&glyph_key, frame_id);
                    if let Some(image_info) = image_info {
                        // TODO(gw): Need a general solution to handle multiple texture pages per tile in WR2!
                        assert!(color_texture_id == TextureId(0) ||
                                color_texture_id == image_info.texture_id);
                        color_texture_id = image_info.texture_id;

                        let x = glyph.x + image_info.user_data.x0 as f32 / device_pixel_ratio - blur_offset;
                        let y = glyph.y - image_info.user_data.y0 as f32 / device_pixel_ratio - blur_offset;

                        let width = image_info.requested_rect.size.width as f32 / device_pixel_ratio;
                        let height = image_info.requested_rect.size.height as f32 / device_pixel_ratio;

                        let uv_rect = image_info.uv_rect();

                        glyphs.push(PackedGlyph {
                            padding: 0,
                            layer: 0,
                            offset: Point2D::zero(),
                            color: details.color,
                            p0: Point2D::new(x, y),
                            p1: Point2D::new(x + width, y + height),
                            st0: uv_rect.top_left,
                            st1: uv_rect.bottom_right,
                        });
                    }
                }

                debug_assert!(!glyphs.is_empty());       // todo(gw): support prims that end up getting cleared
                let mut current_rect = Rect::zero();
                let mut current_glyphs = Vec::new();

                for glyph in glyphs {
                    let glyph_rect = Rect::new(glyph.p0, Size2D::new(glyph.p1.x - glyph.p0.x, glyph.p1.y - glyph.p0.y));
                    let new_rect = current_rect.union(&glyph_rect);
                    if current_glyphs.len() > 0 && (new_rect.size.width > 256.0 || new_rect.size.height > 256.0) {
                        // flush range
                        let xf_rect = TransformedRect::new(&current_rect,
                                                           transform,
                                                           device_pixel_ratio);

                        renderables.push_text(&xf_rect,
                                              tile_info,
                                              layer_index,
                                              prim_index,
                                              current_rect,
                                              color_texture_id,
                                              mem::replace(&mut current_glyphs, Vec::new()));

                        current_glyphs.push(glyph);
                        current_rect = glyph_rect;
                    } else {
                        current_glyphs.push(glyph);
                        current_rect = new_rect;
                    }
                }

                debug_assert!(current_glyphs.len() > 0);
                let xf_rect = TransformedRect::new(&current_rect,
                                                   transform,
                                                   device_pixel_ratio);

                renderables.push_text(&xf_rect,
                                      tile_info,
                                      layer_index,
                                      prim_index,
                                      current_rect,
                                      color_texture_id,
                                      mem::replace(&mut current_glyphs, Vec::new()));
            }
            PrimitiveDetails::Image(ref details) => {
                let image_info = resource_cache.get_image(details.image_key,
                                                          details.image_rendering,
                                                          frame_id);
                let uv_rect = image_info.uv_rect();
                let location = PrimitiveCacheEntry {
                    rect: Rect::new(image_info.pixel_rect.top_left,
                                    Size2D::new(image_info.pixel_rect.bottom_right.x - image_info.pixel_rect.top_left.x,
                                                image_info.pixel_rect.bottom_right.y - image_info.pixel_rect.top_left.y)),
                    st0: uv_rect.top_left,
                    st1: uv_rect.bottom_right,
                };

                renderables.push_image(&xf_rect,
                                       tile_info,
                                       layer_index,
                                       prim_index,
                                       location,
                                       image_info);
            }
            PrimitiveDetails::Gradient(ref details) => {
                let stops = auxiliary_lists.gradient_stops(&details.stops_range);
                for i in 0..(stops.len() - 1) {
                    let (prev_stop, next_stop) = (&stops[i], &stops[i + 1]);
                    let piece_origin;
                    let piece_size;
                    match details.dir {
                        AxisDirection::Horizontal => {
                            let prev_x = util::lerp(self.rect.origin.x, self.rect.max_x(), prev_stop.offset);
                            let next_x = util::lerp(self.rect.origin.x, self.rect.max_x(), next_stop.offset);
                            piece_origin = Point2D::new(prev_x, self.rect.origin.y);
                            piece_size = Size2D::new(next_x - prev_x, self.rect.size.height);
                        }
                        AxisDirection::Vertical => {
                            let prev_y = util::lerp(self.rect.origin.y, self.rect.max_y(), prev_stop.offset);
                            let next_y = util::lerp(self.rect.origin.y, self.rect.max_y(), next_stop.offset);
                            piece_origin = Point2D::new(self.rect.origin.x, prev_y);
                            piece_size = Size2D::new(self.rect.size.width, next_y - prev_y);
                        }
                    }

                    let piece_rect = Rect::new(piece_origin, piece_size);

                    let xf_rect = TransformedRect::new(&piece_rect,
                                                       transform,
                                                       device_pixel_ratio);

                    // Split gradient up into small parts so that it doesn't
                    // blow out the prim cache for large gradients!
                    let x_tile_size = DevicePixel(256);
                    let y_tile_size = DevicePixel(256);
                    let x_tile_count = (xf_rect.bounding_rect.size.width + x_tile_size - DevicePixel(1)).0 / x_tile_size.0;
                    let y_tile_count = (xf_rect.bounding_rect.size.height + y_tile_size - DevicePixel(1)).0 / y_tile_size.0;

                    for y in 0..y_tile_count {
                        let y0 = DevicePixel(xf_rect.bounding_rect.origin.y.0 + y * y_tile_size.0);
                        let y1 = cmp::min(y0 + y_tile_size, xf_rect.bounding_rect.origin.y + xf_rect.bounding_rect.size.height);

                        for x in 0..x_tile_count {
                            let x0 = DevicePixel(xf_rect.bounding_rect.origin.x.0 + x * x_tile_size.0);
                            let x1 = cmp::min(x0 + x_tile_size, xf_rect.bounding_rect.origin.x + xf_rect.bounding_rect.size.width);
                                let rect = rect_from_points(x0, y0, x1, y1);

                                renderables.push_gradient(&rect,
                                                          tile_info,
                                                          layer_index,
                                                          prim_index,
                                                          &prev_stop.color,
                                                          &next_stop.color,
                                                          details.dir,
                                                          &piece_rect);
                        }
                    }
                }
            }
            PrimitiveDetails::BoxShadow(ref details) => {
                // Split box shadows up into small parts so that it doesn't
                // blow out the prim cache for large gradients!
                let x_tile_size = DevicePixel(128);
                let y_tile_size = DevicePixel(128);
                let x_tile_count = (xf_rect.bounding_rect.size.width + x_tile_size - DevicePixel(1)).0 / x_tile_size.0;
                let y_tile_count = (xf_rect.bounding_rect.size.height + y_tile_size - DevicePixel(1)).0 / y_tile_size.0;

                for y in 0..y_tile_count {
                    let y0 = DevicePixel(xf_rect.bounding_rect.origin.y.0 + y * y_tile_size.0);
                    let y1 = cmp::min(y0 + y_tile_size, xf_rect.bounding_rect.origin.y + xf_rect.bounding_rect.size.height);

                    for x in 0..x_tile_count {
                        let x0 = DevicePixel(xf_rect.bounding_rect.origin.x.0 + x * x_tile_size.0);
                        let x1 = cmp::min(x0 + x_tile_size, xf_rect.bounding_rect.origin.x + xf_rect.bounding_rect.size.width);
                            let rect = rect_from_points(x0, y0, x1, y1);

                            // TODO(gw): Cull segments that are inside/outside the src rect
                            //           based on box shadow clip mode for an easy perf win!

                            renderables.push_box_shadow(&rect,
                                                        tile_info,
                                                        layer_index,
                                                        prim_index,
                                                        &details.color,
                                                        details.blur_radius,
                                                        Point2D::new(details.border_radius,
                                                                     details.border_radius),
                                                        details.clip_mode,
                                                        &details.bs_rect,
                                                        &details.src_rect);
                    }
                }
            }
        }
    }

}

#[derive(Debug)]
enum RenderableDetails {
    Rectangle(RenderableRectangleDetails),
    Text(RenderableTextDetails),
    Gradient(RenderableGradientDetails),
    Border,
    BoxShadow(RenderableBoxShadowDetails),
    Image,
}

#[derive(Debug)]
struct RenderableRectangleDetails {
    color: ColorF,
}

#[derive(Debug)]
struct RenderableBoxShadowDetails {
    color: ColorF,
    border_radii: Point2D<f32>,
    blur_radius: f32,
    clip_mode: BoxShadowClipMode,
    bs_rect: Rect<f32>,
    src_rect: Rect<f32>,
}

#[derive(Debug)]
struct RenderableGradientDetails {
    color0: ColorF,
    color1: ColorF,
    piece_rect: Rect<f32>,
    dir: AxisDirection,
}

#[derive(Debug)]
struct RenderableTextDetails {
    rect: Rect<f32>,
    color_texture_id: TextureId,
    glyphs: Vec<PackedGlyph>,
}

#[derive(Debug)]
struct Renderable {
    bounding_rect: Rect<DevicePixel>,
    tile_range: TileRange,
    layer_index: RenderLayerIndex,
    primitive_index: PrimitiveIndex,
    st_offset: DevicePixel,
    texture_id: TextureId,
    is_opaque: bool,
    location: Option<PrimitiveCacheEntry>,
    cache_size: Size2D<DevicePixel>,
    clip_parts: ClipPart,
    details: RenderableDetails,
}

#[derive(Debug)]
struct RenderableList {
    renderables: Vec<Renderable>,
}

impl RenderableList {
    fn new() -> RenderableList {
        RenderableList {
            renderables: Vec::new(),
        }
    }

    fn push_clipped_rect(&mut self,
                         xf_rect: &TransformedRect,
                         tile_info: &TilingInfo,
                         layer_index: RenderLayerIndex,
                         prim_index: PrimitiveIndex,
                         color: &ColorF) {
        if xf_rect.bounding_rect.size.width.0 <= 0 ||
           xf_rect.bounding_rect.size.height.0 <= 0 {
            return;
        }

        self.renderables.push(Renderable {
            bounding_rect: xf_rect.bounding_rect,
            tile_range: tile_info.get_tile_range(&xf_rect.bounding_rect),
            layer_index: layer_index,
            primitive_index: prim_index,
            is_opaque: false,
            texture_id: TextureId(0),
            location: None,
            cache_size: xf_rect.bounding_rect.size,
            st_offset: DevicePixel(0),
            clip_parts: ClipPart::All,
            details: RenderableDetails::Rectangle(RenderableRectangleDetails {
                color: *color,
            }),
        });
    }

    fn push_rect(&mut self,
                 xf_rect: &TransformedRect,
                 tile_info: &TilingInfo,
                 layer_index: RenderLayerIndex,
                 prim_index: PrimitiveIndex,
                 color: &ColorF) {
        if xf_rect.bounding_rect.size.width.0 <= 0 ||
           xf_rect.bounding_rect.size.height.0 <= 0 {
            return;
        }

        self.renderables.push(Renderable {
            bounding_rect: xf_rect.bounding_rect,
            tile_range: tile_info.get_tile_range(&xf_rect.bounding_rect),
            layer_index: layer_index,
            primitive_index: prim_index,
            is_opaque: color.a == 1.0,
            texture_id: TextureId(0),
            location: None,
            cache_size: Size2D::new(DevicePixel(8), DevicePixel(8)),
            st_offset: DevicePixel(2),
            clip_parts: ClipPart::None,
            details: RenderableDetails::Rectangle(RenderableRectangleDetails {
                color: *color,
            }),
        });
    }

    fn push_border(&mut self,
                   xf_rect: &TransformedRect,
                   tile_info: &TilingInfo,
                   layer_index: RenderLayerIndex,
                   prim_index: PrimitiveIndex) {
        if xf_rect.bounding_rect.size.width.0 <= 0 ||
           xf_rect.bounding_rect.size.height.0 <= 0 {
            return;
        }

        self.renderables.push(Renderable {
            bounding_rect: xf_rect.bounding_rect,
            tile_range: tile_info.get_tile_range(&xf_rect.bounding_rect),
            layer_index: layer_index,
            primitive_index: prim_index,
            texture_id: TextureId(0),
            is_opaque: false,
            location: None,
            cache_size: xf_rect.bounding_rect.size,
            st_offset: DevicePixel(0),
            clip_parts: ClipPart::None,
            details: RenderableDetails::Border,
        })
    }

    fn push_text(&mut self,
                xf_rect: &TransformedRect,
                tile_info: &TilingInfo,
                layer_index: RenderLayerIndex,
                prim_index: PrimitiveIndex,
                rect: Rect<f32>,
                color_texture_id: TextureId,
                glyphs: Vec<PackedGlyph>) {
        if xf_rect.bounding_rect.size.width.0 <= 0 ||
           xf_rect.bounding_rect.size.height.0 <= 0 {
            return;
        }

        self.renderables.push(Renderable {
            bounding_rect: xf_rect.bounding_rect,
            tile_range: tile_info.get_tile_range(&xf_rect.bounding_rect),
            layer_index: layer_index,
            primitive_index: prim_index,
            texture_id: TextureId(0),
            is_opaque: false,
            location: None,
            cache_size: xf_rect.bounding_rect.size,
            st_offset: DevicePixel(0),
            clip_parts: ClipPart::None,
            details: RenderableDetails::Text(RenderableTextDetails {
                rect: rect,
                color_texture_id: color_texture_id,
                glyphs: glyphs,
            })
        });
    }

    fn push_image(&mut self,
                xf_rect: &TransformedRect,
                tile_info: &TilingInfo,
                layer_index: RenderLayerIndex,
                prim_index: PrimitiveIndex,
                location: PrimitiveCacheEntry,
                image_info: &TextureCacheItem) {
        self.renderables.push(Renderable {
            bounding_rect: xf_rect.bounding_rect,
            tile_range: tile_info.get_tile_range(&xf_rect.bounding_rect),
            layer_index: layer_index,
            primitive_index: prim_index,
            texture_id: image_info.texture_id,
            is_opaque: image_info.is_opaque,
            location: Some(location),
            cache_size: Size2D::zero(),
            st_offset: DevicePixel(0),
            clip_parts: ClipPart::None,
            details: RenderableDetails::Image,
        });
    }

    fn push_box_shadow(&mut self,
                       rect: &Rect<DevicePixel>,
                       tile_info: &TilingInfo,
                       layer_index: RenderLayerIndex,
                       prim_index: PrimitiveIndex,
                       color: &ColorF,
                       blur_radius: f32,
                       border_radii: Point2D<f32>,
                       clip_mode: BoxShadowClipMode,
                       bs_rect: &Rect<f32>,
                       src_rect: &Rect<f32>) {
        self.renderables.push(Renderable {
            bounding_rect: *rect,
            tile_range: tile_info.get_tile_range(rect),
            layer_index: layer_index,
            primitive_index: prim_index,
            texture_id: TextureId(0),
            is_opaque: false,
            location: None,
            cache_size: rect.size,
            st_offset: DevicePixel(0),
            clip_parts: ClipPart::None,
            details: RenderableDetails::BoxShadow(RenderableBoxShadowDetails {
                color: *color,
                blur_radius: blur_radius,
                border_radii: border_radii,
                clip_mode: clip_mode,
                bs_rect: *bs_rect,
                src_rect: *src_rect,
            }),
        });
    }

    fn push_gradient(&mut self,
                     rect: &Rect<DevicePixel>,
                     tile_info: &TilingInfo,
                     layer_index: RenderLayerIndex,
                     prim_index: PrimitiveIndex,
                     color0: &ColorF,
                     color1: &ColorF,
                     dir: AxisDirection,
                     piece_rect: &Rect<f32>) {
        self.renderables.push(Renderable {
            bounding_rect: *rect,
            tile_range: tile_info.get_tile_range(rect),
            layer_index: layer_index,
            primitive_index: prim_index,
            texture_id: TextureId(0),
            is_opaque: color0.a == 1.0 && color1.a == 1.0,
            location: None,
            cache_size: rect.size,
            st_offset: DevicePixel(0),
            clip_parts: ClipPart::None,
            details: RenderableDetails::Gradient(RenderableGradientDetails {
                color0: *color0,
                color1: *color1,
                dir: dir,
                piece_rect: *piece_rect,
            })
        });
    }
}

struct RenderLayer {
    pipeline_id: PipelineId,
    local_transform: Matrix4D<f32>,
    local_rect: Rect<f32>,
    local_offset: Point2D<f32>,
    primitives: Vec<Primitive>,
    scroll_layer_id: ScrollLayerId,
    opacity: f32,
    transform: Matrix4D<f32>,
    xf_rect: Option<TransformedRect>,
}

impl RenderLayer {
    fn build_resource_list(&self,
                           auxiliary_lists: &AuxiliaryLists,
                           resource_list: &mut ResourceList) {
        for prim in &self.primitives {
            match prim.details {
                PrimitiveDetails::Rectangle(..) => {}
                PrimitiveDetails::Gradient(..) => {}
                PrimitiveDetails::Border(..) => {}
                PrimitiveDetails::BoxShadow(..) => {}
                PrimitiveDetails::Image(ref details) => {
                    resource_list.add_image(details.image_key,
                                            details.image_rendering);
                }
                PrimitiveDetails::Text(ref details) => {
                    let glyphs = auxiliary_lists.glyph_instances(&details.glyph_range);
                    for glyph in glyphs {
                        let glyph = Glyph::new(details.size, details.blur_radius, glyph.index);
                        resource_list.add_glyph(details.font_key, glyph);
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct TransformedRect {
    local_rect: Rect<f32>,
    vertices: [Point4D<f32>; 4],
    bounding_rect: Rect<DevicePixel>,
}

impl TransformedRect {
    fn new(rect: &Rect<f32>,
           transform: &Matrix4D<f32>,
           device_pixel_ratio: f32) -> TransformedRect {
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

        let mut screen_min: Point2D<f32> = Point2D::new( 10000000.0,  10000000.0);
        let mut screen_max: Point2D<f32> = Point2D::new(-10000000.0, -10000000.0);

        for vertex in &vertices {
            let inv_w = 1.0 / vertex.w;
            let vx = vertex.x * inv_w;
            let vy = vertex.y * inv_w;
            screen_min.x = screen_min.x.min(vx);
            screen_min.y = screen_min.y.min(vy);
            screen_max.x = screen_max.x.max(vx);
            screen_max.y = screen_max.y.max(vy);
        }

        let screen_min_dp = Point2D::new(DevicePixel((screen_min.x * device_pixel_ratio).floor() as i32),
                                         DevicePixel((screen_min.y * device_pixel_ratio).floor() as i32));
        let screen_max_dp = Point2D::new(DevicePixel((screen_max.x * device_pixel_ratio).ceil() as i32),
                                         DevicePixel((screen_max.y * device_pixel_ratio).ceil() as i32));

        let screen_rect_dp = Rect::new(screen_min_dp, Size2D::new(screen_max_dp.x - screen_min_dp.x,
                                                                  screen_max_dp.y - screen_min_dp.y));

        TransformedRect {
            local_rect: *rect,
            vertices: vertices,
            bounding_rect: screen_rect_dp,
        }
    }

    fn should_split(&self) -> bool {
        // TODO(gw): Take into account rotation / perspective here!
        let width = self.bounding_rect.size.width;
        let height = self.bounding_rect.size.height;
        let area = width.0 * height.0;

        width.0 > 512 || height.0 > 512 || area > 256*256
    }
}

#[derive(Debug, Eq, PartialEq, Hash, Clone)]
pub struct BatchKey {
    pub shader: CompositeShader,
    pub samplers: [TextureId; MAX_PRIMS_PER_COMPOSITE],
}

impl BatchKey {
    fn new(shader: CompositeShader,
           samplers: [TextureId; MAX_PRIMS_PER_COMPOSITE]) -> BatchKey {
        BatchKey {
            shader: shader,
            samplers: samplers,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum ClipKind {
    ClipOut,
    //ClipIn,
}

#[derive(Clone, Debug)]
pub struct CompositePrimitive {
    bounding_rect: Rect<DevicePixel>,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
}

impl CompositePrimitive {
    fn new(renderable: &Renderable) -> CompositePrimitive {
        let location = renderable.location.as_ref().unwrap();

        CompositePrimitive {
            bounding_rect: renderable.bounding_rect,
            st0: location.st0,
            st1: location.st1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PackedGlyph {
    pub offset: Point2D<DevicePixel>,
    pub layer: u32,
    pub padding: u32,
    pub color: ColorF,
    pub p0: Point2D<f32>,
    pub p1: Point2D<f32>,
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
}

#[derive(Debug)]
pub struct CachedBoxShadow {
    pub rect: Rect<DevicePixel>,
    pub offset: Point2D<DevicePixel>,
    pub layer: u32,
    pub inverted: u32,
    pub color: ColorF,
    pub border_radii: Point2D<f32>,
    pub blur_radius: f32,
    pub padding: u32,
    pub bs_rect: Rect<f32>,
    pub src_rect: Rect<f32>,
}

#[derive(Debug)]
pub struct CachedBorder {
    pub offset: Point2D<DevicePixel>,
    pub layer: u32,
    pub padding: u32,
    pub rect: Rect<DevicePixel>,
    pub local_rect: Rect<f32>,
    pub left_color: ColorF,
    pub top_color: ColorF,
    pub right_color: ColorF,
    pub bottom_color: ColorF,
    pub widths: [f32; 4],
    pub clip: Clip,
}

#[derive(Clone)]
pub struct CachedPrimitive {
    pub rect: Rect<DevicePixel>,
    pub offset: Point2D<DevicePixel>,
    pub layer: u32,
    pub padding: u32,
    pub color: ColorF,
    pub clip: Clip,
}

#[derive(Clone)]
pub struct CachedRectangle {
    pub rect: Rect<DevicePixel>,
    pub color: ColorF,
}

#[derive(Clone)]
pub struct CachedGradient {
    pub rect: Rect<DevicePixel>,
    pub local_rect: Rect<f32>,
    pub offset: Point2D<DevicePixel>,
    pub layer: u32,
    pub dir: AxisDirection,
    pub color0: ColorF,
    pub color1: ColorF,
}

#[derive(Clone, Debug)]
pub struct PrimitiveCacheEntry {
    rect: Rect<DevicePixel>,
    st0: Point2D<f32>,
    st1: Point2D<f32>,
}

impl PrimitiveCacheEntry {
    fn new(origin: &Point2D<DevicePixel>,
           size: &Size2D<DevicePixel>,
           st_offset: DevicePixel,
           target_size: &Size2D<f32>) -> PrimitiveCacheEntry {

        let x = origin.x.0 + st_offset.0;
        let y = origin.y.0 + st_offset.0;
        let w = size.width.0 - 2 * st_offset.0;
        let h = size.height.0 - 2 * st_offset.0;

        let st0 = Point2D::new(x as f32 / target_size.width,
                               y as f32 / target_size.height);

        let st1 = Point2D::new((x + w) as f32 / target_size.width,
                               (y + h) as f32 / target_size.height);

        PrimitiveCacheEntry {
            rect: Rect::new(*origin, *size),
            st0: st0,
            st1: st1,
        }
    }
}

pub struct PrimitiveCache {
    target_size: Size2D<f32>,
    page_allocator: TexturePage,
    entries: HashMap<RenderableId, PrimitiveCacheEntry, BuildHasherDefault<FnvHasher>>,
}

impl PrimitiveCache {
    fn new(size: DevicePixel) -> PrimitiveCache {
        let target_size = Size2D::new(size.0 as f32, size.0 as f32);
        PrimitiveCache {
            page_allocator: TexturePage::new(TextureId(0), size.0 as u32),
            entries: HashMap::with_hasher(Default::default()),
            target_size: target_size,
        }
    }

    fn get(&self, id: RenderableId) -> Option<PrimitiveCacheEntry> {
        self.entries.get(&id).map(|e| e.clone())
    }

    pub fn clear(&mut self) {
        self.page_allocator.clear();
        self.entries.clear();
    }

    fn alloc_prim(&mut self,
                  id: RenderableId,
                  st_offset: DevicePixel,
                  size: &Size2D<DevicePixel>) -> Option<PrimitiveCacheEntry> {
        let alloc_size = Size2D::new(size.width.0 as u32, size.height.0 as u32);
        let origin = self.page_allocator
                         .allocate(&alloc_size, TextureFilter::Linear);

        origin.map(|origin| {
            let origin = Point2D::new(DevicePixel(origin.x as i32), DevicePixel(origin.y as i32));

            let entry = PrimitiveCacheEntry::new(&origin,
                                                 size,
                                                 st_offset,
                                                 &self.target_size);

            self.entries.insert(id, entry.clone());

            entry
        })
    }

    fn allocate(&mut self,
                instances: &mut Vec<RenderableId>,
                renderables: &mut Vec<Renderable>) -> Option<Vec<RenderableId>> {
        let mut jobs = Vec::new();

        for ri in instances {
            let ri = *ri;
            let entry = self.get(ri);

            match entry {
                Some(..) => {}
                None => {
                    let RenderableId(i) = ri;
                    let renderable = &mut renderables[i as usize];
                    if renderable.cache_size.width.0 > 0 &&
                       renderable.cache_size.height.0 > 0 {
                        let location = self.alloc_prim(ri, renderable.st_offset, &renderable.cache_size);
                        match location {
                            Some(location) => {
                                renderable.location = Some(location);
                                jobs.push(ri);
                            }
                            None => {
                                return None;
                            }
                        }
                    } else {
                        debug_assert!(renderable.location.is_some());
                    }
                }
            }
        }

        Some(jobs)
    }
}

#[derive(Debug, Clone)]
pub struct ClipCorner {
    rect: Rect<f32>,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
}

#[derive(Debug, Clone)]
pub struct Clip {
    rect: Rect<f32>,
    clip_kind: ClipKind,
    layer: u32,
    padding: [u32; 2],
    top_left: ClipCorner,
    top_right: ClipCorner,
    bottom_left: ClipCorner,
    bottom_right: ClipCorner,
}

impl Clip {
    pub fn from_clip_region(clip: &ComplexClipRegion, clip_kind: ClipKind) -> Clip {
        Clip {
            rect: clip.rect,
            clip_kind: clip_kind,
            layer: INVALID_LAYER_INDEX,
            padding: [0, 0],
            top_left: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x, clip.rect.origin.y),
                                Size2D::new(clip.radii.top_left.width, clip.radii.top_left.height)),
                outer_radius_x: clip.radii.top_left.width,
                outer_radius_y: clip.radii.top_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            top_right: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.top_right.width,
                                             clip.rect.origin.y),
                                Size2D::new(clip.radii.top_right.width, clip.radii.top_right.height)),
                outer_radius_x: clip.radii.top_right.width,
                outer_radius_y: clip.radii.top_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_left: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x,
                                             clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_left.height),
                                Size2D::new(clip.radii.bottom_left.width, clip.radii.bottom_left.height)),
                outer_radius_x: clip.radii.bottom_left.width,
                outer_radius_y: clip.radii.bottom_left.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
            bottom_right: ClipCorner {
                rect: Rect::new(Point2D::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.bottom_right.width,
                                             clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_right.height),
                                Size2D::new(clip.radii.bottom_right.width, clip.radii.bottom_right.height)),
                outer_radius_x: clip.radii.bottom_right.width,
                outer_radius_y: clip.radii.bottom_right.height,
                inner_radius_x: 0.0,
                inner_radius_y: 0.0,
            },
        }
    }

    pub fn from_border_radius(rect: &Rect<f32>,
                              outer_radius: &BorderRadius,
                              inner_radius: &BorderRadius,
                              clip_kind: ClipKind) -> Clip {
        Clip {
            rect: *rect,
            clip_kind: clip_kind,
            layer: INVALID_LAYER_INDEX,
            padding: [0, 0],
            top_left: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x, rect.origin.y),
                                Size2D::new(outer_radius.top_left.width, outer_radius.top_left.height)),
                outer_radius_x: outer_radius.top_left.width,
                outer_radius_y: outer_radius.top_left.height,
                inner_radius_x: inner_radius.top_left.width,
                inner_radius_y: inner_radius.top_left.height,
            },
            top_right: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x + rect.size.width - outer_radius.top_right.width,
                                             rect.origin.y),
                                Size2D::new(outer_radius.top_right.width, outer_radius.top_right.height)),
                outer_radius_x: outer_radius.top_right.width,
                outer_radius_y: outer_radius.top_right.height,
                inner_radius_x: inner_radius.top_right.width,
                inner_radius_y: inner_radius.top_right.height,
            },
            bottom_left: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x,
                                             rect.origin.y + rect.size.height - outer_radius.bottom_left.height),
                                Size2D::new(outer_radius.bottom_left.width, outer_radius.bottom_left.height)),
                outer_radius_x: outer_radius.bottom_left.width,
                outer_radius_y: outer_radius.bottom_left.height,
                inner_radius_x: inner_radius.bottom_left.width,
                inner_radius_y: inner_radius.bottom_left.height,
            },
            bottom_right: ClipCorner {
                rect: Rect::new(Point2D::new(rect.origin.x + rect.size.width - outer_radius.bottom_right.width,
                                             rect.origin.y + rect.size.height - outer_radius.bottom_right.height),
                                Size2D::new(outer_radius.bottom_right.width, outer_radius.bottom_right.height)),
                outer_radius_x: outer_radius.bottom_right.width,
                outer_radius_y: outer_radius.bottom_right.height,
                inner_radius_x: inner_radius.bottom_right.width,
                inner_radius_y: inner_radius.bottom_right.height,
            },
        }
    }
}

pub struct FrameBuilderConfig {

}

impl FrameBuilderConfig {
    pub fn new() -> FrameBuilderConfig {
        FrameBuilderConfig {

        }
    }
}

#[derive(Debug, Clone)]
pub struct DebugRect {
    pub label: String,
    pub color: ColorF,
    pub rect: Rect<DevicePixel>,
}

#[derive(Debug, Clone)]
pub struct ErrorTile {
    pub rect: Rect<DevicePixel>,
}

#[derive(Debug, Clone)]
pub struct ClearTile {
    pub rect: Rect<DevicePixel>,
}

pub struct TileBatch {
    pub primitives: Vec<CompositePrimitive>,
    pub batches: HashMap<BatchKey, Vec<CompositeTile>, BuildHasherDefault<FnvHasher>>,
}

impl TileBatch {
    fn new() -> TileBatch {
        TileBatch {
            batches: HashMap::with_hasher(Default::default()),
            primitives: Vec::new(),
        }
    }
}

pub struct Pass {
    // prim cache jobs
    pub complex: Vec<CachedPrimitive>,
    pub rectangles: Vec<CachedRectangle>,
    pub borders: Vec<CachedBorder>,
    pub glyphs: Vec<PackedGlyph>,
    pub box_shadows: Vec<CachedBoxShadow>,
    pub gradients: Vec<CachedGradient>,
    pub text_texture_id: TextureId,

    // composite jobs
    // todo: batch these across tiles where possible (if prim ubo size is small enough!)
    pub tile_batches: Vec<TileBatch>,
}

impl Pass {
    fn new() -> Pass {
        Pass {
            complex: Vec::new(),
            rectangles: Vec::new(),
            borders: Vec::new(),
            glyphs: Vec::new(),
            box_shadows: Vec::new(),
            tile_batches: Vec::new(),
            gradients: Vec::new(),
            text_texture_id: TextureId(0),
        }
    }

    fn render_to_cache(&mut self,
                       id: RenderableId,
                       renderables: &Vec<Renderable>,
                       layers: &Vec<RenderLayer>,
                       clips: &Vec<Clip>) {
        // TODO(gw): Remove some of this indirection somehow?
        let RenderableId(rid) = id;
        let renderable = &renderables[rid as usize];

        let RenderLayerIndex(lid) = renderable.layer_index;
        let layer = &layers[lid];

        let PrimitiveIndex(pid) = renderable.primitive_index;
        let prim = &layer.primitives[pid];

        let location = renderable.location.as_ref().unwrap();

        match renderable.details {
            RenderableDetails::Rectangle(ref details) => {
                match renderable.clip_parts {
                    ClipPart::All => {
                        let ClipIndex(clip_index) = prim.clip_index.unwrap();
                        self.complex.push(CachedPrimitive {
                            color: details.color,
                            padding: 0,
                            layer: lid as u32,
                            offset: renderable.bounding_rect.origin,
                            rect: location.rect,
                            clip: clips[clip_index as usize].clone(),
                        });
                    }
                    ClipPart::None => {
                        self.rectangles.push(CachedRectangle {
                            color: details.color,
                            rect: location.rect,
                        });
                    }
                }
            }
            RenderableDetails::Text(ref details) => {
                debug_assert!(self.text_texture_id == TextureId(0) ||
                              self.text_texture_id == details.color_texture_id);
                self.text_texture_id = details.color_texture_id;
                for glyph in &details.glyphs {
                    let mut glyph = glyph.clone();
                    glyph.layer = lid as u32;
                    glyph.offset = location.rect.origin - renderable.bounding_rect.origin;
                    self.glyphs.push(glyph);
                }
            }
            RenderableDetails::Border => {
                match prim.details {
                    PrimitiveDetails::Border(ref details) => {
                        let inner_radius = BorderRadius {
                            top_left: Size2D::new(details.radius.top_left.width - details.left_width,
                                                  details.radius.top_left.width - details.left_width),
                            top_right: Size2D::new(details.radius.top_right.width - details.right_width,
                                                   details.radius.top_right.width - details.right_width),
                            bottom_left: Size2D::new(details.radius.bottom_left.width - details.left_width,
                                                     details.radius.bottom_left.width - details.left_width),
                            bottom_right: Size2D::new(details.radius.bottom_right.width - details.right_width,
                                                      details.radius.bottom_right.width - details.right_width),
                        };

                        let clip = Clip::from_border_radius(&prim.rect,
                                                            &details.radius,
                                                            &inner_radius,
                                                            ClipKind::ClipOut);

                        self.borders.push(CachedBorder {
                            rect: location.rect,
                            offset: renderable.bounding_rect.origin,
                            local_rect: prim.rect,
                            left_color: details.left_color,
                            bottom_color: details.bottom_color,
                            right_color: details.right_color,
                            top_color: details.top_color,
                            widths: [ details.left_width,
                                      details.top_width,
                                      details.right_width,
                                      details.bottom_width
                                    ],
                            layer: lid as u32,
                            padding: 0,
                            clip: clip,
                        });
                    }
                    _ => unreachable!(),
                }
            }
            RenderableDetails::Gradient(ref details) => {
                self.gradients.push(CachedGradient {
                    layer: lid as u32,
                    offset: renderable.bounding_rect.origin,
                    rect: location.rect,
                    local_rect: details.piece_rect,
                    color0: details.color0,
                    color1: details.color1,
                    dir: details.dir,
                });
            }
            RenderableDetails::BoxShadow(ref details) => {
                let inverted = match details.clip_mode {
                    BoxShadowClipMode::Inset => 1,
                    BoxShadowClipMode::Outset => 0,
                    BoxShadowClipMode::None => 0,
                };

                self.box_shadows.push(CachedBoxShadow {
                    color: details.color,
                    padding: 0,
                    layer: lid as u32,
                    offset: renderable.bounding_rect.origin,
                    rect: location.rect,
                    border_radii: details.border_radii,
                    blur_radius: details.blur_radius,
                    inverted: inverted,
                    bs_rect: details.bs_rect,
                    src_rect: details.src_rect,
                });
            }
            RenderableDetails::Image => {
                // TODO: Handle 3d transformed textures!
            }
        }
    }
}

pub struct Tile {
    bsp_tree: BspTree<RenderableInstanceId>,
    pub batches: HashMap<BatchKey, Vec<CompositeTile>, BuildHasherDefault<FnvHasher>>,

    instances: Vec<RenderableId>,
    debug_rects: Vec<DebugRect>,
    error_tiles: Vec<ErrorTile>,
    clear_tiles: Vec<ClearTile>,
}

impl Tile {
    fn new(x0: DevicePixel,
           y0: DevicePixel,
           x1: DevicePixel,
           y1: DevicePixel) -> Tile {
        let tile_rect = rect_from_points(x0, y0, x1, y1);
        Tile {
            batches: HashMap::with_hasher(Default::default()),
            bsp_tree: BspTree::new(tile_rect),
            debug_rects: Vec::new(),
            error_tiles: Vec::new(),
            clear_tiles: Vec::new(),
            instances: Vec::new(),
        }
    }

    pub fn add(&mut self,
               bounding_rect: &Rect<DevicePixel>,
               id: RenderableId) {
        let instance_id = RenderableInstanceId(self.instances.len() as u32);
        self.instances.push(id);
        self.bsp_tree.add(bounding_rect, instance_id);
    }
}

pub struct Frame {
    pub viewport_size: Size2D<u32>,
    pub layer_ubo: Ubo<PackedLayer>, // TODO(gw): Handle batching this, in crazy case where layer count > ubo size!
    pub debug_rects: Vec<DebugRect>,
    pub clear_tiles: Vec<ClearTile>,
    pub error_tiles: Vec<ErrorTile>,
    pub passes: Vec<Pass>,
    pub prim_cache_size: Size2D<f32>,
}

pub struct FrameBuilder {
    screen_rect: Rect<i32>,
    layers: Vec<RenderLayer>,
    layer_stack: Vec<RenderLayerIndex>,
    clips: Vec<Clip>,
    clip_stack: Vec<ClipIndex>,
    device_pixel_ratio: f32,
    debug: bool,
}

impl FrameBuilder {
    pub fn new(viewport_size: Size2D<f32>,
               device_pixel_ratio: f32,
               debug: bool) -> FrameBuilder {
        let viewport_size = Size2D::new(viewport_size.width as i32, viewport_size.height as i32);
        FrameBuilder {
            screen_rect: Rect::new(Point2D::zero(), viewport_size),
            layers: Vec::new(),
            layer_stack: Vec::new(),
            device_pixel_ratio: device_pixel_ratio,
            debug: debug,
            clips: Vec::new(),
            clip_stack: Vec::new(),
        }
    }

    pub fn set_clip(&mut self, clip: Clip) {
        let clip_index = ClipIndex(self.clips.len() as u32);
        // TODO(gw): Handle intersecting clips!
        self.clips.push(clip);
        self.clip_stack.push(clip_index);
    }

    pub fn clear_clip(&mut self) {
        self.clip_stack.pop().unwrap();
    }

    fn add_primitive(&mut self,
                     rect: &Rect<f32>,
                     details: PrimitiveDetails) {
        let current_layer = *self.layer_stack.last().unwrap();
        let RenderLayerIndex(layer_index) = current_layer;
        let layer = &mut self.layers[layer_index as usize];

        let prim = Primitive {
            rect: *rect,
            xf_rect: None,
            clip_index: self.clip_stack.last().map(|i| *i),
            details: details,
        };
        layer.primitives.push(prim);
    }

    pub fn push_layer(&mut self,
                      rect: Rect<f32>,
                      _clip_rect: Rect<f32>,
                      transform: Matrix4D<f32>,
                      opacity: f32,
                      pipeline_id: PipelineId,
                      scroll_layer_id: ScrollLayerId,
                      offset: Point2D<f32>) {
        let layer = RenderLayer {
            primitives: Vec::new(),
            local_rect: rect,
            local_transform: transform,
            local_offset: offset,
            scroll_layer_id: scroll_layer_id,
            opacity: opacity,
            pipeline_id: pipeline_id,
            xf_rect: None,
            transform: Matrix4D::identity(),
        };

        self.layer_stack.push(RenderLayerIndex(self.layers.len()));
        self.layers.push(layer);
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
    }

    pub fn add_solid_rectangle(&mut self,
                               rect: &Rect<f32>,
                               color: &ColorF) {
        if color.a == 0.0 {
            return;
        }

        let prim = RectanglePrimitive {
            color: *color,
        };

        self.add_primitive(rect, PrimitiveDetails::Rectangle(prim));
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

        let prim = BorderPrimitive {
            tl_outer: tl_outer,
            tl_inner: tl_inner,
            tr_outer: tr_outer,
            tr_inner: tr_inner,
            bl_outer: bl_outer,
            bl_inner: bl_inner,
            br_outer: br_outer,
            br_inner: br_inner,
            radius: radius.clone(),
            left_width: left.width,
            top_width: top.width,
            bottom_width: bottom.width,
            right_width: right.width,
            left_color: left_color,
            top_color: top_color,
            bottom_color: bottom_color,
            right_color: right_color,
        };

        self.add_primitive(&rect, PrimitiveDetails::Border(prim));
    }

    pub fn add_gradient(&mut self,
                        rect: Rect<f32>,
                        start_point: Point2D<f32>,
                        end_point: Point2D<f32>,
                        stops: ItemRange) {
        // Fast paths for axis-aligned gradients:
        if start_point.x == end_point.x {
            let rect = Rect::new(Point2D::new(rect.origin.x, start_point.y),
                                 Size2D::new(rect.size.width, end_point.y - start_point.y));
            let prim = GradientPrimitive {
                stops_range: stops,
                dir: AxisDirection::Vertical,
            };
            self.add_primitive(&rect, PrimitiveDetails::Gradient(prim));
        } else if start_point.y == end_point.y {
            let rect = Rect::new(Point2D::new(start_point.x, rect.origin.y),
                                 Size2D::new(end_point.x - start_point.x, rect.size.height));
            let prim = GradientPrimitive {
                stops_range: stops,
                dir: AxisDirection::Horizontal,
            };
            self.add_primitive(&rect, PrimitiveDetails::Gradient(prim));
        } else {
            //println!("TODO: Angle gradients! {:?} {:?} {:?}", start_point, end_point, stops);
        }
    }

    pub fn add_text(&mut self,
                    rect: Rect<f32>,
                    font_key: FontKey,
                    size: Au,
                    blur_radius: Au,
                    color: &ColorF,
                    glyph_range: ItemRange) {
        if color.a == 0.0 {
            return
        }

        let prim = TextPrimitive {
            color: *color,
            font_key: font_key,
            size: size,
            blur_radius: blur_radius,
            glyph_range: glyph_range,
        };

        self.add_primitive(&rect, PrimitiveDetails::Text(prim));
    }

    pub fn add_box_shadow(&mut self,
                          box_bounds: &Rect<f32>,
                          box_offset: &Point2D<f32>,
                          color: &ColorF,
                          blur_radius: f32,
                          spread_radius: f32,
                          border_radius: f32,
                          clip_mode: BoxShadowClipMode) {
        if color.a == 0.0 {
            return
        }

        let bs_rect = compute_box_shadow_rect(box_bounds,
                                              box_offset,
                                              spread_radius);

        let metrics = BoxShadowMetrics::new(&bs_rect,
                                            border_radius,
                                            blur_radius);

        let prim_rect = Rect::new(metrics.tl_outer,
                                  Size2D::new(metrics.br_outer.x - metrics.tl_outer.x,
                                              metrics.br_outer.y - metrics.tl_outer.y));

        let prim = BoxShadowPrimitive {
            metrics: metrics,
            src_rect: *box_bounds,
            bs_rect: bs_rect,
            color: *color,
            blur_radius: blur_radius,
            spread_radius: spread_radius,
            border_radius: border_radius,
            clip_mode: clip_mode,
        };

        self.add_primitive(&prim_rect, PrimitiveDetails::BoxShadow(prim));
    }

    pub fn add_image(&mut self,
                     rect: Rect<f32>,
                     _stretch_size: &Size2D<f32>,
                     image_key: ImageKey,
                     image_rendering: ImageRendering) {
        let prim = ImagePrimitive {
            image_key: image_key,
            image_rendering: image_rendering,
        };

        self.add_primitive(&rect, PrimitiveDetails::Image(prim));
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 frame_id: FrameId,
                 pipeline_auxiliary_lists: &HashMap<PipelineId, AuxiliaryLists, BuildHasherDefault<FnvHasher>>,
                 layer_map: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>) -> Frame {
        //let _pf = util::ProfileScope::new("--build--");
        println!("-- build --");
        // Remove layers that are transparent.
        self.layers.retain(|layer| {
            layer.opacity > 0.0
        });

        // Build layer screen rects.
        // TODO(gw): This can be done earlier once update_layer_transforms() is fixed.
        for layer in &mut self.layers {
            let scroll_layer = &layer_map[&layer.scroll_layer_id];
            let offset_transform = Matrix4D::identity().translate(layer.local_offset.x,
                                                                  layer.local_offset.y,
                                                                  0.0);
            let transform = scroll_layer.world_transform
                                        .mul(&layer.local_transform)
                                        .mul(&offset_transform);
            layer.transform = transform;
            layer.xf_rect = Some(TransformedRect::new(&layer.local_rect,
                                                      &transform,
                                                      self.device_pixel_ratio));
        }

        // Remove layers that are not visible.
        let screen_rect = Rect::new(Point2D::zero(),
                                    Size2D::new(DevicePixel::new(self.screen_rect.size.width as f32, self.device_pixel_ratio),
                                                DevicePixel::new(self.screen_rect.size.height as f32, self.device_pixel_ratio)));
        self.layers.retain(|layer| {
            layer.xf_rect
                 .as_ref()
                 .unwrap()
                 .bounding_rect
                 .intersects(&screen_rect)
        });

        // Cull primitives that aren't visible.
        // TODO(gw): This can easily be sped up for large pages (it's brute force for now).
        let mut layer_ubo = Ubo::new();
        for layer in &mut self.layers {
            layer_ubo.push(&PackedLayer {
                transform: layer.transform,
                inv_transform: layer.transform.invert(),
                blend_info: [0.0, 0.0, 0.0, 0.0],
                screen_vertices: layer.xf_rect.as_ref().unwrap().vertices,
            });

            for prim in &mut layer.primitives {
                prim.xf_rect = Some(TransformedRect::new(&prim.rect,
                                                         &layer.transform,
                                                         self.device_pixel_ratio));
            }

            layer.primitives.retain(|prim| {
                prim.xf_rect
                    .as_ref()
                    .unwrap()
                    .bounding_rect
                    .intersects(&screen_rect)
            });
        }

        // Build resource lists
        let mut resource_list = ResourceList::new(self.device_pixel_ratio);
        for layer in &self.layers {
            let auxiliary_lists = pipeline_auxiliary_lists.get(&layer.pipeline_id)
                                                          .expect("No auxiliary lists?!");
            layer.build_resource_list(auxiliary_lists, &mut resource_list);
        }

        // Rasterize glyphs as required
        resource_cache.add_resource_list(&resource_list, frame_id);
        resource_cache.raster_pending_glyphs(frame_id);

        let x_tile_size = DevicePixel(256);
        let y_tile_size = DevicePixel(256);
        let x_tile_count = (screen_rect.size.width + x_tile_size - DevicePixel(1)).0 / x_tile_size.0;
        let y_tile_count = (screen_rect.size.height + y_tile_size - DevicePixel(1)).0 / y_tile_size.0;

        let tile_info = TilingInfo {
           x_tile_count: x_tile_count,
           y_tile_count: y_tile_count,
           x_tile_size: x_tile_size,
           y_tile_size: y_tile_size,
        };

        // Compile visible primitives to renderables
        let mut rlist = RenderableList::new();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            let layer_index = RenderLayerIndex(layer_index);
            let auxiliary_lists = pipeline_auxiliary_lists.get(&layer.pipeline_id)
                                                          .expect("No auxiliary lists?!");

            for (prim_index, prim) in layer.primitives.iter().enumerate() {
                let prim_index = PrimitiveIndex(prim_index);

                prim.build_renderables(layer_index,
                                       prim_index,
                                       &tile_info,
                                       self.device_pixel_ratio,
                                       &mut rlist,
                                       auxiliary_lists,
                                       resource_cache,
                                       frame_id,
                                       &layer.transform,
                                       &self.clips);
            }
        }

        // Build screen space tiles, which are individual BSP trees.
        let mut tiles = Vec::new();
        for y in 0..y_tile_count {
            let y0 = DevicePixel(y * y_tile_size.0);
            let y1 = y0 + y_tile_size;

            for x in 0..x_tile_count {
                let x0 = DevicePixel(x * x_tile_size.0);
                let x1 = x0 + x_tile_size;

                tiles.push(Tile::new(x0, y0, x1, y1));
            }
        }

        // Add each of the visible primitives to each BSP tile that it touches.
        for (renderable_index, renderable) in rlist.renderables.iter().enumerate() {
            let renderable_id = RenderableId(renderable_index as u32);

            for y in renderable.tile_range.y0..renderable.tile_range.y1 {
                for x in renderable.tile_range.x0..renderable.tile_range.x1 {
                    let tile = &mut tiles[(y*x_tile_count+x) as usize];
                    tile.add(&renderable.bounding_rect, renderable_id);
                }
            }
        }

        for tile in &mut tiles {
            println!("process tile");
            let debug_rects = &mut tile.debug_rects;
            let error_tiles = &mut tile.error_tiles;
            let clear_tiles = &mut tile.clear_tiles;
            let bsp_tree = &mut tile.bsp_tree;
            let batches = &mut tile.batches;
            let instances = &mut tile.instances;
            //let mut layer_buffer: Vec<LayerInfo> = Vec::new();
            let mut samplers = [
                TextureId(0),
                TextureId(0),
                TextureId(0),
                TextureId(0),
            ];

            bsp_tree.split(self.device_pixel_ratio, &mut |rect, cover_indices, partial_indices| {
                if cover_indices.is_empty() && partial_indices.is_empty() {
                    clear_tiles.push(ClearTile {
                        rect: *rect,
                    });
                    return;
                }

                if !partial_indices.is_empty() {
                    error_tiles.push(ErrorTile {
                        rect: *rect,
                    });
                    return;
                }

                cover_indices.sort_by(|a, b| {
                    b.cmp(&a)
                });

                if self.debug {
                    let color = ColorF::new(1.0, 0.0, 0.0, 1.0);
                    let debug_rect = DebugRect {
                        label: String::new(),
                        color: color,
                        rect: *rect,
                    };
                    debug_rects.push(debug_rect);
                }

                println!("before {:?}", cover_indices);

                let mut layer_is_opaque = false;
                let mut tile_is_opaque = false;
                let mut current_layer_index = RenderLayerIndex(0xffffffff);

                cover_indices.retain(|ci| {
                    let need_prim;

                    if tile_is_opaque {
                        need_prim = false;
                    } else {
                        let RenderableInstanceId(ii) = *ci;

                        let RenderableId(ri) = instances[ii as usize];
                        let renderable = &rlist.renderables[ri as usize];

                        let RenderLayerIndex(lid) = renderable.layer_index;
                        let layer = &self.layers[lid];

                        if current_layer_index == renderable.layer_index {
                            need_prim = !layer_is_opaque;
                        } else {
                            need_prim = true;
                            layer_is_opaque = false;
                            current_layer_index = renderable.layer_index;
                        }

                        layer_is_opaque = layer_is_opaque || renderable.is_opaque;

                        if layer_is_opaque && layer.opacity == 1.0 {
                            tile_is_opaque = true;
                        }
                    }

                    need_prim
                });

                println!("after {:?}", cover_indices);

                if cover_indices.len() > MAX_PRIMS_PER_COMPOSITE {
                    error_tiles.push(ErrorTile {
                        rect: *rect,
                    });
                    return;
                }

                let mut composite_tile = CompositeTile::new(rect);
                let mut current_layer_index = RenderLayerIndex(0xffffffff);
                let mut next_prim_index = 0;

                for instance_index in cover_indices.iter().rev() {
                    let RenderableInstanceId(ii) = *instance_index;

                    let RenderableId(ri) = instances[ii as usize];
                    let renderable = &rlist.renderables[ri as usize];

                    let RenderLayerIndex(lid) = renderable.layer_index;
                    let layer = &self.layers[lid];

                    samplers[next_prim_index] = renderable.texture_id;

                    let need_new_layer = renderable.layer_index != current_layer_index;

                    let blend_mode = if need_new_layer {
                        BlendMode::LayerOpacity
                    } else {
                        BlendMode::Accumulate
                    };

                    composite_tile.set_primitive(next_prim_index,
                                                 *instance_index,
                                                 blend_mode,
                                                 layer.opacity);

                    next_prim_index += 1;
                    current_layer_index = renderable.layer_index;
                }

                println!("{:?}", composite_tile);

                let shader = Some(CompositeShader::Prim1);

                match shader {
                    Some(shader) => {
                        let batch_key = BatchKey::new(shader, samplers);
                        let batch = batches.entry(batch_key).or_insert_with(|| {
                            Vec::new()
                        });
                        batch.push(composite_tile);
                    }
                    None => {
                        error_tiles.push(ErrorTile {
                            rect: *rect,
                        });
                    }
                }
            });
        }

        // Step through each tile and allocate renderable instance jobs as required!
        // This is sequential due to the shared primitive cache, but should be quick
        // since the main splitting and job ubo creation can be passed to worker threads!
        let mut passes = vec![Pass::new()];
        let mut prim_cache = PrimitiveCache::new(DevicePixel(1024));
        for tile in &mut tiles {
            if tile.instances.is_empty() {
                continue;
            }

            let render_jobs = prim_cache.allocate(&mut tile.instances,
                                                  &mut rlist.renderables);

            let render_jobs = match render_jobs {
                Some(render_jobs) => render_jobs,
                None => {
                    prim_cache.clear();
                    passes.push(Pass::new());
                    prim_cache.allocate(&mut tile.instances,
                                        &mut rlist.renderables)
                              .expect("TODO Handle edge case failure to fit a single tile in cache!")
                }
            };

            let pass = passes.last_mut().unwrap();

            // create prim ubo from instance list
            let mut prim_ubo = Vec::new();

            for id in render_jobs {
                pass.render_to_cache(id,
                                     &rlist.renderables,
                                     &self.layers,
                                     &self.clips);
            }

            for instance in &tile.instances {
                let RenderableId(i) = *instance;
                prim_ubo.push(CompositePrimitive::new(&rlist.renderables[i as usize]));
            }

            // TODO(gw): Batch between tiles within the same
            //           prim cache pass!!

            let mut tile_batch = TileBatch::new();
            tile_batch.primitives = prim_ubo;
            tile_batch.batches = tile.batches.clone();      // TODO(gw): perf warning - remove clone here!

            pass.tile_batches.push(tile_batch);
        }

        let mut debug_rects = Vec::new();
        let mut clear_tiles = Vec::new();
        let mut error_tiles = Vec::new();

        for tile in tiles {
            debug_rects.extend_from_slice(&tile.debug_rects);
            clear_tiles.extend_from_slice(&tile.clear_tiles);
            error_tiles.extend_from_slice(&tile.error_tiles);
        }

        Frame {
            viewport_size: Size2D::new(self.screen_rect.size.width as u32,
                                       self.screen_rect.size.height as u32),
            layer_ubo: layer_ubo,
            debug_rects: debug_rects,
            error_tiles: error_tiles,
            clear_tiles: clear_tiles,
            passes: passes,
            prim_cache_size: prim_cache.target_size,
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