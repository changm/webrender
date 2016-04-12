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
use std::collections::{HashMap};
//use std::f32;
use std::mem;
use std::hash::{Hash, BuildHasherDefault};
use texture_cache::TexturePage;
use webrender_traits::{ColorF, FontKey, GlyphInstance, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderRadius};

const MAX_PRIMITIVES_PER_PASS: usize = 8;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct ShaderId(pub u32);

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TileLayout {
    Empty,
    L4P1,
    L4P2,
    L4P3,
    L4P4,
    L4P6,
}

#[derive(Debug)]
pub struct EmptyTile {
    pub rect: Rect<i32>,
}

#[derive(Debug)]
pub struct TileL4P1 {
    pub rect: Rect<i32>,
    pub layer_info: [LayerTemplateIndex; 4],
    pub prim_info: [PrimitiveKind; 4],
    pub prim: PackedPrimitive,
}

#[derive(Debug)]
pub struct TileL4P2 {
    pub rect: Rect<i32>,
    pub layer_info: [LayerTemplateIndex; 4],
    pub prim_info: [PrimitiveKind; 4],
    pub prims: [PackedPrimitive; 2],
}

#[derive(Debug)]
pub struct TileL4P3 {
    pub rect: Rect<i32>,
    pub layer_info: [LayerTemplateIndex; 4],
    pub prim_info: [PrimitiveKind; 4],
    pub prims: [PackedPrimitive; 3],
}

#[derive(Debug)]
pub struct TileL4P4 {
    pub rect: Rect<i32>,
    pub layer_info: [LayerTemplateIndex; 4],
    pub prim_info: [PrimitiveKind; 4],
    pub prims: [PackedPrimitive; 4],
}

#[derive(Debug)]
pub struct TileL4P6 {
    pub rect: Rect<i32>,
    pub layer_info: [LayerTemplateIndex; 4],
    pub prim_info: [PrimitiveKind; 8],
    pub prims: [PackedPrimitive; 6],
}

#[derive(Debug)]
pub enum TileData {
    Empty(Vec<EmptyTile>),
    L4P1(Vec<TileL4P1>),
    L4P2(Vec<TileL4P2>),
    L4P3(Vec<TileL4P3>),
    L4P4(Vec<TileL4P4>),
    L4P6(Vec<TileL4P6>),
}

trait PrimitiveHelpers {
    fn get(&self, key: PrimitiveKey) -> PackedPrimitive;
}

impl PrimitiveHelpers for Vec<Primitive> {
    #[inline]
    fn get(&self, key: PrimitiveKey) -> PackedPrimitive {
        let PrimitiveIndex(index) = key.index;
        self[index as usize].packed.clone()
    }
}

impl TileData {
    fn add_tile(&mut self,
                layout: TileLayout,
                tile: &Tile,
                primitives: &Vec<Primitive>) {
        match *self {
            TileData::Empty(ref mut tiles) => {
                assert!(layout == TileLayout::Empty);
                tiles.push(EmptyTile {
                    rect: tile.screen_rect,
                });
            }
            TileData::L4P1(ref mut tiles) => {
                assert!(layout == TileLayout::L4P1);
                let tile_layer = &tile.layers[0];
                let prim_key = tile_layer.primitives[0];
                let tile = TileL4P1 {
                    rect: tile.screen_rect,
                    layer_info: [
                                  tile_layer.layer_index,
                                  LayerTemplateIndex(0),
                                  LayerTemplateIndex(0),
                                  LayerTemplateIndex(0)
                                ],
                    prim_info: [
                                 prim_key.kind,
                                 PrimitiveKind::Invalid,
                                 PrimitiveKind::Invalid,
                                 PrimitiveKind::Invalid
                               ],
                    prim: primitives.get(prim_key),
                };
                tiles.push(tile);
            }
            TileData::L4P2(ref mut tiles) => {
                assert!(layout == TileLayout::L4P2);
                let tile_layer = &tile.layers[0];
                let pk0 = tile_layer.primitives[0];
                let pk1 = tile_layer.primitives[1];
                let tile = TileL4P2 {
                    rect: tile.screen_rect,
                    layer_info: [
                                  tile_layer.layer_index,
                                  LayerTemplateIndex(0),
                                  LayerTemplateIndex(0),
                                  LayerTemplateIndex(0)
                                ],
                    prim_info: [
                                 pk0.kind,
                                 pk1.kind,
                                 PrimitiveKind::Invalid,
                                 PrimitiveKind::Invalid
                               ],
                    prims: [
                            primitives.get(pk0),
                            primitives.get(pk1),
                           ],
                };
                tiles.push(tile);
            }
            TileData::L4P3(ref mut tiles) => {
                assert!(layout == TileLayout::L4P3);
                let tile_layer = &tile.layers[0];
                let pk0 = tile_layer.primitives[0];
                let pk1 = tile_layer.primitives[1];
                let pk2 = tile_layer.primitives[2];
                let tile = TileL4P3 {
                    rect: tile.screen_rect,
                    layer_info: [
                                  tile_layer.layer_index,
                                  LayerTemplateIndex(0),
                                  LayerTemplateIndex(0),
                                  LayerTemplateIndex(0)
                                ],
                    prim_info: [
                                 pk0.kind,
                                 pk1.kind,
                                 pk2.kind,
                                 PrimitiveKind::Invalid
                               ],
                    prims: [
                            primitives.get(pk0),
                            primitives.get(pk1),
                            primitives.get(pk2),
                           ],
                };
                tiles.push(tile);
            }
            TileData::L4P4(ref mut tiles) => {
                assert!(layout == TileLayout::L4P4);
                let tile_layer = &tile.layers[0];
                let mut packed_tile = TileL4P4 {
                    rect: tile.screen_rect,
                    layer_info: [
                                  tile_layer.layer_index,
                                  LayerTemplateIndex(0),
                                  LayerTemplateIndex(0),
                                  LayerTemplateIndex(0)
                                ],
                    prim_info: [
                                PrimitiveKind::Invalid,
                                PrimitiveKind::Invalid,
                                PrimitiveKind::Invalid,
                                PrimitiveKind::Invalid,
                               ],
                    prims: unsafe { mem::uninitialized() },
                };
                tile.visit_primitives(4, |key, index| {
                    packed_tile.prim_info[index] = key.kind;
                    packed_tile.prims[index] = primitives.get(key);
                });
                tiles.push(packed_tile);
            }
            TileData::L4P6(ref mut tiles) => {
                assert!(layout == TileLayout::L4P6);
                let tile_layer = &tile.layers[0];
                let mut packed_tile = TileL4P6 {
                    rect: tile.screen_rect,
                    layer_info: [
                                  tile_layer.layer_index,
                                  LayerTemplateIndex(0),
                                  LayerTemplateIndex(0),
                                  LayerTemplateIndex(0)
                                ],
                    prim_info: [
                                PrimitiveKind::Invalid,
                                PrimitiveKind::Invalid,
                                PrimitiveKind::Invalid,
                                PrimitiveKind::Invalid,
                                PrimitiveKind::Invalid,
                                PrimitiveKind::Invalid,
                                PrimitiveKind::Invalid,
                                PrimitiveKind::Invalid,
                               ],
                    prims: unsafe { mem::uninitialized() },
                };
                tile.visit_primitives(6, |key, index| {
                    packed_tile.prim_info[index] = key.kind;
                    packed_tile.prims[index] = primitives.get(key);
                });
                tiles.push(packed_tile);
            }
        }
    }
}

#[derive(Debug)]
pub struct TileBatch {
    pub data: TileData,
    pub shader_id: ShaderId,
}

impl TileBatch {
    fn new(shader_id: ShaderId, tile_layout: TileLayout) -> TileBatch {
        let data = match tile_layout {
            TileLayout::Empty => TileData::Empty(Vec::new()),
            TileLayout::L4P1 => TileData::L4P1(Vec::new()),
            TileLayout::L4P2 => TileData::L4P2(Vec::new()),
            TileLayout::L4P3 => TileData::L4P3(Vec::new()),
            TileLayout::L4P4 => TileData::L4P4(Vec::new()),
            TileLayout::L4P6 => TileData::L4P6(Vec::new()),
        };

        TileBatch {
            shader_id: shader_id,
            data: data,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum PrimitiveKind {
    Rectangle = 0,
    //SetClip,
    //ClearClip,
    Image,
    Gradient,
    Text,

    Invalid,
}

#[derive(Debug, Clone, Copy)]
pub enum TechniqueCountKind {
    Equal,
    LessEqual,
    DontCare,
}

#[derive(Debug, Clone)]
pub struct TechniqueDescriptor {
    pub primitive_count: usize,
    pub primitive_count_kind: TechniqueCountKind,
    pub layer_count: usize,
    pub layer_count_kind: TechniqueCountKind,
    pub primitive_kinds: [Option<PrimitiveKind>; MAX_PRIMITIVES_PER_PASS],
    pub tile_layout: TileLayout,
    pub shader_id: ShaderId,
}

impl TechniqueDescriptor {
    pub fn new(primitive_count: usize,
               primitive_count_kind: TechniqueCountKind,
               layer_count: usize,
               layer_count_kind: TechniqueCountKind,
               primitives: &[PrimitiveKind],
               tile_layout: TileLayout,
               shader_id: ShaderId) -> TechniqueDescriptor {
        let mut primitive_kinds = [None; MAX_PRIMITIVES_PER_PASS];

        for (index, prim) in primitives.iter().enumerate() {
            primitive_kinds[index] = Some(*prim);
        }

        TechniqueDescriptor {
            primitive_count: primitive_count,
            primitive_count_kind: primitive_count_kind,
            layer_count: layer_count,
            layer_count_kind: layer_count_kind,
            primitive_kinds: primitive_kinds,
            tile_layout: tile_layout,
            shader_id: shader_id,
        }
    }

    fn can_draw(&self, tile: &Tile) -> bool {
        let prim_count_ok = match self.primitive_count_kind {
            TechniqueCountKind::Equal => tile.prim_count as usize == self.primitive_count,
            TechniqueCountKind::LessEqual => tile.prim_count as usize <= self.primitive_count,
            TechniqueCountKind::DontCare => true,
        };
        if !prim_count_ok {
            return false;
        }

        let layer_count_ok = match self.layer_count_kind {
            TechniqueCountKind::Equal => tile.layers.len() == self.layer_count,
            TechniqueCountKind::LessEqual => tile.layers.len() <= self.layer_count,
            TechniqueCountKind::DontCare => true,
        };
        if !layer_count_ok {
            return false;
        }

        let mut test_index = 0;
        for layer in &tile.layers {
            for prim_key in &layer.primitives {
                if test_index == MAX_PRIMITIVES_PER_PASS {
                    return true;
                }
                if let Some(required_prim) = self.primitive_kinds[test_index] {
                    if prim_key.kind != required_prim {
                        return false;
                    }
                }
                test_index += 1;
            }
        }

        true
    }
}

pub struct TileFrame {
    pub viewport_size: Size2D<i32>,
    pub text_buffer: TextBuffer,
    pub color_texture_id: TextureId,
    pub mask_texture_id: TextureId,
    pub layer_ubo: Ubo<LayerTemplateIndex, PackedLayer>,
    pub batches: Vec<TileBatch>,
}

#[derive(Debug, Clone)]
pub struct PackedPrimitive {
    pub p0: Point2D<f32>,
    pub p1: Point2D<f32>,
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
    pub color: ColorF,
}

struct Primitive {
    xf_rect: TransformedRect,
    is_opaque: bool,
    packed: PackedPrimitive,
}

pub struct TileBuilder {
    screen_rect: Rect<i32>,
    layer_templates: Vec<LayerTemplate>,
    layer_instances: Vec<LayerInstance>,
    layer_stack: Vec<LayerTemplateIndex>,
    primitives: Vec<Primitive>,
    color_texture_id: TextureId,
    mask_texture_id: TextureId,
    scroll_offset: Point2D<f32>,
    techniques: Vec<TechniqueDescriptor>,
    text_buffer: TextBuffer,
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

#[derive(Debug, Clone)]
pub struct ClipCorner {
    position: Point2D<f32>,
    outer_radius_x: f32,
    outer_radius_y: f32,
    inner_radius_x: f32,
    inner_radius_y: f32,
    padding: Point2D<f32>,
}

/*
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
*/

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

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
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

#[derive(Clone, Debug)]
pub struct PackedLayer {
    inv_transform: Matrix4,
    screen_vertices: [Point4D<f32>; 4],
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

    fn maybe_insert_and_get_index(&mut self, key: KEY, data: &TYPE) -> usize {
        let map = &mut self.map;
        let items = &mut self.items;

        *map.entry(key).or_insert_with(|| {
            let index = items.len();
            items.push(data.clone());
            index
        })
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct LayerTemplateIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct LayerInstanceIndex(u32);

#[derive(Debug)]
struct TransformedRect {
    local_rect: Rect<f32>,
    vertices: [Point4D<f32>; 4],
    screen_rect: Rect<i32>,
}

struct LayerTemplate {
    transform: Matrix4,
    packed: PackedLayer,
}

struct LayerInstance {
    layer_index: LayerTemplateIndex,
    primitives: Vec<PrimitiveKey>,
}

#[derive(Debug)]
struct TileLayer {
    layer_index: LayerTemplateIndex,
    primitives: Vec<PrimitiveKey>,
}

impl TileLayer {
    fn new(layer_index: LayerTemplateIndex) -> TileLayer {
        TileLayer {
            layer_index: layer_index,
            primitives: Vec::new(),
        }
    }
}

pub struct Tile {
    pub screen_rect: Rect<i32>,
    layers: Vec<TileLayer>,
    children: Vec<Tile>,
    prim_count: u32,
}

impl Tile {
    fn new(screen_rect: Rect<i32>) -> Tile {
        Tile {
            screen_rect: screen_rect,
            layers: Vec::new(),
            children: Vec::new(),
            prim_count: 0,
        }
    }

    fn visit_primitives<F>(&self,
                           max_count: usize,
                           mut f: F) where F: FnMut(PrimitiveKey, usize) {
        let mut prim_count = 0;

        for layer in &self.layers {
            for prim_key in &layer.primitives {
                f(*prim_key, prim_count);
                prim_count += 1;
                if prim_count == max_count {
                    return;
                }
            }
        }
    }

    fn add_primitive(&mut self,
                     primitive_key: PrimitiveKey,
                     layer_index: LayerTemplateIndex,
                     primitives: &Vec<Primitive>) {
        let PrimitiveIndex(prim_index) = primitive_key.index;
        let primitive = &primitives[prim_index as usize];

        if primitive.xf_rect.screen_rect.intersects(&self.screen_rect) {

            // Check if this primitive supercedes all existing primitives in this
            // tile - this is a very important optimization to allow the CPU to create
            // small tiles that can use the simple tiling pass shader.
            // TODO(gw): This doesn't work with 3d transforms (it assumes axis aligned rects for now)!!
            if primitive.is_opaque &&
               primitive.xf_rect.screen_rect.contains(&self.screen_rect.origin) &&
               primitive.xf_rect.screen_rect.contains(&self.screen_rect.bottom_right()) {
                self.layers.clear();
                self.prim_count = 0;
            }

            let need_new_layer = self.layers.is_empty() ||
                                 self.layers.last().unwrap().layer_index != layer_index;

            if need_new_layer {
                self.layers.push(TileLayer::new(layer_index));
            }

            self.layers.last_mut().unwrap().primitives.push(primitive_key);
            self.prim_count += 1;
        }
    }

    fn split_if_needed(&mut self,
                       primitives: &Vec<Primitive>) {
        let try_split = self.screen_rect.size.width > 15 &&
                        self.screen_rect.size.height > 15 &&
                        self.prim_count > 2;

        if try_split {
            let new_width = self.screen_rect.size.width / 2;
            let left_rect = Rect::new(self.screen_rect.origin, Size2D::new(new_width, self.screen_rect.size.height));
            let right_rect = Rect::new(self.screen_rect.origin + Point2D::new(new_width, 0),
                                       Size2D::new(self.screen_rect.size.width - new_width, self.screen_rect.size.height));

            let new_height = self.screen_rect.size.height / 2;
            let top_rect = Rect::new(self.screen_rect.origin, Size2D::new(self.screen_rect.size.width, new_height));
            let bottom_rect = Rect::new(self.screen_rect.origin + Point2D::new(0, new_height),
                                        Size2D::new(self.screen_rect.size.width, self.screen_rect.size.height - new_height));

            let mut left = Tile::new(left_rect);
            let mut right = Tile::new(right_rect);

            let mut top = Tile::new(top_rect);
            let mut bottom = Tile::new(bottom_rect);

            for layer in &self.layers {
                for prim_key in &layer.primitives {
                    let prim_key = *prim_key;

                    left.add_primitive(prim_key,
                                       layer.layer_index,
                                       primitives);

                    right.add_primitive(prim_key,
                                        layer.layer_index,
                                        primitives);

                    top.add_primitive(prim_key,
                                      layer.layer_index,
                                      primitives);

                    bottom.add_primitive(prim_key,
                                         layer.layer_index,
                                         primitives);
                }
            }

            // TODO(gw): Investigate different heuristics for selecting which axis
            //           to do the split on!
            let h_min = cmp::min(left.prim_count, right.prim_count);
            let v_min = cmp::min(top.prim_count, bottom.prim_count);

            if (self.screen_rect.size.width > 63 || self.screen_rect.size.height > 63) || h_min < self.prim_count || v_min < self.prim_count {
                self.layers.clear();
                self.prim_count = 0;

                if h_min < v_min {
                    self.children.push(left);
                    self.children.push(right);
                } else if v_min < h_min {
                    self.children.push(top);
                    self.children.push(bottom);
                } else if self.screen_rect.size.width > self.screen_rect.size.height {
                    self.children.push(left);
                    self.children.push(right);
                } else {
                    self.children.push(top);
                    self.children.push(bottom);
                }

                for child in &mut self.children {
                    child.split_if_needed(primitives);
                }
            }
        }
    }
}

impl TileBuilder {
    pub fn new(viewport_size: Size2D<f32>,
               scroll_offset: Point2D<f32>,
               techniques: Vec<TechniqueDescriptor>) -> TileBuilder {
        TileBuilder {
            layer_templates: Vec::new(),
            layer_instances: Vec::new(),
            layer_stack: Vec::new(),
            primitives: Vec::new(),
            color_texture_id: TextureId(0),
            mask_texture_id: TextureId(0),
            scroll_offset: scroll_offset,
            screen_rect: Rect::new(Point2D::zero(),
                                   Size2D::new(viewport_size.width as i32, viewport_size.height as i32)),
            techniques: techniques,
            text_buffer: TextBuffer::new(TEXT_TARGET_SIZE),
        }
    }

    fn should_add_prim(&self, rect: &Rect<f32>) -> Option<TransformedRect> {
        let current_layer = *self.layer_stack.last().unwrap();
        let LayerTemplateIndex(current_layer_index) = current_layer;
        let layer = &self.layer_templates[current_layer_index as usize];

        let xf_rect = transform_rect(rect, &layer.transform);
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
            xf_rect: xf_rect,
            is_opaque: is_opaque,
            packed: packed,
        });
        self.layer_instances.last_mut().unwrap().primitives.push(prim_key);
    }

    pub fn add_gradient(&mut self,
                        _rect: Rect<f32>,
                        _start_point: &Point2D<f32>,
                        _end_point: &Point2D<f32>,
                        _stops: &ItemRange,
                        _auxiliary_lists: &AuxiliaryLists) {
/*
        let source_stops = auxiliary_lists.gradient_stops(stops);

        let mut stops = Vec::new();
        for stop in source_stops {
            stops.push(PackedGradientStop {
                offset: [stop.offset, 0.0, 0.0, 0.0],
                color: stop.color,
            });
        }
        let stops_key = self.primitives.add_gradient_stops(stops);

        let gradient_key = self.primitives.add_gradient(rect,
                                                        start_point,
                                                        end_point,
                                                        stops_key,
                                                        source_stops.len() as u32);
        self.add_primitive(gradient_key);
        */
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

            self.add_primitive(PrimitiveKind::Text,
                               xf_rect,
                               PackedPrimitive {
                                   p0: run.rect.origin,
                                   p1: run.rect.bottom_right(),
                                   st0: run.st0,
                                   st1: run.st1,
                                   color: *color,
                               },
                               false);
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
                               },
                               false);          // todo: handle this - big opt potential!
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

/*
    pub fn set_clip(&mut self, clip: Clip) {
        let xf_rect = self.transform_rect(&clip.rect);
        let clip_key = self.primitives.add_set_clip(xf_rect, clip);
        self.add_primitive(clip_key);
    }

    pub fn clear_clip(&mut self) {
        let clip_key = self.primitives.add_clear_clip();
        self.add_primitive(clip_key);
    }
*/

    pub fn push_layer(&mut self,
                      rect: Rect<f32>,
                      transform: Matrix4,
                      _: f32) {
        // TODO(gw): Not 3d transform correct!
        let transform = transform.translate(self.scroll_offset.x,
                                            self.scroll_offset.y,
                                            0.0);

        let layer_rect = transform_rect(&rect, &transform);

        let template = LayerTemplate {
            transform: transform,
            packed: PackedLayer {
                //blend_info: [opacity, 0.0, 0.0, 0.0],
                //p0: rect.origin,
                //p1: rect.bottom_right(),
                inv_transform: transform.invert(),
                screen_vertices: layer_rect.vertices,
            },
        };

        self.layer_stack.push(LayerTemplateIndex(self.layer_templates.len() as u32));
        self.layer_templates.push(template);
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
    }

    fn add_tile_to_batches(&self,
                          tile: &Tile,
                          batches: &mut Vec<TileBatch>) {
        for child in &tile.children {
            self.add_tile_to_batches(child, batches);
        }

        if tile.children.len() == 0 {
            for technique in &self.techniques {
                if technique.can_draw(tile) {
                    //println!("technique {:?} draws {:?}", technique, tile.layers);

                    let mut batch_index = batches.iter().position(|batch| {
                        batch.shader_id == technique.shader_id
                    });

                    if batch_index.is_none() {
                        batch_index = Some(batches.len());
                        batches.push(TileBatch::new(technique.shader_id, technique.tile_layout));
                    }

                    let batch = &mut batches[batch_index.unwrap()];
                    batch.data.add_tile(technique.tile_layout, tile, &self.primitives);

                    return;
                }
            }

            panic!("ERROR: Can't find technique for {:?}", tile.layers);
        }
    }

    // TODO(gw): This is grossly inefficient! But it should allow us to check the GPU
    //           perf on real world pages / demos, then we can worry about the CPU perf...
    pub fn build(self) -> TileFrame {
        let mut root_tile = Tile::new(self.screen_rect);
        let mut layer_ubo = Ubo::new();

        for layer_instance in &self.layer_instances {
            let LayerTemplateIndex(layer_index) = layer_instance.layer_index;
            let layer_template = &self.layer_templates[layer_index as usize].packed;
            layer_ubo.maybe_insert_and_get_index(layer_instance.layer_index, layer_template);

            for primitive_key in &layer_instance.primitives {
                root_tile.add_primitive(*primitive_key,
                                        layer_instance.layer_index,
                                        &self.primitives);
            }
        }

        root_tile.split_if_needed(&self.primitives);

        let mut batches = Vec::new();
        self.add_tile_to_batches(&root_tile, &mut batches);

        TileFrame {
            viewport_size: self.screen_rect.size,
            color_texture_id: self.color_texture_id,
            text_buffer: self.text_buffer,
            mask_texture_id: self.mask_texture_id,
            layer_ubo: layer_ubo,
            batches: batches,
        }
    }
}

fn transform_rect(rect: &Rect<f32>,
                  transform: &Matrix4) -> TransformedRect {
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
