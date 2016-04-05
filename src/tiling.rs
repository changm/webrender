/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch_builder::BorderSideHelpers;
use device::{TextureId, TextureFilter};
use euclid::{Point2D, Rect, Matrix4, Size2D, Point4D};
use fnv::FnvHasher;
use frame::FrameId;
use internal_types::{AxisDirection, GlyphKey, DevicePixel, PackedColor};
use renderer::{BLUR_INFLATION_FACTOR, UboBindLocation, TEXT_TARGET_SIZE};
use resource_cache::ResourceCache;
use std::cmp::{self, Ordering};
use std::collections::{HashMap, HashSet};
use std::f32;
use std::hash::{Hash, BuildHasherDefault};
use std::mem;
use std::ops;
use texture_cache::TexturePage;
use util;
use webrender_traits::{self, ColorF, FontKey, GlyphInstance, ImageKey, ImageRendering, ComplexClipRegion};
use webrender_traits::{BorderSide, BorderDisplayItem, BorderStyle, ItemRange, AuxiliaryLists, BorderRadius};

#[derive(Copy, Clone, Debug)]
struct vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl vec3 {
    fn from_point_4d(p: &Point4D<f32>) -> vec3 {
        vec3 {
            x: p.x / p.w,
            y: p.y / p.w,
            z: p.z / p.w,
        }
    }

    fn new(x: f32, y: f32, z: f32) -> vec3 {
        vec3 {
            x: x,
            y: y,
            z: z,
        }
    }
}

fn dot(a: vec3, b: vec3) -> f32 {
    a.x * b.x +
    a.y * b.y +
    a.z * b.z
}

fn cross(a: vec3, b: vec3) -> vec3 {
    vec3 {
        x: a.y * b.z - a.z * b.y,
        y: a.z * b.x - a.x * b.z,
        z: a.x * b.y - a.y * b.x,
    }
}

fn len(v: vec3) -> f32 {
    (v.x*v.x + v.y*v.y + v.z*v.z).sqrt()
}

fn normalize(v: vec3) -> vec3 {
    let inv_len = 1.0 / len(v);

    let r = vec3 {
        x: v.x * inv_len,
        y: v.y * inv_len,
        z: v.z * inv_len,
    };

    r
}

fn mix(x: vec3, y: vec3, a: f32) -> vec3 {
    vec3 {
        x: x.x * (1.0 - a) + y.x * a,
        y: x.y * (1.0 - a) + y.y * a,
        z: x.z * (1.0 - a) + y.z * a,
    }
}

fn sub(a: vec3, b: vec3) -> vec3 {
    vec3 {
        x: a.x - b.x,
        y: a.y - b.y,
        z: a.z - b.z,
    }
}

fn add(a: vec3, b: vec3) -> vec3 {
    vec3 {
        x: a.x + b.x,
        y: a.y + b.y,
        z: a.z + b.z,
    }
}

fn mul(a: vec3, f: f32) -> vec3 {
    vec3 {
        x: a.x * f,
        y: a.y * f,
        z: a.z * f,
    }
}

fn ray_plane(normal: vec3, point: vec3, ray_origin: vec3, ray_dir: vec3, t: &mut f32) -> bool {
    let denom = dot(normal, ray_dir);
    if denom > 1e-6 {
        let d = sub(point, ray_origin);
        *t = dot(d, normal) / denom;
        return *t >= 0.0;
    }

    return false;
}

fn untransform(r: vec3, n: vec3, a: vec3, inv_transform: &Matrix4) -> Point4D<f32> {
    let p = vec3::new(r.x, r.y, -100.0);
    let d = vec3::new(0.0, 0.0, 1.0);

    let mut t = 0.0;
    ray_plane(n, a, p, d, &mut t);
    let c = add(p, mul(d, t));// mix(p, d, t);

    let out = inv_transform.transform_point4d(&Point4D::new(c.x, c.y, c.z, 1.0));

    out
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct TechniqueParams {
    pub layer_count: usize,
    pub rect_count: usize,
    pub image_count: usize,
    pub text_count: usize,
}

#[derive(Debug, Clone)]
pub struct TextRun {
    pub glyphs: Vec<PackedGlyph>,
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
    pub rect: Rect<f32>,
}

pub struct TextBuffer {
    pub texture_size: f32,
    pub page_allocator: TexturePage,
    pub texts: HashMap<TextPrimitiveIndex, TextRun>,
}

impl TextBuffer {
    fn new(size: u32) -> TextBuffer {
        TextBuffer {
            texture_size: size as f32,
            page_allocator: TexturePage::new(TextureId(0), size),
            texts: HashMap::new(),
        }
    }

    fn push_text(&mut self,
                 text_index: TextPrimitiveIndex,
                 old_rect: &Rect<f32>,
                 glyphs: &GlyphPrimitive) {
        if self.texts.contains_key(&text_index) {
            return;
        }

        let mut rect = Rect::zero();
        for glyph in &glyphs.glyphs {
            rect = rect.union(&Rect::new(glyph.p0, Size2D::new(glyph.p1.x - glyph.p0.x, glyph.p1.y - glyph.p0.y)));
        }

        let size = Size2D::new(rect.size.width.ceil() as u32, rect.size.height.ceil() as u32);

        let origin = self.page_allocator
                         .allocate(&size, TextureFilter::Linear)
                         .expect("handle no texture space!");

        let mut text = TextRun {
            glyphs: Vec::new(),
            st0: Point2D::new(origin.x as f32 / self.texture_size,
                              origin.y as f32 / self.texture_size),
            st1: Point2D::new((origin.x + size.width) as f32 / self.texture_size,
                              (origin.y + size.height) as f32 / self.texture_size),
            rect: rect,
        };

        let d = Point2D::new(origin.x as f32, origin.y as f32) - rect.origin;
        for glyph in &glyphs.glyphs {
            text.glyphs.push(PackedGlyph {
                st0: glyph.st0,
                st1: glyph.st1,
                p0: glyph.p0 + d,
                p1: glyph.p1 + d,
            });
        }

        self.texts.insert(text_index, text);
    }

    fn get(&self, text_index: TextPrimitiveIndex) -> &TextRun {
        &self.texts[&text_index]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ColorBufferKey(u32);

#[derive(Debug, Clone)]
pub struct ColorBuffer {
    pub values: Vec<u8>,
}

impl ColorBuffer {
    fn new() -> ColorBuffer {
        ColorBuffer {
            values: Vec::new(),
        }
    }

    fn push_color(&mut self, color: &ColorF) -> ColorBufferKey {
        let key = self.values.len() as u32 / 4;
        let color = PackedColor::from_color(color);
        self.values.push(color.b);
        self.values.push(color.g);
        self.values.push(color.r);
        self.values.push(color.a);
        ColorBufferKey(key)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstantBufferKey(u32);

#[derive(Debug, Clone)]
pub struct ConstantBuffer {
    pub values: Vec<f32>,
}

impl ConstantBuffer {
    fn new() -> ConstantBuffer {
        ConstantBuffer {
            values: Vec::new(),
        }
    }

    fn push(&mut self, x: f32, y: f32, z: f32, w: f32) -> ConstantBufferKey {
        let key = self.values.len() as u32 / 4;
        self.values.push(x);
        self.values.push(y);
        self.values.push(z);
        self.values.push(w);
        ConstantBufferKey(key)
    }
}

pub struct UniformBuffer {
    max_ubo_size: usize,

    pub layer_ubos: Vec<Ubo<LayerTemplateIndex, PackedLayer>>,
    pub rect_ubos: Vec<Ubo<RectanglePrimitiveIndex, PackedRectangle>>,
    pub clip_ubos: Vec<Ubo<ClipPrimitiveIndex, Clip>>,
    pub image_ubos: Vec<Ubo<ImagePrimitiveIndex, PackedImage>>,
    pub gradient_ubos: Vec<Ubo<GradientPrimitiveIndex, PackedGradient>>,
    pub gradient_stop_ubos: Vec<ArrayUbo<GradientStopPrimitiveIndex, PackedGradientStop>>,
    pub text_ubos: Vec<Ubo<TextPrimitiveIndex, PackedText>>,
    //pub glyph_ubos: Vec<ArrayUbo<GlyphPrimitiveIndex, PackedGlyph>>,
    pub cmd_ubo: Vec<PackedCommand>,

    layer_ubo: Ubo<LayerTemplateIndex, PackedLayer>,
    rect_ubo: Ubo<RectanglePrimitiveIndex, PackedRectangle>,
    clip_ubo: Ubo<ClipPrimitiveIndex, Clip>,
    image_ubo: Ubo<ImagePrimitiveIndex, PackedImage>,
    gradient_ubo: Ubo<GradientPrimitiveIndex, PackedGradient>,
    gradient_stop_ubo: ArrayUbo<GradientStopPrimitiveIndex, PackedGradientStop>,
    text_ubo: Ubo<TextPrimitiveIndex, PackedText>,
    //glyph_ubo: ArrayUbo<GlyphPrimitiveIndex, PackedGlyph>,
}

impl UniformBuffer {
    fn new(max_ubo_size: usize) -> UniformBuffer {
        UniformBuffer {
            max_ubo_size: max_ubo_size,

            layer_ubos: Vec::new(),
            rect_ubos: Vec::new(),
            clip_ubos: Vec::new(),
            image_ubos: Vec::new(),
            gradient_ubos: Vec::new(),
            gradient_stop_ubos: Vec::new(),
            text_ubos: Vec::new(),
            //glyph_ubos: Vec::new(),

            layer_ubo: Ubo::new(),
            rect_ubo: Ubo::new(),
            clip_ubo: Ubo::new(),
            image_ubo: Ubo::new(),
            gradient_ubo: Ubo::new(),
            gradient_stop_ubo: ArrayUbo::new(),
            text_ubo: Ubo::new(),
            //glyph_ubo: ArrayUbo::new(),
            cmd_ubo: Vec::new(),
        }
    }

    fn finalize(&mut self) {
        self.layer_ubos.push(mem::replace(&mut self.layer_ubo, Ubo::new()));
        self.rect_ubos.push(mem::replace(&mut self.rect_ubo, Ubo::new()));
        self.clip_ubos.push(mem::replace(&mut self.clip_ubo, Ubo::new()));
        self.image_ubos.push(mem::replace(&mut self.image_ubo, Ubo::new()));
        self.gradient_ubos.push(mem::replace(&mut self.gradient_ubo, Ubo::new()));
        self.gradient_stop_ubos.push(mem::replace(&mut self.gradient_stop_ubo, ArrayUbo::new()));
        //self.glyph_ubos.push(mem::replace(&mut self.glyph_ubo, ArrayUbo::new()));
        self.text_ubos.push(mem::replace(&mut self.text_ubo, Ubo::new()));

        assert!(self.cmd_ubo.len() < ((self.max_ubo_size - 16) / mem::size_of::<Command>()));        // A reasonable estimate for now...
    }
}

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct ClipCorner {
    position: Point2D<DevicePixel>,
    outer_radius_x: DevicePixel,
    outer_radius_y: DevicePixel,
    inner_radius_x: DevicePixel,
    inner_radius_y: DevicePixel,
    padding: Point2D<DevicePixel>,
}

impl ClipCorner {
    pub fn invalid() -> ClipCorner {
        ClipCorner {
            position: Point2D::zero(),
            outer_radius_x: DevicePixel::from_u32(0),
            outer_radius_y: DevicePixel::from_u32(0),
            inner_radius_x: DevicePixel::from_u32(0),
            inner_radius_y: DevicePixel::from_u32(0),
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
    pub fn invalid() -> Clip {
        Clip {
            rect: Rect::zero(),
            top_left: ClipCorner::invalid(),
            top_right: ClipCorner::invalid(),
            bottom_left: ClipCorner::invalid(),
            bottom_right: ClipCorner::invalid(),
        }
    }

    pub fn from_clip_region(clip: &ComplexClipRegion,
                            device_pixel_ratio: f32) -> Clip {
        Clip {
            rect: clip.rect,
            top_left: ClipCorner {
                position: Point2D::new(DevicePixel::new(clip.rect.origin.x + clip.radii.top_left.width, device_pixel_ratio),
                                       DevicePixel::new(clip.rect.origin.y + clip.radii.top_left.height, device_pixel_ratio)),
                outer_radius_x: DevicePixel::new(clip.radii.top_left.width, device_pixel_ratio),
                outer_radius_y: DevicePixel::new(clip.radii.top_left.height, device_pixel_ratio),
                inner_radius_x: DevicePixel::from_u32(0),
                inner_radius_y: DevicePixel::from_u32(0),
                padding: Point2D::zero(),
            },
            top_right: ClipCorner {
                position: Point2D::new(DevicePixel::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.top_right.width, device_pixel_ratio),
                                       DevicePixel::new(clip.rect.origin.y + clip.radii.top_right.height, device_pixel_ratio)),
                outer_radius_x: DevicePixel::new(clip.radii.top_right.width, device_pixel_ratio),
                outer_radius_y: DevicePixel::new(clip.radii.top_right.height, device_pixel_ratio),
                inner_radius_x: DevicePixel::from_u32(0),
                inner_radius_y: DevicePixel::from_u32(0),
                padding: Point2D::zero(),
            },
            bottom_left: ClipCorner {
                position: Point2D::new(DevicePixel::new(clip.rect.origin.x + clip.radii.bottom_left.width, device_pixel_ratio),
                                       DevicePixel::new(clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_left.height, device_pixel_ratio)),
                outer_radius_x: DevicePixel::new(clip.radii.bottom_left.width, device_pixel_ratio),
                outer_radius_y: DevicePixel::new(clip.radii.bottom_left.height, device_pixel_ratio),
                inner_radius_x: DevicePixel::from_u32(0),
                inner_radius_y: DevicePixel::from_u32(0),
                padding: Point2D::zero(),
            },
            bottom_right: ClipCorner {
                position: Point2D::new(DevicePixel::new(clip.rect.origin.x + clip.rect.size.width - clip.radii.bottom_right.width, device_pixel_ratio),
                                       DevicePixel::new(clip.rect.origin.y + clip.rect.size.height - clip.radii.bottom_right.height, device_pixel_ratio)),
                outer_radius_x: DevicePixel::new(clip.radii.bottom_right.width, device_pixel_ratio),
                outer_radius_y: DevicePixel::new(clip.radii.bottom_right.height, device_pixel_ratio),
                inner_radius_x: DevicePixel::from_u32(0),
                inner_radius_y: DevicePixel::from_u32(0),
                padding: Point2D::zero(),
            },
        }
    }

    pub fn from_border_radius(rect: &Rect<f32>,
                              outer_radius: &BorderRadius,
                              inner_radius: &BorderRadius,
                              device_pixel_ratio: f32) -> Clip {
        Clip {
            rect: *rect,
            top_left: ClipCorner {
                position: Point2D::new(DevicePixel::new(rect.origin.x + outer_radius.top_left.width, device_pixel_ratio),
                                       DevicePixel::new(rect.origin.y + outer_radius.top_left.height, device_pixel_ratio)),
                outer_radius_x: DevicePixel::new(outer_radius.top_left.width, device_pixel_ratio),
                outer_radius_y: DevicePixel::new(outer_radius.top_left.height, device_pixel_ratio),
                inner_radius_x: DevicePixel::new(inner_radius.top_left.width, device_pixel_ratio),
                inner_radius_y: DevicePixel::new(inner_radius.top_left.height, device_pixel_ratio),
                padding: Point2D::zero(),
            },
            top_right: ClipCorner {
                position: Point2D::new(DevicePixel::new(rect.origin.x + rect.size.width - outer_radius.top_right.width, device_pixel_ratio),
                                       DevicePixel::new(rect.origin.y + outer_radius.top_right.height, device_pixel_ratio)),
                outer_radius_x: DevicePixel::new(outer_radius.top_right.width, device_pixel_ratio),
                outer_radius_y: DevicePixel::new(outer_radius.top_right.height, device_pixel_ratio),
                inner_radius_x: DevicePixel::new(inner_radius.top_right.width, device_pixel_ratio),
                inner_radius_y: DevicePixel::new(inner_radius.top_right.height, device_pixel_ratio),
                padding: Point2D::zero(),
            },
            bottom_left: ClipCorner {
                position: Point2D::new(DevicePixel::new(rect.origin.x + outer_radius.bottom_left.width, device_pixel_ratio),
                                       DevicePixel::new(rect.origin.y + rect.size.height - outer_radius.bottom_left.height, device_pixel_ratio)),
                outer_radius_x: DevicePixel::new(outer_radius.bottom_left.width, device_pixel_ratio),
                outer_radius_y: DevicePixel::new(outer_radius.bottom_left.height, device_pixel_ratio),
                inner_radius_x: DevicePixel::new(inner_radius.bottom_left.width, device_pixel_ratio),
                inner_radius_y: DevicePixel::new(inner_radius.bottom_left.height, device_pixel_ratio),
                padding: Point2D::zero(),
            },
            bottom_right: ClipCorner {
                position: Point2D::new(DevicePixel::new(rect.origin.x + rect.size.width - outer_radius.bottom_right.width, device_pixel_ratio),
                                       DevicePixel::new(rect.origin.y + rect.size.height - outer_radius.bottom_right.height, device_pixel_ratio)),
                outer_radius_x: DevicePixel::new(outer_radius.bottom_right.width, device_pixel_ratio),
                outer_radius_y: DevicePixel::new(outer_radius.bottom_right.height, device_pixel_ratio),
                inner_radius_x: DevicePixel::new(inner_radius.bottom_right.width, device_pixel_ratio),
                inner_radius_y: DevicePixel::new(inner_radius.bottom_right.height, device_pixel_ratio),
                padding: Point2D::zero(),
            },
        }
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct RectanglePrimitiveIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct GradientPrimitiveIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct GradientStopPrimitiveIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct TextPrimitiveIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct GlyphPrimitiveIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ImagePrimitiveIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct ClipPrimitiveIndex(u32);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum PrimitiveKey {
    Rectangle(RectanglePrimitiveIndex),
    SetClip(ClipPrimitiveIndex),
    ClearClip(ClipPrimitiveIndex),
    Image(ImagePrimitiveIndex),
    Gradient(GradientPrimitiveIndex),
    Text(TextPrimitiveIndex),
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct PackedCommand(pub u32);

pub enum Command {
    SetLayer,
    DrawRectangle,
    SetClip,
    ClearClip,
    DrawImage,
    DrawGradient,
    DrawText,
}

impl PackedCommand {
    pub fn new(cmd: Command, index: usize) -> PackedCommand {
        debug_assert!(index < 65536);       // ensure u16 (should be bounded by max ubo size anyway)
        PackedCommand(index as u32 | ((cmd as u32) << 24))
    }

    pub fn raw(value: u32) -> PackedCommand {
        PackedCommand(value)
    }
}

#[derive(Debug)]
struct ImagePrimitive {
    xf_rect: TransformedRect,
    packed: PackedImage,
}

#[derive(Clone, Debug)]
pub struct PackedImage {
    p0: Point2D<f32>,
    st0: Point2D<f32>,
    p1: Point2D<f32>,
    st1: Point2D<f32>,
}

#[derive(Clone, Debug)]
pub struct PackedGlyph {
    pub p0: Point2D<f32>,
    pub p1: Point2D<f32>,
    pub st0: Point2D<f32>,
    pub st1: Point2D<f32>,
}

pub struct GlyphPrimitive {
    glyphs: Vec<PackedGlyph>,
}

#[derive(Debug)]
pub struct TextPrimitive {
    xf_rect: TransformedRect,
    glyph_index: GlyphPrimitiveIndex,
    packed: PackedText,
}

#[derive(Clone, Debug)]
pub struct PackedText {
    p0: Point2D<f32>,
    st0: Point2D<f32>,
    p1: Point2D<f32>,
    st1: Point2D<f32>,
    color: ColorF,
}

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

#[derive(Clone, Debug)]
pub struct PackedRectangle {
    p0: Point2D<f32>,
    p1: Point2D<f32>,
    color: ColorF,
    //color_key: ColorBufferKey,
    //padding: [u32; 3],
}

#[derive(Clone, Debug)]
pub struct PackedLayer {
    blend_info: [f32; 4],
    p0: Point2D<f32>,
    p1: Point2D<f32>,
    inv_transform: Matrix4,
    screen_vertices: [Point4D<f32>; 4],
}

#[derive(Debug)]
pub struct PackedTile {
    pub rect: Rect<i32>,
    pub rect_ubo_index: usize,
    pub layer_ubo_index: usize,
    pub clip_ubo_index: usize,
    pub image_ubo_index: usize,
    pub gradient_ubo_index: usize,
    pub gradient_stop_ubo_index: usize,
    pub text_ubo_index: usize,
    pub cmd_index: usize,
    pub cmd_count: usize,
    pub technique_params: TechniqueParams,
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

    fn can_fit(&self, keys: &HashSet<KEY, BuildHasherDefault<FnvHasher>>, kind: UboBindLocation, max_ubo_size: usize) -> bool {
        let max_item_count = kind.get_array_len(max_ubo_size);
        let new_item_count = keys.iter().filter(|key| !self.map.contains_key(key)).count();
        let item_count = self.items.len() + new_item_count;
        item_count < max_item_count
    }

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

#[derive(Debug)]
pub struct ArrayUbo<KEY: Eq + Hash, TYPE> {
    pub items: Vec<TYPE>,
    map: HashMap<KEY, usize, BuildHasherDefault<FnvHasher>>,
}

impl<KEY: Eq + Hash + Copy, TYPE: Clone> ArrayUbo<KEY, TYPE> {
    fn new() -> ArrayUbo<KEY, TYPE> {
        ArrayUbo {
            items: Vec::new(),
            map: HashMap::with_hasher(Default::default()),
        }
    }

    fn used_size(&self) -> usize {
        self.required_bytes(self.items.len())
    }

    fn required_bytes(&self, item_count: usize) -> usize {
        item_count * mem::size_of::<TYPE>()
    }

    fn can_fit<F>(&self,
                  keys: &HashSet<KEY, BuildHasherDefault<FnvHasher>>,
                  kind: UboBindLocation,
                  max_ubo_size: usize,
                  f: F) -> bool
                  where F: Fn(&KEY) -> usize {
        let max_item_count = kind.get_array_len(max_ubo_size);
        let new_item_count = keys.iter()
                                 .filter(|key| !self.map.contains_key(key))
                                 .map(|key| {
                                   f(key)
                                 })
                                 .fold(0, ops::Add::add);
        let item_count = self.items.len() + new_item_count;
        item_count < max_item_count
    }

    fn maybe_insert_and_get_index(&mut self, key: KEY, data: &[TYPE]) -> usize {
        let map = &mut self.map;
        let items = &mut self.items;

        *map.entry(key).or_insert_with(|| {
            let index = items.len();
            for item in data {
                items.push(item.clone());
            }
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

#[derive(Debug)]
struct RectanglePrimitive {
    xf_rect: TransformedRect,
    packed: PackedRectangle,
}

struct PrimitiveBuffer {
    rectangles: Vec<RectanglePrimitive>,
    clips: Vec<Clip>,
    images: Vec<ImagePrimitive>,
    gradients: Vec<GradientPrimitive>,
    gradient_stops: Vec<GradientStopPrimitive>,
    glyphs: Vec<GlyphPrimitive>,
    texts: Vec<TextPrimitive>,
}

impl PrimitiveBuffer {
    fn new() -> PrimitiveBuffer {
        PrimitiveBuffer {
            rectangles: Vec::new(),
            clips: Vec::new(),
            images: Vec::new(),
            gradients: Vec::new(),
            gradient_stops: Vec::new(),
            glyphs: Vec::new(),
            texts: Vec::new(),
        }
    }

    fn print_stats(&self) {
        println!("PB: r={} c={} i={} g={} t={}", self.rectangles.len(), self.clips.len(), self.images.len(), self.gradients.len(), self.texts.len());
        println!("PB: total prims = {}", self.rectangles.len() + self.clips.len() + self.images.len() + self.gradients.len() + self.texts.len());
    }

    fn get_xf_rect_and_opacity(&self, primitive_key: &PrimitiveKey) -> (&TransformedRect, bool) {
        match primitive_key {
            &PrimitiveKey::Rectangle(index) => {
                let RectanglePrimitiveIndex(index) = index;
                let rect = &self.rectangles[index as usize];
                (&rect.xf_rect, rect.packed.color.a == 1.0)
            }
            &PrimitiveKey::Text(index) => {
                let TextPrimitiveIndex(index) = index;
                (&self.texts[index as usize].xf_rect, false)
            }
            &PrimitiveKey::Image(index) => {
                let ImagePrimitiveIndex(index) = index;
                let image = &self.images[index as usize];
                (&image.xf_rect, false)
            }
            _ => {
                panic!("todo");
            }
        }
    }

    fn get_local_rect(&self, primitive_key: &PrimitiveKey) -> &Rect<f32> {
        match primitive_key {
            &PrimitiveKey::Rectangle(index) => {
                let RectanglePrimitiveIndex(index) = index;
                &self.rectangles[index as usize].xf_rect.local_rect
            }
            &PrimitiveKey::SetClip(index) => {
                let ClipPrimitiveIndex(index) = index;
                &self.clips[index as usize].rect
            }
            &PrimitiveKey::ClearClip(index) => {
                let ClipPrimitiveIndex(index) = index;
                &self.clips[index as usize].rect
            }
            &PrimitiveKey::Image(index) => {
                let ImagePrimitiveIndex(index) = index;
                &self.images[index as usize].xf_rect.local_rect
            }
            &PrimitiveKey::Gradient(index) => {
                let GradientPrimitiveIndex(index) = index;
                &self.gradients[index as usize].rect
            }
            &PrimitiveKey::Text(index) => {
                let TextPrimitiveIndex(index) = index;
                &self.texts[index as usize].xf_rect.local_rect
            }
        }
    }

    fn add_rectangle(&mut self,
                     xf_rect: TransformedRect,
                     color: ColorF) -> PrimitiveKey {
        let index = self.rectangles.len();
        self.rectangles.push(RectanglePrimitive {
            packed: PackedRectangle {
                p0: xf_rect.local_rect.origin,
                p1: xf_rect.local_rect.bottom_right(),
                color: color,
                //color_key: color_key,
                //padding: [0, 0, 0],
            },
            xf_rect: xf_rect,
        });
        PrimitiveKey::Rectangle(RectanglePrimitiveIndex(index as u32))
    }

    fn add_set_clip(&mut self,
                    clip: Clip) -> PrimitiveKey {
        let index = self.clips.len();
        self.clips.push(clip);
        PrimitiveKey::SetClip(ClipPrimitiveIndex(index as u32))
    }

    fn add_clear_clip(&mut self) -> PrimitiveKey {
        PrimitiveKey::ClearClip(ClipPrimitiveIndex((self.clips.len() - 1) as u32))
    }

    fn add_image(&mut self,
                 xf_rect: TransformedRect,
                 st0: Point2D<f32>,
                 st1: Point2D<f32>) -> PrimitiveKey {
        let index = self.images.len();
        self.images.push(ImagePrimitive {
            packed: PackedImage {
                p0: xf_rect.local_rect.origin,
                p1: xf_rect.local_rect.bottom_right(),
                st0: st0,
                st1: st1,
            },
            xf_rect: xf_rect,
        });
        PrimitiveKey::Image(ImagePrimitiveIndex(index as u32))
    }

    fn add_gradient(&mut self,
                    rect: Rect<f32>,
                    start_point: &Point2D<f32>,
                    end_point: &Point2D<f32>,
                    stops_index: GradientStopPrimitiveIndex,
                    stop_count: u32) -> PrimitiveKey {
        let angle = (start_point.y - end_point.y).atan2(end_point.x - start_point.x);
        let (sin_angle, cos_angle) = angle.sin_cos();
        let rx0 = start_point.x * cos_angle - start_point.y * sin_angle;
        let rx1 = end_point.x * cos_angle - end_point.y * sin_angle;
        let d = rx1 - rx0;

        let index = self.gradients.len();
        self.gradients.push(GradientPrimitive {
            rect: rect,
            stops_index: stops_index,
            packed: PackedGradient {
                p0: rect.origin,
                p1: rect.bottom_right(),
                constants: [ sin_angle, cos_angle, d, rx0 ],
                stop_start_index: 0,
                stop_count: stop_count,
                pad0: 0,
                pad1: 0,
            },
        });
        PrimitiveKey::Gradient(GradientPrimitiveIndex(index as u32))
    }

    fn add_gradient_stops(&mut self, stops: Vec<PackedGradientStop>) -> GradientStopPrimitiveIndex {
        let index = self.gradient_stops.len();
        self.gradient_stops.push(GradientStopPrimitive {
            stops: stops,
        });
        GradientStopPrimitiveIndex(index as u32)
    }

    fn add_text(&mut self,
                xf_rect: TransformedRect,
                color: ColorF,
                glyph_index: GlyphPrimitiveIndex) -> PrimitiveKey {
        let index = self.texts.len();
        self.texts.push(TextPrimitive {
            packed: PackedText {
                p0: xf_rect.local_rect.origin,
                p1: xf_rect.local_rect.bottom_right(),
                color: color,
                st0: Point2D::zero(),
                st1: Point2D::zero(),
            },
            xf_rect: xf_rect,
            glyph_index: glyph_index,
        });
        PrimitiveKey::Text(TextPrimitiveIndex(index as u32))
    }

    fn add_glyphs(&mut self, glyphs: Vec<PackedGlyph>) -> GlyphPrimitiveIndex {
        let index = self.glyphs.len();
        self.glyphs.push(GlyphPrimitive {
            glyphs: glyphs,
        });
        GlyphPrimitiveIndex(index as u32)
    }

    fn get_rect(&self, index: RectanglePrimitiveIndex) -> &RectanglePrimitive {
        let RectanglePrimitiveIndex(index) = index;
        &self.rectangles[index as usize]
    }

    fn get_clip(&self, index: ClipPrimitiveIndex) -> &Clip {
        let ClipPrimitiveIndex(index) = index;
        &self.clips[index as usize]
    }

    fn get_image(&self, index: ImagePrimitiveIndex) -> &ImagePrimitive {
        let ImagePrimitiveIndex(index) = index;
        &self.images[index as usize]
    }

    fn get_gradient(&self, index: GradientPrimitiveIndex) -> &GradientPrimitive {
        let GradientPrimitiveIndex(index) = index;
        &self.gradients[index as usize]
    }

    fn get_gradient_stop(&self, index: GradientStopPrimitiveIndex) -> &GradientStopPrimitive {
        let GradientStopPrimitiveIndex(index) = index;
        &self.gradient_stops[index as usize]
    }

    fn get_text(&self, index: TextPrimitiveIndex) -> &TextPrimitive {
        let TextPrimitiveIndex(index) = index;
        &self.texts[index as usize]
    }

    fn get_glyph(&self, index: GlyphPrimitiveIndex) -> &GlyphPrimitive {
        let GlyphPrimitiveIndex(index) = index;
        &self.glyphs[index as usize]
    }
}

struct LayerTemplate {
    transform: Matrix4,
    packed: PackedLayer,
}

struct LayerInstance {
    layer_index: LayerTemplateIndex,
    primitives: Vec<PrimitiveKey>,
}

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
    pub prim_count: usize,
    children: Vec<Tile>,

    required_layers: HashSet<LayerTemplateIndex, BuildHasherDefault<FnvHasher>>,
    required_rects: HashSet<RectanglePrimitiveIndex, BuildHasherDefault<FnvHasher>>,
    required_clips: HashSet<ClipPrimitiveIndex, BuildHasherDefault<FnvHasher>>,
    required_images: HashSet<ImagePrimitiveIndex, BuildHasherDefault<FnvHasher>>,
    required_gradients: HashSet<GradientPrimitiveIndex, BuildHasherDefault<FnvHasher>>,
    required_gradient_stops: HashSet<GradientStopPrimitiveIndex, BuildHasherDefault<FnvHasher>>,
    //required_glyphs: HashSet<GlyphPrimitiveIndex, BuildHasherDefault<FnvHasher>>,
    required_texts: HashSet<TextPrimitiveIndex, BuildHasherDefault<FnvHasher>>,
}

impl Tile {
    fn new(screen_rect: Rect<i32>) -> Tile {
        Tile {
            screen_rect: screen_rect,
            layers: Vec::new(),
            children: Vec::new(),
            prim_count: 0,

            required_layers: HashSet::with_hasher(Default::default()),
            required_rects: HashSet::with_hasher(Default::default()),
            required_clips: HashSet::with_hasher(Default::default()),
            required_images: HashSet::with_hasher(Default::default()),
            required_gradients: HashSet::with_hasher(Default::default()),
            required_gradient_stops: HashSet::with_hasher(Default::default()),
            //required_glyphs: HashSet::with_hasher(Default::default()),
            required_texts: HashSet::with_hasher(Default::default()),
        }
    }

    fn add_primitive(&mut self,
                     primitive_key: PrimitiveKey,
                     layer_index: LayerTemplateIndex,
                     primitives: &PrimitiveBuffer,
                     /*text_buffer: &mut TextBuffer*/) {
        let (xf_rect, is_opaque) = primitives.get_xf_rect_and_opacity(&primitive_key);

        if xf_rect.screen_rect.intersects(&self.screen_rect) {

            // Check if this primitive supercedes all existing primitives in this
            // tile - this is a very important optimization to allow the CPU to create
            // small tiles that can use the simple tiling pass shader.
            // TODO(gw): This doesn't work with 3d transforms (it assumes axis aligned rects for now)!!
            if is_opaque &&
               xf_rect.screen_rect.contains(&self.screen_rect.origin) &&
               xf_rect.screen_rect.contains(&self.screen_rect.bottom_right()) {
                self.layers.clear();
            }

            let need_new_layer = self.layers.is_empty() ||
                                 self.layers.last().unwrap().layer_index != layer_index;

            if need_new_layer {
                self.layers.push(TileLayer::new(layer_index));
            }

            self.layers.last_mut().unwrap().primitives.push(primitive_key);
            self.prim_count += 1;

/*
            match primitive_key {
                PrimitiveKey::Rectangle(index) => {
                    self.required_rects.insert(index);
                }
                PrimitiveKey::SetClip(index) => {
                    self.required_clips.insert(index);
                }
                PrimitiveKey::ClearClip(index) => {
                    // Already handled by matching SetClip
                }
                PrimitiveKey::Image(index) => {
                    self.required_images.insert(index);
                }
                PrimitiveKey::Gradient(index) => {
                    let gradient_prim = primitives.get_gradient(index);
                    self.required_gradient_stops.insert(gradient_prim.stops_index);
                    self.required_gradients.insert(index);
                }
                PrimitiveKey::Text(index) => {
                    let text_prim = primitives.get_text(index);
                    let glyph_prim = primitives.get_glyph(text_prim.glyph_index);
                    text_buffer.push_text(index, &text_prim.xf_rect.local_rect, glyph_prim);
                    //self.required_glyphs.insert(text_prim.glyph_index);
                    self.required_texts.insert(index);
                }
            }*/
        }
    }

    fn split_if_needed(&mut self,
                       primitives: &PrimitiveBuffer,
                       text_buffer: &mut TextBuffer) {
        let try_split = self.screen_rect.size.width > 15 &&
                        self.screen_rect.size.height > 15 &&
                        self.prim_count > 4;

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

            if h_min < self.prim_count || v_min < self.prim_count {
                self.layers.clear();
                self.prim_count = 0;

                if h_min < v_min {
                    self.children.push(left);
                    self.children.push(right);
                } else {
                    self.children.push(top);
                    self.children.push(bottom);
                }

                for child in &mut self.children {
                    child.split_if_needed(primitives, text_buffer);
                }
            }
        }

        for layer in &self.layers {
            self.required_layers.insert(layer.layer_index);

            for primitive_key in &layer.primitives {
                match *primitive_key {
                    PrimitiveKey::Rectangle(index) => {
                        self.required_rects.insert(index);
                    }
                    PrimitiveKey::SetClip(index) => {
                        self.required_clips.insert(index);
                    }
                    PrimitiveKey::ClearClip(index) => {
                        // Already handled by matching SetClip
                    }
                    PrimitiveKey::Image(index) => {
                        self.required_images.insert(index);
                    }
                    PrimitiveKey::Gradient(index) => {
                        let gradient_prim = primitives.get_gradient(index);
                        self.required_gradient_stops.insert(gradient_prim.stops_index);
                        self.required_gradients.insert(index);
                    }
                    PrimitiveKey::Text(index) => {
                        let text_prim = primitives.get_text(index);
                        let glyph_prim = primitives.get_glyph(text_prim.glyph_index);
                        text_buffer.push_text(index, &text_prim.xf_rect.local_rect, glyph_prim);
                        //self.required_glyphs.insert(text_prim.glyph_index);
                        self.required_texts.insert(index);
                    }
                }
            }
        }
    }

    fn build(&mut self,
             uniforms: &mut UniformBuffer,
             text_buffer: &TextBuffer,
             primitives: &PrimitiveBuffer,
             layer_templates: &Vec<LayerTemplate>,
             max_ubo_size: usize,
             packed_tiles: &mut Vec<PackedTile>) {
        if self.prim_count > 0 {
            if !uniforms.layer_ubo.can_fit(&self.required_layers, UboBindLocation::Layers, max_ubo_size) {
                let ubo = mem::replace(&mut uniforms.layer_ubo, Ubo::new());
                uniforms.layer_ubos.push(ubo);
            }

            if !uniforms.rect_ubo.can_fit(&self.required_rects, UboBindLocation::Rectangles, max_ubo_size) {
                let ubo = mem::replace(&mut uniforms.rect_ubo, Ubo::new());
                uniforms.rect_ubos.push(ubo);
            }

            if !uniforms.clip_ubo.can_fit(&self.required_clips, UboBindLocation::Clips, max_ubo_size) {
                let ubo = mem::replace(&mut uniforms.clip_ubo, Ubo::new());
                uniforms.clip_ubos.push(ubo);
            }

            if !uniforms.image_ubo.can_fit(&self.required_images, UboBindLocation::Images, max_ubo_size) {
                let ubo = mem::replace(&mut uniforms.image_ubo, Ubo::new());
                uniforms.image_ubos.push(ubo);
            }

            if !uniforms.gradient_ubo.can_fit(&self.required_gradients, UboBindLocation::Gradients, max_ubo_size) {
                let ubo = mem::replace(&mut uniforms.gradient_ubo, Ubo::new());
                uniforms.gradient_ubos.push(ubo);
            }

            if !uniforms.gradient_stop_ubo.can_fit(&self.required_gradient_stops, UboBindLocation::GradientStops, max_ubo_size, |key| {
                let gradient_stop_prim = primitives.get_gradient_stop(*key);
                gradient_stop_prim.stops.len()
            }) {
                let ubo = mem::replace(&mut uniforms.gradient_stop_ubo, ArrayUbo::new());
                uniforms.gradient_stop_ubos.push(ubo);
            }

            if !uniforms.text_ubo.can_fit(&self.required_texts, UboBindLocation::Texts, max_ubo_size) {
                let ubo = mem::replace(&mut uniforms.text_ubo, Ubo::new());
                uniforms.text_ubos.push(ubo);
            }

/*
            if !uniforms.glyph_ubo.can_fit(&self.required_glyphs, UboBindLocation::Glyphs, max_ubo_size, |key| {
                let glyph_prim = primitives.get_glyph(*key);
                glyph_prim.glyphs.len()
            }) {
                let ubo = mem::replace(&mut uniforms.glyph_ubo, ArrayUbo::new());
                uniforms.glyph_ubos.push(ubo);
            }
*/

            let mut layer_cmd_lists = Vec::new();

            let mut technique_params = TechniqueParams {
                layer_count: self.layers.len(),
                text_count: 0,
                image_count: 0,
                rect_count: 0,
            };

            for tile_layer in self.layers.iter().rev() {
                let LayerTemplateIndex(layer_index) = tile_layer.layer_index;
                let layer_template = &layer_templates[layer_index as usize].packed;

                let packed_layer_index = uniforms.layer_ubo.maybe_insert_and_get_index(tile_layer.layer_index,
                                                                                      layer_template);

                let mut cmds_for_this_layer = Vec::new();
                cmds_for_this_layer.push(PackedCommand::new(Command::SetLayer, packed_layer_index));

                for prim_key in tile_layer.primitives.iter().rev() {
                    match prim_key {
                        &PrimitiveKey::Rectangle(index) => {
                            technique_params.rect_count += 1;
                            let rect_prim = primitives.get_rect(index);
                            let rect_index = uniforms.rect_ubo.maybe_insert_and_get_index(index, &rect_prim.packed);
                            //println!("\t\t\trect {:?}", rect_prim.packed);
                            cmds_for_this_layer.push(PackedCommand::new(Command::DrawRectangle, rect_index));
                        }
                        &PrimitiveKey::SetClip(index) => {
                            panic!("todo");
                            let clip_prim = primitives.get_clip(index);
                            let clip_index = uniforms.clip_ubo.maybe_insert_and_get_index(index, &clip_prim);
                            //println!("\t\t\tset clip {:?}", clip_prim);
                            cmds_for_this_layer.push(PackedCommand::new(Command::SetClip, clip_index));
                        }
                        &PrimitiveKey::ClearClip(index) => {
                            panic!("todo");
                            let clip_prim = primitives.get_clip(index);
                            let clip_index = uniforms.clip_ubo.maybe_insert_and_get_index(index, &clip_prim);
                            //println!("\t\t\tclear clip {:?}", clip_prim);
                            cmds_for_this_layer.push(PackedCommand::new(Command::ClearClip, clip_index));
                        }
                        &PrimitiveKey::Image(index) => {
                            technique_params.image_count += 1;
                            let image_prim = primitives.get_image(index);
                            let image_index = uniforms.image_ubo.maybe_insert_and_get_index(index, &image_prim.packed);
                            //println!("\t\t\timage {:?}", image_prim.packed);
                            cmds_for_this_layer.push(PackedCommand::new(Command::DrawImage, image_index));
                        }
                        &PrimitiveKey::Gradient(index) => {
                            panic!("todo");
                            let gradient_prim = primitives.get_gradient(index);
                            let gradient_stop_prim = primitives.get_gradient_stop(gradient_prim.stops_index);

                            let gradient_stop_index = uniforms.gradient_stop_ubo.maybe_insert_and_get_index(gradient_prim.stops_index,
                                                                                                   &gradient_stop_prim.stops);

                            let mut packed_gradient = gradient_prim.packed.clone();
                            packed_gradient.stop_start_index = gradient_stop_index as u32;

                            //println!("\t\t\tgradient {:?}", packed_gradient);

                            let gradient_index = uniforms.gradient_ubo.maybe_insert_and_get_index(index, &packed_gradient);
                            cmds_for_this_layer.push(PackedCommand::new(Command::DrawGradient, gradient_index));
                        }
                        &PrimitiveKey::Text(index) => {
                            technique_params.text_count += 1;
                            let text_prim = primitives.get_text(index);
                            let glyph_prim = primitives.get_glyph(text_prim.glyph_index);

                            //let glyph_index = uniforms.glyph_ubo.maybe_insert_and_get_index(text_prim.glyph_index,
                            //                                                       &glyph_prim.glyphs);

                            let text = text_buffer.get(index);

                            let mut packed_text = text_prim.packed.clone();
                            packed_text.st0 = text.st0;
                            packed_text.st1 = text.st1;
                            packed_text.p0 = text.rect.origin;
                            packed_text.p1 = text.rect.bottom_right();

                            //packed_text.glyph_start_index = glyph_index as u32;
                            //packed_text.glyph_end_index = (glyph_index + glyph_prim.glyphs.len()) as u32;

                            //println!("\t\t\ttext {:?}", packed_text);

                            let text_index = uniforms.text_ubo.maybe_insert_and_get_index(index, &packed_text);
                            cmds_for_this_layer.push(PackedCommand::new(Command::DrawText, text_index));
                        }
                    }
                }

                if cmds_for_this_layer.len() > 1 {
                    layer_cmd_lists.push(cmds_for_this_layer);
                }
            }

            let cmd_first_index = uniforms.cmd_ubo.len();

            for layer in layer_cmd_lists.iter() {
                for cmd in layer.iter() {
                    uniforms.cmd_ubo.push(*cmd);
                }
            }

            let cmd_count = uniforms.cmd_ubo.len() - cmd_first_index;

/*
            if cmd_count == 5 {
                println!("{:?} -> cmds = {:?}", self.screen_rect, &uniforms.cmd_ubo[cmd_first_index..cmd_first_index + cmd_count]);
            }
*/

            let packed_tile = PackedTile {
                rect: self.screen_rect,
                layer_ubo_index: uniforms.layer_ubos.len(),
                rect_ubo_index: uniforms.rect_ubos.len(),
                clip_ubo_index: uniforms.clip_ubos.len(),
                image_ubo_index: uniforms.image_ubos.len(),
                gradient_ubo_index: uniforms.gradient_ubos.len(),
                gradient_stop_ubo_index: uniforms.gradient_stop_ubos.len(),
                text_ubo_index: uniforms.text_ubos.len(),
                cmd_index: cmd_first_index,
                cmd_count: cmd_count,
                technique_params: technique_params,
            };

            packed_tiles.push(packed_tile);
        }

        for child in &mut self.children {
            child.build(uniforms,
                        text_buffer,
                        primitives,
                        layer_templates,
                        max_ubo_size,
                        packed_tiles);
        }
    }
}

pub struct TileFrame {
    pub uniforms: UniformBuffer,
    pub text_buffer: TextBuffer,
    pub viewport_size: Size2D<i32>,
    pub tiles: Vec<PackedTile>,
    pub color_texture_id: TextureId,
    pub mask_texture_id: TextureId,
    //pub scroll_offset: Point2D<f32>,
}

pub struct TileBuilder {
    layer_templates: Vec<LayerTemplate>,
    layer_instances: Vec<LayerInstance>,
    layer_stack: Vec<LayerTemplateIndex>,
    primitives: PrimitiveBuffer,
    device_pixel_ratio: f32,
    color_texture_id: TextureId,
    mask_texture_id: TextureId,
    scroll_offset: Point2D<f32>,
}

impl TileBuilder {
    pub fn new(device_pixel_ratio: f32, scroll_offset: Point2D<f32>) -> TileBuilder {
        TileBuilder {
            layer_templates: Vec::new(),
            layer_instances: Vec::new(),
            layer_stack: Vec::new(),
            primitives: PrimitiveBuffer::new(),
            device_pixel_ratio: device_pixel_ratio,
            color_texture_id: TextureId(0),
            mask_texture_id: TextureId(0),
            scroll_offset: scroll_offset,
        }
    }

    fn transform_rect(&self, rect: &Rect<f32>) -> TransformedRect {
        let current_layer = *self.layer_stack.last().unwrap();
        let LayerTemplateIndex(current_layer_index) = current_layer;
        let layer = &self.layer_templates[current_layer_index as usize];
        transform_rect(rect, &layer.transform)
    }

    fn add_primitive(&mut self, prim_key: PrimitiveKey) {
        let current_layer = *self.layer_stack.last().unwrap();
        let LayerTemplateIndex(current_layer_index) = current_layer;

        if self.layer_instances.is_empty() ||
           self.layer_instances.last().unwrap().layer_index != current_layer {
            let instance = LayerInstance {
                layer_index: current_layer,
                primitives: Vec::new(),
            };
            self.layer_instances.push(instance);
        }

        self.layer_instances.last_mut().unwrap().primitives.push(prim_key);
    }

    pub fn add_gradient(&mut self,
                        rect: Rect<f32>,
                        start_point: &Point2D<f32>,
                        end_point: &Point2D<f32>,
                        stops: &ItemRange,
                        auxiliary_lists: &AuxiliaryLists) {
        return;

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
    }

    pub fn add_text(&mut self,
                    rect: Rect<f32>,
                    font_key: FontKey,
                    size: Au,
                    blur_radius: Au,
                    color: &ColorF,
                    src_glyphs: &[GlyphInstance],
                    resource_cache: &ResourceCache,
                    frame_id: FrameId,
                    device_pixel_ratio: f32) {
        if color.a == 0.0 {
            //println!("drop zero alpha text {:?}", rect);
            return
        }

        // Logic below to pick the primary render item depends on len > 0!
        assert!(src_glyphs.len() > 0);
        let mut glyph_key = GlyphKey::new(font_key, size, blur_radius, src_glyphs[0].index);
        let blur_offset = blur_radius.to_f32_px() * (BLUR_INFLATION_FACTOR as f32) / 2.0;
        let mut glyphs = Vec::new();

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

        let glyphs_key = self.primitives.add_glyphs(glyphs);

        let xf_rect = self.transform_rect(&rect);
        let text_key = self.primitives.add_text(xf_rect,
                                                *color,
                                                glyphs_key);
        self.add_primitive(text_key);
    }

    pub fn add_image(&mut self,
                     rect: Rect<f32>,
                     stretch_size: &Size2D<f32>,
                     image_key: ImageKey,
                     image_rendering: ImageRendering,
                     resource_cache: &ResourceCache,
                     frame_id: FrameId) {
        let image_info = resource_cache.get_image(image_key, image_rendering, frame_id);
        let uv_rect = image_info.uv_rect();

        assert!(self.color_texture_id == TextureId(0) || self.color_texture_id == image_info.texture_id);
        self.color_texture_id = image_info.texture_id;

        let xf_rect = self.transform_rect(&rect);
        let image_key = self.primitives.add_image(xf_rect, uv_rect.top_left, uv_rect.bottom_right);
        self.add_primitive(image_key);
    }

    pub fn add_rectangle(&mut self,
                         rect: Rect<f32>,
                         color: ColorF) {
        if color.a == 0.0 {
            return;
        }
        let xf_rect = self.transform_rect(&rect);
        let rect_key = self.primitives.add_rectangle(xf_rect, color);
        self.add_primitive(rect_key);
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
                                                &inner_radius,
                                                self.device_pixel_ratio);

            self.set_clip(clip);
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
            self.clear_clip();
        }
    }

    pub fn set_clip(&mut self, clip: Clip) {
        return;

        let clip_key = self.primitives.add_set_clip(clip);
        self.add_primitive(clip_key);
    }

    pub fn clear_clip(&mut self) {
        return;

        let clip_key = self.primitives.add_clear_clip();
        self.add_primitive(clip_key);
    }

    pub fn push_layer(&mut self,
                      rect: Rect<f32>,
                      transform: Matrix4,
                      opacity: f32) {
        // TODO(gw): Not 3d transform correct!
        let transform = transform.translate(self.scroll_offset.x,
                                            self.scroll_offset.y,
                                            0.0);

        let layer_rect = transform_rect(&rect, &transform);

        let template = LayerTemplate {
            transform: transform,
            packed: PackedLayer {
                blend_info: [opacity, 0.0, 0.0, 0.0],
                p0: rect.origin,
                p1: rect.bottom_right(),
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

    // TODO(gw): This is grossly inefficient! But it should allow us to check the GPU
    //           perf on real world pages / demos, then we can worry about the CPU perf...
    pub fn build(&self,
                 viewport_size: Size2D<i32>,
                 tile_size: Size2D<i32>,
                 max_ubo_size: usize) -> TileFrame {
        let mut text_buffer = TextBuffer::new(TEXT_TARGET_SIZE);

        let mut root_tile = Tile::new(Rect::new(Point2D::zero(), viewport_size));
        for layer_instance in &self.layer_instances {
            for primitive_key in &layer_instance.primitives {
                root_tile.add_primitive(*primitive_key,
                                        layer_instance.layer_index,
                                        &self.primitives,
                                        /*&mut text_buffer*/);
            }
        }

        root_tile.split_if_needed(&self.primitives,
                                  &mut text_buffer);

        // Build UBO batches for each tile
        let mut packed_tiles = Vec::new();
        let mut uniforms = UniformBuffer::new(max_ubo_size);
        root_tile.build(&mut uniforms,
                        &text_buffer,
                        &self.primitives,
                        &self.layer_templates,
                        max_ubo_size,
                        &mut packed_tiles);
        uniforms.finalize();

        TileFrame {
            uniforms: uniforms,
            text_buffer: text_buffer,
            viewport_size: viewport_size,
            tiles: packed_tiles,
            color_texture_id: self.color_texture_id,
            mask_texture_id: self.mask_texture_id,
            //scroll_offset: Point2D::new(-viewport_rect.origin.x as f32,
            //                            -viewport_rect.origin.y as f32),
        }
    }
}

#[inline]
fn clamp(min: i32, value: i32, max: i32) -> i32 {
    cmp::min(max, cmp::max(min, value))
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
