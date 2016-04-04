/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

//use aabbtree::AABBTreeNode;
use app_units::Au;
use batch_builder;
use euclid::{Rect, Size2D};
use fnv::FnvHasher;
use internal_types::{BorderRadiusRasterOp, BoxShadowRasterOp, DrawListItemIndex};
use internal_types::{Glyph, GlyphKey, RasterItem, DevicePixel};
use resource_cache::ResourceCache;
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::hash::BuildHasherDefault;
use tessellator;
use webrender_traits::{AuxiliaryLists, BorderRadius, BorderStyle, BoxShadowClipMode};
use webrender_traits::{FontKey, ImageFormat, ImageKey, ImageRendering, PipelineId};
use webrender_traits::{SpecificDisplayItem};

type RequiredImageSet = HashSet<(ImageKey, ImageRendering), BuildHasherDefault<FnvHasher>>;
type RequiredGlyphMap = HashMap<FontKey, HashSet<Glyph>, BuildHasherDefault<FnvHasher>>;
type RequiredRasterSet = HashSet<RasterItem, BuildHasherDefault<FnvHasher>>;

pub struct ResourceList {
    required_images: RequiredImageSet,
    required_glyphs: RequiredGlyphMap,
    required_rasters: RequiredRasterSet,
    device_pixel_ratio: f32,
}

impl ResourceList {
    pub fn new(device_pixel_ratio: f32) -> ResourceList {
        ResourceList {
            required_glyphs: HashMap::with_hasher(Default::default()),
            required_images: HashSet::with_hasher(Default::default()),
            required_rasters: HashSet::with_hasher(Default::default()),
            device_pixel_ratio: device_pixel_ratio,
        }
    }

    pub fn add_image(&mut self,
                     key: ImageKey,
                     image_rendering: ImageRendering) {
        self.required_images.insert((key, image_rendering));
    }

    pub fn add_glyph(&mut self, font_key: FontKey, glyph: Glyph) {
        match self.required_glyphs.entry(font_key) {
            Occupied(entry) => {
                entry.into_mut().insert(glyph);
            }
            Vacant(entry) => {
                let mut hash_set = HashSet::new();
                hash_set.insert(glyph);
                entry.insert(hash_set);
            }
        }
    }

    pub fn add_radius_raster(&mut self,
                             outer_radius: &Size2D<f32>,
                             inner_radius: &Size2D<f32>,
                             inverted: bool,
                             index: Option<u32>,
                             image_format: ImageFormat) {
        let outer_radius_x = DevicePixel::new(outer_radius.width, self.device_pixel_ratio);
        let outer_radius_y = DevicePixel::new(outer_radius.height, self.device_pixel_ratio);
        let inner_radius_x = DevicePixel::new(inner_radius.width, self.device_pixel_ratio);
        let inner_radius_y = DevicePixel::new(inner_radius.height, self.device_pixel_ratio);
        if let Some(raster_item) = BorderRadiusRasterOp::create(outer_radius_x,
                                                                outer_radius_y,
                                                                inner_radius_x,
                                                                inner_radius_y,
                                                                inverted,
                                                                index,
                                                                image_format) {
            self.required_rasters.insert(RasterItem::BorderRadius(raster_item));
        }
    }

    /// NB: Only adds non-tessellated border radii.
    pub fn add_radius_raster_for_border_radii(&mut self, radii: &BorderRadius) {
        let zero_size = Size2D::new(0.0, 0.0);
        self.add_radius_raster(&radii.top_left, &zero_size, false, None, ImageFormat::A8);
        self.add_radius_raster(&radii.top_right, &zero_size, false, None, ImageFormat::A8);
        self.add_radius_raster(&radii.bottom_left, &zero_size, false, None, ImageFormat::A8);
        self.add_radius_raster(&radii.bottom_right, &zero_size, false, None, ImageFormat::A8);
    }

    pub fn add_box_shadow_corner(&mut self,
                                 blur_radius: f32,
                                 border_radius: f32,
                                 box_rect: &Rect<f32>,
                                 inverted: bool) {
        if let Some(raster_item) = BoxShadowRasterOp::create_corner(blur_radius,
                                                                    border_radius,
                                                                    box_rect,
                                                                    inverted,
                                                                    self.device_pixel_ratio) {
            self.required_rasters.insert(RasterItem::BoxShadow(raster_item));
        }
    }

    pub fn add_box_shadow_edge(&mut self,
                               blur_radius: f32,
                               border_radius: f32,
                               box_rect: &Rect<f32>,
                               inverted: bool) {
        if let Some(raster_item) = BoxShadowRasterOp::create_edge(blur_radius,
                                                                  border_radius,
                                                                  box_rect,
                                                                  inverted,
                                                                  self.device_pixel_ratio) {
            self.required_rasters.insert(RasterItem::BoxShadow(raster_item));
        }
    }

    pub fn for_each_image<F>(&self, mut f: F) where F: FnMut(ImageKey, ImageRendering) {
        for &(image_id, image_rendering) in &self.required_images {
            f(image_id, image_rendering);
        }
    }

    pub fn for_each_raster<F>(&self, mut f: F) where F: FnMut(&RasterItem) {
        for raster_item in &self.required_rasters {
            f(raster_item);
        }
    }

    pub fn for_each_glyph<F>(&self, mut f: F) where F: FnMut(&GlyphKey) {
        for (font_id, glyphs) in &self.required_glyphs {
            let mut glyph_key = GlyphKey::new(font_id.clone(), Au(0), Au(0), 0);

            for glyph in glyphs {
                glyph_key.size = glyph.size;
                glyph_key.index = glyph.index;
                glyph_key.blur_radius = glyph.blur_radius;

                f(&glyph_key);
            }
        }
    }
}

/*
pub trait BuildRequiredResources {
    fn build_resource_list(&mut self,
                           resource_cache: &ResourceCache,
                           pipeline_auxiliary_lists: &HashMap<PipelineId,
                                                              AuxiliaryLists,
                                                              BuildHasherDefault<FnvHasher>>);
}

impl BuildRequiredResources for AABBTreeNode {
    fn build_resource_list(&mut self,
                           resource_cache: &ResourceCache,
                           pipeline_auxiliary_lists: &HashMap<PipelineId,
                                                              AuxiliaryLists,
                                                              BuildHasherDefault<FnvHasher>>) {
        //let _pf = util::ProfileScope::new("  build_resource_list");
        let mut resource_list = ResourceList::new(resource_cache.device_pixel_ratio());

        for item in &self.items {
            let draw_list = resource_cache.get_draw_list(item.draw_list_id);
            let auxiliary_lists =
                pipeline_auxiliary_lists.get(&draw_list.pipeline_id)
                                        .expect("No auxiliary lists for pipeline?!");

            let DrawListItemIndex(index) = item.item_index;
            let display_item = &draw_list.items[index as usize];

            match display_item.item {
                SpecificDisplayItem::Image(ref info) => {
                    resource_list.add_image(info.image_key, info.image_rendering);
                }
                SpecificDisplayItem::Text(ref info) => {
                    let glyphs = auxiliary_lists.glyph_instances(&info.glyphs);
                    for glyph in glyphs {
                        let glyph = Glyph::new(info.size, info.blur_radius, glyph.index);
                        resource_list.add_glyph(info.font_key, glyph);
                    }
                }
                SpecificDisplayItem::BoxShadow(ref info) => {
                    let box_rect = batch_builder::compute_box_shadow_rect(&info.box_bounds,
                                                                          &info.offset,
                                                                          info.spread_radius);
                    resource_list.add_box_shadow_corner(info.blur_radius,
                                                        info.border_radius,
                                                        &box_rect,
                                                        false);
                    resource_list.add_box_shadow_edge(info.blur_radius,
                                                      info.border_radius,
                                                      &box_rect,
                                                      false);
                    if info.clip_mode == BoxShadowClipMode::Inset {
                        resource_list.add_box_shadow_corner(info.blur_radius,
                                                            info.border_radius,
                                                            &box_rect,
                                                            true);
                        resource_list.add_box_shadow_edge(info.blur_radius,
                                                          info.border_radius,
                                                          &box_rect,
                                                          true);
                    }
                }
                SpecificDisplayItem::WebGL(..) => {}
                SpecificDisplayItem::Rectangle(..) => {}
                SpecificDisplayItem::Gradient(..) => {}
                SpecificDisplayItem::Border(ref info) => {}
            }
        }

        self.resource_list = Some(resource_list);
    }
}

fn add_border_radius_raster(outer_radius: &Size2D<f32>,
                            inner_radius: &Size2D<f32>,
                            can_tessellate: bool,
                            resource_cache: &ResourceCache,
                            resource_list: &mut ResourceList) {
    let quad_count = if can_tessellate {
        tessellator::quad_count_for_border_corner(outer_radius,
                                                  resource_cache.device_pixel_ratio())
    } else {
        1
    };
    for rect_index in 0..quad_count {
        let index = if can_tessellate {
            Some(rect_index)
        } else {
            None
        };
        resource_list.add_radius_raster(outer_radius,
                                        inner_radius,
                                        false,
                                        index,
                                        ImageFormat::A8);
    }
}

*/
