/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use app_units::Au;
use batch::{MAX_MATRICES_PER_BATCH, OffsetParams};
use device::{TextureId, TextureFilter};
use euclid::{Rect, Point2D, Point3D, Point4D, Size2D, Matrix4};
use fnv::FnvHasher;
use geometry::ray_intersects_rect;
use internal_types::{AxisDirection, LowLevelFilterOp, CompositionOp, DrawListItemIndex};
use internal_types::{BatchUpdateList, ChildLayerIndex, DrawListId, RenderTile, GlyphKey};
use internal_types::{CompositeBatchInfo, CompositeBatchJob, MaskRegion, PackedSceneVertex};
use internal_types::{RendererFrame, StackingContextInfo, BatchInfo, DrawCall, StackingContextIndex};
use internal_types::{ANGLE_FLOAT_TO_FIXED, MAX_RECT, BatchUpdate, BatchUpdateOp, DrawLayer};
use internal_types::{DrawCommand, ClearInfo, RenderTargetId, DrawListGroupId};
use layer::{Layer, ScrollingState};
use node_compiler::NodeCompiler;
use internal_types::{ANGLE_FLOAT_TO_FIXED, MAX_RECT, BatchUpdate, BatchUpdateOp, DrawLayer, Glyph};
use internal_types::{DrawCommand, ClearInfo, RenderTargetId, DrawListGroupId, DisplayItemId, BasicRotationAngle};
//use layer::Layer;
//use node_compiler::NodeCompiler;
use renderer::CompositionOpHelpers;
use resource_cache::ResourceCache;
//use resource_list::BuildRequiredResources;
use scene::{SceneStackingContext, ScenePipeline, Scene, SceneItem, SpecificSceneItem};
use scoped_threadpool;
use std::collections::{HashMap, HashSet};
use std::hash::BuildHasherDefault;
use std::mem;
use texture_cache::TexturePage;
use util;
use webrender_traits::{AuxiliaryLists, PipelineId, Epoch, ScrollPolicy, ScrollLayerId};
use webrender_traits::{StackingContext, FilterOp, ImageFormat, MixBlendMode, StackingLevel};
use webrender_traits::{ScrollEventPhase, ScrollLayerInfo};

#[cfg(target_os = "macos")]
const CAN_OVERSCROLL: bool = true;

#[cfg(not(target_os = "macos"))]
const CAN_OVERSCROLL: bool = false;

use webrender_traits::{PipelineId, Epoch, ScrollPolicy, ScrollLayerId, StackingContext};
use webrender_traits::{FilterOp, ImageFormat, MixBlendMode, StackingLevel};
use webrender_traits::{SpecificDisplayItem, ColorF};
use renderer::BLUR_INFLATION_FACTOR;
//use resource_list::ResourceList;
use spatial_hash::{self, SpatialHash};
use batch_builder::BorderSideHelpers;
use tessellator;
use tessellator::BorderCornerTessellation;

#[derive(Copy, Clone, PartialEq, PartialOrd, Debug)]
pub struct FrameId(pub u32);

pub struct DrawListGroup {
    pub id: DrawListGroupId,

    // Together, these define the granularity that batches
    // can be created at. When compiling nodes, if either
    // the scroll layer or render target are different from
    // the current batch, it must be broken and a new batch started.
    // This automatically handles the case of CompositeBatch, because
    // for a composite batch to be present, the next draw list must be
    // in a different render target!
    pub scroll_layer_id: ScrollLayerId,
    //pub render_target_id: RenderTargetId,

    pub draw_list_ids: Vec<DrawListId>,
}

impl DrawListGroup {
    fn new(id: DrawListGroupId,
           scroll_layer_id: ScrollLayerId,
           /*render_target_id: RenderTargetId*/) -> DrawListGroup {
        DrawListGroup {
            id: id,
            scroll_layer_id: scroll_layer_id,
            //render_target_id: render_target_id,
            draw_list_ids: Vec::new(),
        }
    }

    fn can_add(&self,
               scroll_layer_id: ScrollLayerId,
               /*render_target_id: RenderTargetId*/) -> bool {
        let scroll_ok = scroll_layer_id == self.scroll_layer_id;
        let target_ok = true; //render_target_id == self.render_target_id;
        let size_ok = true; //self.draw_list_ids.len() < MAX_MATRICES_PER_BATCH;
        scroll_ok && target_ok && size_ok
    }

    fn push(&mut self, draw_list_id: DrawListId) {
        self.draw_list_ids.push(draw_list_id);
    }
}

struct FlattenContext<'a> {
    resource_cache: &'a mut ResourceCache,
    scene: &'a Scene,
    pipeline_sizes: &'a mut HashMap<PipelineId, Size2D<f32>>,
    current_draw_list_group: Option<DrawListGroup>,
    device_pixel_ratio: f32,
    spatial_hash: SpatialHash,
}

#[derive(Debug)]
struct FlattenInfo {
    viewport_size: Size2D<f32>,
    current_clip_rect: Rect<f32>,
    default_scroll_layer_id: ScrollLayerId,
    actual_scroll_layer_id: ScrollLayerId,
    fixed_scroll_layer_id: ScrollLayerId,
    offset_from_origin: Point2D<f32>,
    offset_from_current_layer: Point2D<f32>,
    transform: Matrix4,
    perspective: Matrix4,
}

/*
#[derive(Debug)]
pub enum FrameRenderItem {
    Clear(ClearInfo),
    CompositeBatch(CompositeBatchInfo),
    DrawListBatch(DrawListGroupId),
}

pub struct RenderTarget {
    id: RenderTargetId,
    size: Size2D<f32>,
    origin: Point2D<f32>,
    items: Vec<FrameRenderItem>,
    texture_id: Option<TextureId>,
    children: Vec<RenderTarget>,

    page_allocator: Option<TexturePage>,
    texture_id_list: Vec<TextureId>,
}

impl RenderTarget {
    fn new(id: RenderTargetId,
           origin: Point2D<f32>,
           size: Size2D<f32>,
           texture_id: Option<TextureId>) -> RenderTarget {
        RenderTarget {
            id: id,
            size: size,
            origin: origin,
            items: Vec::new(),
            texture_id: texture_id,
            children: Vec::new(),
            texture_id_list: Vec::new(),
            page_allocator: None,
        }
    }

    fn allocate_target_rect(&mut self,
                            width: f32,
                            height: f32,
                            device_pixel_ratio: f32,
                            resource_cache: &mut ResourceCache) -> (Point2D<f32>, TextureId) {
        // If the target is more than 512x512 (an arbitrary choice), assign it
        // to an exact sized render target - assuming that there probably aren't
        // many of them. This minimises GPU memory wastage if there are just a small
        // number of large targets. Otherwise, attempt to allocate inside a shared render
        // target texture - this allows composite batching to take place when
        // there are a lot of small targets (which is more efficient).
        if width < 512.0 && height < 512.0 {
            if self.page_allocator.is_none() {
                let texture_size = 2048;
                let device_pixel_size = texture_size * device_pixel_ratio as u32;

                let texture_id = resource_cache.allocate_render_target(device_pixel_size,
                                                                       device_pixel_size,
                                                                       ImageFormat::RGBA8);
                self.texture_id_list.push(texture_id);
                self.page_allocator = Some(TexturePage::new(texture_id, texture_size));
            }

            // TODO(gw): This has accuracy issues if the size of a rendertarget is
            //           not scene pixel aligned!
            let size = Size2D::new(width as u32, height as u32);
            let allocated_origin = self.page_allocator
                                       .as_mut()
                                       .unwrap()
                                       .allocate(&size, TextureFilter::Linear);
            if let Some(allocated_origin) = allocated_origin {
                let origin = Point2D::new(allocated_origin.x as f32,
                                          allocated_origin.y as f32);
                return (origin, self.page_allocator.as_ref().unwrap().texture_id())
            }
        }

        let device_pixel_width = width as u32 * device_pixel_ratio as u32;
        let device_pixel_height = height as u32 * device_pixel_ratio as u32;

        let texture_id = resource_cache.allocate_render_target(device_pixel_width,
                                                               device_pixel_height,
                                                               ImageFormat::RGBA8);
        self.texture_id_list.push(texture_id);

        (Point2D::zero(), texture_id)
    }

    fn collect_and_sort_visible_batches(&mut self,
                                        resource_cache: &mut ResourceCache,
                                        draw_list_groups: &HashMap<DrawListGroupId, DrawListGroup, BuildHasherDefault<FnvHasher>>,
                                        layers: &HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>,
                                        stacking_context_info: &[StackingContextInfo],
                                        device_pixel_ratio: f32) -> DrawLayer {
        let mut commands = vec![];
        for item in &self.items {
            match item {
                &FrameRenderItem::Clear(ref info) => {
                    commands.push(DrawCommand::Clear(info.clone()));
                }
                &FrameRenderItem::CompositeBatch(ref info) => {
                    commands.push(DrawCommand::CompositeBatch(info.clone()));
                }
                &FrameRenderItem::DrawListBatch(draw_list_group_id) => {
                    let draw_list_group = &draw_list_groups[&draw_list_group_id];
                    debug_assert!(draw_list_group.draw_list_ids.len() <= MAX_MATRICES_PER_BATCH);

                    let layer = &layers[&draw_list_group.scroll_layer_id];
                    let mut matrix_palette =
                        vec![Matrix4::identity(); draw_list_group.draw_list_ids.len()];
                    let mut offset_palette =
                        vec![OffsetParams::identity(); draw_list_group.draw_list_ids.len()];

                    // Update batch matrices
                    for (index, draw_list_id) in draw_list_group.draw_list_ids.iter().enumerate() {
                        let draw_list = resource_cache.get_draw_list(*draw_list_id);

                        let StackingContextIndex(stacking_context_id) = draw_list.stacking_context_index.unwrap();
                        let context = &stacking_context_info[stacking_context_id];

                        let transform = layer.world_transform.mul(&context.transform);
                        matrix_palette[index] = transform;

                        offset_palette[index].stacking_context_x0 = context.offset_from_layer.x;
                        offset_palette[index].stacking_context_y0 = context.offset_from_layer.y;
                    }

                    let mut batch_info = BatchInfo::new(matrix_palette, offset_palette);

                    // Collect relevant draws from each node in the tree.
                    for node in &layer.aabb_tree.nodes {
                        if node.is_visible {
                            debug_assert!(node.compiled_node.is_some());
                            let compiled_node = node.compiled_node.as_ref().unwrap();

                            let batch_list = compiled_node.batch_list.iter().find(|batch_list| {
                                batch_list.draw_list_group_id == draw_list_group_id
                            });

                            if let Some(batch_list) = batch_list {
                                let mut region = MaskRegion::new();

                                let vertex_buffer_id = compiled_node.vertex_buffer_id.unwrap();

                                // TODO(gw): Support mask regions for nested render targets
                                //           with transforms.
                                if self.texture_id.is_none() {
                                    // Mask out anything outside this AABB tree node.
                                    // This is a requirement to ensure paint order is correctly
                                    // maintained since the batches are built in parallel.
                                    region.add_mask(node.split_rect, layer.world_transform);

                                    // Mask out anything outside this viewport. This is used
                                    // for things like clipping content that is outside a
                                    // transformed iframe.
                                    region.add_mask(Rect::new(layer.world_origin, layer.viewport_size),
                                                    layer.local_transform);

                                }

                                for batch in &batch_list.batches {
                                    region.draw_calls.push(DrawCall {
                                        tile_params: batch.tile_params.clone(),     // TODO(gw): Move this instead?
                                        clip_rects: batch.clip_rects.clone(),
                                        vertex_buffer_id: vertex_buffer_id,
                                        color_texture_id: batch.color_texture_id,
                                        mask_texture_id: batch.mask_texture_id,
                                        first_instance: batch.first_instance,
                                        instance_count: batch.instance_count,
                                    });
                                }

                                batch_info.regions.push(region);
                            }
                        }
                    }

                    // Finally, add the batch + draw calls
                    commands.push(DrawCommand::Batch(batch_info));
                }
            }
        }

        let mut child_layers = Vec::new();

        for child in &mut self.children {
            let child_layer = child.collect_and_sort_visible_batches(resource_cache,
                                                                     draw_list_groups,
                                                                     layers,
                                                                     stacking_context_info,
                                                                     device_pixel_ratio);

            child_layers.push(child_layer);
        }

        DrawLayer::new(self.origin,
                       self.size,
                       self.texture_id,
                       commands,
                       child_layers)
    }

    fn reset(&mut self,
             pending_updates: &mut BatchUpdateList,
             resource_cache: &mut ResourceCache) {
        self.texture_id_list.clear();
        resource_cache.free_old_render_targets();

        for mut child in &mut self.children.drain(..) {
            child.reset(pending_updates,
                        resource_cache);
        }

        self.items.clear();
        self.page_allocator = None;
    }

    fn push_clear(&mut self, clear_info: ClearInfo) {
        self.items.push(FrameRenderItem::Clear(clear_info));
    }

    fn push_composite(&mut self,
                      op: CompositionOp,
                      texture_id: TextureId,
                      target: Rect<f32>,
                      transform: &Matrix4,
                      child_layer_index: ChildLayerIndex) {
        // TODO(gw): Relax the restriction on batch breaks for FB reads
        //           once the proper render target allocation code is done!
        let need_new_batch = op.needs_framebuffer() || match self.items.last() {
            Some(&FrameRenderItem::CompositeBatch(ref info)) => {
                info.operation != op || info.texture_id != texture_id
            }
            Some(&FrameRenderItem::Clear(..)) |
            Some(&FrameRenderItem::DrawListBatch(..)) |
            None => {
                true
            }
        };

        if need_new_batch {
            self.items.push(FrameRenderItem::CompositeBatch(CompositeBatchInfo {
                operation: op,
                texture_id: texture_id,
                jobs: Vec::new(),
            }));
        }

        // TODO(gw): This seems a little messy - restructure how current batch works!
        match self.items.last_mut().unwrap() {
            &mut FrameRenderItem::CompositeBatch(ref mut batch) => {
                let job = CompositeBatchJob {
                    rect: target,
                    transform: *transform,
                    child_layer_index: child_layer_index,
                };
                batch.jobs.push(job);
            }
            _ => {
                unreachable!();
            }
        }
    }

    fn push_draw_list_group(&mut self, draw_list_group_id: DrawListGroupId) {
        self.items.push(FrameRenderItem::DrawListBatch(draw_list_group_id));
    }
}*/

pub struct Frame {
    //pub layers: HashMap<ScrollLayerId, Layer, BuildHasherDefault<FnvHasher>>,
    pub pipeline_epoch_map: HashMap<PipelineId, Epoch, BuildHasherDefault<FnvHasher>>,
    pub pipeline_auxiliary_lists: HashMap<PipelineId,
                                          AuxiliaryLists,
                                          BuildHasherDefault<FnvHasher>>,
    pub pending_updates: BatchUpdateList,
    //pub root: Option<RenderTarget>,
    pub stacking_context_info: Vec<StackingContextInfo>,
    //next_render_target_id: RenderTargetId,
    //next_draw_list_group_id: DrawListGroupId,
    //draw_list_groups: HashMap<DrawListGroupId, DrawListGroup, BuildHasherDefault<FnvHasher>>,
    root_scroll_layer_id: Option<ScrollLayerId>,
    id: FrameId,

    spatial_hash: Option<SpatialHash>,
}

enum SceneItemKind<'a> {
    StackingContext(&'a SceneStackingContext, PipelineId),
    Pipeline(&'a ScenePipeline)
}

#[derive(Clone)]
struct SceneItemWithZOrder {
    item: SceneItem,
    z_index: i32,
}

impl<'a> SceneItemKind<'a> {
    fn collect_scene_items(&self, scene: &Scene) -> Vec<SceneItem> {
        let mut background_and_borders = Vec::new();
        let mut positioned_content = Vec::new();
        let mut block_background_and_borders = Vec::new();
        let mut floats = Vec::new();
        let mut content = Vec::new();
        let mut outlines = Vec::new();

        let stacking_context = match *self {
            SceneItemKind::StackingContext(stacking_context, _) => {
                &stacking_context.stacking_context
            }
            SceneItemKind::Pipeline(pipeline) => {
                if let Some(background_draw_list) = pipeline.background_draw_list {
                    background_and_borders.push(SceneItem {
                        stacking_level: StackingLevel::BackgroundAndBorders,
                        specific: SpecificSceneItem::DrawList(background_draw_list),
                    });
                }

                &scene.stacking_context_map
                      .get(&pipeline.root_stacking_context_id)
                      .unwrap()
                      .stacking_context
            }
        };

        for display_list_id in &stacking_context.display_lists {
            let display_list = &scene.display_list_map[display_list_id];
            for item in &display_list.items {
                match item.stacking_level {
                    StackingLevel::BackgroundAndBorders => {
                        background_and_borders.push(item.clone());
                    }
                    StackingLevel::BlockBackgroundAndBorders => {
                        block_background_and_borders.push(item.clone());
                    }
                    StackingLevel::PositionedContent => {
                        let z_index = match item.specific {
                            SpecificSceneItem::StackingContext(id, _) => {
                                scene.stacking_context_map
                                     .get(&id)
                                     .unwrap()
                                     .stacking_context
                                     .z_index
                            }
                            SpecificSceneItem::DrawList(..) |
                            SpecificSceneItem::Iframe(..) => {
                                // TODO(gw): Probably wrong for an iframe?
                                0
                            }
                        };

                        positioned_content.push(SceneItemWithZOrder {
                            item: item.clone(),
                            z_index: z_index,
                        });
                    }
                    StackingLevel::Floats => {
                        floats.push(item.clone());
                    }
                    StackingLevel::Content => {
                        content.push(item.clone());
                    }
                    StackingLevel::Outlines => {
                        outlines.push(item.clone());
                    }
                }
            }
        }

        positioned_content.sort_by(|a, b| {
            a.z_index.cmp(&b.z_index)
        });

        let mut result = Vec::new();
        result.extend(background_and_borders);
        result.extend(positioned_content.iter().filter_map(|item| {
            if item.z_index < 0 {
                Some(item.item.clone())
            } else {
                None
            }
        }));
        result.extend(block_background_and_borders);
        result.extend(floats);
        result.extend(content);
        result.extend(positioned_content.iter().filter_map(|item| {
            if item.z_index < 0 {
                None
            } else {
                Some(item.item.clone())
            }
        }));
        result.extend(outlines);
        result
    }
}

trait StackingContextHelpers {
    fn needs_composition_operation_for_mix_blend_mode(&self) -> bool;
    fn composition_operations(&self, auxiliary_lists: &AuxiliaryLists) -> Vec<CompositionOp>;
}

impl StackingContextHelpers for StackingContext {
    fn needs_composition_operation_for_mix_blend_mode(&self) -> bool {
        match self.mix_blend_mode {
            MixBlendMode::Normal => false,
            MixBlendMode::Multiply |
            MixBlendMode::Screen |
            MixBlendMode::Overlay |
            MixBlendMode::Darken |
            MixBlendMode::Lighten |
            MixBlendMode::ColorDodge |
            MixBlendMode::ColorBurn |
            MixBlendMode::HardLight |
            MixBlendMode::SoftLight |
            MixBlendMode::Difference |
            MixBlendMode::Exclusion |
            MixBlendMode::Hue |
            MixBlendMode::Saturation |
            MixBlendMode::Color |
            MixBlendMode::Luminosity => true,
        }
    }

    fn composition_operations(&self, auxiliary_lists: &AuxiliaryLists) -> Vec<CompositionOp> {
        let mut composition_operations = vec![];
        if self.needs_composition_operation_for_mix_blend_mode() {
            composition_operations.push(CompositionOp::MixBlend(self.mix_blend_mode));
        }
        for filter in auxiliary_lists.filters(&self.filters) {
            match *filter {
                FilterOp::Blur(radius) => {
                    composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Blur(
                        radius,
                        AxisDirection::Horizontal)));
                    composition_operations.push(CompositionOp::Filter(LowLevelFilterOp::Blur(
                        radius,
                        AxisDirection::Vertical)));
                }
                FilterOp::Brightness(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Brightness(Au::from_f32_px(amount))));
                }
                FilterOp::Contrast(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Contrast(Au::from_f32_px(amount))));
                }
                FilterOp::Grayscale(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Grayscale(Au::from_f32_px(amount))));
                }
                FilterOp::HueRotate(angle) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::HueRotate(f32::round(
                                    angle * ANGLE_FLOAT_TO_FIXED) as i32)));
                }
                FilterOp::Invert(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Invert(Au::from_f32_px(amount))));
                }
                FilterOp::Opacity(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Opacity(Au::from_f32_px(amount))));
                }
                FilterOp::Saturate(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Saturate(Au::from_f32_px(amount))));
                }
                FilterOp::Sepia(amount) => {
                    composition_operations.push(CompositionOp::Filter(
                            LowLevelFilterOp::Sepia(Au::from_f32_px(amount))));
                }
            }
        }

        composition_operations
    }
}

impl Frame {
    pub fn new() -> Frame {
        Frame {
            pipeline_epoch_map: HashMap::with_hasher(Default::default()),
            pending_updates: BatchUpdateList::new(),
            pipeline_auxiliary_lists: HashMap::with_hasher(Default::default()),
            //root: None,
            //layers: HashMap::with_hasher(Default::default()),
            stacking_context_info: Vec::new(),
            //next_render_target_id: RenderTargetId(0),
            //next_draw_list_group_id: DrawListGroupId(0),
            //draw_list_groups: HashMap::with_hasher(Default::default()),
            root_scroll_layer_id: None,
            id: FrameId(0),

            //packed_scene: None,
            spatial_hash: None,
        }
    }

    pub fn reset(&mut self, resource_cache: &mut ResourceCache)
                 -> HashMap<ScrollLayerId, ScrollingState, BuildHasherDefault<FnvHasher>> {
        //self.draw_list_groups.clear();
        self.pipeline_epoch_map.clear();
        //self.packed_scene = None;
        //self.next_display_item_id = 0;
        self.stacking_context_info.clear();

        //if let Some(mut root) = self.root.take() {
        //    root.reset(&mut self.pending_updates, resource_cache);
        //}

        // Free any render targets from last frame.
        // TODO: This should really re-use existing targets here...
        /*
        let mut old_layer_scrolling_states = HashMap::with_hasher(Default::default());
        for (layer_id, mut old_layer) in &mut self.layers.drain() {
            old_layer.reset(&mut self.pending_updates);
            old_layer_scrolling_states.insert(layer_id, old_layer.scrolling);
        }
        */

        // Advance to the next frame.
        self.id.0 += 1;

        old_layer_scrolling_states
    }

/*
    fn next_render_target_id(&mut self) -> RenderTargetId {
        let RenderTargetId(render_target_id) = self.next_render_target_id;
        self.next_render_target_id = RenderTargetId(render_target_id + 1);
        RenderTargetId(render_target_id)
    }
*/

/*
    fn next_draw_list_group_id(&mut self) -> DrawListGroupId {
        let DrawListGroupId(draw_list_group_id) = self.next_draw_list_group_id;
        self.next_draw_list_group_id = DrawListGroupId(draw_list_group_id + 1);
        DrawListGroupId(draw_list_group_id)
    }
*/

    pub fn pending_updates(&mut self) -> BatchUpdateList {
        mem::replace(&mut self.pending_updates, BatchUpdateList::new())
    }

/*
    pub fn get_scroll_layer(&self,
                            cursor: &Point2D<f32>,
                            scroll_layer_id: ScrollLayerId,
                            parent_transform: &Matrix4) -> Option<ScrollLayerId> {
        self.layers.get(&scroll_layer_id).and_then(|layer| {
            let transform = parent_transform.mul(&layer.local_transform);

            for child_layer_id in &layer.children {
                if let Some(layer_id) = self.get_scroll_layer(cursor,
                                                              *child_layer_id,
                                                              &transform) {
                    return Some(layer_id);
                }
            }

            match scroll_layer_id.info {
                ScrollLayerInfo::Fixed => {
                    None
                }
                ScrollLayerInfo::Scrollable(..) => {
                    let inv = transform.invert();
                    let z0 = -10000.0;
                    let z1 =  10000.0;

                    let p0 = inv.transform_point4d(&Point4D::new(cursor.x, cursor.y, z0, 1.0));
                    let p0 = Point3D::new(p0.x / p0.w,
                                          p0.y / p0.w,
                                          p0.z / p0.w);
                    let p1 = inv.transform_point4d(&Point4D::new(cursor.x, cursor.y, z1, 1.0));
                    let p1 = Point3D::new(p1.x / p1.w,
                                          p1.y / p1.w,
                                          p1.z / p1.w);

                    let layer_rect = Rect::new(layer.world_origin, layer.viewport_size);

                    if ray_intersects_rect(p0, p1, layer_rect) {
                        Some(scroll_layer_id)
                    } else {
                        None
                    }
                }
            }
        })
    }
*/

    pub fn scroll(&mut self,
                  mut delta: Point2D<f32>,
                  cursor: Point2D<f32>,
                  phase: ScrollEventPhase) {
        let root_scroll_layer_id = match self.root_scroll_layer_id {
            Some(root_scroll_layer_id) => root_scroll_layer_id,
            None => return,
        };

        panic!("todo");
        /*
        let scroll_layer_id = match self.get_scroll_layer(&cursor,
                                                          root_scroll_layer_id,
                                                          &Matrix4::identity()) {
            Some(scroll_layer_id) => scroll_layer_id,
            None => return,
        };

        let layer = self.layers.get_mut(&scroll_layer_id).unwrap();
        if layer.scrolling.started_bouncing_back && phase == ScrollEventPhase::Move(false) {
            return
        }

        let overscroll_amount = layer.overscroll_amount();
        let overscrolling = overscroll_amount.width != 0.0 || overscroll_amount.height != 0.0;
        if overscrolling {
            if overscroll_amount.width != 0.0 {
                delta.x /= overscroll_amount.width.abs()
            }
            if overscroll_amount.height != 0.0 {
                delta.y /= overscroll_amount.height.abs()
            }
        }

        let is_unscrollable = layer.layer_size.width <= layer.viewport_size.width &&
            layer.layer_size.height <= layer.viewport_size.height;

        if layer.layer_size.width > layer.viewport_size.width {
            layer.scrolling.offset.x = layer.scrolling.offset.x + delta.x;
            if is_unscrollable || !CAN_OVERSCROLL {
                layer.scrolling.offset.x = layer.scrolling.offset.x.min(0.0);
                layer.scrolling.offset.x = layer.scrolling.offset.x.max(-layer.layer_size.width +
                                                                        layer.viewport_size.width);
            }
        }

        if layer.layer_size.height > layer.viewport_size.height {
            layer.scrolling.offset.y = layer.scrolling.offset.y + delta.y;
            if is_unscrollable || !CAN_OVERSCROLL {
                layer.scrolling.offset.y = layer.scrolling.offset.y.min(0.0);
                layer.scrolling.offset.y =
                    layer.scrolling.offset.y.max(-layer.layer_size.height +
                                                 layer.viewport_size.height);
            }
        }

        if phase == ScrollEventPhase::Start || phase == ScrollEventPhase::Move(true) {
            layer.scrolling.started_bouncing_back = false
        } else if overscrolling &&
                ((delta.x < 1.0 && delta.y < 1.0) || phase == ScrollEventPhase::End) {
            layer.scrolling.started_bouncing_back = true
        }

        layer.scrolling.offset.x = layer.scrolling.offset.x.round();
        layer.scrolling.offset.y = layer.scrolling.offset.y.round();

        layer.stretch_overscroll_spring();
    }

    pub fn tick_scrolling_bounce_animations(&mut self) {
        for (_, layer) in &mut self.layers {
            layer.tick_scrolling_bounce_animation()
        }*/
    }

    pub fn create(&mut self,
                  scene: &Scene,
                  resource_cache: &mut ResourceCache,
                  pipeline_sizes: &mut HashMap<PipelineId, Size2D<f32>>,
                  device_pixel_ratio: f32) {
        if let Some(root_pipeline_id) = scene.root_pipeline_id {
            if let Some(root_pipeline) = scene.pipeline_map.get(&root_pipeline_id) {
                let old_layer_scrolling_states = self.reset(resource_cache);

                self.pipeline_auxiliary_lists = scene.pipeline_auxiliary_lists.clone();

                let root_stacking_context = scene.stacking_context_map
                                                 .get(&root_pipeline.root_stacking_context_id)
                                                 .unwrap();

                let root_scroll_layer_id = root_stacking_context.stacking_context
                                                                .scroll_layer_id
                                                                .expect("root layer must be a scroll layer!");
                self.root_scroll_layer_id = Some(root_scroll_layer_id);

/*
                let root_target_id = self.next_render_target_id();

                let mut root_target = RenderTarget::new(root_target_id,
                                                        Point2D::zero(),
                                                        root_pipeline.viewport_size,
                                                        None);
*/

                // Insert global position: fixed elements layer
/*
                debug_assert!(self.layers.is_empty());
                let root_fixed_layer_id = ScrollLayerId::create_fixed(root_pipeline_id);
                self.layers.insert(root_fixed_layer_id,
                                   Layer::new(root_stacking_context.stacking_context.overflow.origin,
                                              root_stacking_context.stacking_context.overflow.size,
                                              root_pipeline.viewport_size,
                                              Matrix4::identity()));
*/

                // Work around borrow check on resource cache
                {
                    let mut context = FlattenContext {
                        resource_cache: resource_cache,
                        scene: scene,
                        pipeline_sizes: pipeline_sizes,
                        current_draw_list_group: None,
                        device_pixel_ratio: device_pixel_ratio,
                        //packed_scene: PackedScene::new(),
                        spatial_hash: SpatialHash::new(Size2D::new(800, 600), Size2D::new(80, 60)),
                    };

                    let parent_info = FlattenInfo {
                        viewport_size: root_pipeline.viewport_size,
                        offset_from_origin: Point2D::zero(),
                        offset_from_current_layer: Point2D::zero(),
                        default_scroll_layer_id: root_scroll_layer_id,
                        actual_scroll_layer_id: root_scroll_layer_id,
                        fixed_scroll_layer_id: root_fixed_layer_id,
                        current_clip_rect: MAX_RECT,
                        transform: Matrix4::identity(),
                        perspective: Matrix4::identity(),
                    };

                    let root_pipeline = SceneItemKind::Pipeline(root_pipeline);
                    self.flatten(root_pipeline,
                                 &parent_info,
                                 &mut context,
                                 //&mut root_target,
                                 0);
                    //self.packed_scene = Some(context.packed_scene);
                    self.spatial_hash = Some(context.spatial_hash);
                    //self.root = Some(root_target);

/*
                    if let Some(last_draw_list_group) = context.current_draw_list_group.take() {
                        self.draw_list_groups.insert(last_draw_list_group.id,
                                                     last_draw_list_group);
                    }*/
                }

                // TODO(gw): These are all independent - can be run through thread pool if it shows up in the profile!
                /*
                for (scroll_layer_id, layer) in &mut self.layers {
                    let scrolling_state = match old_layer_scrolling_states.get(&scroll_layer_id) {
                        Some(old_scrolling_state) => *old_scrolling_state,
                        None => ScrollingState::new(),
                    };

                    layer.finalize(&scrolling_state);
                }*/
            }
        }
    }

    fn add_items_to_target(&mut self,
                           scene_items: &[SceneItem],
                           info: &FlattenInfo,
                           context: &mut FlattenContext,
                           _level: i32) {
        let stacking_context_index = StackingContextIndex(self.stacking_context_info.len());
        self.stacking_context_info.push(StackingContextInfo {
            offset_from_layer: info.offset_from_current_layer,
            local_clip_rect: info.current_clip_rect,
            transform: info.transform,
            perspective: info.perspective,
        });

        // hack hack hack!
        for item in scene_items {
            match item.specific {
                SpecificSceneItem::DrawList(draw_list_id) => {
                    let draw_list = context.resource_cache.get_draw_list(draw_list_id);

                    //let layer = self.layers.get_mut(&info.actual_scroll_layer_id).unwrap();
                    for (item_index, item) in draw_list.items.iter().enumerate() {
                        let item_index = DrawListItemIndex(item_index as u32);
                        let rect = item.rect
                                       .translate(&info.offset_from_current_layer);

                        let v_tl = info.transform.transform_point4d(&Point4D::new(rect.origin.x,
                                                                                  rect.origin.y,
                                                                                  0.0,
                                                                                  1.0));

                        let v_tr = info.transform.transform_point4d(&Point4D::new(rect.top_right().x,
                                                                                  rect.top_right().y,
                                                                                  0.0,
                                                                                  1.0));

                        let v_bl = info.transform.transform_point4d(&Point4D::new(rect.bottom_left().x,
                                                                                  rect.bottom_left().y,
                                                                                  0.0,
                                                                                  1.0));

                        let v_br = info.transform.transform_point4d(&Point4D::new(rect.bottom_right().x,
                                                                                  rect.bottom_right().y,
                                                                                  0.0,
                                                                                  1.0));

                        match item.item {
                            SpecificDisplayItem::WebGL(ref info) => {
                                panic!("todo");
                            }
                            SpecificDisplayItem::Image(ref image_info) => {
                                let image_info = context.resource_cache.get_image(image_info.image_key,
                                                                                  image_info.image_rendering,
                                                                                  self.id);

                                let color = ColorF::new(1.0, 1.0, 1.0, 1.0);

                                let vertices = [
                                    PackedSceneVertex::new(&v_tl, &color, &image_info.uv_rect.top_left),
                                    PackedSceneVertex::new(&v_bl, &color, &image_info.uv_rect.bottom_left),
                                    PackedSceneVertex::new(&v_br, &color, &image_info.uv_rect.bottom_right),
                                    PackedSceneVertex::new(&v_tr, &color, &image_info.uv_rect.top_right),
                                ];

                                context.spatial_hash.add_color_rectangle(&rect,
                                                                         &color,
                                                                         &info.transform,
                                                                         &Size2D::zero(),
                                                                         &Size2D::zero(),
                                                                         &Point2D::zero());
                            }
                            SpecificDisplayItem::Text(ref text_info) => {
                                let mut glyph_key = GlyphKey::new(text_info.font_key,
                                                                  text_info.size,
                                                                  text_info.blur_radius,
                                                                  text_info.glyphs[0].index);
                                let blur_offset = text_info.blur_radius.to_f32_px() * (BLUR_INFLATION_FACTOR as f32) / 2.0;

                                for glyph in &text_info.glyphs {
                                    glyph_key.index = glyph.index;
                                    let image_info = context.resource_cache.get_glyph(&glyph_key, self.id);
                                    if let Some(image_info) = image_info {
                                        let x = glyph.x + image_info.user_data.x0 as f32 / context.device_pixel_ratio - blur_offset;
                                        let y = glyph.y - image_info.user_data.y0 as f32 / context.device_pixel_ratio - blur_offset;

                                        let width = image_info.requested_rect.size.width as f32 / context.device_pixel_ratio;
                                        let height = image_info.requested_rect.size.height as f32 / context.device_pixel_ratio;

                                        let v_tl = info.transform.transform_point4d(&Point4D::new(x,
                                                                                                  y,
                                                                                                  0.0,
                                                                                                  1.0));

                                        let v_tr = info.transform.transform_point4d(&Point4D::new(x + width,
                                                                                                  y,
                                                                                                  0.0,
                                                                                                  1.0));

                                        let v_bl = info.transform.transform_point4d(&Point4D::new(x,
                                                                                                  y + height,
                                                                                                  0.0,
                                                                                                  1.0));

                                        let v_br = info.transform.transform_point4d(&Point4D::new(x + width,
                                                                                                  y + height,
                                                                                                  0.0,
                                                                                                  1.0));

                                        let vertices = [
                                            PackedSceneVertex::new(&v_tl, &text_info.color, &image_info.uv_rect.top_left),
                                            PackedSceneVertex::new(&v_bl, &text_info.color, &image_info.uv_rect.bottom_left),
                                            PackedSceneVertex::new(&v_br, &text_info.color, &image_info.uv_rect.bottom_right),
                                            PackedSceneVertex::new(&v_tr, &text_info.color, &image_info.uv_rect.top_right),
                                        ];

                                        context.spatial_hash.add_color_rectangle(&Rect::new(Point2D::new(x, y), Size2D::new(width, height)),
                                                                                 &text_info.color,
                                                                                 &info.transform,
                                                                                 &Size2D::zero(),
                                                                                 &Size2D::zero(),
                                                                                 &Point2D::zero());
                                    }
                                }
                            }
                            SpecificDisplayItem::Rectangle(ref rect_info) => {
                                //println!("Rectangle {:?}", rect_info);
                                let image_info = context.resource_cache.get_dummy_color_image();

                                let vertices = [
                                    PackedSceneVertex::new(&v_tl, &rect_info.color, &image_info.uv_rect.top_left),
                                    PackedSceneVertex::new(&v_bl, &rect_info.color, &image_info.uv_rect.bottom_left),
                                    PackedSceneVertex::new(&v_br, &rect_info.color, &image_info.uv_rect.bottom_right),
                                    PackedSceneVertex::new(&v_tr, &rect_info.color, &image_info.uv_rect.top_right),
                                ];

                                context.spatial_hash.add_color_rectangle(&rect,
                                                                         &rect_info.color,
                                                                         &info.transform,
                                                                         &Size2D::zero(),
                                                                         &Size2D::zero(),
                                                                         &Point2D::zero());
                            }
                            SpecificDisplayItem::Gradient(ref info) => {
                                panic!("todo");
                            }
                            SpecificDisplayItem::BoxShadow(ref info) => {
                                panic!("todo");
                            }
                            SpecificDisplayItem::Border(ref  border_info) => {
                                let radius = &border_info.radius;
                                let left = &border_info.left;
                                let right = &border_info.right;
                                let top = &border_info.top;
                                let bottom = &border_info.bottom;

                                //println!("r = {:?} {:?} {:?}", rect, radius, left);

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

                                let r0 = Rect::new(tl_outer,
                                                   Size2D::new(tl_inner.x - tl_outer.x, tl_inner.y - tl_outer.y));

                                let r1 = Rect::new(Point2D::new(bl_outer.x, bl_inner.y),
                                                   Size2D::new(bl_inner.x - bl_outer.x, bl_outer.y - bl_inner.y));

                                let r2 = Rect::new(br_inner,
                                                   Size2D::new(br_outer.x - br_inner.x, br_outer.y - br_inner.y));

                                let r3 = Rect::new(Point2D::new(tr_inner.x, tr_outer.y),
                                                   Size2D::new(tr_outer.x - tr_inner.x, tr_inner.y - tr_outer.y));

                                for rect_index in 0..tessellator::quad_count_for_border_corner(&radius.top_left,
                                                                                               context.device_pixel_ratio) {
                                    let tessellated_rect = r0.tessellate_border_corner(&radius.top_left,
                                                                                       &border_info.top_left_inner_radius(),
                                                                                        context.device_pixel_ratio,
                                                                                        BasicRotationAngle::Upright,
                                                                                        rect_index);
                                    context.spatial_hash.add_color_rectangle(&tessellated_rect,
                                                                             &left.color,
                                                                             &info.transform,
                                                                             &radius.top_left,
                                                                             &border_info.top_left_inner_radius(),
                                                                             &tl_inner);
                                }

                                for rect_index in 0..tessellator::quad_count_for_border_corner(&radius.bottom_left,
                                                                                               context.device_pixel_ratio) {
                                    let tessellated_rect = r1.tessellate_border_corner(&radius.bottom_left,
                                                                                       &border_info.bottom_left_inner_radius(),
                                                                                        context.device_pixel_ratio,
                                                                                        BasicRotationAngle::Clockwise270,
                                                                                        rect_index);
                                    context.spatial_hash.add_color_rectangle(&tessellated_rect,
                                                                             &top.color,
                                                                             &info.transform,
                                                                             &radius.bottom_left,
                                                                             &border_info.bottom_left_inner_radius(),
                                                                             &bl_inner);
                                }

                                for rect_index in 0..tessellator::quad_count_for_border_corner(&radius.bottom_right,
                                                                                               context.device_pixel_ratio) {
                                    let tessellated_rect = r2.tessellate_border_corner(&radius.bottom_right,
                                                                                       &border_info.bottom_right_inner_radius(),
                                                                                        context.device_pixel_ratio,
                                                                                        BasicRotationAngle::Clockwise180,
                                                                                        rect_index);
                                    context.spatial_hash.add_color_rectangle(&tessellated_rect,
                                                                             &right.color,
                                                                             &info.transform,
                                                                             &radius.bottom_right,
                                                                             &border_info.bottom_right_inner_radius(),
                                                                             &br_inner);
                                }

                                for rect_index in 0..tessellator::quad_count_for_border_corner(&radius.top_right,
                                                                                               context.device_pixel_ratio) {
                                    let tessellated_rect = r3.tessellate_border_corner(&radius.top_right,
                                                                                       &border_info.top_right_inner_radius(),
                                                                                        context.device_pixel_ratio,
                                                                                        BasicRotationAngle::Clockwise90,
                                                                                        rect_index);
                                    context.spatial_hash.add_color_rectangle(&tessellated_rect,
                                                                             &bottom.color,
                                                                             &info.transform,
                                                                             &radius.top_right,
                                                                             &border_info.top_right_inner_radius(),
                                                                             &tr_inner);
                                }
                            }
                        }
                    }

                    // Store draw context
                    //draw_list.stacking_context_index = Some(stacking_context_index);

                    /*
                    let needs_new_draw_group = match context.current_draw_list_group {
                        Some(ref draw_list_group) => {
                            !draw_list_group.can_add(info.actual_scroll_layer_id,
                                                     /*target.id*/)
                        }
                        None => {
                            true
                        }
                    };

                    if needs_new_draw_group {
                        if let Some(draw_list_group) = context.current_draw_list_group.take() {
                            self.draw_list_groups.insert(draw_list_group.id,
                                                         draw_list_group);
                        }

                        let draw_list_group_id = self.next_draw_list_group_id();

                        let new_draw_list_group = DrawListGroup::new(draw_list_group_id,
                                                                     info.actual_scroll_layer_id,
                                                                     /*target.id*/);

                        panic!("todo");
                        //target.push_draw_list_group(draw_list_group_id);

                        context.current_draw_list_group = Some(new_draw_list_group);
                    }

                    context.current_draw_list_group.as_mut().unwrap().push(draw_list_id);

                    let draw_list_group_id = context.current_draw_list_group.as_ref().unwrap().id;
                    let layer = self.layers.get_mut(&info.actual_scroll_layer_id).unwrap();
                    for (item_index, item) in draw_list.items.iter().enumerate() {
                        let item_index = DrawListItemIndex(item_index as u32);
                        let rect = item.rect
                                       .translate(&info.offset_from_current_layer);
                        layer.insert(rect,
                                     draw_list_group_id,
                                     draw_list_id,
                                     item_index);
                    }
                    */
                }
                SpecificSceneItem::StackingContext(id, pipeline_id) => {
                    let stacking_context = context.scene
                                                  .stacking_context_map
                                                  .get(&id)
                                                  .unwrap();

                    let child = SceneItemKind::StackingContext(stacking_context, pipeline_id);
                    self.flatten(child,
                                 info,
                                 context,
                                 //target,
                                 _level+1);
                }
                SpecificSceneItem::Iframe(ref iframe_info) => {
                    panic!("todo");
                    /*
                    let pipeline = context.scene
                                          .pipeline_map
                                          .get(&iframe_info.id);

                    context.pipeline_sizes.insert(iframe_info.id,
                                                  iframe_info.bounds.size);

                    if let Some(pipeline) = pipeline {
                        let iframe = SceneItemKind::Pipeline(pipeline);

                        let iframe_fixed_layer_id = ScrollLayerId::create_fixed(pipeline.pipeline_id);

                        let iframe_info = FlattenInfo {
                            viewport_size: iframe_info.bounds.size,
                            offset_from_origin: info.offset_from_origin + iframe_info.bounds.origin,
                            offset_from_current_layer: info.offset_from_current_layer + iframe_info.bounds.origin,
                            default_scroll_layer_id: info.default_scroll_layer_id,
                            actual_scroll_layer_id: info.actual_scroll_layer_id,
                            fixed_scroll_layer_id: iframe_fixed_layer_id,
                            current_clip_rect: MAX_RECT,
                            transform: info.transform,
                            perspective: info.perspective,
                        };

                        let iframe_stacking_context = context.scene
                                                             .stacking_context_map
                                                             .get(&pipeline.root_stacking_context_id)
                                                             .unwrap();

                        let layer_origin = iframe_stacking_context.stacking_context.overflow.origin +
                                            iframe_info.offset_from_current_layer;
                        let layer_size = iframe_stacking_context.stacking_context.overflow.size;

                        self.layers.insert(iframe_fixed_layer_id,
                                           Layer::new(layer_origin,
                                                      layer_size,
                                                      iframe_info.viewport_size,
                                                      iframe_info.transform));

                        self.flatten(iframe,
                                     &iframe_info,
                                     context,
                                     target,
                                     _level+1);
                    }
                    */
                }
            }
        }
    }

    fn flatten(&mut self,
               scene_item: SceneItemKind,
               parent_info: &FlattenInfo,
               context: &mut FlattenContext,
               //target: &mut RenderTarget,
               level: i32) {
        let _pf = util::ProfileScope::new("  flatten");

        let (stacking_context, local_clip_rect, pipeline_id) = match scene_item {
            SceneItemKind::StackingContext(stacking_context, pipeline_id) => {
                let stacking_context = &stacking_context.stacking_context;

                let local_clip_rect = parent_info.current_clip_rect
                                                 .translate(&-stacking_context.bounds.origin)
                                                 .intersection(&stacking_context.overflow);

                (stacking_context, local_clip_rect, pipeline_id)
            }
            SceneItemKind::Pipeline(pipeline) => {
                self.pipeline_epoch_map.insert(pipeline.pipeline_id, pipeline.epoch);

                let stacking_context = &context.scene.stacking_context_map
                                               .get(&pipeline.root_stacking_context_id)
                                               .unwrap()
                                               .stacking_context;

                (stacking_context, Some(MAX_RECT), pipeline.pipeline_id)
            }
        };

        if let Some(local_clip_rect) = local_clip_rect {
            let scene_items = scene_item.collect_scene_items(&context.scene);
            if !scene_items.is_empty() {
                let composition_operations = {
                    let auxiliary_lists = self.pipeline_auxiliary_lists
                                              .get(&pipeline_id)
                                              .expect("No auxiliary lists?!");
                    stacking_context.composition_operations(auxiliary_lists)
                };

                // Build world space transform
                let origin = parent_info.offset_from_current_layer + stacking_context.bounds.origin;
                let local_transform = if composition_operations.is_empty() {
                    Matrix4::identity().translate(origin.x, origin.y, 0.0)
                                       .mul(&stacking_context.transform)
                                       .translate(-origin.x, -origin.y, 0.0)
                } else {
                    Matrix4::identity()
                };

                let transform = parent_info.perspective.mul(&parent_info.transform)
                                                       .mul(&local_transform);

                // Build world space perspective transform
                let perspective = Matrix4::identity().translate(origin.x, origin.y, 0.0)
                                                     .mul(&stacking_context.perspective)
                                                     .translate(-origin.x, -origin.y, 0.0);

                let mut info = FlattenInfo {
                    viewport_size: parent_info.viewport_size,
                    offset_from_origin: parent_info.offset_from_origin + stacking_context.bounds.origin,
                    offset_from_current_layer: parent_info.offset_from_current_layer + stacking_context.bounds.origin,
                    default_scroll_layer_id: parent_info.default_scroll_layer_id,
                    actual_scroll_layer_id: parent_info.default_scroll_layer_id,
                    fixed_scroll_layer_id: parent_info.fixed_scroll_layer_id,
                    current_clip_rect: local_clip_rect,
                    transform: transform,
                    perspective: perspective,
                };

                match (stacking_context.scroll_policy, stacking_context.scroll_layer_id) {
                    (ScrollPolicy::Fixed, _scroll_layer_id) => {
                        debug_assert!(_scroll_layer_id.is_none());
                        info.actual_scroll_layer_id = info.fixed_scroll_layer_id;
                    }
                    (ScrollPolicy::Scrollable, Some(scroll_layer_id)) => {
                        /*
                        debug_assert!(!self.layers.contains_key(&scroll_layer_id));
                        let layer = Layer::new(parent_info.offset_from_origin,
                                               stacking_context.overflow.size,
                                               parent_info.viewport_size,
                                               transform);
                        if parent_info.actual_scroll_layer_id != scroll_layer_id {
                            self.layers.get_mut(&parent_info.actual_scroll_layer_id).unwrap().add_child(scroll_layer_id);
                        }
                        self.layers.insert(scroll_layer_id, layer);*/
                        info.default_scroll_layer_id = scroll_layer_id;
                        info.actual_scroll_layer_id = scroll_layer_id;
                        info.offset_from_current_layer = Point2D::zero();
                        info.transform = Matrix4::identity();
                        info.perspective = Matrix4::identity();
                    }
                    (ScrollPolicy::Scrollable, None) => {
                        // Nothing to do - use defaults as set above.
                    }
                }

                // When establishing a new 3D context, clear Z. This is only needed if there
                // are child stacking contexts, otherwise it is a redundant clear.
                if stacking_context.establishes_3d_context &&
                   stacking_context.has_stacking_contexts {
                    //panic!("todo");
                    /*
                    target.push_clear(ClearInfo {
                        clear_color: false,
                        clear_z: true,
                        clear_stencil: true,
                    });
                    */
                }

                // TODO: Account for scroll offset with transforms!
                if composition_operations.is_empty() {
                    self.add_items_to_target(&scene_items,
                                             &info,
                                             //target,
                                             context,
                                             level);
                } else {
                    panic!("todo");
                    /*
                    // TODO(gw): This makes the reftests work (mix_blend_mode) and
                    //           inline_stacking_context, but it seems wrong.
                    //           Need to investigate those and see what the root
                    //           issue is...
                    let empty_stacking_context = stacking_context.bounds.size.width == 0.0 ||
                                                 stacking_context.bounds.size.height == 0.0;
                    let target_size = if empty_stacking_context {
                        stacking_context.overflow.size
                    } else {
                        stacking_context.bounds.size
                    };
                    let target_origin = Point2D::new(info.offset_from_origin.x,
                                                     info.offset_from_origin.y);
                    let unfiltered_target_rect = Rect::new(target_origin, target_size);
                    let mut target_rect = unfiltered_target_rect;
                    for composition_operation in &composition_operations {
                        target_rect = composition_operation.target_rect(&target_rect);
                    }

                    let child_layer_index = ChildLayerIndex(target.children.len() as u32);

                    let render_target_size = Size2D::new(target_rect.size.width,
                                                         target_rect.size.height);
                    let render_target_id = self.next_render_target_id();

                    let (origin, texture_id) =
                        target.allocate_target_rect(target_rect.size.width,
                                                    target_rect.size.height,
                                                    context.device_pixel_ratio,
                                                    context.resource_cache);

                    let mut new_target = RenderTarget::new(render_target_id,
                                                           origin,
                                                           render_target_size,
                                                           Some(texture_id));

                    let local_transform =
                        Matrix4::identity().translate(origin.x, origin.y, 0.0)
                                           .mul(&stacking_context.transform)
                                           .translate(-origin.x, -origin.y, 0.0);
                    for composition_operation in composition_operations {
                        target.push_composite(composition_operation,
                                              texture_id,
                                              target_rect,
                                              &local_transform,
                                              child_layer_index);
                    }

                    info.offset_from_current_layer = Point2D::zero();

                    self.add_items_to_target(&scene_items,
                                             &info,
                                             &mut new_target,
                                             context,
                                             level);

                    target.children.push(new_target);
                    */
                }
            }
        }
    }

    pub fn build(&mut self,
                 resource_cache: &mut ResourceCache,
                 thread_pool: &mut scoped_threadpool::Pool,
                 device_pixel_ratio: f32)
                 -> RendererFrame {
        // Traverse layer trees to calculate visible nodes
/*        for (_, layer) in &mut self.layers {
            layer.cull();
        }*/

        // Build resource list for newly visible nodes
        //self.update_resource_lists(resource_cache, thread_pool);

        // Update texture cache and build list of raster jobs.
        //self.update_texture_cache_and_build_raster_jobs(resource_cache);

        // Rasterize needed glyphs on worker threads
        //self.raster_glyphs(thread_pool, resource_cache);

        // Compile nodes that have become visible
        //self.compile_visible_nodes(thread_pool, resource_cache, device_pixel_ratio);

        /*
        // Update the batch cache from newly compiled nodes
        self.update_batch_cache();
        */

        // Update the layer transform matrices
        //self.update_layer_transforms();

        // Collect the visible batches into a frame
        //let frame = self.collect_and_sort_visible_batches(resource_cache, device_pixel_ratio);
        let frame = self.build_frame();

        resource_cache.expire_old_resources(self.id);

        frame
    }

/*
    fn update_layer_transform(&mut self,
                              layer_id: ScrollLayerId,
                              parent_transform: &Matrix4) {
        // TODO(gw): This is an ugly borrow check workaround to clone these.
        //           Restructure this to avoid the clones!
        let (layer_transform, layer_children) = {
            match self.layers.get_mut(&layer_id) {
                Some(layer) => {
                    layer.world_transform = parent_transform.mul(&layer.local_transform)
                                                            .translate(layer.world_origin.x,
                                                                       layer.world_origin.y, 0.0)
                                                            .translate(layer.scrolling.offset.x,
                                                                       layer.scrolling.offset.y,
                                                                       0.0);
                    (layer.world_transform, layer.children.clone())
                }
                None => {
                    return;
                }
            }
        };

        for child_layer_id in layer_children {
            self.update_layer_transform(child_layer_id, &layer_transform);
        }
    }

    fn update_layer_transforms(&mut self) {
        if let Some(root_scroll_layer_id) = self.root_scroll_layer_id {
            self.update_layer_transform(root_scroll_layer_id, &Matrix4::identity());
        }

        // Update any fixed layers
        let mut fixed_layers = Vec::new();
        for (layer_id, _) in &self.layers {
            match layer_id.info {
                ScrollLayerInfo::Scrollable(..) => {}
                ScrollLayerInfo::Fixed => {
                    fixed_layers.push(*layer_id);
                }
            }
        }

        for layer_id in fixed_layers {
            self.update_layer_transform(layer_id, &Matrix4::identity());
        }
    }

    pub fn update_resource_lists(&mut self,
                                 resource_cache: &ResourceCache,
                                 thread_pool: &mut scoped_threadpool::Pool) {
        let _pf = util::ProfileScope::new("  update_resource_lists");

        for (_, layer) in &mut self.layers {
            let nodes = &mut layer.aabb_tree.nodes;
            let pipeline_auxiliary_lists = &self.pipeline_auxiliary_lists;

            thread_pool.scoped(|scope| {
                for node in nodes {
                    if node.is_visible && node.resource_list.is_none() {
                        scope.execute(move || {
                            node.build_resource_list(resource_cache, pipeline_auxiliary_lists);
                        });
                    }
                }
            });
        }
    }
*/

/*
    pub fn update_texture_cache_and_build_raster_jobs(&mut self,
                                                      resource_cache: &mut ResourceCache) {
        let _pf = util::ProfileScope::new("  update_texture_cache_and_build_raster_jobs");

        let frame_id = self.id;
        for (_, layer) in &self.layers {
            for node in &layer.aabb_tree.nodes {
                if node.is_visible {
                    let resource_list = node.resource_list.as_ref().unwrap();
                    resource_cache.add_resource_list(resource_list, frame_id);
                }
            }
        }
    }
*/

    pub fn raster_glyphs(&mut self,
                         thread_pool: &mut scoped_threadpool::Pool,
                         resource_cache: &mut ResourceCache) {
        let _pf = util::ProfileScope::new("  raster_glyphs");
        //resource_cache.raster_pending_glyphs(thread_pool, self.id);
    }

    fn build_frame(&mut self) -> RendererFrame {
        /*let mut tiles = Vec::new();

        for (_, layer) in &mut self.layers {
            let nodes = &mut layer.aabb_tree.nodes;
            for node in nodes {
                if node.is_visible {
                    let mut tile = RenderTile::new(node.actual_rect);
                    for item_id in &node.items {
                        tile.push(*item_id);
                    }
                    tiles.push(tile);
                }
            }
        }

        let packed_scene = match self.packed_scene {
            Some(ref packed_scene) => {
                packed_scene.clone()
            }
            None => {
                PackedScene::new()
            }
        };

        let debug_info = match self.spatial_hash {
            Some(ref spatial_hash) => {
                spatial_hash.build_debug_info()
            }
            None => {
                spatial_hash::DebugInfo::empty()
            }
        };
*/

        let (tiles, tile_size) = match self.spatial_hash {
            Some(ref spatial_hash) => {
                (spatial_hash.build(), spatial_hash.tile_size)
            }
            None => {
                (Vec::new(), Size2D::zero())
            }
        };

        RendererFrame::new(self.pipeline_epoch_map.clone(),
                           tiles,
                           tile_size,
                           //packed_scene,
                           //debug_info
                           )
    }

/*
    pub fn compile_visible_nodes(&mut self,
                                 thread_pool: &mut scoped_threadpool::Pool,
                                 resource_cache: &ResourceCache,
                                 device_pixel_ratio: f32) {
        let _pf = util::ProfileScope::new("  compile_visible_nodes");

        let layers = &mut self.layers;
        let stacking_context_info = &self.stacking_context_info;
        let draw_list_groups = &self.draw_list_groups;
        let frame_id = self.id;
        let pipeline_auxiliary_lists = &self.pipeline_auxiliary_lists;

        thread_pool.scoped(|scope| {
            for (_, layer) in layers {
                let nodes = &mut layer.aabb_tree.nodes;
                for node in nodes {
                    if node.is_visible && node.compiled_node.is_none() {
                        scope.execute(move || {
                            node.compile(resource_cache,
                                         frame_id,
                                         device_pixel_ratio,
                                         stacking_context_info,
                                         draw_list_groups,
                                         pipeline_auxiliary_lists);
                        });
                    }
                }
            }
        });
    }*/

/*
    pub fn update_batch_cache(&mut self) {
        let _pf = util::ProfileScope::new("  update_batch_cache");

        // Allocate and update VAOs
        for (_, layer) in &mut self.layers {
            for node in &mut layer.aabb_tree.nodes {
                if node.is_visible {
                    let compiled_node = node.compiled_node.as_mut().unwrap();
                    if let Some(vertex_buffer) = compiled_node.vertex_buffer.take() {
                        debug_assert!(compiled_node.vertex_buffer_id.is_none());

                        self.pending_updates.push(BatchUpdate {
                            id: vertex_buffer.id,
                            op: BatchUpdateOp::Create(vertex_buffer.vertices),
                        });

                        compiled_node.vertex_buffer_id = Some(vertex_buffer.id);
                    }
                }
            }
        }
    }

    pub fn collect_and_sort_visible_batches(&mut self,
                                            resource_cache: &mut ResourceCache,
                                            device_pixel_ratio: f32)
                                            -> RendererFrame {
        let root_layer = match self.root {
            Some(ref mut root) => {
                 root.collect_and_sort_visible_batches(resource_cache,
                                                       &self.draw_list_groups,
                                                       &self.layers,
                                                       &self.stacking_context_info,
                                                       device_pixel_ratio)
            }
            None => {
                DrawLayer::new(Point2D::zero(),
                               Size2D::zero(),
                               None,
                               Vec::new(),
                               Vec::new())
            }
        };

        let layers_bouncing_back = self.collect_layers_bouncing_back();
        RendererFrame::new(self.pipeline_epoch_map.clone(), layers_bouncing_back, root_layer)
    }

    fn collect_layers_bouncing_back(&self)
                                    -> HashSet<ScrollLayerId, BuildHasherDefault<FnvHasher>> {
        let mut layers_bouncing_back = HashSet::with_hasher(Default::default());
        for (scroll_layer_id, layer) in &self.layers {
            if layer.scrolling.started_bouncing_back {
                let overscroll_amount = layer.overscroll_amount();
                if overscroll_amount.width.abs() >= 0.1 || overscroll_amount.height.abs() >= 0.1 {
                    layers_bouncing_back.insert(*scroll_layer_id);
                }
            }
        }
        layers_bouncing_back
    }*/
}
