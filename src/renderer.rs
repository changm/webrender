/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use batch::{RasterBatch, VertexBufferId};
use debug_render::DebugRenderer;
use device::{Device, ProgramId, TextureId, UniformLocation, VertexFormat, GpuProfile};
use device::{TextureFilter, VAOId, VBOId, VertexUsageHint, FileWatcherHandler};
use euclid::{Rect, Matrix4, Point2D, Size2D};
use fnv::FnvHasher;
use gleam::gl;
use internal_types::{RendererFrame, ResultMsg, TextureUpdateOp, BatchUpdateOp, BatchUpdateList};
use internal_types::{TextureUpdateDetails, TextureUpdateList, PackedVertex, RenderTargetMode};
use internal_types::{ORTHO_NEAR_PLANE, ORTHO_FAR_PLANE, DevicePixel};
use internal_types::{PackedVertexForTextureCacheUpdate, CompositionOp, ChildLayerIndex};
use internal_types::{AxisDirection, LowLevelFilterOp, DrawCommand, DrawLayer, ANGLE_FLOAT_TO_FIXED};
use internal_types::{BasicRotationAngle, FontVertex};
use ipc_channel::ipc;
use profiler::{Profiler, BackendProfileCounters};
use profiler::{RendererProfileTimers, RendererProfileCounters};
use render_backend::RenderBackend;
use std::collections::HashMap;
use std::f32;
use std::hash::BuildHasherDefault;
use std::mem;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use std::usize;
use tessellator::BorderCornerTessellation;
use texture_cache::{BorderType, TextureCache, TextureInsertOp};
use tiling::{self, PackedCommand, TileFrame, TextBuffer, PackedTile};
use time::precise_time_ns;
use webrender_traits::{ColorF, Epoch, PipelineId, RenderNotifier};
use webrender_traits::{ImageFormat, MixBlendMode, RenderApiSender};
use offscreen_gl_context::{NativeGLContext, NativeGLContextMethods};
use util::RectHelpers;

pub const TEXT_TARGET_SIZE: u32 = 1024;

pub const BLUR_INFLATION_FACTOR: u32 = 3;
pub const MAX_RASTER_OP_SIZE: u32 = 2048;

const MAX_CACHED_QUAD_VAOS: usize = 8;

// TODO(gw): HACK! Need to support lighten/darken mix-blend-mode properly on android...

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
const GL_BLEND_MIN: gl::GLuint = gl::MIN;

#[cfg(any(target_os = "android", target_os = "gonk"))]
const GL_BLEND_MIN: gl::GLuint = gl::FUNC_ADD;

#[cfg(not(any(target_os = "android", target_os = "gonk")))]
const GL_BLEND_MAX: gl::GLuint = gl::MAX;

#[cfg(any(target_os = "android", target_os = "gonk"))]
const GL_BLEND_MAX: gl::GLuint = gl::FUNC_ADD;

#[derive(Debug)]
struct TileBatchInstance {
    cmd_index: u32,
    cmd_count: u32,
    scroll_offset: Point2D<f32>,
    rect: Rect<f32>,
}

struct TileBatch {
    layer_ubo_index: usize,
    rect_ubo_index: usize,
    clip_ubo_index: usize,
    image_ubo_index: usize,
    gradient_ubo_index: usize,
    stop_ubo_index: usize,
    text_ubo_index: usize,
    //glyph_ubo_index: u32,
    instances: Vec<TileBatchInstance>,
}

impl TileBatch {
    fn new(tile: &PackedTile) -> TileBatch {
        TileBatch {
            layer_ubo_index: tile.layer_ubo_index,
            rect_ubo_index: tile.rect_ubo_index,
            clip_ubo_index: tile.clip_ubo_index,
            image_ubo_index: tile.image_ubo_index,
            gradient_ubo_index: tile.gradient_ubo_index,
            stop_ubo_index: tile.gradient_stop_ubo_index,
            text_ubo_index: tile.text_ubo_index,
            //current_glyph_ubo: 0,
            instances: Vec::new(),
        }
    }

    fn can_add(&self, tile: &PackedTile) -> bool {
        self.layer_ubo_index == tile.layer_ubo_index &&
        self.rect_ubo_index == tile.rect_ubo_index &&
        self.clip_ubo_index == tile.clip_ubo_index &&
        self.image_ubo_index == tile.image_ubo_index &&
        self.gradient_ubo_index == tile.gradient_ubo_index &&
        self.stop_ubo_index == tile.gradient_stop_ubo_index &&
        self.text_ubo_index == tile.text_ubo_index// &&
       //current_glyph_ubo == glyph_ubos[tile.glyph_ubo_index];
    }

    fn add_tile(&mut self, tile: &PackedTile) {
        self.instances.push(TileBatchInstance {
            cmd_index: tile.cmd_index as u32,
            cmd_count: tile.cmd_count as u32,
            scroll_offset: Point2D::zero(),
            rect: Rect::new(Point2D::new(tile.rect.origin.x as f32, tile.rect.origin.y as f32),
                            Size2D::new(tile.rect.size.width as f32, tile.rect.size.height as f32)),
        });
    }

    fn flush(&self) {
        panic!("todo");
    }
}

#[derive(Debug, Copy, Clone)]
pub enum UboBindLocation {
    Misc,
    IndexBuffer,
    Rectangles,
    Glyphs,
    Texts,
    Images,
    Gradients,
    GradientStops,
    Layers,
    Clips,
    Instances,
}

impl UboBindLocation {
    pub fn get_item_size_and_fixed_size(&self) -> (usize, usize) {
        match *self {
            UboBindLocation::Misc => (0, mem::size_of::<MiscTileData>()),
            UboBindLocation::IndexBuffer => (4 * mem::size_of::<u32>(), 0),
            UboBindLocation::Rectangles => (mem::size_of::<tiling::PackedRectangle>(), 0),
            UboBindLocation::Glyphs => (mem::size_of::<tiling::PackedGlyph>(), 0),
            UboBindLocation::Texts => (mem::size_of::<tiling::PackedText>(), 0),
            UboBindLocation::Images => (mem::size_of::<tiling::PackedImage>(), 0),
            UboBindLocation::Gradients => (mem::size_of::<tiling::PackedGradient>(), 0),
            UboBindLocation::GradientStops => (mem::size_of::<tiling::PackedGradientStop>(), 0),
            UboBindLocation::Layers => (mem::size_of::<tiling::PackedLayer>(), 0),
            UboBindLocation::Clips => (mem::size_of::<tiling::Clip>(), 0),
            UboBindLocation::Instances => (mem::size_of::<TileBatchInstance>(), 0),
        }
    }

    pub fn get_array_len(&self, max_ubo_size: usize) -> usize {
        let (item_size, fixed_size) = self.get_item_size_and_fixed_size();
        (self.get_max_ubo_size(max_ubo_size) - fixed_size) / item_size
    }

    pub fn get_max_ubo_size(&self, max_ubo_size: usize) -> usize {
        // To work around an apparent bug in the nVidia shader compiler :(
        match *self {
            UboBindLocation::Misc => mem::size_of::<MiscTileData>(),
            UboBindLocation::IndexBuffer => max_ubo_size,
            UboBindLocation::Rectangles => max_ubo_size,
            UboBindLocation::Glyphs => max_ubo_size,
            UboBindLocation::Texts => max_ubo_size,
            UboBindLocation::Images => 4096,
            UboBindLocation::Gradients => 4096,
            UboBindLocation::GradientStops => 4096,
            UboBindLocation::Layers => max_ubo_size,
            UboBindLocation::Clips => 8192,
            UboBindLocation::Instances => max_ubo_size,
        }
    }
}

struct UniformBufferBinding {
    name: &'static str,
    location: UboBindLocation,
}

const UBO_BINDINGS: [UniformBufferBinding; 11] = [
    UniformBufferBinding {
        name: "Misc",
        location: UboBindLocation::Misc,
    },
    UniformBufferBinding {
        name: "IndexBuffer",
        location: UboBindLocation::IndexBuffer,
    },
    UniformBufferBinding {
        name: "Rectangles",
        location: UboBindLocation::Rectangles
    },
    UniformBufferBinding {
        name: "Glyphs",
        location: UboBindLocation::Glyphs
    },
    UniformBufferBinding {
        name: "Texts",
        location: UboBindLocation::Texts
    },
    UniformBufferBinding {
        name: "Images",
        location: UboBindLocation::Images
    },
    UniformBufferBinding {
        name: "Gradients",
        location: UboBindLocation::Gradients
    },
    UniformBufferBinding {
        name: "GradientStops",
        location: UboBindLocation::GradientStops
    },
    UniformBufferBinding {
        name: "Layers",
        location: UboBindLocation::Layers
    },
    UniformBufferBinding {
        name: "Clips",
        location: UboBindLocation::Clips
    },
    UniformBufferBinding {
        name: "Instances",
        location: UboBindLocation::Instances
    },
];

/*
#[derive(Debug, Copy, Clone)]
pub struct PackedPrimitiveKey(u32);

impl PackedPrimitiveKey {
    pub fn new(key: &PrimitiveKey) -> PackedPrimitiveKey {
        debug_assert!(key.index < 65536);       // ensure u16 (should be bounded by max ubo size anyway)
        PackedPrimitiveKey(key.index | ((key.kind as u32) << 24))
    }
}*/

struct MiscTileData {
    background_color: ColorF,
}

#[derive(Clone, Copy)]
struct VertexBuffer {
    vao_id: VAOId,
}

pub trait CompositionOpHelpers {
    fn needs_framebuffer(&self) -> bool;
}

impl CompositionOpHelpers for CompositionOp {
    fn needs_framebuffer(&self) -> bool {
        match *self {
            CompositionOp::MixBlend(MixBlendMode::Normal) => unreachable!(),

            CompositionOp::MixBlend(MixBlendMode::Screen) |
            CompositionOp::MixBlend(MixBlendMode::Overlay) |
            CompositionOp::MixBlend(MixBlendMode::ColorDodge) |
            CompositionOp::MixBlend(MixBlendMode::ColorBurn) |
            CompositionOp::MixBlend(MixBlendMode::HardLight) |
            CompositionOp::MixBlend(MixBlendMode::SoftLight) |
            CompositionOp::MixBlend(MixBlendMode::Difference) |
            CompositionOp::MixBlend(MixBlendMode::Exclusion) |
            CompositionOp::MixBlend(MixBlendMode::Hue) |
            CompositionOp::MixBlend(MixBlendMode::Saturation) |
            CompositionOp::MixBlend(MixBlendMode::Color) |
            CompositionOp::MixBlend(MixBlendMode::Luminosity) => true,
            CompositionOp::Filter(_) |
            CompositionOp::MixBlend(MixBlendMode::Multiply) |
            CompositionOp::MixBlend(MixBlendMode::Darken) |
            CompositionOp::MixBlend(MixBlendMode::Lighten) => false,
        }
    }
}

struct RenderContext {
    blend_program_id: ProgramId,
    filter_program_id: ProgramId,
    device_pixel_ratio: f32,
    framebuffer_size: Size2D<u32>,
}

struct FileWatcher {
    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,
    result_tx: Sender<ResultMsg>,
}

impl FileWatcherHandler for FileWatcher {
    fn file_changed(&self, path: PathBuf) {
        self.result_tx.send(ResultMsg::RefreshShader(path)).ok();
        let mut notifier = self.notifier.lock();
        notifier.as_mut().unwrap().as_mut().unwrap().new_frame_ready();
    }
}

pub struct Renderer {
    result_rx: Receiver<ResultMsg>,
    device: Device,
    pending_texture_updates: Vec<TextureUpdateList>,
    pending_batch_updates: Vec<BatchUpdateList>,
    pending_shader_updates: Vec<PathBuf>,
    current_frame: Option<RendererFrame>,
    device_pixel_ratio: f32,
    vertex_buffers: HashMap<VertexBufferId, Vec<VertexBufferAndOffset>, BuildHasherDefault<FnvHasher>>,
    raster_batches: Vec<RasterBatch>,
    quad_vertex_buffer: Option<VBOId>,
    cached_quad_vaos: Vec<VAOId>,
    simple_triangles_vao: Option<VAOId>,
    raster_op_vao: Option<VAOId>,

    quad_program_id: ProgramId,
    u_quad_transform_array: UniformLocation,
    u_quad_offset_array: UniformLocation,
    u_tile_params: UniformLocation,
    u_clip_rects: UniformLocation,
    u_atlas_params: UniformLocation,

    blit_program_id: ProgramId,

    border_program_id: ProgramId,

    blend_program_id: ProgramId,
    u_blend_params: UniformLocation,

    filter_program_id: ProgramId,
    u_filter_params: UniformLocation,

    box_shadow_program_id: ProgramId,

    blur_program_id: ProgramId,
    u_direction: UniformLocation,

    mask_program_id: ProgramId,

    scene_program_id: ProgramId,
    complex_program_id: ProgramId,
    text_program_id: ProgramId,

    notifier: Arc<Mutex<Option<Box<RenderNotifier>>>>,

    enable_profiler: bool,
    enable_msaa: bool,
    debug: DebugRenderer,
    backend_profile_counters: BackendProfileCounters,
    profile_counters: RendererProfileCounters,
    profiler: Profiler,
    last_time: u64,

    max_raster_op_size: u32,
    raster_op_target_a8: TextureId,
    raster_op_target_rgba8: TextureId,
    temporary_fb_texture: TextureId,
    text_composite_target: TextureId,

    //gpu_profile: GpuProfile,
    quad_vao_id: VAOId,
    quad_vao_index_count: gl::GLint,
}

impl Renderer {
    pub fn new(options: RendererOptions) -> (Renderer, RenderApiSender) {
        let (api_tx, api_rx) = ipc::channel().unwrap();
        let (payload_tx, payload_rx) = ipc::bytes_channel().unwrap();
        let (result_tx, result_rx) = channel();

        let notifier = Arc::new(Mutex::new(None));

        let file_watch_handler = FileWatcher {
            result_tx: result_tx.clone(),
            notifier: notifier.clone(),
        };

        let mut device = Device::new(options.resource_path.clone(),
                                     options.device_pixel_ratio,
                                     Box::new(file_watch_handler));
        device.begin_frame();

        let quad_program_id = ProgramId(0);// device.create_program("quad");
        let blit_program_id = ProgramId(0);//device.create_program("blit");
        let border_program_id = ProgramId(0);//device.create_program("border");
        let blend_program_id = ProgramId(0);//device.create_program("blend");
        let filter_program_id = ProgramId(0);//device.create_program("filter");
        let box_shadow_program_id = ProgramId(0);//device.create_program("box_shadow");
        let blur_program_id = ProgramId(0);//device.create_program("blur");
        let mask_program_id = ProgramId(0);//device.create_program("mask");
        let max_raster_op_size = MAX_RASTER_OP_SIZE * options.device_pixel_ratio as u32;

        let max_ubo_size = gl::get_integer_v(gl::MAX_UNIFORM_BLOCK_SIZE) as usize;
        println!("Max UBO size = {}", max_ubo_size);

        //println!("{:?}: {:?}", UboBindLocation::Misc, UboBindLocation::Misc.get_array_len(max_ubo_size));
        println!("{:?}: {:?}", UboBindLocation::IndexBuffer, UboBindLocation::IndexBuffer.get_array_len(max_ubo_size));
        println!("{:?}: {:?}", UboBindLocation::Rectangles, UboBindLocation::Rectangles.get_array_len(max_ubo_size));
        println!("{:?}: {:?}", UboBindLocation::Glyphs, UboBindLocation::Glyphs.get_array_len(max_ubo_size));
        println!("{:?}: {:?}", UboBindLocation::Texts, UboBindLocation::Texts.get_array_len(max_ubo_size));
        println!("{:?}: {:?}", UboBindLocation::Images, UboBindLocation::Images.get_array_len(max_ubo_size));
        println!("{:?}: {:?}", UboBindLocation::Gradients, UboBindLocation::Gradients.get_array_len(max_ubo_size));
        println!("{:?}: {:?}", UboBindLocation::GradientStops, UboBindLocation::GradientStops.get_array_len(max_ubo_size));
        println!("{:?}: {:?}", UboBindLocation::Layers, UboBindLocation::Layers.get_array_len(max_ubo_size));
        println!("{:?}: {:?}", UboBindLocation::Clips, UboBindLocation::Clips.get_array_len(max_ubo_size));
        println!("{:?}: {:?}", UboBindLocation::Instances, UboBindLocation::Instances.get_array_len(max_ubo_size));

        let mut preamble = String::new();

        preamble.push_str(&format!("#define UBO_INDEX_COUNT {:?}\n", UboBindLocation::IndexBuffer.get_array_len(max_ubo_size)));
        preamble.push_str(&format!("#define UBO_RECTANGLE_COUNT {:?}\n", UboBindLocation::Rectangles.get_array_len(max_ubo_size)));
        preamble.push_str(&format!("#define UBO_GLYPH_COUNT {:?}\n", UboBindLocation::Glyphs.get_array_len(max_ubo_size)));
        preamble.push_str(&format!("#define UBO_TEXT_COUNT {:?}\n", UboBindLocation::Texts.get_array_len(max_ubo_size)));
        preamble.push_str(&format!("#define UBO_IMAGE_COUNT {:?}\n", UboBindLocation::Images.get_array_len(max_ubo_size)));
        preamble.push_str(&format!("#define UBO_GRADIENT_COUNT {:?}\n", UboBindLocation::Gradients.get_array_len(max_ubo_size)));
        preamble.push_str(&format!("#define UBO_GRADIENT_STOP_COUNT {:?}\n", UboBindLocation::GradientStops.get_array_len(max_ubo_size)));
        preamble.push_str(&format!("#define UBO_LAYER_COUNT {:?}\n", UboBindLocation::Layers.get_array_len(max_ubo_size)));
        preamble.push_str(&format!("#define UBO_CLIP_COUNT {:?}\n", UboBindLocation::Clips.get_array_len(max_ubo_size)));
        preamble.push_str(&format!("#define UBO_INSTANCE_COUNT {:?}\n", UboBindLocation::Instances.get_array_len(max_ubo_size)));

        let scene_program_id = device.create_program_with_prefix("tiling_opaque", "shared_tiling", Some(preamble.clone()));
        let text_program_id = device.create_program("text", "shared_other");
        let complex_program_id = device.create_program_with_prefix("tiling_complex", "shared_tiling", Some(preamble));

        for info in &UBO_BINDINGS {
            let block_index = gl::get_uniform_block_index(scene_program_id.0, info.name);
            gl::uniform_block_binding(scene_program_id.0, block_index, info.location as u32);
            println!("S: {} {} -> {}", info.name, block_index, info.location as u32);

            let block_index = gl::get_uniform_block_index(complex_program_id.0, info.name);
            gl::uniform_block_binding(complex_program_id.0, block_index, info.location as u32);
            println!("C: {} {} -> {}", info.name, block_index, info.location as u32);
        }

        let texture_ids = device.create_texture_ids(1024);
        let mut texture_cache = TextureCache::new(texture_ids);
        let white_pixels: Vec<u8> = vec![
            0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
            0xff, 0xff, 0xff, 0xff,
        ];
        let mask_pixels: Vec<u8> = vec![
            0xff, 0xff,
            0xff, 0xff,
        ];
        // TODO: Ensure that the white texture can never get evicted when the cache supports LRU eviction!
        let white_image_id = texture_cache.new_item_id();
        texture_cache.insert(white_image_id,
                             0,
                             0,
                             2,
                             2,
                             ImageFormat::RGBA8,
                             TextureFilter::Linear,
                             TextureInsertOp::Blit(white_pixels),
                             BorderType::SinglePixel);

        let dummy_mask_image_id = texture_cache.new_item_id();
        texture_cache.insert(dummy_mask_image_id,
                             0,
                             0,
                             2,
                             2,
                             ImageFormat::A8,
                             TextureFilter::Linear,
                             TextureInsertOp::Blit(mask_pixels),
                             BorderType::SinglePixel);

        let debug_renderer = DebugRenderer::new(&mut device);

        let raster_op_target_a8 = device.create_texture_ids(1)[0];
        device.init_texture(raster_op_target_a8,
                            max_raster_op_size,
                            max_raster_op_size,
                            ImageFormat::A8,
                            TextureFilter::Nearest,
                            RenderTargetMode::RenderTarget,
                            None);

        let raster_op_target_rgba8 = device.create_texture_ids(1)[0];
        device.init_texture(raster_op_target_rgba8,
                            max_raster_op_size,
                            max_raster_op_size,
                            ImageFormat::RGBA8,
                            TextureFilter::Nearest,
                            RenderTargetMode::RenderTarget,
                            None);

        let text_composite_target = device.create_texture_ids(1)[0];
        device.init_texture(text_composite_target,
                            TEXT_TARGET_SIZE * options.device_pixel_ratio as u32,
                            TEXT_TARGET_SIZE * options.device_pixel_ratio as u32,
                            ImageFormat::RGBA8,
                            TextureFilter::Linear,
                            RenderTargetMode::RenderTarget,
                            None);

        let temporary_fb_texture = device.create_texture_ids(1)[0];

        let x0 = 0.0;
        let y0 = 0.0;
        let x1 = 1.0;//options.tile_size.width as f32;
        let y1 = 1.0;//options.tile_size.height as f32;

        let quad_indices: [u16; 6] = [ 0, 1, 2, 2, 3, 1 ];
        let quad_vertices = [
            PackedVertex {
                pos: [x0, y0],
                rect: [x0, y0, x1, y1],
            },
            PackedVertex {
                pos: [x1, y0],
                rect: [x0, y0, x1, y1],
            },
            PackedVertex {
                pos: [x0, y1],
                rect: [x0, y0, x1, y1],
            },
            PackedVertex {
                pos: [x1, y1],
                rect: [x0, y0, x1, y1],
            },
        ];
/*
        let mut quad_indices: Vec<u16> = Vec::new();
        let mut quad_vertices = Vec::new();
        let sub_tile_size = 32;
        let sub_tile_x_count = (options.tile_size.width + sub_tile_size - 1) / sub_tile_size;
        let sub_tile_y_count = (options.tile_size.height + sub_tile_size - 1) / sub_tile_size;

        for y in 0..sub_tile_y_count {
            for x in 0..sub_tile_x_count {
                let base_index = quad_vertices.len() as u16;
                quad_indices.push(base_index + 0);
                quad_indices.push(base_index + 1);
                quad_indices.push(base_index + 2);
                quad_indices.push(base_index + 2);
                quad_indices.push(base_index + 3);
                quad_indices.push(base_index + 1);

                let x0 = (x * sub_tile_size) as f32;
                let y0 = (y * sub_tile_size) as f32;
                let x1 = (x0 + sub_tile_size as f32).min(options.tile_size.width as f32);
                let y1 = (y0 + sub_tile_size as f32).min(options.tile_size.height as f32);

                quad_vertices.extend_from_slice(&[
                    PackedVertex {
                        pos: [x0, y0],
                        rect: [x0, y0, x1, y1],
                    },
                    PackedVertex {
                        pos: [x1, y0],
                        rect: [x0, y0, x1, y1],
                    },
                    PackedVertex {
                        pos: [x0, y1],
                        rect: [x0, y0, x1, y1],
                    },
                    PackedVertex {
                        pos: [x1, y1],
                        rect: [x0, y0, x1, y1],
                    },
                ]);
            }
        }
*/

        let quad_vao_id = device.create_vao(VertexFormat::Triangles, None);
        device.bind_vao(quad_vao_id);
        device.update_vao_indices(quad_vao_id, &quad_indices, VertexUsageHint::Static);
        device.update_vao_main_vertices(quad_vao_id, &quad_vertices, VertexUsageHint::Static);

        device.end_frame();

        let backend_notifier = notifier.clone();

        // We need a reference to the webrender context from the render backend in order to share
        // texture ids
        let context_handle = NativeGLContext::current_handle();

        let tile_size = options.tile_size;
        let (device_pixel_ratio, enable_aa) = (options.device_pixel_ratio, options.enable_aa);
        let payload_tx_for_backend = payload_tx.clone();
        thread::spawn(move || {
            let mut backend = RenderBackend::new(api_rx,
                                                 payload_rx,
                                                 payload_tx_for_backend,
                                                 result_tx,
                                                 device_pixel_ratio,
                                                 white_image_id,
                                                 dummy_mask_image_id,
                                                 texture_cache,
                                                 enable_aa,
                                                 backend_notifier,
                                                 context_handle,
                                                 max_ubo_size,
                                                 tile_size);
            backend.run();
        });

        let mut renderer = Renderer {
            result_rx: result_rx,
            device: device,
            current_frame: None,
            vertex_buffers: HashMap::with_hasher(Default::default()),
            raster_batches: Vec::new(),
            quad_vertex_buffer: None,
            simple_triangles_vao: None,
            raster_op_vao: None,
            cached_quad_vaos: Vec::new(),
            pending_texture_updates: Vec::new(),
            pending_batch_updates: Vec::new(),
            pending_shader_updates: Vec::new(),
            border_program_id: border_program_id,
            device_pixel_ratio: options.device_pixel_ratio,
            blend_program_id: blend_program_id,
            filter_program_id: filter_program_id,
            quad_program_id: quad_program_id,
            blit_program_id: blit_program_id,
            box_shadow_program_id: box_shadow_program_id,
            blur_program_id: blur_program_id,
            mask_program_id: mask_program_id,
            scene_program_id: scene_program_id,
            complex_program_id: complex_program_id,
            text_program_id: text_program_id,
            u_blend_params: UniformLocation::invalid(),
            u_filter_params: UniformLocation::invalid(),
            u_direction: UniformLocation::invalid(),
            u_quad_offset_array: UniformLocation::invalid(),
            u_quad_transform_array: UniformLocation::invalid(),
            u_atlas_params: UniformLocation::invalid(),
            u_tile_params: UniformLocation::invalid(),
            u_clip_rects: UniformLocation::invalid(),
            notifier: notifier,
            debug: debug_renderer,
            backend_profile_counters: BackendProfileCounters::new(),
            profile_counters: RendererProfileCounters::new(),
            profiler: Profiler::new(),
            enable_profiler: options.enable_profiler,
            enable_msaa: options.enable_msaa,
            last_time: 0,
            raster_op_target_a8: raster_op_target_a8,
            raster_op_target_rgba8: raster_op_target_rgba8,
            temporary_fb_texture: temporary_fb_texture,
            text_composite_target: text_composite_target,
            max_raster_op_size: max_raster_op_size,
            //gpu_profile: GpuProfile::new(),
            quad_vao_id: quad_vao_id,
            quad_vao_index_count: quad_indices.len() as gl::GLint,
        };

        renderer.update_uniform_locations();

        let sender = RenderApiSender::new(api_tx, payload_tx);
        (renderer, sender)
    }

    #[cfg(any(target_os = "android", target_os = "gonk"))]
    fn enable_msaa(&self, _: bool) {
    }

    #[cfg(not(any(target_os = "android", target_os = "gonk")))]
    fn enable_msaa(&self, enable_msaa: bool) {
        if self.enable_msaa {
            if enable_msaa {
                gl::enable(gl::MULTISAMPLE);
            } else {
                gl::disable(gl::MULTISAMPLE);
            }
        }
    }

    fn update_uniform_locations(&mut self) {
        self.u_quad_transform_array = self.device.get_uniform_location(self.quad_program_id, "uMatrixPalette");
        self.u_quad_offset_array = self.device.get_uniform_location(self.quad_program_id, "uOffsets");
        self.u_tile_params = self.device.get_uniform_location(self.quad_program_id, "uTileParams");
        self.u_clip_rects = self.device.get_uniform_location(self.quad_program_id, "uClipRects");
        self.u_atlas_params = self.device.get_uniform_location(self.quad_program_id, "uAtlasParams");
        self.u_blend_params = self.device.get_uniform_location(self.blend_program_id, "uBlendParams");
        self.u_filter_params = self.device.get_uniform_location(self.filter_program_id, "uFilterParams");
        self.u_direction = self.device.get_uniform_location(self.blur_program_id, "uDirection");
    }

    pub fn set_render_notifier(&self, notifier: Box<RenderNotifier>) {
        let mut notifier_arc = self.notifier.lock().unwrap();
        *notifier_arc = Some(notifier);
    }

    pub fn current_epoch(&self, pipeline_id: PipelineId) -> Option<Epoch> {
        self.current_frame.as_ref().and_then(|frame| {
            frame.pipeline_epoch_map.get(&pipeline_id).map(|epoch| *epoch)
        })
    }

    pub fn update(&mut self) {
        // Pull any pending results and return the most recent.
        while let Ok(msg) = self.result_rx.try_recv() {
            match msg {
                ResultMsg::UpdateTextureCache(update_list) => {
                    self.pending_texture_updates.push(update_list);
                }
                ResultMsg::NewFrame(frame, update_list, profile_counters) => {
                    self.backend_profile_counters = profile_counters;
                    self.pending_batch_updates.push(update_list);
                    self.current_frame = Some(frame);
                }
                ResultMsg::RefreshShader(path) => {
                    self.pending_shader_updates.push(path);
                }
            }
        }
    }

    pub fn render(&mut self, framebuffer_size: Size2D<u32>) {
        let mut profile_timers = RendererProfileTimers::new();

        let gpu_ns = 0;//self.gpu_profile.begin();

        profile_timers.cpu_time.profile(|| {
            self.device.begin_frame();

            gl::disable(gl::SCISSOR_TEST);
            gl::clear_color(1.0, 1.0, 1.0, 0.0);
            gl::clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);

            self.update_shaders();
            self.update_texture_cache();
            self.update_batches();
            self.draw_frame(framebuffer_size);
        });

        let current_time = precise_time_ns();
        let ns = current_time - self.last_time;
        self.profile_counters.frame_time.set(ns);

        //self.gpu_profile.end();
        profile_timers.gpu_time.set(gpu_ns);

        if self.enable_profiler {
            self.profiler.draw_profile(&self.backend_profile_counters,
                                       &self.profile_counters,
                                       &profile_timers,
                                       &mut self.debug);
        }

        self.profile_counters.reset();
        self.profile_counters.frame_counter.inc();

        let debug_size = Size2D::new((framebuffer_size.width as f32 / self.device_pixel_ratio) as u32,
                                     (framebuffer_size.height as f32 / self.device_pixel_ratio) as u32);
        self.debug.render(&mut self.device, &debug_size);
        self.device.end_frame();
        self.last_time = current_time;
    }

    pub fn layers_are_bouncing_back(&self) -> bool {
        match self.current_frame {
            None => false,
            Some(ref current_frame) => !current_frame.layers_bouncing_back.is_empty(),
        }
    }

    fn update_batches(&mut self) {
        let mut pending_batch_updates = mem::replace(&mut self.pending_batch_updates, vec![]);
        for update_list in pending_batch_updates.drain(..) {
            for update in update_list.updates {
                match update.op {
                    BatchUpdateOp::Create(vertices) => {
                        panic!("todo");
                        /*
                        if self.quad_vertex_buffer.is_none() {
                            self.quad_vertex_buffer = Some(self.device.create_quad_vertex_buffer())
                        }

                        let vao_id = match self.cached_quad_vaos.pop() {
                            Some(quad_vao_id) => quad_vao_id,
                            None => {
                                self.device.create_vao(VertexFormat::Rectangles,
                                                       Some(self.quad_vertex_buffer.unwrap()))
                            }
                        };

                        self.device.bind_vao(vao_id);

                        self.device.update_vao_aux_vertices(vao_id,
                                                            &vertices,
                                                            VertexUsageHint::Static);

                        self.vertex_buffers.insert(update.id, vec![
                            VertexBufferAndOffset {
                                buffer: VertexBuffer {
                                    vao_id: vao_id,
                                },
                                offset: 0,
                            }
                        ]);*/
                    }
                    BatchUpdateOp::Destroy => {
                        let vertex_buffers_and_offsets =
                            self.vertex_buffers.remove(&update.id).unwrap();
                        for vertex_buffer_and_offset in vertex_buffers_and_offsets.into_iter() {
                            if self.cached_quad_vaos.len() < MAX_CACHED_QUAD_VAOS &&
                                    vertex_buffer_and_offset.offset == 0 {
                                self.cached_quad_vaos.push(vertex_buffer_and_offset.buffer.vao_id);
                            } else {
                                self.device.delete_vao(vertex_buffer_and_offset.buffer.vao_id);
                            }
                        }
                    }
                }
            }
        }
    }

    fn update_shaders(&mut self) {
        let update_uniforms = !self.pending_shader_updates.is_empty();

        for path in self.pending_shader_updates.drain(..) {
            panic!("todo");
            //self.device.refresh_shader(path);
        }

        if update_uniforms {
            self.update_uniform_locations();
        }
    }

    fn update_texture_cache(&mut self) {
        let mut pending_texture_updates = mem::replace(&mut self.pending_texture_updates, vec![]);
        for update_list in pending_texture_updates.drain(..) {
            for update in update_list.updates {
                match update.op {
                    TextureUpdateOp::Create(width, height, format, filter, mode, maybe_bytes) => {
                        // TODO: clean up match
                        match maybe_bytes {
                            Some(bytes) => {
                                self.device.init_texture(update.id,
                                                         width,
                                                         height,
                                                         format,
                                                         filter,
                                                         mode,
                                                         Some(bytes.as_slice()));
                            }
                            None => {
                                self.device.init_texture(update.id,
                                                         width,
                                                         height,
                                                         format,
                                                         filter,
                                                         mode,
                                                         None);
                            }
                        }
                    }
                    TextureUpdateOp::Grow(new_width,
                                          new_height,
                                          format,
                                          filter,
                                          mode) => {
                        self.device.resize_texture(update.id,
                                                   new_width,
                                                   new_height,
                                                   format,
                                                   filter,
                                                   mode);
                    }
                    TextureUpdateOp::Update(x, y, width, height, details) => {
                        match details {
                            TextureUpdateDetails::Raw => {
                                self.device.update_raw_texture(update.id, x, y, width, height);
                            }
                            TextureUpdateDetails::Blit(bytes) => {
                                self.device.update_texture(
                                    update.id,
                                    x,
                                    y,
                                    width, height,
                                    bytes.as_slice());
                            }
                            TextureUpdateDetails::Blur(bytes,
                                                       glyph_size,
                                                       radius,
                                                       unblurred_glyph_texture_image,
                                                       horizontal_blur_texture_image,
                                                       border_type) => {
                                let radius =
                                    f32::ceil(radius.to_f32_px() * self.device_pixel_ratio) as u32;
                                self.device.update_texture(
                                    unblurred_glyph_texture_image.texture_id,
                                    unblurred_glyph_texture_image.pixel_uv.x,
                                    unblurred_glyph_texture_image.pixel_uv.y,
                                    glyph_size.width,
                                    glyph_size.height,
                                    bytes.as_slice());

                                let blur_program_id = self.blur_program_id;

                                let white = ColorF::new(1.0, 1.0, 1.0, 1.0);
                                let (width, height) = (width as f32, height as f32);

                                let zero_point = Point2D::new(0.0, 0.0);
                                let dest_texture_size = Size2D::new(width as f32, height as f32);
                                let source_texture_size = Size2D::new(glyph_size.width as f32,
                                                                      glyph_size.height as f32);
                                let blur_radius = radius as f32;

                                self.add_rect_to_raster_batch(horizontal_blur_texture_image.texture_id,
                                                              unblurred_glyph_texture_image.texture_id,
                                                              blur_program_id,
                                                              Some(AxisDirection::Horizontal),
                                                              &Rect::new(horizontal_blur_texture_image.pixel_uv,
                                                                         Size2D::new(width as u32, height as u32)),
                                                              border_type,
                                                              |texture_rect| {
                                    [
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.origin,
                                            &white,
                                            &Point2D::new(0.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.top_right(),
                                            &white,
                                            &Point2D::new(1.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_left(),
                                            &white,
                                            &Point2D::new(0.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_right(),
                                            &white,
                                            &Point2D::new(1.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &unblurred_glyph_texture_image.texel_uv.origin,
                                            &unblurred_glyph_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                    ]
                                });

                                let source_texture_size = Size2D::new(width as f32, height as f32);

                                self.add_rect_to_raster_batch(update.id,
                                                              horizontal_blur_texture_image.texture_id,
                                                              blur_program_id,
                                                              Some(AxisDirection::Vertical),
                                                              &Rect::new(Point2D::new(x as u32, y as u32),
                                                                         Size2D::new(width as u32, height as u32)),
                                                              border_type,
                                                              |texture_rect| {
                                    [
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.origin,
                                            &white,
                                            &Point2D::new(0.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.top_right(),
                                            &white,
                                            &Point2D::new(1.0, 0.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_left(),
                                            &white,
                                            &Point2D::new(0.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_right(),
                                            &white,
                                            &Point2D::new(1.0, 1.0),
                                            &zero_point,
                                            &zero_point,
                                            &horizontal_blur_texture_image.texel_uv.origin,
                                            &horizontal_blur_texture_image.texel_uv.bottom_right(),
                                            &dest_texture_size,
                                            &source_texture_size,
                                            blur_radius),
                                    ]
                                });
                            }
                            TextureUpdateDetails::BorderRadius(outer_rx,
                                                               outer_ry,
                                                               inner_rx,
                                                               inner_ry,
                                                               index,
                                                               inverted,
                                                               border_type) => {
                                // From here on out everything is in device coordinates.
                                let border_program_id = self.border_program_id;
                                let color = if inverted {
                                    ColorF::new(0.0, 0.0, 0.0, 1.0)
                                } else {
                                    ColorF::new(1.0, 1.0, 1.0, 1.0)
                                };

                                let border_radii_outer = Point2D::new(outer_rx.as_f32(), outer_ry.as_f32());
                                let border_radii_inner = Point2D::new(inner_rx.as_f32(), inner_ry.as_f32());

                                let zero_point = Point2D::new(0.0, 0.0);
                                let zero_size = Size2D::new(0.0, 0.0);

                                self.add_rect_to_raster_batch(update.id,
                                                              TextureId(0),
                                                              border_program_id,
                                                              None,
                                                              &Rect::new(Point2D::new(x as u32, y as u32),
                                                                         Size2D::new(width as u32, height as u32)),
                                                              border_type,
                                                              |texture_rect| {
                                    let border_radii_outer_size =
                                        Size2D::new(border_radii_outer.x,
                                                    border_radii_outer.y);
                                    let border_radii_inner_size =
                                        Size2D::new(border_radii_inner.x,
                                                    border_radii_inner.y);
                                    let untessellated_rect =
                                        Rect::new(texture_rect.origin, border_radii_outer_size);
                                    let tessellated_rect =
                                        match index {
                                            None => untessellated_rect,
                                            Some(index) => {
                                                untessellated_rect.tessellate_border_corner(
                                                    &border_radii_outer_size,
                                                    &border_radii_inner_size,
                                                    1.0,
                                                    BasicRotationAngle::Upright,
                                                    index)
                                            }
                                        };

                                    let border_position =
                                        untessellated_rect.bottom_right() -
                                        (tessellated_rect.origin - texture_rect.origin);

                                    [
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.origin,
                                            &color,
                                            &zero_point,
                                            &border_radii_outer,
                                            &border_radii_inner,
                                            &border_position,
                                            &zero_point,
                                            &zero_size,
                                            &zero_size,
                                            0.0),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.top_right(),
                                            &color,
                                            &zero_point,
                                            &border_radii_outer,
                                            &border_radii_inner,
                                            &border_position,
                                            &zero_point,
                                            &zero_size,
                                            &zero_size,
                                            0.0),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_left(),
                                            &color,
                                            &zero_point,
                                            &border_radii_outer,
                                            &border_radii_inner,
                                            &border_position,
                                            &zero_point,
                                            &zero_size,
                                            &zero_size,
                                            0.0),
                                        PackedVertexForTextureCacheUpdate::new(
                                            &texture_rect.bottom_right(),
                                            &color,
                                            &zero_point,
                                            &border_radii_outer,
                                            &border_radii_inner,
                                            &border_position,
                                            &zero_point,
                                            &zero_size,
                                            &zero_size,
                                            0.0),
                                    ]
                                });
                            }
                            TextureUpdateDetails::BoxShadow(blur_radius,
                                                            border_radius,
                                                            box_rect_size,
                                                            raster_origin,
                                                            inverted,
                                                            border_type) => {
                                self.update_texture_cache_for_box_shadow(
                                    update.id,
                                    &Rect::new(Point2D::new(x, y),
                                               Size2D::new(width, height)),
                                    &Rect::new(
                                        Point2D::new(raster_origin.x, raster_origin.y),
                                        Size2D::new(box_rect_size.width, box_rect_size.height)),
                                    blur_radius,
                                    border_radius,
                                    inverted,
                                    border_type)
                            }
                        }
                    }
                }
            }
        }

        self.flush_raster_batches();
    }

    fn update_texture_cache_for_box_shadow(&mut self,
                                           update_id: TextureId,
                                           texture_rect: &Rect<u32>,
                                           box_rect: &Rect<DevicePixel>,
                                           blur_radius: DevicePixel,
                                           border_radius: DevicePixel,
                                           inverted: bool,
                                           border_type: BorderType) {
        debug_assert!(border_type == BorderType::SinglePixel);
        let box_shadow_program_id = self.box_shadow_program_id;

        let blur_radius = blur_radius.as_f32();

        let color = if inverted {
            ColorF::new(1.0, 1.0, 1.0, 0.0)
        } else {
            ColorF::new(1.0, 1.0, 1.0, 1.0)
        };

        let zero_point = Point2D::new(0.0, 0.0);
        let zero_size = Size2D::new(0.0, 0.0);

        self.add_rect_to_raster_batch(update_id,
                                      TextureId(0),
                                      box_shadow_program_id,
                                      None,
                                      &texture_rect,
                                      border_type,
                                      |texture_rect| {
            let box_rect_top_left = Point2D::new(box_rect.origin.x.as_f32() + texture_rect.origin.x,
                                                 box_rect.origin.y.as_f32() + texture_rect.origin.y);
            let box_rect_bottom_right = Point2D::new(box_rect_top_left.x + box_rect.size.width.as_f32(),
                                                     box_rect_top_left.y + box_rect.size.height.as_f32());
            let border_radii = Point2D::new(border_radius.as_f32(),
                                            border_radius.as_f32());

            [
                PackedVertexForTextureCacheUpdate::new(&texture_rect.origin,
                                                       &color,
                                                       &zero_point,
                                                       &border_radii,
                                                       &zero_point,
                                                       &box_rect_top_left,
                                                       &box_rect_bottom_right,
                                                       &zero_size,
                                                       &zero_size,
                                                       blur_radius),
                PackedVertexForTextureCacheUpdate::new(&texture_rect.top_right(),
                                                       &color,
                                                       &zero_point,
                                                       &border_radii,
                                                       &zero_point,
                                                       &box_rect_top_left,
                                                       &box_rect_bottom_right,
                                                       &zero_size,
                                                       &zero_size,
                                                       blur_radius),
                PackedVertexForTextureCacheUpdate::new(&texture_rect.bottom_left(),
                                                       &color,
                                                       &zero_point,
                                                       &border_radii,
                                                       &zero_point,
                                                       &box_rect_top_left,
                                                       &box_rect_bottom_right,
                                                       &zero_size,
                                                       &zero_size,
                                                       blur_radius),
                PackedVertexForTextureCacheUpdate::new(&texture_rect.bottom_right(),
                                                       &color,
                                                       &zero_point,
                                                       &border_radii,
                                                       &zero_point,
                                                       &box_rect_top_left,
                                                       &box_rect_bottom_right,
                                                       &zero_size,
                                                       &zero_size,
                                                       blur_radius),
            ]
        });
    }

    fn add_rect_to_raster_batch<F>(&mut self,
                                   dest_texture_id: TextureId,
                                   color_texture_id: TextureId,
                                   program_id: ProgramId,
                                   blur_direction: Option<AxisDirection>,
                                   dest_rect: &Rect<u32>,
                                   border_type: BorderType,
                                   f: F)
                                   where F: Fn(&Rect<f32>) -> [PackedVertexForTextureCacheUpdate; 4] {
        // FIXME(pcwalton): Use a hash table if this linear search shows up in the profile.
        for batch in &mut self.raster_batches {
            if batch.add_rect_if_possible(dest_texture_id,
                                          color_texture_id,
                                          program_id,
                                          blur_direction,
                                          dest_rect,
                                          border_type,
                                          &f) {
                return;
            }
        }

        let raster_op_target = if self.device.texture_has_alpha(dest_texture_id) {
            self.raster_op_target_rgba8
        } else {
            self.raster_op_target_a8
        };

        let mut raster_batch = RasterBatch::new(raster_op_target,
                                                self.max_raster_op_size,
                                                program_id,
                                                blur_direction,
                                                color_texture_id,
                                                dest_texture_id);

        let added = raster_batch.add_rect_if_possible(dest_texture_id,
                                                      color_texture_id,
                                                      program_id,
                                                      blur_direction,
                                                      dest_rect,
                                                      border_type,
                                                      &f);
        debug_assert!(added);
        self.raster_batches.push(raster_batch);
    }

    fn flush_raster_batches(&mut self) {
        let batches = mem::replace(&mut self.raster_batches, vec![]);
        if !batches.is_empty() {
            //println!("flushing {:?} raster batches", batches.len());

            gl::disable(gl::DEPTH_TEST);
            gl::disable(gl::SCISSOR_TEST);

            // Disable MSAA here for raster ops
            self.enable_msaa(false);

            let projection = Matrix4::ortho(0.0,
                                            self.max_raster_op_size as f32,
                                            0.0,
                                            self.max_raster_op_size as f32,
                                            ORTHO_NEAR_PLANE,
                                            ORTHO_FAR_PLANE);

            // All horizontal blurs must complete before anything else.
            let mut remaining_batches = vec![];
            for batch in batches.into_iter() {
                if batch.blur_direction != Some(AxisDirection::Horizontal) {
                    remaining_batches.push(batch);
                    continue
                }

                self.set_up_gl_state_for_texture_cache_update(batch.page_allocator.texture_id(),
                                                              batch.color_texture_id,
                                                              batch.program_id,
                                                              batch.blur_direction,
                                                              &projection);
                self.perform_gl_texture_cache_update(batch);
            }

            // Flush the remaining batches.
            for batch in remaining_batches.into_iter() {
                self.set_up_gl_state_for_texture_cache_update(batch.page_allocator.texture_id(),
                                                              batch.color_texture_id,
                                                              batch.program_id,
                                                              batch.blur_direction,
                                                              &projection);
                self.perform_gl_texture_cache_update(batch);
            }
        }
    }

    fn set_up_gl_state_for_texture_cache_update(&mut self,
                                                target_texture_id: TextureId,
                                                color_texture_id: TextureId,
                                                program_id: ProgramId,
                                                blur_direction: Option<AxisDirection>,
                                                projection: &Matrix4) {
        if !self.device.texture_has_alpha(target_texture_id) {
            gl::enable(gl::BLEND);
            gl::blend_func(gl::SRC_ALPHA, gl::ZERO);
        } else {
            gl::disable(gl::BLEND);
        }

        self.device.bind_render_target(Some(target_texture_id));
        gl::viewport(0, 0, self.max_raster_op_size as gl::GLint, self.max_raster_op_size as gl::GLint);

        self.device.bind_program(program_id, &projection);

        self.device.bind_color_texture(color_texture_id);
        self.device.bind_mask_texture(TextureId(0));

        match blur_direction {
            Some(AxisDirection::Horizontal) => {
                self.device.set_uniform_2f(self.u_direction, 1.0, 0.0)
            }
            Some(AxisDirection::Vertical) => {
                self.device.set_uniform_2f(self.u_direction, 0.0, 1.0)
            }
            None => {}
        }
    }

    fn perform_gl_texture_cache_update(&mut self, batch: RasterBatch) {
        let vao_id = match self.raster_op_vao {
            Some(ref mut vao_id) => *vao_id,
            None => {
                let vao_id = self.device.create_vao(VertexFormat::RasterOp, None);
                self.raster_op_vao = Some(vao_id);
                vao_id
            }
        };
        self.device.bind_vao(vao_id);

        self.device.update_vao_indices(vao_id, &batch.indices[..], VertexUsageHint::Dynamic);
        self.device.update_vao_main_vertices(vao_id,
                                             &batch.vertices[..],
                                             VertexUsageHint::Dynamic);

        self.profile_counters.vertices.add(batch.indices.len());
        self.profile_counters.draw_calls.inc();

        //println!("drawing triangles due to GL texture cache update");
        self.device.draw_triangles_u16(0, batch.indices.len() as gl::GLint);

        for blit_job in batch.blit_jobs {
            self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                              blit_job.dest_origin.x as i32,
                                              blit_job.dest_origin.y as i32,
                                              blit_job.src_origin.x as i32,
                                              blit_job.src_origin.y as i32,
                                              blit_job.size.width as i32,
                                              blit_job.size.height as i32);

            match blit_job.border_type {
                BorderType::SinglePixel => {
                    // Single pixel corners
                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32 - 1,
                                                      blit_job.dest_origin.y as i32 - 1,
                                                      blit_job.src_origin.x as i32,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      (blit_job.dest_origin.x + blit_job.size.width) as i32,
                                                      blit_job.dest_origin.y as i32 - 1,
                                                      (blit_job.src_origin.x + blit_job.size.width) as i32 - 1,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32 - 1,
                                                      (blit_job.dest_origin.y + blit_job.size.height) as i32,
                                                      blit_job.src_origin.x as i32,
                                                      (blit_job.src_origin.y + blit_job.size.height) as i32 - 1,
                                                      1,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      (blit_job.dest_origin.x + blit_job.size.width) as i32,
                                                      (blit_job.dest_origin.y + blit_job.size.height) as i32,
                                                      (blit_job.src_origin.x + blit_job.size.width) as i32 - 1,
                                                      (blit_job.src_origin.y + blit_job.size.height) as i32 - 1,
                                                      1,
                                                      1);

                    // Horizontal edges
                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32,
                                                      blit_job.dest_origin.y as i32 - 1,
                                                      blit_job.src_origin.x as i32,
                                                      blit_job.src_origin.y as i32,
                                                      blit_job.size.width as i32,
                                                      1);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32,
                                                      (blit_job.dest_origin.y + blit_job.size.height) as i32,
                                                      blit_job.src_origin.x as i32,
                                                      (blit_job.src_origin.y + blit_job.size.height) as i32 - 1,
                                                      blit_job.size.width as i32,
                                                      1);

                    // Vertical edges
                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      blit_job.dest_origin.x as i32 - 1,
                                                      blit_job.dest_origin.y as i32,
                                                      blit_job.src_origin.x as i32,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      blit_job.size.height as i32);

                    self.device.read_framebuffer_rect(blit_job.dest_texture_id,
                                                      (blit_job.dest_origin.x + blit_job.size.width) as i32,
                                                      blit_job.dest_origin.y as i32,
                                                      (blit_job.src_origin.x + blit_job.size.width) as i32 - 1,
                                                      blit_job.src_origin.y as i32,
                                                      1,
                                                      blit_job.size.height as i32);

                }
                BorderType::_NoBorder => {}
            }
        }
    }

/*
    fn draw_layer(&mut self,
                  layer: &DrawLayer,
                  render_context: &RenderContext) {
        // Draw child layers first, to ensure that dependent render targets
        // have been built before they are read as a texture.
        for child in &layer.child_layers {
            self.draw_layer(child,
                            render_context);
        }

        self.device.bind_render_target(layer.texture_id);

        // TODO(gw): This may not be needed in all cases...
        let layer_origin = Point2D::new((layer.origin.x * self.device_pixel_ratio).round() as gl::GLint,
                                        (layer.origin.y * self.device_pixel_ratio).round() as gl::GLint);

        let layer_size = Size2D::new((layer.size.width * self.device_pixel_ratio).round() as u32,
                                     (layer.size.height * self.device_pixel_ratio).round() as u32);

        let layer_origin = match layer.texture_id {
            Some(..) => {
                layer_origin
            }
            None => {
                let inverted_y0 = render_context.framebuffer_size.height as gl::GLint -
                                  layer_size.height as gl::GLint -
                                  layer_origin.y;

                Point2D::new(layer_origin.x, inverted_y0)
            }
        };

        gl::scissor(layer_origin.x,
                    layer_origin.y,
                    layer_size.width as gl::GLint,
                    layer_size.height as gl::GLint);

        gl::viewport(layer_origin.x,
                     layer_origin.y,
                     layer_size.width as gl::GLint,
                     layer_size.height as gl::GLint);
        let clear_color = if layer.texture_id.is_some() {
            ColorF::new(0.0, 0.0, 0.0, 0.0)
        } else {
            ColorF::new(1.0, 1.0, 1.0, 0.0)
        };
        gl::clear_color(clear_color.r, clear_color.g, clear_color.b, clear_color.a);
        gl::clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);

        let projection = Matrix4::ortho(0.0,
                                        layer.size.width,
                                        layer.size.height,
                                        0.0,
                                        ORTHO_NEAR_PLANE,
                                        ORTHO_FAR_PLANE);

        for cmd in &layer.commands {
            match cmd {
                &DrawCommand::Clear(ref info) => {
                    let mut clear_bits = 0;
                    if info.clear_color {
                        clear_bits |= gl::COLOR_BUFFER_BIT;
                    }
                    if info.clear_z {
                        clear_bits |= gl::DEPTH_BUFFER_BIT;
                    }
                    if info.clear_stencil {
                        clear_bits |= gl::STENCIL_BUFFER_BIT;
                    }
                    gl::clear(clear_bits);
                }
                &DrawCommand::Batch(ref info) => {
                    // TODO: probably worth sorting front to back to minimize overdraw (if profiling shows fragment / rop bound)

                    self.enable_msaa(true);

                    if layer.texture_id.is_some() {
                        gl::disable(gl::BLEND);
                    } else {
                        gl::enable(gl::BLEND);
                        gl::blend_func_separate(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA,
                                                gl::ONE, gl::ONE);
                        gl::blend_equation(gl::FUNC_ADD);
                    }

                    self.device.bind_program(self.quad_program_id,
                                             &projection);

                    if !info.offset_palette.is_empty() {
                        // TODO(gw): Avoid alloc here...
                        let mut floats = Vec::new();
                        for vec in &info.offset_palette {
                            floats.push(vec.stacking_context_x0);
                            floats.push(vec.stacking_context_y0);
                            floats.push(vec.render_target_x0);
                            floats.push(vec.render_target_y0);
                        }

                        self.device.set_uniform_vec4_array(self.u_quad_offset_array,
                                                           &floats);
                    }

                    self.device.set_uniform_mat4_array(self.u_quad_transform_array,
                                                       &info.matrix_palette);

                    // Render any masks to the stencil buffer.
                    for region in &info.regions {
                        let layer_rect = Rect::new(layer.origin, layer.size);
                        let mut valid_mask_count = 0;
                        for mask in &region.masks {

                            // If the mask is larger than the viewport / scissor rect
                            // then there is no need to draw it.
                            if mask.transform == Matrix4::identity() &&
                               mask.rect.contains_rect(&layer_rect) {
                                continue;
                            }

                            // First time we find a valid mask, clear stencil and setup render states
                            if valid_mask_count == 0 {
                                gl::clear(gl::STENCIL_BUFFER_BIT);
                                gl::enable(gl::STENCIL_TEST);
                                gl::color_mask(false, false, false, false);
                                gl::depth_mask(false);
                                gl::stencil_mask(0xff);
                                gl::stencil_func(gl::ALWAYS, 1, 0xff);
                                gl::stencil_op(gl::KEEP, gl::INCR, gl::INCR)
                            }

                            // TODO(gw): The below is a copy pasta and can be trivially optimized.
                            let (mut indices, mut vertices) = (vec![], vec![]);
                            indices.push(0);
                            indices.push(1);
                            indices.push(2);
                            indices.push(2);
                            indices.push(3);
                            indices.push(1);

                            let color = ColorF::new(0.0, 0.0, 0.0, 0.0);

                            let x0 = mask.rect.origin.x;
                            let y0 = mask.rect.origin.y;
                            let x1 = x0 + mask.rect.size.width;
                            let y1 = y0 + mask.rect.size.height;

                            vertices.extend_from_slice(&[
                                PackedVertex::from_components(
                                    x0, y0,
                                    &color,
                                    0.0, 0.0,
                                    0.0, 0.0),
                                PackedVertex::from_components(
                                    x1, y0,
                                    &color,
                                    1.0, 0.0,
                                    1.0, 0.0),
                                PackedVertex::from_components(
                                    x0, y1,
                                    &color,
                                    0.0, 1.0,
                                    0.0, 1.0),
                                PackedVertex::from_components(
                                    x1, y1,
                                    &color,
                                    1.0, 1.0,
                                    1.0, 1.0),
                            ]);

                            let wvp = projection.mul(&mask.transform);
                            self.device.bind_program(self.mask_program_id, &wvp);

                            draw_simple_triangles(&mut self.simple_triangles_vao,
                                                  &mut self.device,
                                                  &mut self.profile_counters,
                                                  &indices[..],
                                                  &vertices[..],
                                                  TextureId(0));

                            valid_mask_count += 1;
                        }

                        // If any masks were found, enable stencil test rejection.
                        // TODO(gw): This may be faster to switch the logic and
                        //           rely on sfail!
                        if valid_mask_count > 0 {
                            gl::stencil_op(gl::KEEP, gl::KEEP, gl::KEEP);
                            gl::stencil_func(gl::EQUAL, valid_mask_count, 0xff);
                            gl::color_mask(true, true, true, true);
                            gl::depth_mask(true);
                            self.device.bind_program(self.quad_program_id,
                                                     &projection);
                        }

                        for draw_call in &region.draw_calls {
                            let vao_id = self.get_or_create_similar_vao_with_offset(
                                draw_call.vertex_buffer_id,
                                VertexFormat::Rectangles,
                                draw_call.first_instance);
                            self.device.bind_vao(vao_id);

                            if !draw_call.tile_params.is_empty() {
                                // TODO(gw): Avoid alloc here...
                                let mut floats = Vec::new();
                                for vec in &draw_call.tile_params {
                                    floats.push(vec.u0);
                                    floats.push(vec.v0);
                                    floats.push(vec.u_size);
                                    floats.push(vec.v_size);
                                }

                                self.device.set_uniform_vec4_array(self.u_tile_params,
                                                                   &floats);
                            }

                            if !draw_call.clip_rects.is_empty() {
                                // TODO(gw): Avoid alloc here...
                                let mut floats = Vec::new();
                                for rect in &draw_call.clip_rects {
                                    floats.push(rect.origin.x);
                                    floats.push(rect.origin.y);
                                    floats.push(rect.origin.x + rect.size.width);
                                    floats.push(rect.origin.y + rect.size.height);
                                }

                                self.device.set_uniform_vec4_array(self.u_clip_rects,
                                                                   &floats);
                            }

                            self.device.bind_mask_texture(draw_call.mask_texture_id);
                            self.device.bind_color_texture(draw_call.color_texture_id);

                            // TODO(gw): Although a minor cost, this is an extra hashtable lookup for every
                            //           draw call, when the batch textures are (almost) always the same.
                            //           This could probably be cached or provided elsewhere.
                            let color_size = self.device
                                                 .get_texture_dimensions(draw_call.color_texture_id);
                            let mask_size = self.device
                                                .get_texture_dimensions(draw_call.mask_texture_id);
                            self.device.set_uniform_4f(self.u_atlas_params,
                                                       color_size.0 as f32,
                                                       color_size.1 as f32,
                                                       mask_size.0 as f32,
                                                       mask_size.1 as f32);

                            self.profile_counters.draw_calls.inc();

                            self.device
                                .draw_triangles_instanced_u16(0, 6, draw_call.instance_count as i32);
                        }

                        // Disable stencil test if it was used
                        if valid_mask_count > 0 {
                            gl::disable(gl::STENCIL_TEST);
                        }
                    }
                }
                &DrawCommand::CompositeBatch(ref info) => {
                    let needs_fb = info.operation.needs_framebuffer();

                    let alpha;
                    if needs_fb {
                        match info.operation {
                            CompositionOp::MixBlend(blend_mode) => {
                                gl::disable(gl::BLEND);
                                self.device.bind_program(render_context.blend_program_id,
                                                         &projection);
                                self.device.set_uniform_4f(self.u_blend_params,
                                                           blend_mode as i32 as f32,
                                                           0.0,
                                                           0.0,
                                                           0.0);
                            }
                            _ => unreachable!(),
                        }
                        self.device.bind_mask_texture(self.temporary_fb_texture);
                        alpha = 1.0;
                    } else {
                        gl::enable(gl::BLEND);

                        let program;
                        let mut filter_params = None;
                        match info.operation {
                            CompositionOp::Filter(LowLevelFilterOp::Brightness(
                                    amount)) => {
                                gl::blend_func(gl::CONSTANT_COLOR, gl::ZERO);
                                gl::blend_equation(gl::FUNC_ADD);
                                gl::blend_color(amount.to_f32_px(),
                                                amount.to_f32_px(),
                                                amount.to_f32_px(),
                                                1.0);
                                alpha = 1.0;
                                program = self.blit_program_id;
                            }
                            CompositionOp::Filter(LowLevelFilterOp::Opacity(amount)) => {
                                gl::blend_func_separate(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA,
                                                        gl::ONE, gl::ONE);
                                gl::blend_equation(gl::FUNC_ADD);
                                alpha = amount.to_f32_px();
                                program = self.blit_program_id;
                            }
                            CompositionOp::Filter(filter_op) => {
                                alpha = 1.0;
                                program = render_context.filter_program_id;

                                let (opcode, amount, param0, param1) = match filter_op {
                                    LowLevelFilterOp::Blur(radius,
                                                           AxisDirection::Horizontal) => {
                                        gl::blend_func_separate(gl::SRC_ALPHA,
                                                                gl::ONE_MINUS_SRC_ALPHA,
                                                                gl::ONE,
                                                                gl::ONE);
                                        (0.0,
                                         radius.to_f32_px() * self.device_pixel_ratio,
                                         1.0,
                                         0.0)
                                    }
                                    LowLevelFilterOp::Blur(radius,
                                                           AxisDirection::Vertical) => {
                                        gl::blend_func_separate(gl::SRC_ALPHA,
                                                                gl::ONE_MINUS_SRC_ALPHA,
                                                                gl::ONE,
                                                                gl::ONE);
                                        (0.0,
                                         radius.to_f32_px() * self.device_pixel_ratio,
                                         0.0,
                                         1.0)
                                    }
                                    LowLevelFilterOp::Contrast(amount) => {
                                        gl::disable(gl::BLEND);
                                        (1.0, amount.to_f32_px(), 0.0, 0.0)
                                    }
                                    LowLevelFilterOp::Grayscale(amount) => {
                                        gl::disable(gl::BLEND);
                                        (2.0, amount.to_f32_px(), 0.0, 0.0)
                                    }
                                    LowLevelFilterOp::HueRotate(angle) => {
                                        gl::disable(gl::BLEND);
                                        (3.0,
                                         (angle as f32) / ANGLE_FLOAT_TO_FIXED,
                                         0.0,
                                         0.0)
                                    }
                                    LowLevelFilterOp::Invert(amount) => {
                                        gl::disable(gl::BLEND);
                                        (4.0, amount.to_f32_px(), 0.0, 0.0)
                                    }
                                    LowLevelFilterOp::Saturate(amount) => {
                                        gl::disable(gl::BLEND);
                                        (5.0, amount.to_f32_px(), 0.0, 0.0)
                                    }
                                    LowLevelFilterOp::Sepia(amount) => {
                                        gl::disable(gl::BLEND);
                                        (6.0, amount.to_f32_px(), 0.0, 0.0)
                                    }
                                    LowLevelFilterOp::Brightness(_) |
                                    LowLevelFilterOp::Opacity(_) => {
                                        // Expressible using GL blend modes, so not handled
                                        // here.
                                        unreachable!()
                                    }
                                };

                                filter_params = Some((opcode, amount, param0, param1));
                            }
                            CompositionOp::MixBlend(MixBlendMode::Multiply) => {
                                gl::blend_func(gl::DST_COLOR, gl::ZERO);
                                gl::blend_equation(gl::FUNC_ADD);
                                program = self.blit_program_id;
                                alpha = 1.0;
                            }
                            CompositionOp::MixBlend(MixBlendMode::Darken) => {
                                gl::blend_func(gl::ONE, gl::ONE);
                                gl::blend_equation(GL_BLEND_MIN);
                                program = self.blit_program_id;
                                alpha = 1.0;
                            }
                            CompositionOp::MixBlend(MixBlendMode::Lighten) => {
                                gl::blend_func(gl::ONE, gl::ONE);
                                gl::blend_equation(GL_BLEND_MAX);
                                program = self.blit_program_id;
                                alpha = 1.0;
                            }
                            _ => unreachable!(),
                        }

                        self.device.bind_program(program, &projection);

                        if let Some(ref filter_params) = filter_params {
                            self.device.set_uniform_4f(self.u_filter_params,
                                                       filter_params.0,
                                                       filter_params.1,
                                                       filter_params.2,
                                                       filter_params.3);
                        }
                    }

                    let (mut indices, mut vertices) = (vec![], vec![]);
                    for job in &info.jobs {
                        let p0 = Point2D::new(job.rect.origin.x as f32, job.rect.origin.y as f32);
                        let p1 = Point2D::new(job.rect.max_x() as f32, job.rect.max_y() as f32);

                        // TODO(glennw): No need to re-init this FB working copy texture
                        // every time...
                        if needs_fb {
                            let fb_rect_size = Size2D::new(job.rect.size.width as f32 * render_context.device_pixel_ratio,
                                                           job.rect.size.height as f32 * render_context.device_pixel_ratio);

                            let inverted_y0 = layer.size.height -
                                              job.rect.size.height as f32 -
                                              p0.y;
                            let fb_rect_origin = Point2D::new(
                                p0.x * render_context.device_pixel_ratio,
                                inverted_y0 * render_context.device_pixel_ratio);

                            self.device.init_texture_if_necessary(self.temporary_fb_texture,
                                                                  fb_rect_size.width as u32,
                                                                  fb_rect_size.height as u32,
                                                                  ImageFormat::RGBA8,
                                                                  TextureFilter::Nearest,
                                                                  RenderTargetMode::None);
                            self.device.read_framebuffer_rect(
                                self.temporary_fb_texture,
                                0,
                                0,
                                fb_rect_origin.x as i32,
                                fb_rect_origin.y as i32,
                                fb_rect_size.width as i32,
                                fb_rect_size.height as i32);
                        }

                        let vertex_count = vertices.len() as u16;
                        indices.push(vertex_count + 0);
                        indices.push(vertex_count + 1);
                        indices.push(vertex_count + 2);
                        indices.push(vertex_count + 2);
                        indices.push(vertex_count + 3);
                        indices.push(vertex_count + 1);

                        let color = ColorF::new(1.0, 1.0, 1.0, alpha);

                        let ChildLayerIndex(child_layer_index) = job.child_layer_index;
                        let src_target = &layer.child_layers[child_layer_index as usize];
                        debug_assert!(src_target.texture_id.unwrap() == info.texture_id);

                        let pixel_uv = Rect::new(
                            Point2D::new(src_target.origin.x as u32,
                                         src_target.origin.y as u32),
                            Size2D::new(src_target.size.width as u32,
                                        src_target.size.height as u32));

                        let (texture_width, texture_height) = self.device.get_texture_dimensions(info.texture_id);
                        let texture_width = texture_width as f32 / self.device_pixel_ratio;
                        let texture_height = texture_height as f32 / self.device_pixel_ratio;
                        let texture_uv = Rect::new(
                            Point2D::new(
                                pixel_uv.origin.x as f32 / texture_width,
                                pixel_uv.origin.y as f32 / texture_height),
                            Size2D::new(pixel_uv.size.width as f32 / texture_width,
                                        pixel_uv.size.height as f32 / texture_height));

                        let tl = job.transform.transform_point(&p0);
                        let tr = job.transform.transform_point(&Point2D::new(p1.x, p0.y));
                        let br = job.transform.transform_point(&p1);
                        let bl = job.transform.transform_point(&Point2D::new(p0.x, p1.y));

                        if needs_fb {
                            vertices.extend_from_slice(&[
                                PackedVertex::from_components(
                                    tl.x, tl.y,
                                    &color,
                                    texture_uv.origin.x, texture_uv.max_y(),
                                    0.0, 1.0),
                                PackedVertex::from_components(
                                    tr.x, tr.y,
                                    &color,
                                    texture_uv.max_x(), texture_uv.max_y(),
                                    1.0, 1.0),
                                PackedVertex::from_components(
                                    bl.x, bl.y,
                                    &color,
                                    texture_uv.origin.x, texture_uv.origin.y,
                                    0.0, 0.0),
                                PackedVertex::from_components(
                                    br.x, br.y,
                                    &color,
                                    texture_uv.max_x(), texture_uv.origin.y,
                                    1.0, 0.0),
                            ]);
                        } else {
                            vertices.extend_from_slice(&[
                                PackedVertex::from_components_unscaled_muv(
                                    tl.x, tl.y,
                                    &color,
                                    texture_uv.origin.x, texture_uv.max_y(),
                                    texture_width as u16, texture_height as u16),
                                PackedVertex::from_components_unscaled_muv(
                                    tr.x, tr.y,
                                    &color,
                                    texture_uv.max_x(), texture_uv.max_y(),
                                    texture_width as u16, texture_height as u16),
                                PackedVertex::from_components_unscaled_muv(
                                    bl.x, bl.y,
                                    &color,
                                    texture_uv.origin.x, texture_uv.origin.y,
                                    texture_width as u16, texture_height as u16),
                                PackedVertex::from_components_unscaled_muv(
                                    br.x, br.y,
                                    &color,
                                    texture_uv.max_x(), texture_uv.origin.y,
                                    texture_width as u16, texture_height as u16),
                            ]);
                        }
                    }

                    draw_simple_triangles(&mut self.simple_triangles_vao,
                                          &mut self.device,
                                          &mut self.profile_counters,
                                          &indices[..],
                                          &vertices[..],
                                          info.texture_id);
                }
            }
        }
    }
*/

    fn draw_text(&mut self,
                 text_buffer: &TextBuffer,
                 texture: TextureId) {
        self.device.bind_render_target(Some(self.text_composite_target));
        gl::viewport(0, 0, (TEXT_TARGET_SIZE * self.device_pixel_ratio as u32) as gl::GLint, (TEXT_TARGET_SIZE * self.device_pixel_ratio as u32) as gl::GLint);

        gl::clear_color(1.0, 1.0, 1.0, 0.0);
        gl::clear(gl::COLOR_BUFFER_BIT);

        let projection = Matrix4::ortho(0.0,
                                        TEXT_TARGET_SIZE as f32,
                                        0.0,
                                        TEXT_TARGET_SIZE as f32,
                                        ORTHO_NEAR_PLANE,
                                        ORTHO_FAR_PLANE);

        let (mut indices, mut vertices) = (vec![], vec![]);

        for (_, run) in &text_buffer.texts {
            for glyph in &run.glyphs {
                let base_index = vertices.len() as u16;
                indices.push(base_index + 0);
                indices.push(base_index + 1);
                indices.push(base_index + 2);
                indices.push(base_index + 2);
                indices.push(base_index + 3);
                indices.push(base_index + 1);

                vertices.extend_from_slice(&[
                    FontVertex {
                        x: glyph.p0.x,
                        y: glyph.p0.y,
                        s: glyph.st0.x,
                        t: glyph.st0.y,
                    },
                    FontVertex {
                        x: glyph.p1.x,
                        y: glyph.p0.y,
                        s: glyph.st1.x,
                        t: glyph.st0.y,
                    },
                    FontVertex {
                        x: glyph.p0.x,
                        y: glyph.p1.y,
                        s: glyph.st0.x,
                        t: glyph.st1.y,
                    },
                    FontVertex {
                        x: glyph.p1.x,
                        y: glyph.p1.y,
                        s: glyph.st1.x,
                        t: glyph.st1.y,
                    },
                ]);
            }
        }

        let vao_id = self.device.create_vao(VertexFormat::Font, None);

        self.device.bind_program(self.text_program_id, &projection);
        self.device.bind_color_texture(texture);
        self.device.bind_vao(vao_id);
        self.device.update_vao_indices(vao_id, &indices[..], VertexUsageHint::Dynamic);
        self.device.update_vao_main_vertices(vao_id, &vertices[..], VertexUsageHint::Dynamic);
        self.device.draw_triangles_u16(0, indices.len() as gl::GLint);
        self.device.delete_vao(vao_id);
    }

    fn draw_tile_frame(&mut self,
                       tile_frame: &TileFrame,
                       framebuffer_size: &Size2D<u32>) {
        let projection = Matrix4::ortho(0.0,
                                        tile_frame.viewport_size.width as f32,
                                        tile_frame.viewport_size.height as f32,
                                        0.0,
                                        ORTHO_NEAR_PLANE,
                                        ORTHO_FAR_PLANE);

        let mut g = GpuProfile::new();
        g.begin();

        gl::disable(gl::BLEND);
        gl::depth_mask(false);
        gl::stencil_mask(0xff);
        gl::disable(gl::STENCIL_TEST);

        self.draw_text(&tile_frame.text_buffer, tile_frame.color_texture_id);

        g.end();
        let text_ns = g.get();
        g.begin();

        self.device.bind_render_target(None);
        gl::viewport(0, 0, framebuffer_size.width as i32, framebuffer_size.height as i32);

        gl::clear(gl::STENCIL_BUFFER_BIT);
        gl::enable(gl::STENCIL_TEST);
        gl::stencil_func(gl::ALWAYS, 1, 0xff);
        gl::stencil_op(gl::REPLACE, gl::REPLACE, gl::REPLACE);

        let misc_data = MiscTileData {
            background_color: ColorF::new(1.0, 1.0, 1.0, 1.0),
        };

        let misc_ubos = gl::gen_buffers(2);
        let misc_ubo = misc_ubos[0];
        let cmd_ubo = misc_ubos[1];

        gl::bind_buffer(gl::UNIFORM_BUFFER, misc_ubo);
        gl::buffer_data_raw(gl::UNIFORM_BUFFER, &misc_data, gl::STATIC_DRAW);

        gl::bind_buffer(gl::UNIFORM_BUFFER, cmd_ubo);
        gl::buffer_data(gl::UNIFORM_BUFFER, &tile_frame.uniforms.cmd_ubo, gl::STATIC_DRAW);

        let layer_ubos = gl::gen_buffers(tile_frame.uniforms.layer_ubos.len() as gl::GLint);
        let rect_ubos = gl::gen_buffers(tile_frame.uniforms.rect_ubos.len() as gl::GLint);
        let clip_ubos = gl::gen_buffers(tile_frame.uniforms.clip_ubos.len() as gl::GLint);
        let image_ubos = gl::gen_buffers(tile_frame.uniforms.image_ubos.len() as gl::GLint);
        let gradient_ubos = gl::gen_buffers(tile_frame.uniforms.gradient_ubos.len() as gl::GLint);
        let gradient_stop_ubos = gl::gen_buffers(tile_frame.uniforms.gradient_stop_ubos.len() as gl::GLint);
        //let glyph_ubos = gl::gen_buffers(tile_frame.uniforms.glyph_ubos.len() as gl::GLint);
        let text_ubos = gl::gen_buffers(tile_frame.uniforms.text_ubos.len() as gl::GLint);

        for ubo_index in 0..layer_ubos.len() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, layer_ubos[ubo_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &tile_frame.uniforms.layer_ubos[ubo_index].items, gl::STATIC_DRAW);
        }

        for ubo_index in 0..rect_ubos.len() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, rect_ubos[ubo_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &tile_frame.uniforms.rect_ubos[ubo_index].items, gl::STATIC_DRAW);
        }

        for ubo_index in 0..clip_ubos.len() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, clip_ubos[ubo_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &tile_frame.uniforms.clip_ubos[ubo_index].items, gl::STATIC_DRAW);
        }

        for ubo_index in 0..image_ubos.len() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, image_ubos[ubo_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &tile_frame.uniforms.image_ubos[ubo_index].items, gl::STATIC_DRAW);
        }

        for ubo_index in 0..gradient_ubos.len() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, gradient_ubos[ubo_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &tile_frame.uniforms.gradient_ubos[ubo_index].items, gl::STATIC_DRAW);
        }

        for ubo_index in 0..gradient_stop_ubos.len() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, gradient_stop_ubos[ubo_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &tile_frame.uniforms.gradient_stop_ubos[ubo_index].items, gl::STATIC_DRAW);
        }

/*
        for ubo_index in 0..glyph_ubos.len() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, glyph_ubos[ubo_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &tile_frame.uniforms.glyph_ubos[ubo_index].items, gl::STATIC_DRAW);
        }
*/

        for ubo_index in 0..text_ubos.len() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, text_ubos[ubo_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &tile_frame.uniforms.text_ubos[ubo_index].items, gl::STATIC_DRAW);
        }

        gl::bind_buffer(gl::UNIFORM_BUFFER, 0);
        gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Misc as u32, misc_ubo);
        gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::IndexBuffer as u32, cmd_ubo);

        self.device.bind_program(self.scene_program_id, &projection);
        self.device.bind_mask_texture(self.text_composite_target);
        self.device.bind_vao(self.quad_vao_id);

        let mut tile_batches: Vec<TileBatch> = Vec::new();

        for (tile_index, tile) in tile_frame.tiles.iter().enumerate() {
            let need_new_batch = match tile_batches.last() {
                Some(ref batch) => !batch.can_add(tile),
                None => true,
            };

            if need_new_batch {
                tile_batches.push(TileBatch::new(tile));
            }

            let batch = tile_batches.last_mut().unwrap();
            batch.add_tile(tile);

            if self.enable_profiler {
                let tile_x0 = tile.rect.origin.x as f32;
                let tile_y0 = tile.rect.origin.y as f32;
                let tile_x1 = tile_x0 + tile.rect.size.width as f32;
                let tile_y1 = tile_y0 + tile.rect.size.height as f32;

                self.debug.add_line(tile_x0,
                                    tile_y0,
                                    &ColorF::new(0.0, 0.0, 0.0, 1.0),
                                    tile_x1,
                                    tile_y0,
                                    &ColorF::new(0.0, 0.0, 0.0, 1.0));
                self.debug.add_line(tile_x0,
                                    tile_y1,
                                    &ColorF::new(0.0, 0.0, 0.0, 1.0),
                                    tile_x1,
                                    tile_y1,
                                    &ColorF::new(0.0, 0.0, 0.0, 1.0));
                self.debug.add_line(tile_x0,
                                    tile_y0,
                                    &ColorF::new(0.0, 0.0, 0.0, 1.0),
                                    tile_x0,
                                    tile_y1,
                                    &ColorF::new(0.0, 0.0, 0.0, 1.0));
                self.debug.add_line(tile_x1,
                                    tile_y0,
                                    &ColorF::new(0.0, 0.0, 0.0, 1.0),
                                    tile_x1,
                                    tile_y1,
                                    &ColorF::new(0.0, 0.0, 0.0, 1.0));
                self.debug.add_text((tile_x0 + tile_x1) * 0.5,
                                    (tile_y0 + tile_y1) * 0.5,
                                    &format!("{}", tile.cmd_count),
                                    &ColorF::new(1.0, 0.0, 1.0, 1.0));
            }
        }

        let mut current_layer_ubo = usize::MAX;
        let mut current_rect_ubo = usize::MAX;
        let mut current_clip_ubo = usize::MAX;
        let mut current_image_ubo = usize::MAX;
        let mut current_gradient_ubo = usize::MAX;
        let mut current_stop_ubo = usize::MAX;
        let mut current_text_ubo = usize::MAX;
        //let current_glyph_ubo: u32,

        let batch_ubos = gl::gen_buffers(tile_batches.len() as gl::GLint);

        for (batch_index, batch) in tile_batches.iter().enumerate() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, batch_ubos[batch_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &batch.instances, gl::STATIC_DRAW);
            gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Instances as u32, batch_ubos[batch_index]);

            if current_layer_ubo != batch.layer_ubo_index {
                current_layer_ubo = batch.layer_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Layers as u32, layer_ubos[current_layer_ubo]);
            }

            if current_rect_ubo != batch.rect_ubo_index {
                current_rect_ubo = batch.rect_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Rectangles as u32, rect_ubos[current_rect_ubo]);
            }

            if current_clip_ubo != batch.clip_ubo_index {
                current_clip_ubo = batch.clip_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Clips as u32, clip_ubos[current_clip_ubo]);
            }

            if current_image_ubo != batch.image_ubo_index {
                current_image_ubo = batch.image_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Images as u32, image_ubos[current_image_ubo]);
            }

            if current_gradient_ubo != batch.gradient_ubo_index {
                current_gradient_ubo = batch.gradient_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Gradients as u32, gradient_ubos[current_gradient_ubo]);
            }

            if current_stop_ubo != batch.stop_ubo_index {
                current_stop_ubo = batch.stop_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::GradientStops as u32, gradient_stop_ubos[current_stop_ubo]);
            }

            if current_text_ubo != batch.text_ubo_index {
                current_text_ubo = batch.text_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Texts as u32, text_ubos[current_text_ubo]);
            }
/*
            if current_glyph_ubo != glyph_ubos[tile.glyph_ubo_index] {
                current_glyph_ubo = glyph_ubos[tile.glyph_ubo_index];
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Glyphs as u32, current_glyph_ubo);
            }
*/

            self.device.draw_indexed_triangles_instanced_u16(self.quad_vao_index_count, batch.instances.len() as gl::GLint);
        }

        g.end();
        let tile_ns = g.get();
        g.begin();

        gl::stencil_op(gl::KEEP, gl::KEEP, gl::KEEP);
        gl::stencil_func(gl::EQUAL, 0, 0xff);

        self.device.bind_program(self.complex_program_id, &projection);

        current_layer_ubo = usize::MAX;
        current_rect_ubo = usize::MAX;
        current_clip_ubo = usize::MAX;
        current_image_ubo = usize::MAX;
        current_gradient_ubo = usize::MAX;
        current_stop_ubo = usize::MAX;
        current_text_ubo = usize::MAX;

        for (batch_index, batch) in tile_batches.iter().enumerate() {
            //gl::bind_buffer(gl::UNIFORM_BUFFER, batch_ubos[batch_index]);
            gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Instances as u32, batch_ubos[batch_index]);

            if current_layer_ubo != batch.layer_ubo_index {
                current_layer_ubo = batch.layer_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Layers as u32, layer_ubos[current_layer_ubo]);
            }

            if current_rect_ubo != batch.rect_ubo_index {
                current_rect_ubo = batch.rect_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Rectangles as u32, rect_ubos[current_rect_ubo]);
            }

            if current_clip_ubo != batch.clip_ubo_index {
                current_clip_ubo = batch.clip_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Clips as u32, clip_ubos[current_clip_ubo]);
            }

            if current_image_ubo != batch.image_ubo_index {
                current_image_ubo = batch.image_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Images as u32, image_ubos[current_image_ubo]);
            }

            if current_gradient_ubo != batch.gradient_ubo_index {
                current_gradient_ubo = batch.gradient_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Gradients as u32, gradient_ubos[current_gradient_ubo]);
            }

            if current_stop_ubo != batch.stop_ubo_index {
                current_stop_ubo = batch.stop_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::GradientStops as u32, gradient_stop_ubos[current_stop_ubo]);
            }

            if current_text_ubo != batch.text_ubo_index {
                current_text_ubo = batch.text_ubo_index;
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Texts as u32, text_ubos[current_text_ubo]);
            }
/*
            if current_glyph_ubo != glyph_ubos[tile.glyph_ubo_index] {
                current_glyph_ubo = glyph_ubos[tile.glyph_ubo_index];
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UboBindLocation::Glyphs as u32, current_glyph_ubo);
            }
*/

            self.device.draw_indexed_triangles_instanced_u16(self.quad_vao_index_count, batch.instances.len() as gl::GLint);
        }

        gl::disable(gl::STENCIL_TEST);

        gl::delete_buffers(&misc_ubos);
        gl::delete_buffers(&layer_ubos);
        gl::delete_buffers(&rect_ubos);
        gl::delete_buffers(&clip_ubos);
        gl::delete_buffers(&image_ubos);
        gl::delete_buffers(&gradient_ubos);
        gl::delete_buffers(&gradient_stop_ubos);
        gl::delete_buffers(&text_ubos);
        gl::delete_buffers(&batch_ubos);
        //gl::delete_buffers(&glyph_ubos);

        g.end();
        let complex_ns = g.get();

        let text_ms = text_ns as f64 / 1000000.0;
        let tile_ms = tile_ns as f64 / 1000000.0;
        let complex_ms = complex_ns as f64 / 1000000.0;

        println!("text={} tile={} complex={}", text_ms, tile_ms, complex_ms);
    }

    fn draw_frame(&mut self, framebuffer_size: Size2D<u32>) {
        if let Some(frame) = self.current_frame.take() {
            // TODO: cache render targets!

            let render_context = RenderContext {
                blend_program_id: self.blend_program_id,
                filter_program_id: self.filter_program_id,
                device_pixel_ratio: self.device_pixel_ratio,
                framebuffer_size: framebuffer_size,
            };

            // TODO(gw): Doesn't work well with transforms.
            //           Look into this...
            gl::disable(gl::DEPTH_TEST);
            gl::disable(gl::SCISSOR_TEST);
            gl::disable(gl::BLEND);

            if let Some(ref tile_frame) = frame.tile_frame {
                self.draw_tile_frame(tile_frame, &framebuffer_size);
            }

            // Restore frame - avoid borrow checker!
            self.current_frame = Some(frame);
        }
    }

    fn get_or_create_similar_vao_with_offset(&mut self,
                                             source_vertex_buffer_id: VertexBufferId,
                                             format: VertexFormat,
                                             offset: u32)
                                             -> VAOId {
        let source_vertex_buffers_and_offsets =
            self.vertex_buffers.get_mut(&source_vertex_buffer_id)
                               .expect("Didn't find source vertex buffer ID in \
                                        `get_or_create_similar_vao_with_offset()`!");
        if let Some(vertex_buffer_and_offset) =
                source_vertex_buffers_and_offsets.iter().find(|vertex_buffer| {
                    vertex_buffer.offset == offset
                }) {
            return vertex_buffer_and_offset.buffer.vao_id
        }

        let vao =
            self.device.create_similar_vao(format,
                                           source_vertex_buffers_and_offsets[0].buffer.vao_id,
                                           offset);
        source_vertex_buffers_and_offsets.push(VertexBufferAndOffset {
            buffer: VertexBuffer {
                vao_id: vao,
            },
            offset: offset,
        });
        vao
    }
}

#[derive(Clone, Debug)]
pub struct RendererOptions {
    pub device_pixel_ratio: f32,
    pub resource_path: PathBuf,
    pub enable_aa: bool,
    pub enable_msaa: bool,
    pub enable_profiler: bool,
    pub tile_size: Size2D<i32>,
}

fn draw_simple_triangles(simple_triangles_vao: &mut Option<VAOId>,
                         device: &mut Device,
                         profile_counters: &mut RendererProfileCounters,
                         indices: &[u16],
                         vertices: &[PackedVertex],
                         texture: TextureId) {
    let vao_id = match *simple_triangles_vao {
        Some(ref mut vao_id) => *vao_id,
        None => {
            let vao_id = device.create_vao(VertexFormat::Triangles, None);
            *simple_triangles_vao = Some(vao_id);
            vao_id
        }
    };
    device.bind_color_texture(texture);
    device.bind_vao(vao_id);
    device.update_vao_indices(vao_id, &indices[..], VertexUsageHint::Dynamic);
    device.update_vao_main_vertices(vao_id, &vertices[..], VertexUsageHint::Dynamic);

    profile_counters.vertices.add(indices.len());
    profile_counters.draw_calls.inc();

    device.draw_triangles_u16(0, indices.len() as gl::GLint);
}

struct VertexBufferAndOffset {
    buffer: VertexBuffer,
    offset: u32,
}

