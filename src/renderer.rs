/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

use batch::{RasterBatch};
use debug_render::DebugRenderer;
use device::{Device, ProgramId, TextureId, UniformLocation, VertexFormat, GpuProfile};
use device::{TextureFilter, VAOId, VertexUsageHint, FileWatcherHandler};
use euclid::{Rect, Matrix4D, Point2D, Size2D};
use gleam::gl;
use internal_types::{RendererFrame, ResultMsg, TextureUpdateOp};
use internal_types::{TextureUpdateDetails, TextureUpdateList, PackedVertex, RenderTargetMode};
use internal_types::{ORTHO_NEAR_PLANE, ORTHO_FAR_PLANE};
use internal_types::{PackedVertexForTextureCacheUpdate, CompositionOp};
use internal_types::{AxisDirection, DevicePixel};
use internal_types::{FontVertex};
use ipc_channel::ipc;
use profiler::{Profiler, BackendProfileCounters};
use profiler::{RendererProfileTimers, RendererProfileCounters};
use render_backend::RenderBackend;
use std::f32;
use std::mem;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread;
use texture_cache::{BorderType, TextureCache, TextureInsertOp};
use tiling::{Frame, FrameBuilderConfig, PrimitiveShader, TextBuffer};
use time::precise_time_ns;
use webrender_traits::{ColorF, Epoch, PipelineId, RenderNotifier};
use webrender_traits::{ImageFormat, MixBlendMode, RenderApiSender};
use offscreen_gl_context::{NativeGLContext, NativeGLContextMethods};

pub const TEXT_TARGET_SIZE: u32 = 1024;

pub const BLUR_INFLATION_FACTOR: u32 = 3;
pub const MAX_RASTER_OP_SIZE: u32 = 2048;

const UBO_BIND_COMMANDS: u32 = 0;
const UBO_BIND_PRIMITIVES: u32 = 1;
const UBO_BIND_LAYERS: u32 = 2;
const UBO_BIND_CLIPS: u32 = 3;

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

fn create_prim_shader(name: &'static str, device: &mut Device) -> ProgramId {
    let program_id = device.create_program(name, "prim_shared");

    let cmds_index = gl::get_uniform_block_index(program_id.0, "Commands");
    gl::uniform_block_binding(program_id.0, cmds_index, UBO_BIND_COMMANDS);

    let layer_index = gl::get_uniform_block_index(program_id.0, "Layers");
    gl::uniform_block_binding(program_id.0, layer_index, UBO_BIND_LAYERS);

    let prim_index = gl::get_uniform_block_index(program_id.0, "Primitives");
    gl::uniform_block_binding(program_id.0, prim_index, UBO_BIND_PRIMITIVES);

    let clip_index = gl::get_uniform_block_index(program_id.0, "Clips");
    gl::uniform_block_binding(program_id.0, clip_index, UBO_BIND_CLIPS);

    println!("PrimitiveShader {}: cmds={} layers={} prims={} clips={}", name, cmds_index, layer_index, prim_index, clip_index);

    program_id
}

pub struct Renderer {
    result_rx: Receiver<ResultMsg>,
    device: Device,
    pending_texture_updates: Vec<TextureUpdateList>,
    pending_shader_updates: Vec<PathBuf>,
    current_frame: Option<RendererFrame>,
    device_pixel_ratio: f32,
    raster_batches: Vec<RasterBatch>,
    raster_op_vao: Option<VAOId>,

    quad_program_id: ProgramId,
    u_quad_transform_array: UniformLocation,
    u_quad_offset_array: UniformLocation,
    u_tile_params: UniformLocation,
    u_clip_rects: UniformLocation,
    u_atlas_params: UniformLocation,

    blend_program_id: ProgramId,
    u_blend_params: UniformLocation,

    filter_program_id: ProgramId,
    u_filter_params: UniformLocation,

    box_shadow_program_id: ProgramId,

    blur_program_id: ProgramId,
    u_direction: UniformLocation,

    opaque_rect_shader: ProgramId,
    opaque_rect_shader_clip: ProgramId,
    opaque_image_shader: ProgramId,
    text_rect_shader: ProgramId,
    text_program_id: ProgramId,
    generic2_shader: ProgramId,
    generic4_shader: ProgramId,
    error_shader: ProgramId,

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
    text_composite_target: TextureId,
    //tiling_render_targets: [TextureId; 2],

    gpu_profile_text: GpuProfile,
    gpu_profile_tiling: GpuProfile,
    gpu_profile_complex: GpuProfile,
    quad_vao_id: VAOId,
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
        let blend_program_id = ProgramId(0);//device.create_program("blend");
        let filter_program_id = ProgramId(0);//device.create_program("filter");
        let box_shadow_program_id = device.create_program("box_shadow", "shared_other");
        let blur_program_id = ProgramId(0);//device.create_program("blur");
        let max_raster_op_size = MAX_RASTER_OP_SIZE * options.device_pixel_ratio as u32;
        //let max_ubo_size = gl::get_integer_v(gl::MAX_UNIFORM_BLOCK_SIZE) as usize;
        //println!("MAX_UBO_SIZE = {}", max_ubo_size);

        let text_program_id = device.create_program("text", "shared_other");
        let opaque_rect_shader = create_prim_shader("prim_opaque_rect", &mut device);
        let opaque_rect_shader_clip = create_prim_shader("prim_opaque_rect_clip", &mut device);
        let opaque_image_shader = create_prim_shader("prim_opaque_image", &mut device);
        let text_rect_shader = create_prim_shader("prim_rect_text", &mut device);
        let error_shader = create_prim_shader("prim_error", &mut device);
        let generic2_shader = create_prim_shader("prim_generic_2", &mut device);
        let generic4_shader = create_prim_shader("prim_generic_4", &mut device);

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

        let x0 = 0.0;
        let y0 = 0.0;
        let x1 = 1.0;
        let y1 = 1.0;

        // TODO(gw): Consider separate VBO for quads vs border corners if VS ever shows up in profile!
        let quad_indices: [u16; 6] = [ 0, 1, 2, 3, 4, 5 ];
        let quad_vertices = [
            PackedVertex {
                pos: [x0, y0, 0.0],
            },
            PackedVertex {
                pos: [x1, y0, 0.0],
            },
            PackedVertex {
                pos: [x0, y1, 0.0],
            },
            PackedVertex {
                pos: [x0, y1, 1.0],
            },
            PackedVertex {
                pos: [x1, y1, 1.0],
            },
            PackedVertex {
                pos: [x1, y0, 1.0],
            },
        ];

        let quad_vao_id = device.create_vao(VertexFormat::Triangles, None);
        device.bind_vao(quad_vao_id);
        device.update_vao_indices(quad_vao_id, &quad_indices, VertexUsageHint::Static);
        device.update_vao_main_vertices(quad_vao_id, &quad_vertices, VertexUsageHint::Static);

        device.end_frame();

        let backend_notifier = notifier.clone();

        // We need a reference to the webrender context from the render backend in order to share
        // texture ids
        let context_handle = NativeGLContext::current_handle();

        let config = FrameBuilderConfig::new();

        //let tile_size = options.tile_size;
        //let allow_splitting = options.allow_splitting;
        let (device_pixel_ratio, enable_aa) = (options.device_pixel_ratio, options.enable_aa);
        let payload_tx_for_backend = payload_tx.clone();
        //let descriptors = shader_list.descriptors.clone();
        thread::spawn(move || {
            let mut backend = RenderBackend::new(api_rx,
                                                 payload_rx,
                                                 payload_tx_for_backend,
                                                 result_tx,
                                                 device_pixel_ratio,
                                                 white_image_id,
                                                 texture_cache,
                                                 enable_aa,
                                                 backend_notifier,
                                                 context_handle,
                                                 config);
            backend.run();
        });

        let mut renderer = Renderer {
            result_rx: result_rx,
            device: device,
            current_frame: None,
            raster_batches: Vec::new(),
            raster_op_vao: None,
            pending_texture_updates: Vec::new(),
            pending_shader_updates: Vec::new(),
            device_pixel_ratio: options.device_pixel_ratio,
            blend_program_id: blend_program_id,
            filter_program_id: filter_program_id,
            quad_program_id: quad_program_id,
            box_shadow_program_id: box_shadow_program_id,
            blur_program_id: blur_program_id,
            opaque_rect_shader: opaque_rect_shader,
            opaque_rect_shader_clip: opaque_rect_shader_clip,
            opaque_image_shader: opaque_image_shader,
            generic2_shader: generic2_shader,
            generic4_shader: generic4_shader,
            text_rect_shader: text_rect_shader,
            text_program_id: text_program_id,
            error_shader: error_shader,
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
            text_composite_target: text_composite_target,
            //tiling_render_targets: [TextureId(0), TextureId(0)],
            max_raster_op_size: max_raster_op_size,
            gpu_profile_text: GpuProfile::new(),
            gpu_profile_tiling: GpuProfile::new(),
            gpu_profile_complex: GpuProfile::new(),
            quad_vao_id: quad_vao_id,
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
                ResultMsg::NewFrame(frame, profile_counters) => {
                    self.backend_profile_counters = profile_counters;
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

        // Block CPU waiting for last frame's GPU profiles to arrive.
        // In general this shouldn't block unless heavily GPU limited.
        let text_ns = self.gpu_profile_text.get();
        let tiling_ns = self.gpu_profile_tiling.get();
        let complex_ns = self.gpu_profile_complex.get();

        profile_timers.cpu_time.profile(|| {
            self.device.begin_frame();

            gl::disable(gl::SCISSOR_TEST);
            gl::clear_color(1.0, 1.0, 1.0, 0.0);
            gl::clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT | gl::STENCIL_BUFFER_BIT);

            //self.update_shaders();
            self.update_texture_cache();
            self.draw_frame(framebuffer_size);
        });

        let current_time = precise_time_ns();
        let ns = current_time - self.last_time;
        self.profile_counters.frame_time.set(ns);

        profile_timers.gpu_time_text.set(text_ns);
        profile_timers.gpu_time_tiling.set(tiling_ns);
        profile_timers.gpu_time_complex.set(complex_ns);

        let gpu_ns = text_ns + tiling_ns + complex_ns;
        profile_timers.gpu_time_total.set(gpu_ns);

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

/*
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
*/

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

            let projection = Matrix4D::ortho(0.0,
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
                                                projection: &Matrix4D<f32>) {
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

    fn draw_text(&mut self,
                 text_buffer: &TextBuffer,
                 texture: TextureId) {
        self.device.bind_render_target(Some(self.text_composite_target));
        gl::viewport(0, 0, (TEXT_TARGET_SIZE * self.device_pixel_ratio as u32) as gl::GLint, (TEXT_TARGET_SIZE * self.device_pixel_ratio as u32) as gl::GLint);

        gl::clear_color(1.0, 1.0, 1.0, 0.0);
        gl::clear(gl::COLOR_BUFFER_BIT);

        let projection = Matrix4D::ortho(0.0,
                                         TEXT_TARGET_SIZE as f32,
                                         0.0,
                                         TEXT_TARGET_SIZE as f32,
                                         ORTHO_NEAR_PLANE,
                                         ORTHO_FAR_PLANE);

        let (mut indices, mut vertices) = (vec![], vec![]);

        for glyph in &text_buffer.glyphs {
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

        let vao_id = self.device.create_vao(VertexFormat::Font, None);

        self.device.bind_program(self.text_program_id, &projection);
        self.device.bind_color_texture(texture);
        self.device.bind_vao(vao_id);
        self.device.update_vao_indices(vao_id, &indices[..], VertexUsageHint::Dynamic);
        self.device.update_vao_main_vertices(vao_id, &vertices[..], VertexUsageHint::Dynamic);
        self.device.draw_triangles_u16(0, indices.len() as gl::GLint);
        self.device.delete_vao(vao_id);
    }

/*
    fn add_debug_rect(&mut self, rect: &Rect<i32>, label: String, c: &ColorF) {
        let tile_x0 = rect.origin.x as f32;
        let tile_y0 = rect.origin.y as f32;
        let tile_x1 = tile_x0 + (rect.size.width-1) as f32;
        let tile_y1 = tile_y0 + (rect.size.height-1) as f32;

//        self.debug.add_quad(tile_x0,
//                            tile_y0,
//                            tile_x1,
//                            tile_y1,
//                            c0,
//                            c1);

        self.debug.add_line(tile_x0,
                            tile_y0,
                            c,//&ColorF::new(0.0, 0.0, 0.0, 1.0),
                            tile_x1,
                            tile_y0,
                            c);//&ColorF::new(0.0, 0.0, 0.0, 1.0));
        self.debug.add_line(tile_x0,
                            tile_y1,
                            c,//&ColorF::new(0.0, 0.0, 0.0, 1.0),
                            tile_x1,
                            tile_y1,
                            c);//&ColorF::new(0.0, 0.0, 0.0, 1.0));
        self.debug.add_line(tile_x0,
                            tile_y0,
                            c,//&ColorF::new(0.0, 0.0, 0.0, 1.0),
                            tile_x0,
                            tile_y1,
                            c);//&ColorF::new(0.0, 0.0, 0.0, 1.0));
        self.debug.add_line(tile_x1,
                            tile_y0,
                            c,//&ColorF::new(0.0, 0.0, 0.0, 1.0),
                            tile_x1,
                            tile_y1,
                            c);//&ColorF::new(0.0, 0.0, 0.0, 1.0));
        self.debug.add_text((tile_x0 + tile_x1) * 0.5,
                            (tile_y0 + tile_y1) * 0.5,
                            &label,
                            c);//&ColorF::new(0.0, 0.0, 0.0, 1.0));
    }

    fn draw_tiling_pass(&mut self, pass: &RenderPass, projection: &Matrix4, initial_color: &ColorF) {
        for &(shader_id, ref batch) in &pass.batches {
            //let shader_id = *shader_id;
            if shader_id == self.composite_shader_id {
                panic!("todo");
                self.composite_shader.draw(&mut self.device, &projection, &batch.data, initial_color);
            } else {
                let ShaderId(shader_index) = shader_id;
                let shader = &self.shader_list.shaders[shader_index as usize];
                shader.draw(&mut self.device, &projection, &batch.data, initial_color);
            }
        }
    }

    fn draw_tile_frame(&mut self,
                       tile_frame: &TileFrame,
                       framebuffer_size: &Size2D<u32>) {
        //println!("draw_tile_frame {:?} {:?}", framebuffer_size, tile_frame.render_target_size);

        self.gpu_profile_text.begin();

        gl::disable(gl::STENCIL_TEST);
        gl::depth_mask(false);
        gl::stencil_mask(0xff);

        if tile_frame.text_buffer.glyphs.len() > 0 {
            gl::disable(gl::BLEND);
            self.draw_text(&tile_frame.text_buffer, tile_frame.color_texture_id);
        }

        self.gpu_profile_text.end();
        self.gpu_profile_tiling.begin();

        if tile_frame.passes.len() > 0 {
            gl::enable(gl::STENCIL_TEST);
            gl::stencil_mask(0xff);
            gl::stencil_func(gl::EQUAL, 0, 0xff);
            gl::stencil_op(gl::KEEP, gl::INVERT, gl::INVERT);

            /*
            let render_target_pixel_size = tile_frame.render_target_size * self.device_pixel_ratio as u32;

            if self.tiling_render_targets[0] == TextureId(0) {
                // hack hack hack
                let ids = self.device.create_texture_ids(2);
                self.tiling_render_targets[0] = ids[0];
                self.tiling_render_targets[1] = ids[1];
                self.device.init_texture(self.tiling_render_targets[0],
                                         render_target_pixel_size,
                                         render_target_pixel_size,
                                         ImageFormat::RGBA8,
                                         TextureFilter::Linear,
                                         RenderTargetMode::RenderTarget,
                                         None);

                self.device.init_texture(self.tiling_render_targets[1],
                                         render_target_pixel_size,
                                         render_target_pixel_size,
                                         ImageFormat::RGBA8,
                                         TextureFilter::Linear,
                                         RenderTargetMode::RenderTarget,
                                         None);
            }

            gl::clear_color(1.0, 1.0, 1.0, 1.0);
            gl::viewport(0, 0, render_target_pixel_size as i32, render_target_pixel_size as i32);
            self.device.bind_render_target(Some(self.tiling_render_targets[0]));
            gl::clear(gl::COLOR_BUFFER_BIT);
            self.device.bind_render_target(Some(self.tiling_render_targets[1]));
            gl::clear(gl::COLOR_BUFFER_BIT);
*/

            self.device.bind_mask_texture(self.text_composite_target);
            self.device.bind_vao(self.quad_vao_id);
            let ubos = gl::gen_buffers(1);
            let layer_ubo = ubos[0];

            gl::bind_buffer(gl::UNIFORM_BUFFER, layer_ubo);
            gl::buffer_data(gl::UNIFORM_BUFFER, &tile_frame.layer_ubo.items, gl::STATIC_DRAW);
            gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_LAYERS, layer_ubo);
            //gl::enable(gl::BLEND);

            let pass_count = tile_frame.passes.len();
            let mut current_target_index = 0;
            for pass_index in 0..pass_count {
                let pass_index = pass_count - pass_index - 1;

                self.device.bind_tiling_texture(self.tiling_render_targets[1 - current_target_index]);

                let (pw, ph, initial_color) = if pass_index == 0 {
                    self.device.bind_render_target(None);
                    gl::viewport(0, 0, framebuffer_size.width as i32, framebuffer_size.height as i32);
                    (tile_frame.viewport_size.width,
                     tile_frame.viewport_size.height,
                     ColorF::new(1.0, 1.0, 1.0, 1.0))
                } else {
                    continue;
                    //panic!("todo");
                    /*
                    self.device.bind_render_target(Some(self.tiling_render_targets[current_target_index]));
                    // hack hack hack ugh
                    gl::viewport(0, 0, render_target_pixel_size as i32, render_target_pixel_size as i32);
                    (tile_frame.render_target_size as i32,
                     tile_frame.render_target_size as i32,
                     ColorF::new(0.0, 0.0, 0.0, 0.0))
                     */
                };

                let projection = Matrix4::ortho(0.0,
                                                pw as f32,
                                                ph as f32,
                                                0.0,
                                                ORTHO_NEAR_PLANE,
                                                ORTHO_FAR_PLANE);

                current_target_index = 1 - current_target_index;
                self.draw_tiling_pass(&tile_frame.passes[pass_index], &projection, &initial_color);
            }

            gl::delete_buffers(&ubos);
            gl::disable(gl::STENCIL_TEST);

            let projection = Matrix4::ortho(0.0,
                                            tile_frame.viewport_size.width as f32,
                                            tile_frame.viewport_size.height as f32,
                                            0.0,
                                            ORTHO_NEAR_PLANE,
                                            ORTHO_FAR_PLANE);

            let initial_color = ColorF::new(1.0, 1.0, 1.0, 1.0);
            self.empty_shader.draw(&mut self.device, &projection, &tile_frame.special_tiles.empty, &initial_color);
            self.error_shader.draw(&mut self.device, &projection, &tile_frame.special_tiles.error, &initial_color);
        } else {
            self.device.bind_render_target(None);
            gl::viewport(0, 0, framebuffer_size.width as i32, framebuffer_size.height as i32);
            gl::clear_color(1.0, 1.0, 1.0, 1.0);
            gl::clear(gl::COLOR_BUFFER_BIT);
        }

        self.gpu_profile_tiling.end();

        for tile in &tile_frame.debug_info {
            self.add_debug_rect(&tile.rect, format!("{}/{}", tile.pass_count, tile.prim_count), &ColorF::new(0.0, 1.0, 0.0, 1.0));
        }
        if self.enable_profiler {
            //for tile in &tile_frame.simple_tiles {
            //    self.add_debug_rect(&tile.rect, format!("{}", tile.info[0]), &ColorF::new(0.0, 1.0, 0.0, 1.0), &ColorF::new(0.0, 0.5, 0.0, 0.5));
            //}
            //for tile in &tile_frame.unhandled_tiles {
            //    //self.add_debug_rect(&tile.rect, format!("{}/{}", tile.info[0], tile.info[1]), &ColorF::new(1.0, 0.0, 0.0, 1.0), &ColorF::new(0.5, 0.0, 0.0, 0.5));
            //}
            //for tile in &tile_frame.empty_tiles {
            //    //self.add_debug_rect(&tile.rect, format!("{}/{}", tile.info[0], tile.info[1]), &ColorF::new(0.0, 0.0, 1.0, 1.0), &ColorF::new(0.0, 0.0, 0.5, 0.5));
            //}
        }
    }
*/

    fn draw_tile_frame(&mut self,
                       frame: &Frame,
                       framebuffer_size: &Size2D<u32>) {
        self.gpu_profile_text.begin();

        gl::depth_mask(false);
        gl::disable(gl::STENCIL_TEST);

        if frame.text_buffer.glyphs.len() > 0 {
            gl::disable(gl::BLEND);
            self.draw_text(&frame.text_buffer, frame.color_texture_id);
        }

        self.gpu_profile_text.end();
        self.gpu_profile_tiling.begin();

        gl::enable(gl::STENCIL_TEST);
        gl::stencil_mask(0xff);
        gl::stencil_func(gl::EQUAL, 0, 0xff);
        gl::stencil_op(gl::KEEP, gl::INVERT, gl::INVERT);

        let layer_ubos = gl::gen_buffers(frame.layer_ubos.len() as gl::GLint);
        let prim_ubos = gl::gen_buffers(frame.primitive_ubos.len() as gl::GLint);
        let clip_ubos = gl::gen_buffers(frame.clip_ubos.len() as gl::GLint);

        for (layer_index, layer_ubo) in frame.layer_ubos.iter().enumerate() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, layer_ubos[layer_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &layer_ubo.items, gl::STATIC_DRAW);
        }

        for (prim_index, prim_ubo) in frame.primitive_ubos.iter().enumerate() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, prim_ubos[prim_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &prim_ubo.items, gl::STATIC_DRAW);
        }

        for (clip_index, clip_ubo) in frame.clip_ubos.iter().enumerate() {
            gl::bind_buffer(gl::UNIFORM_BUFFER, clip_ubos[clip_index]);
            gl::buffer_data(gl::UNIFORM_BUFFER, &clip_ubo.items, gl::STATIC_DRAW);
        }

        self.device.bind_color_texture(frame.color_texture_id);
        self.device.bind_mask_texture(self.text_composite_target);
        self.device.bind_vao(self.quad_vao_id);

        for pass in &frame.passes {
            // TODO(gw): Set up render target / proj for pass...
            self.device.bind_render_target(None);
            gl::viewport(0, 0, framebuffer_size.width as i32, framebuffer_size.height as i32);

            let projection = Matrix4D::ortho(0.0,
                                             pass.viewport_size.width as f32,
                                             pass.viewport_size.height as f32,
                                             0.0,
                                             ORTHO_NEAR_PLANE,
                                             ORTHO_FAR_PLANE);

            for batch in &pass.batches {
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_LAYERS, layer_ubos[batch.layer_ubo_index]);
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_PRIMITIVES, prim_ubos[batch.prim_ubo_index]);
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_CLIPS, clip_ubos[batch.clip_ubo_index]);

                let cmd_ubos = gl::gen_buffers(1);
                let cmd_ubo = cmd_ubos[0];

                gl::bind_buffer(gl::UNIFORM_BUFFER, cmd_ubo);
                gl::buffer_data(gl::UNIFORM_BUFFER, &batch.commands, gl::STATIC_DRAW);
                gl::bind_buffer_base(gl::UNIFORM_BUFFER, UBO_BIND_COMMANDS, cmd_ubo);

                let program_id = match batch.shader {
                    PrimitiveShader::OpaqueRectangle => self.opaque_rect_shader,
                    PrimitiveShader::OpaqueRectangleClip => self.opaque_rect_shader_clip,
                    PrimitiveShader::OpaqueRectangleText => self.text_rect_shader,
                    PrimitiveShader::OpaqueImage => self.opaque_image_shader,
                    PrimitiveShader::Generic2 => self.generic2_shader,
                    PrimitiveShader::Generic4 => self.generic4_shader,
                    PrimitiveShader::Error => self.error_shader,
                };
                self.device.bind_program(program_id, &projection);
                self.device.draw_indexed_triangles_instanced_u16(6, batch.commands.len() as i32);

                gl::delete_buffers(&cmd_ubos)
            }
        }

        gl::delete_buffers(&layer_ubos);
        gl::delete_buffers(&prim_ubos);
        gl::delete_buffers(&clip_ubos);
        gl::disable(gl::STENCIL_TEST);

        self.gpu_profile_tiling.end();
    }

    fn draw_frame(&mut self, framebuffer_size: Size2D<u32>) {
        if let Some(frame) = self.current_frame.take() {
            // TODO: cache render targets!

            // TODO(gw): Doesn't work well with transforms.
            //           Look into this...
            gl::disable(gl::DEPTH_TEST);
            gl::disable(gl::SCISSOR_TEST);
            gl::disable(gl::BLEND);

            if let Some(ref frame) = frame.frame {
                self.draw_tile_frame(frame, &framebuffer_size);
            }

            // Restore frame - avoid borrow checker!
            self.current_frame = Some(frame);
        }
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
    pub allow_splitting: bool,
}
