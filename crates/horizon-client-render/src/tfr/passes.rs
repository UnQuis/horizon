//! WGSL shader sources, wgpu pipelines/dispatch helpers and GPU resources for TFR.
//!
//! Standalone; the frame loop/integrator wires external scene inputs and calls record methods.
//!
//! Public API:
//! - WGSL/constants: `MOTION_WGSL`, `PREPARE_WGSL`, `COMPOSE_WGSL`, `BLIT_WGSL`, `DEPTH_COPY_WGSL`,
//!   `SUPPORTED_COLOR_FORMAT` (`Rgba16Float`), `SUPPORTED_EDGE_FORMAT` (`Rgba8Unorm`),
//!   `SUPPORTED_DEPTH_COPY_FORMAT` (`R32Float`), `MOTION_SCALE` (0.5).
//! - CPU params/helpers: `MotionParams`, `PrepareParams`, `ComposeParams`, `TfrSize`,
//!   `history_uv_from_forward_motion`, `deterministic_fill_select`, `FILL_NEUTRAL_RGBA`,
//!   `formats_supported`, `assert_supported_formats`.
//! - Pipelines/dispatch: `TfrPipelines::new`, bind group layouts, pipelines,
//!   `encode_motion|prepare|compose|blit|depth_copy`, and bind-group creators
//!   `motion_bind_group|prepare_bind_group|compose_bind_group|blit_bind_group|depth_copy_bind_group`.
//! - GPU resources: `TfrGpuResources::{new,resize,ensure}`, owned textures/views/sampler/sizes,
//!   and convenience record methods `record_depth_copy|record_motion|record_prepare|record_compose|record_blit`
//!   that create bind groups internally so the integrator does not assemble them manually.
//!
//! Conventions: Horizon reversed-Z (near=1, far/sky~0), WebGPU clip depth [0,1],
//! UV origin top-left. Rotation-only camera-relative VPs (`projection * pose * rotation`).
//!
//! Supported storage formats (must match WGSL declarations and be storage-capable):
//! - color/motion/output: `wgpu::TextureFormat::Rgba16Float` (intermediate HDR target;
//!   TFR must not composite directly into SDR/swapchain formats such as `Bgra8UnormSrgb`)
//! - edge: `wgpu::TextureFormat::Rgba8Unorm` (filterable + storage, sampled in compose)
//! - history depth copy: `wgpu::TextureFormat::R32Float` (sampleable copy of `Depth32Float`)
//!
//! Bindings:
//! - `MOTION_WGSL` (`@compute main`, 8x8): 0 depth `texture_2d<f32>` (R32Float copy), 1 `rgba16float`
//!   motion (RG forward motion, B validity, A edge), 2 `uniform MotionParams`.
//! - `PREPARE_WGSL` (`@compute main`, 8x8): 0 scene color, 1 scene depth, 2 sampler, 3 `rgba16float`
//!   prepared color, 4 `rgba8unorm` edge, 5 `uniform PrepareParams`.
//! - `COMPOSE_WGSL` (`@compute main`, 8x8): 0 history,1 sampler,2 motion,3 sampler,4 edge,5 sampler,
//!   6 `rgba16float` output, 7 `uniform ComposeParams`.
//! - `BLIT_WGSL` (`@compute main`, 8x8): 0 src, 1 sampler, 2 `rgba16float` output (scaled).
//! - `DEPTH_COPY_WGSL` (`@compute main`, 8x8): 0 `texture_depth_2d`, 1 `r32float` storage output.

/// Motion-field compute shader.
pub const MOTION_WGSL: &str = include_str!("motion.wgsl");
/// Prepared-history pass (color preserved, edge in separate target).
pub const PREPARE_WGSL: &str = include_str!("prepare.wgsl");
/// Synthetic composition pass.
pub const COMPOSE_WGSL: &str = include_str!("compose.wgsl");
/// Real-path blit/scale pass.
pub const BLIT_WGSL: &str = include_str!("blit.wgsl");
/// Depth32Float -> sampleable R32Float copy pass.
pub const DEPTH_COPY_WGSL: &str = include_str!("depth_copy.wgsl");

/// Supported color/motion/output storage format for TFR WGSL pipelines.
pub const SUPPORTED_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Supported edge storage format (filterable, storage-capable) for TFR WGSL pipelines.
pub const SUPPORTED_EDGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
/// Supported sampleable history depth format produced by depth copy.
pub const SUPPORTED_DEPTH_COPY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
/// Motion/edge field resolution relative to scene size.
pub const MOTION_SCALE: f32 = 0.5;

/// Dispatch/output size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TfrSize {
    pub width: u32,
    pub height: u32,
}

impl TfrSize {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub const fn is_valid(self) -> bool {
        self.width > 0 && self.height > 0
    }

    pub const fn motion_size(self) -> Self {
        Self {
            width: ((self.width as f32) * MOTION_SCALE).ceil().max(1.0) as u32,
            height: ((self.height as f32) * MOTION_SCALE).ceil().max(1.0) as u32,
        }
    }
}

/// Motion pass uniform parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MotionParams {
    /// `cur_vp * T(ref_cam-cur_cam) * ref_vp^-1` for rotation-only camera-relative VPs.
    pub geo_reproject: [[f32; 4]; 4],
    /// `cur_vp * ref_vp^-1` (no translation parallax).
    pub sky_reproject: [[f32; 4]; 4],
    /// x,y: scene size; z,w: motion output size.
    pub sizes: [f32; 4],
    /// x: w epsilon; y: bounds epsilon; z: far/sky depth threshold (near 0 in reversed-Z); w: reserved.
    pub thresholds: [f32; 4],
}

/// Prepared-history uniform parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrepareParams {
    /// x,y: texel size in UV; z: edge scale; w: reserved.
    pub params: [f32; 4],
}

/// Synthetic composition uniform parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ComposeParams {
    /// x: reprojection strength [0,1]; y: history stability [0,0.35];
    /// z: reserved; w: use external edge texture (0/1).
    pub mode: [f32; 4],
    /// x,y: source/history texture size in texels; z,w: reserved.
    pub texel: [f32; 4],
}

impl MotionParams {
    pub fn new(
        geo_reproject: [[f32; 4]; 4],
        sky_reproject: [[f32; 4]; 4],
        scene_size: [u32; 2],
        output_size: [u32; 2],
        w_epsilon: f32,
        bounds_epsilon: f32,
        far_depth_threshold: f32,
    ) -> Self {
        Self {
            geo_reproject,
            sky_reproject,
            sizes: [
                scene_size[0] as f32,
                scene_size[1] as f32,
                output_size[0] as f32,
                output_size[1] as f32,
            ],
            thresholds: [w_epsilon, bounds_epsilon, far_depth_threshold, 0.0],
        }
    }
}

impl PrepareParams {
    pub fn new(texel_size: [f32; 2], edge_scale: f32) -> Self {
        Self {
            params: [texel_size[0], texel_size[1], edge_scale, 0.0],
        }
    }
}

impl ComposeParams {
    pub fn new(
        reprojection_strength: f32,
        history_stability: f32,
        source_size: [u32; 2],
        use_edge_texture: bool,
    ) -> Self {
        Self {
            mode: [
                reprojection_strength,
                history_stability,
                0.0,
                if use_edge_texture { 1.0 } else { 0.0 },
            ],
            texel: [source_size[0] as f32, source_size[1] as f32, 0.0, 0.0],
        }
    }
}

/// Backward history sample location for a forward motion vector `m = target-current_source`.
/// Composition searches near `current_uv - m * strength`.
pub fn history_uv_from_forward_motion(current_uv: [f32; 2], motion: [f32; 2], strength: f32) -> [f32; 2] {
    [
        current_uv[0] - motion[0] * strength,
        current_uv[1] - motion[1] * strength,
    ]
}

/// Deterministic fill selection rule (mirrors `compose.wgsl`): pick the first candidate whose
/// motion validity and edge are acceptable; if none, return neutral fallback.
pub fn deterministic_fill_select(valid_edge_ok: impl IntoIterator<Item = bool>) -> Option<usize> {
    valid_edge_ok.into_iter().position(|ok| ok)
}

/// Neutral color used by deterministic fill when no valid candidate exists.
pub const FILL_NEUTRAL_RGBA: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

/// Returns whether `(color_format, edge_format)` are supported by the current TFR WGSL.
pub fn formats_supported(color_format: wgpu::TextureFormat, edge_format: wgpu::TextureFormat) -> bool {
    color_format == SUPPORTED_COLOR_FORMAT && edge_format == SUPPORTED_EDGE_FORMAT
}

/// Validates pipeline formats. Panics with a clear message if unsupported.
pub fn assert_supported_formats(color_format: wgpu::TextureFormat, edge_format: wgpu::TextureFormat) {
    assert!(
        formats_supported(color_format, edge_format),
        "unsupported TFR formats: color={color_format:?} edge={edge_format:?}; \
supported color={SUPPORTED_COLOR_FORMAT:?} edge={SUPPORTED_EDGE_FORMAT:?}. \
TFR targets an intermediate Rgba16Float color target (not SDR/swapchain)."
    );
}

fn workgroups(width: u32, height: u32) -> (u32, u32, u32) {
    ((width + 7) / 8, (height + 7) / 8, 1)
}

/// Compiled pipelines, layouts and dispatch helpers for TFR passes.
pub struct TfrPipelines {
    pub motion_layout: wgpu::BindGroupLayout,
    pub prepare_layout: wgpu::BindGroupLayout,
    pub compose_layout: wgpu::BindGroupLayout,
    pub blit_layout: wgpu::BindGroupLayout,
    pub depth_copy_layout: wgpu::BindGroupLayout,
    pub motion_pipeline: wgpu::ComputePipeline,
    pub prepare_pipeline: wgpu::ComputePipeline,
    pub compose_pipeline: wgpu::ComputePipeline,
    pub blit_pipeline: wgpu::ComputePipeline,
    pub depth_copy_pipeline: wgpu::ComputePipeline,
    pub color_format: wgpu::TextureFormat,
    pub edge_format: wgpu::TextureFormat,
}

impl TfrPipelines {
    /// Create pipelines for the given storage texture formats.
    ///
    /// Supported formats must match the WGSL storage declarations:
    /// - `color_format`: `Rgba16Float` (motion/prepared/output; intermediate HDR target)
    /// - `edge_format`: `Rgba8Unorm` (filterable edge target sampled in compose)
    ///
    /// Panics if formats are unsupported.
    pub fn new(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        edge_format: wgpu::TextureFormat,
    ) -> Self {
        assert_supported_formats(color_format, edge_format);

        let motion_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TFR motion"),
            source: wgpu::ShaderSource::Wgsl(MOTION_WGSL.into()),
        });
        let prepare_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TFR prepare"),
            source: wgpu::ShaderSource::Wgsl(PREPARE_WGSL.into()),
        });
        let compose_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TFR compose"),
            source: wgpu::ShaderSource::Wgsl(COMPOSE_WGSL.into()),
        });
        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TFR blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });
        let depth_copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TFR depth copy"),
            source: wgpu::ShaderSource::Wgsl(DEPTH_COPY_WGSL.into()),
        });

        let motion_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TFR motion bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: color_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let prepare_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TFR prepare bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: color_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: edge_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let filtering_sampler = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let float_tex = |binding, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let compose_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TFR compose bindings"),
            entries: &[
                float_tex(0, true),
                filtering_sampler(1),
                float_tex(2, true),
                filtering_sampler(3),
                float_tex(4, true),
                filtering_sampler(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: color_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TFR blit bindings"),
            entries: &[
                float_tex(0, true),
                filtering_sampler(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: color_format,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let depth_copy_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TFR depth copy bindings"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: SUPPORTED_DEPTH_COPY_FORMAT,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let motion_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TFR motion pipeline"),
            bind_group_layouts: &[Some(&motion_layout)],
            ..Default::default()
        });
        let prepare_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TFR prepare pipeline"),
            bind_group_layouts: &[Some(&prepare_layout)],
            ..Default::default()
        });
        let compose_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TFR compose pipeline"),
            bind_group_layouts: &[Some(&compose_layout)],
            ..Default::default()
        });
        let blit_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TFR blit pipeline"),
            bind_group_layouts: &[Some(&blit_layout)],
            ..Default::default()
        });
        let depth_copy_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TFR depth copy pipeline"),
            bind_group_layouts: &[Some(&depth_copy_layout)],
            ..Default::default()
        });

        let motion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TFR motion"),
            layout: Some(&motion_pl),
            module: &motion_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let prepare_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TFR prepare"),
            layout: Some(&prepare_pl),
            module: &prepare_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let compose_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TFR compose"),
            layout: Some(&compose_pl),
            module: &compose_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let blit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TFR blit"),
            layout: Some(&blit_pl),
            module: &blit_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let depth_copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TFR depth copy"),
            layout: Some(&depth_copy_pl),
            module: &depth_copy_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            motion_layout,
            prepare_layout,
            compose_layout,
            blit_layout,
            depth_copy_layout,
            motion_pipeline,
            prepare_pipeline,
            compose_pipeline,
            blit_pipeline,
            depth_copy_pipeline,
            color_format,
            edge_format,
        }
    }

    pub fn motion_bind_group(
        &self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        motion_view: &wgpu::TextureView,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TFR motion bg"),
            layout: &self.motion_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(motion_view) },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        })
    }

    pub fn prepare_bind_group(
        &self,
        device: &wgpu::Device,
        scene_color: &wgpu::TextureView,
        scene_depth: &wgpu::TextureView,
        scene_sampler: &wgpu::Sampler,
        prepared_color: &wgpu::TextureView,
        edge: &wgpu::TextureView,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TFR prepare bg"),
            layout: &self.prepare_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(scene_color) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(scene_depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(scene_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(prepared_color) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(edge) },
                wgpu::BindGroupEntry { binding: 5, resource: params.as_entire_binding() },
            ],
        })
    }

    pub fn compose_bind_group(
        &self,
        device: &wgpu::Device,
        history: &wgpu::TextureView,
        history_sampler: &wgpu::Sampler,
        motion: &wgpu::TextureView,
        motion_sampler: &wgpu::Sampler,
        edge: &wgpu::TextureView,
        edge_sampler: &wgpu::Sampler,
        output: &wgpu::TextureView,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TFR compose bg"),
            layout: &self.compose_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(history) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(history_sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(motion) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(motion_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(edge) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(edge_sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 7, resource: params.as_entire_binding() },
            ],
        })
    }

    pub fn blit_bind_group(
        &self,
        device: &wgpu::Device,
        src: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        dst: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TFR blit bg"),
            layout: &self.blit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(dst) },
            ],
        })
    }

    pub fn depth_copy_bind_group(
        &self,
        device: &wgpu::Device,
        src_depth: &wgpu::TextureView,
        dst_r32: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TFR depth copy bg"),
            layout: &self.depth_copy_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_depth) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(dst_r32) },
            ],
        })
    }

    pub fn encode_motion(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let (x, y, z) = workgroups(width, height);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("TFR motion"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.motion_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }

    pub fn encode_prepare(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let (x, y, z) = workgroups(width, height);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("TFR prepare"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.prepare_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }

    pub fn encode_compose(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let (x, y, z) = workgroups(width, height);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("TFR compose"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.compose_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }

    pub fn encode_blit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let (x, y, z) = workgroups(width, height);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("TFR blit"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }

    pub fn encode_depth_copy(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        width: u32,
        height: u32,
    ) {
        let (x, y, z) = workgroups(width, height);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("TFR depth copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.depth_copy_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }
}

/// Owned TFR GPU textures, views and convenience record helpers.
pub struct TfrGpuResources {
    pub scene_size: TfrSize,
    pub motion_size: TfrSize,
    pub history_color: wgpu::Texture,
    pub history_color_view: wgpu::TextureView,
    pub history_depth: wgpu::Texture,
    pub history_depth_view: wgpu::TextureView,
    pub motion: wgpu::Texture,
    pub motion_view: wgpu::TextureView,
    pub edge: wgpu::Texture,
    pub edge_view: wgpu::TextureView,
    pub compose_out: wgpu::Texture,
    pub compose_out_view: wgpu::TextureView,
    pub linear_sampler: wgpu::Sampler,
}

impl TfrGpuResources {
    /// Create resources for `width x height` scene. Returns `None` if size is invalid
    /// or required texture limits/formats cannot be satisfied (caller should fall back to real path).
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Option<Self> {
        let scene_size = TfrSize::new(width, height);
        if !scene_size.is_valid() {
            return None;
        }
        let limits = device.limits();
        let max = limits.max_texture_dimension_2d;
        if width > max || height > max {
            return None;
        }
        let motion_size = scene_size.motion_size();
        if motion_size.width > max || motion_size.height > max {
            return None;
        }

        let history_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TFR history color"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SUPPORTED_COLOR_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let history_color_view = history_color.create_view(&wgpu::TextureViewDescriptor::default());

        let history_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TFR history depth (R32Float copy)"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SUPPORTED_DEPTH_COPY_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let history_depth_view = history_depth.create_view(&wgpu::TextureViewDescriptor::default());

        let motion = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TFR motion"),
            size: wgpu::Extent3d {
                width: motion_size.width,
                height: motion_size.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SUPPORTED_COLOR_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let motion_view = motion.create_view(&wgpu::TextureViewDescriptor::default());

        let edge = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TFR edge"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SUPPORTED_EDGE_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let edge_view = edge.create_view(&wgpu::TextureViewDescriptor::default());

        let compose_out = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("TFR compose out"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SUPPORTED_COLOR_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let compose_out_view = compose_out.create_view(&wgpu::TextureViewDescriptor::default());

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TFR linear sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        Some(Self {
            scene_size,
            motion_size,
            history_color,
            history_color_view,
            history_depth,
            history_depth_view,
            motion,
            motion_view,
            edge,
            edge_view,
            compose_out,
            compose_out_view,
            linear_sampler,
        })
    }

    /// Recreate resources if size changed.
    pub fn ensure(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        if self.scene_size == TfrSize::new(width, height) {
            return true;
        }
        self.resize(device, width, height)
    }

    /// Force resize/recreate resources. Returns false if requested size is unsupported.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) -> bool {
        match Self::new(device, width, height) {
            Some(next) => {
                *self = next;
                true
            }
            None => false,
        }
    }

    pub fn record_depth_copy(
        &self,
        device: &wgpu::Device,
        pipelines: &TfrPipelines,
        encoder: &mut wgpu::CommandEncoder,
        src_depth: &wgpu::TextureView,
    ) {
        let bg = pipelines.depth_copy_bind_group(device, src_depth, &self.history_depth_view);
        pipelines.encode_depth_copy(encoder, &bg, self.scene_size.width, self.scene_size.height);
    }

    pub fn record_motion(
        &self,
        device: &wgpu::Device,
        pipelines: &TfrPipelines,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        params: &wgpu::Buffer,
    ) {
        let bg = pipelines.motion_bind_group(device, depth_view, &self.motion_view, params);
        pipelines.encode_motion(encoder, &bg, self.motion_size.width, self.motion_size.height);
    }

    pub fn record_prepare(
        &self,
        device: &wgpu::Device,
        pipelines: &TfrPipelines,
        encoder: &mut wgpu::CommandEncoder,
        scene_color: &wgpu::TextureView,
        scene_depth: &wgpu::TextureView,
        scene_sampler: &wgpu::Sampler,
        prepared_color: &wgpu::TextureView,
        edge: &wgpu::TextureView,
        params: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        let bg = pipelines.prepare_bind_group(
            device,
            scene_color,
            scene_depth,
            scene_sampler,
            prepared_color,
            edge,
            params,
        );
        pipelines.encode_prepare(encoder, &bg, width, height);
    }

    pub fn record_compose(
        &self,
        device: &wgpu::Device,
        pipelines: &TfrPipelines,
        encoder: &mut wgpu::CommandEncoder,
        history: &wgpu::TextureView,
        history_sampler: &wgpu::Sampler,
        edge: &wgpu::TextureView,
        edge_sampler: &wgpu::Sampler,
        output: &wgpu::TextureView,
        params: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) {
        let bg = pipelines.compose_bind_group(
            device,
            history,
            history_sampler,
            &self.motion_view,
            &self.linear_sampler,
            edge,
            edge_sampler,
            output,
            params,
        );
        pipelines.encode_compose(encoder, &bg, width, height);
    }

    pub fn record_blit(
        &self,
        device: &wgpu::Device,
        pipelines: &TfrPipelines,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::TextureView,
        src_sampler: &wgpu::Sampler,
        dst: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) {
        let bg = pipelines.blit_bind_group(device, src, src_sampler, dst);
        pipelines.encode_blit(encoder, &bg, width, height);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_wgsl(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name} WGSL parse error: {e}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} WGSL validation error: {e:?}"));
    }

    #[test]
    fn wgsl_sources_validate() {
        for (name, src) in [
            ("motion", MOTION_WGSL),
            ("prepare", PREPARE_WGSL),
            ("compose", COMPOSE_WGSL),
            ("blit", BLIT_WGSL),
            ("depth_copy", DEPTH_COPY_WGSL),
        ] {
            assert!(!src.trim().is_empty(), "{name} shader is empty");
            validate_wgsl(name, src);
        }
    }

    #[test]
    fn param_layouts_match_expected_size() {
        assert_eq!(std::mem::size_of::<MotionParams>(), 4 * (16 + 16 + 4 + 4));
        assert_eq!(std::mem::size_of::<PrepareParams>(), 16);
        assert_eq!(std::mem::size_of::<ComposeParams>(), 32);
    }

    #[test]
    fn backward_lookup_uses_negative_forward_motion() {
        let uv = history_uv_from_forward_motion([0.5, 0.5], [0.1, -0.05], 1.0);
        assert!((uv[0] - 0.4).abs() < 1.0e-6);
        assert!((uv[1] - 0.55).abs() < 1.0e-6);
    }

    #[test]
    fn deterministic_fill_picks_first_valid_else_neutral() {
        assert_eq!(deterministic_fill_select([false, false, true, true]), Some(2));
        assert_eq!(deterministic_fill_select([false, false]), None);
        assert_eq!(FILL_NEUTRAL_RGBA, [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn format_support_matches_wgsl_contract() {
        assert!(formats_supported(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::Rgba8Unorm
        ));
        assert!(!formats_supported(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8Unorm
        ));
        assert!(!formats_supported(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureFormat::R32Float
        ));
    }

    #[test]
    #[should_panic(expected = "unsupported TFR formats")]
    fn unsupported_formats_are_rejected() {
        assert_supported_formats(
            wgpu::TextureFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8Unorm,
        );
    }

    #[test]
    fn size_and_motion_scale() {
        let s = TfrSize::new(1920, 1080);
        assert_eq!(s.motion_size(), TfrSize::new(960, 540));
        assert_eq!(TfrSize::new(3, 3).motion_size(), TfrSize::new(2, 2));
        assert!(!TfrSize::new(0, 10).is_valid());
    }

    #[test]
    fn gpu_resources_smoke_if_device_available() {
        let instance = crate::gfx::test_gpu::instance();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
            ..Default::default()
        }))
        .ok() else {
            return;
        };
        let limits = adapter.limits();
        let Some((device, _queue)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tfr-test"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults().using_resolution(limits),
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        }))
        .ok() else {
            return;
        };
        let Some(mut res) = TfrGpuResources::new(&device, 64, 32) else {
            return;
        };
        assert_eq!(res.scene_size, TfrSize::new(64, 32));
        assert_eq!(res.motion_size, TfrSize::new(32, 16));
        assert!(res.ensure(&device, 64, 32));
        assert!(res.resize(&device, 48, 24));
        assert_eq!(res.scene_size, TfrSize::new(48, 24));
        let _pipelines = TfrPipelines::new(&device, SUPPORTED_COLOR_FORMAT, SUPPORTED_EDGE_FORMAT);
    }
}
