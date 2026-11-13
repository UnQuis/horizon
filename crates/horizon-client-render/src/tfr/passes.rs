//! WGSL shader sources and wgpu pipelines/dispatch helpers for TFR.
//!
//! Standalone; the frame loop/integrator creates bind groups and calls encode methods.
//!
//! Conventions: Horizon reversed-Z (near=1, far/sky~0), WebGPU clip depth [0,1],
//! UV origin top-left. Rotation-only camera-relative VPs (`projection * pose * rotation`).
//!
//! Supported storage formats (must match WGSL declarations and be storage-capable):
//! - color/motion/output: `wgpu::TextureFormat::Rgba16Float` (intermediate HDR target;
//!   TFR must not composite directly into SDR/swapchain formats such as `Bgra8UnormSrgb`)
//! - edge: `wgpu::TextureFormat::Rgba8Unorm` (filterable + storage, sampled in compose)
//!
//! - `MOTION_WGSL` (`@compute main`, 8x8)
//!   - 0: `texture_2d<f32>` depth
//!   - 1: `texture_storage_2d<rgba16float, write>` motion (RG forward motion target_uv-source_uv, B validity, A edge)
//!   - 2: `uniform MotionParams`
//! - `PREPARE_WGSL` (`@compute main`, 8x8)
//!   - 0: scene color `texture_2d<f32>`
//!   - 1: scene depth `texture_2d<f32>`
//!   - 2: sampler
//!   - 3: prepared color storage `rgba16float` (alpha preserved from source)
//!   - 4: edge storage `rgba8unorm`
//!   - 5: `uniform PrepareParams`
//! - `COMPOSE_WGSL` (`@compute main`, 8x8, synthetic)
//!   - 0 history tex, 1 history sampler, 2 motion tex, 3 motion sampler,
//!     4 edge tex, 5 edge sampler, 6 output storage, 7 `uniform ComposeParams`
//! - `BLIT_WGSL` (`@compute main`, 8x8, real/direct scaled copy)
//!   - 0 src tex, 1 sampler, 2 output storage

/// Motion-field compute shader.
pub const MOTION_WGSL: &str = include_str!("motion.wgsl");
/// Prepared-history pass (color preserved, edge in separate target).
pub const PREPARE_WGSL: &str = include_str!("prepare.wgsl");
/// Synthetic composition pass.
pub const COMPOSE_WGSL: &str = include_str!("compose.wgsl");
/// Real-path blit/scale pass.
pub const BLIT_WGSL: &str = include_str!("blit.wgsl");

/// Supported color/motion/output storage format for TFR WGSL pipelines.
pub const SUPPORTED_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Supported edge storage format (filterable, storage-capable) for TFR WGSL pipelines.
pub const SUPPORTED_EDGE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

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
pub fn history_uv_from_forward_motion(
    current_uv: [f32; 2],
    motion: [f32; 2],
    strength: f32,
) -> [f32; 2] {
    [
        current_uv[0] - motion[0] * strength,
        current_uv[1] - motion[1] * strength,
    ]
}

/// Returns whether `(color_format, edge_format)` are supported by the current TFR WGSL.
pub fn formats_supported(
    color_format: wgpu::TextureFormat,
    edge_format: wgpu::TextureFormat,
) -> bool {
    color_format == SUPPORTED_COLOR_FORMAT && edge_format == SUPPORTED_EDGE_FORMAT
}

/// Validates pipeline formats. Panics with a clear message if unsupported.
pub fn assert_supported_formats(
    color_format: wgpu::TextureFormat,
    edge_format: wgpu::TextureFormat,
) {
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
    pub motion_pipeline: wgpu::ComputePipeline,
    pub prepare_pipeline: wgpu::ComputePipeline,
    pub compose_pipeline: wgpu::ComputePipeline,
    pub blit_pipeline: wgpu::ComputePipeline,
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

        Self {
            motion_layout,
            prepare_layout,
            compose_layout,
            blit_layout,
            motion_pipeline,
            prepare_pipeline,
            compose_pipeline,
            blit_pipeline,
            color_format,
            edge_format,
        }
    }

    /// Dispatch motion-field generation over `width x height` output pixels.
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

    /// Dispatch prepared-history generation over `width x height` output pixels.
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

    /// Dispatch synthetic composition over `width x height` output pixels.
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

    /// Dispatch real-path blit/scale over `width x height` output pixels.
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
}
