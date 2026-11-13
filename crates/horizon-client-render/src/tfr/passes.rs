//! WGSL shader sources and wgpu pass/pipeline builders for TFR.
//!
//! Standalone; the frame loop wires resources and dispatch order.
//!
//! Conventions: Horizon reversed-Z (near=1, far/sky~0), WebGPU clip depth [0,1],
//! UV origin top-left. Rotation-only camera-relative VPs (`projection * pose * rotation`).
//!
//! - `MOTION_WGSL` (`@compute main`, 8x8)
//!   - 0: `texture_2d<f32>` depth (load)
//!   - 1: `texture_storage_2d<rgba16float, write>` motion (RG forward motion target_uv-source_uv, B validity, A edge)
//!   - 2: `uniform MotionParams`
//! - `PREPARE_WGSL` (`@compute main`, 8x8)
//!   - 0: scene color `texture_2d<f32>`
//!   - 1: scene depth `texture_2d<f32>`
//!   - 2: sampler
//!   - 3: prepared color storage `rgba16float` (alpha preserved from source)
//!   - 4: edge storage `r8unorm`
//!   - 5: `uniform PrepareParams`
//! - `COMPOSE_WGSL`
//!   - `fs_compose` (synthetic): 0 history tex,1 history sampler,2 motion tex,3 motion sampler,
//!     4 edge tex,5 edge sampler,6 output storage,7 `uniform ComposeParams`
//!   - `fs_blit` (real/direct copy): 0 src tex,1 dst storage

/// Motion-field compute shader.
pub const MOTION_WGSL: &str = include_str!("motion.wgsl");
/// Prepared-history pass (color preserved, edge in separate target).
pub const PREPARE_WGSL: &str = include_str!("prepare.wgsl");
/// Synthetic composition and direct blit.
pub const COMPOSE_WGSL: &str = include_str!("compose.wgsl");

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_sources_are_non_empty_and_parse() {
        for (name, src) in [
            ("motion", MOTION_WGSL),
            ("prepare", PREPARE_WGSL),
            ("compose", COMPOSE_WGSL),
        ] {
            assert!(!src.trim().is_empty(), "{name} shader is empty");
            naga::front::wgsl::parse_str(src).unwrap_or_else(|e| {
                panic!("{name} WGSL parse error: {e}");
            });
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
        // Forward motion says content moved +x by 0.1; current pixel should sample history at -0.1.
        let uv = history_uv_from_forward_motion([0.5, 0.5], [0.1, -0.05], 1.0);
        assert!((uv[0] - 0.4).abs() < 1.0e-6);
        assert!((uv[1] - 0.55).abs() < 1.0e-6);
    }
}
