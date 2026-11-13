# TFR (Temporal Frame Reprojection) — design (Horizon, clean room)

**Spec source (only):** `TFR_DESCRIPTION.md` (repo root). No Luxium code/resources were consulted.

**Scope (M1):** architecture, core↔GPU interface, render integration points, TAA/scale/frame_gen/post/HDR policy, invalidation plan. WGSL shaders and GPU pass implementations are Agent #2 zone (`crates/horizon-client-render/src/tfr/*.wgsl`, `tfr/passes.rs`, `tfr/math.rs` as assigned); this doc defines their contract.

## 1. Goals and non-goals

- **Goal:** alternate real (fully rendered world) and synthetic (reprojected from history) frames to reduce world-render cost while keeping HUD/crosshair/first-person view and other live overlays current every presented frame, per `TFR_DESCRIPTION.md`.
- **Default:** TFR **off** (no behavior change). All resources freed when off.
- **Clean-room:** behavior follows the spec + Horizon’s existing renderer contracts only.
- **Non-goals (initial):** no Luxium compatibility, no shader-pack-specific TFR path unless explicitly enabled later, no changes to Mojang assets/logic beyond integration hooks.

## 2. Module layout and ownership

Planned (all under `crates/horizon-client-render/src/tfr/`):

- `mod.rs` — public surface, re-exports, feature wiring into renderer.
- `core.rs` (or `state.rs`) — **CPU-only, GPU-free, unit-testable** state machine and decisions (Agent #3 tests target this). **Lead/impl zone.**
- `history.rs` — CPU-side history metadata (camera, matrices, world id, sizes, validity flags) and invalidation bookkeeping; GPU resources owned by passes but described here. **Lead/impl zone.**
- `config.rs` — config struct + validation/clamping + change detection. **Lead/impl zone.**
- `passes.rs`, `math.rs`, `*.wgsl` — GPU passes, shaders, math helpers for motion/prepare/composite. **Agent #2 zone.**

Integration hooks (Lead zone, minimal where touching shared files):

- `crates/horizon-client-render/src/graphics.rs` — settings fields, load/save, preset/key plumbing as needed.
- `crates/horizon-client-render/src/gfx.rs` — renderer fields/resources, frame scheduling, real/synthetic branching, target management, present/composite ordering.
- `crates/horizon-client-render/src/gfx/world_post.rs` — coordination with scale/upscale/TAA and history source selection (prepared vs direct scene) where required.
- `crates/horizon-client/src/game/render_tick.rs` (+ `game.rs` as needed) — plumb config/camera/world identity inputs to `Frame`/renderer.

## 3. Core API (CPU-only, no wgpu types)

Designed so Agent #3 can write conformance-style tests without a device. Types live in `tfr::core`/`tfr::config` and use plain math/value types (`glam`/arrays) only.

```rust
pub struct TfrConfig {
    pub enabled: bool,
    pub reprojection_strength: f32,   // [0.0, 1.0], default 1.0
    pub history_stability: f32,       // [0.0, 0.35], default 0.06
    pub cut_distance: f32,            // [0.25, 16.0] world units, default 9.25
    pub cut_angle_deg: f32,           // [5.0, 90.0] degrees, default 14.0
}
impl TfrConfig { pub fn sanitized(self) -> Self; /* clamp to spec ranges */ }

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind { Real, Synthetic }

#[derive(Clone, Copy, Debug)]
pub struct CameraSample {
    pub world_pos: [f64; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub view: [[f32; 4]; 4],          // or glam::Mat4; convention documented below
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldKey(pub u64); // or (dim, session/generation id) stable per loaded world

#[derive(Clone, Copy, Debug)]
pub struct FrameContext {
    pub world: WorldKey,
    pub scene_size: [u32; 2],         // isolated scene target size
    pub present_size: [u32; 2],
    pub camera: CameraSample,
    pub history_ready: bool,          // GPU/history resources valid & sized & same world
    pub prepared_history_ready: bool, // when preparation mode required
    pub deps_ready: bool,             // pipelines/bindings available for chosen path
    pub scaling_changed: bool,        // forces invalidation per policy
    pub config_changed: bool,
}

pub struct TfrState { /* private: enabled, last_real history meta, synthetic_count, ... */ }

pub enum Decision {
    Real { reason: RealReason },
    Synthetic,
}
pub enum RealReason {
    Disabled, NoHistory, WorldMismatch, SizeMismatch, MissingDeps,
    LimitReached, CutDistance, CutAngle, Invalidated, ErrorFallback,
}

impl TfrState {
    pub fn new() -> Self;
    pub fn invalidate(&mut self, reason: &'static str);
    pub fn on_config_changed(&mut self, cfg: TfrConfig);
    pub fn begin_frame(&mut self, cfg: TfrConfig, ctx: FrameContext) -> Decision;
    pub fn commit_real(&mut self, meta: RealFrameMeta);
    pub fn commit_synthetic(&mut self);
    pub fn commit_fallback_to_real(&mut self);
}
```

Notes:
- `begin_frame` is pure decision logic given `ctx`: enforces “≤1 synthetic after a successful real”, cut thresholds, and all spec preconditions. It must not assume GPU work succeeded.
- `commit_*` called after the frame path actually completes (atomic history update on successful real only).
- Cut detection: squared distance vs `cut_distance^2`; angular delta via wrapped yaw/pitch delta metric (fixed, documented, deterministic) vs `cut_angle_deg`.
- “Meaningful camera motion” for motion-field recompute is a smaller internal epsilon than cut thresholds (decision/output of core helper), per spec.

## 4. History and resources (GPU, Agent #2 implements; interface fixed here)

### 4.1 Logical artifacts (spec terms)
- **Scene target (real frames):** color + depth, isolated from presentation until composite; linear-filterable color sampling.
- **Prepared history (when required):** color (+ optional edge metadata in alpha) at scene size, linear.
- **Motion field:** lower-res target (0.5× scene per axis, min 1), ≥16-bit float components for `delta_uv.xy` + validity + edge/disocclusion measure; linear sampling, clamp-to-edge.
- **Presentation target:** final image users see (swapchain/offscreen chain input) after composite; live overlays drawn after.

### 4.2 Suggested formats (to confirm against `Renderer::frame_format`/HDR caps)
- Scene color/history color: follow render `frame_format` family used for world (`Rgba16Float`/`Rgba16Unorm`/swapchain-unorm) so HDR marks/linear-over-white survive as today; exact choice fixed with HDR path in M2 (default: match world target format).
- Prepared history: same channel model as history color (alpha available for edge metadata).
- Depth history: sampleable depth-compatible copy (as `world_post` TAA does: float view/copy strategy compatible with backend; avoid depth-texture `textureLoad` pitfalls noted in `world_post.wgsl`).
- Motion field: `Rgba16Float` (or `Rg16Float` + separate validity/edge) — RG = delta UV, B = validity/confidence, A = edge/depth-break measure.

### 4.3 Per-real-frame history snapshot (CPU meta + GPU handles)
- `world: WorldKey`
- `camera`: world position, orientation, `view`, `proj`, `view_proj`, `inv_view_proj`
- `scene_size`, `present_size`, scaling mode flags
- references/indices to: scene color, scene depth, prepared color (valid flag)
- `synthetics_since_real: u8` (0 or 1)

History becomes **invalid** on: TFR disable, any TFR config change, any change affecting scene/history resolution or source mode (incl. `world_post` scale/upscale/resolution, window resize), no active world, world/world-key mismatch, buffer size mismatch, (re)alloc failure, missing required deps/pipelines, failed/aborted real pass, explicit lifecycle events making it untrusted. On invalidation: clear validity and stop synthetic use; free TFR-exclusive targets when disabled.

## 5. GPU pass contract (Agent #2 implements)

All passes deterministic for identical inputs/state/path.

1. **Real world pass (TFR active):** render world into isolated scene color+depth (viewport/scissor sized to scene). Preserve/restore global output state. On success update history atomically (meta + valid flags, reset synthetic counter). On failure: drop “special pass” state, restore output, invalidate history, safe fallback present.
2. **Prepare history (conditional):** in: scene color, depth, texel size, mode/quality params; out: prepared color (+ edge alpha). Required when internal world scaling mode needs a prepared source (see §6). Edge-aware, stable for temporal reuse.
3. **Motion field (synthetic, if meaningful motion):** in: previous depth, geo reprojection matrix, sky/far reprojection matrix (no camera translation), sizes; out: per-texel `delta_uv`, validity, edge/break metric. Compute target UV by reprojection+perspective divide; invalidate on bad `w`, OOB (with small tolerance), depth/far handling per spec; sky/far uses no-translation matrix.
4. **Final composite:** in: history source (prepared if required&valid else scene color history), motion field (if used), reprojection strength, history stability, frame mode, edge metadata presence. 
   - Real mode: sample history/source without motion displacement (or direct present of scene) per chosen integration. 
   - Synthetic: apply motion with limited iterative refinement (only while samples valid/in-bounds), clamp to safe interior, deterministic hole fill from valid neighborhood (inward), edge-safe sampling to avoid depth leaks, normalize/clamp output to valid range, opaque write.
   - `history_stability` suppresses high-frequency instability only in reprojection-consistent neighborhood; must not sample non-reprojected history to create camera-lag ghosting; respect depth edges.

## 6. Integration points in Horizon

### 6.1 `gfx.rs` (`Renderer`, `Frame`, render loop)
- Add TFR resources/handles and `tfr::core::TfrState`/config snapshot (or `Option<Tfr>` bundle) to `Renderer`; include in `submit`/reuse/reset flows as appropriate (drop on renderer rebuild/surface reconfig where sizes change).
- **Scheduling:** before world rendering, build `FrameContext` and call `begin_frame`. 
  - `Synthetic`: skip full world render/scene population; run prepare-if-needed dependency (usually none, uses stored history), motion-field (if meaningful motion), composite reconstructed world into presentation path, then render **live** elements (hand, HUD, screen effects, see-through UI elements as applicable) on top.
  - `Real`: render world into isolated scene target at scene resolution; update/validate history inputs; run prepare if required; composite/present world; render live elements; `commit_real` only after successful completion.
- **Live overlays (must run every presented frame):** hand/held item pass, HUD, screen quads/camera overlays, debug/crosshair/outline as per existing ordering — drawn after world image is in the presentation target, without re-rendering world. Ensure outline/high-contrast/after-terrain cases remain correct relative to reconstructed world.
- **Resize/recreate:** any change to `config.width/height`, `world_post` target sizes, or render format path invalidates TFR history.
- **Readback/screenshot/capture/offscreen:** composite produces a normal presentable frame so existing paths (F2, agent capture, offscreen) see the final image; document that synthetic frames are valid outputs.

### 6.2 `gfx/world_post.rs` (scale, FXAA, TAA, upscale)
- Current: world renders at scaled size, post resolves/copies/FXAA/TAA to screen; TAA jitters projection and reprojects history via depth (`TaaUniform { reproject, jitter }`, Halton-like sequence, previous unjittered `view_proj`+camera).
- **LSR / internal scaling mapping:** `world_post.scale`/`upscale` define world render resolution vs output; TFR’s “prepared history” mode applies when source history must be taken from a preprocessed/scaled scene representation per spec. Scene target size = world render size in that mode.
- **TAA interaction (policy):** TFR and TAA are both temporal and assume different frame cadences/history semantics. **Proposed:** when TFR is enabled and eligible, TAA temporal accumulation/jitter path is bypassed for presented frames (or TFR takes precedence and forces TAA off for that renderer state) to avoid conflicting histories and jitter semantics; FXAA/copy/upscale spatial stages remain available only where non-conflicting (to be finalized in M2). Design must guarantee determinism and no double-history reuse. (Explicitly revisited if product wants both; spec requires no conflict with internal scaling.)
- **Matrices:** reuse the same camera convention as `world_post::reprojection`/TAA (`unjittered view_proj`, previous/current camera) for geo reprojection; sky/far matrix variant without translation per spec.

### 6.3 HDR (`gfx/hdr.rs`, `hdr.wgsl`, marks, emission stats)
- History/composite formats must remain compatible with HDR mark/resolve/layered pipeline (`fs_resolve`, coverage/layered HUD, paper/peak/UI brightness, lift/marks in alpha). TFR must not break emissive mark propagation or force incorrect SDR assumptions.
- Real-frame world marks are generated as today into scene/history; synthetic composite must preserve/produce values that downstream HDR passes expect (or run the same resolve/composite contract) — exact placement (composite in linear/HDR-aware space vs before resolve) fixed in M2 with format choice.
- Coordinate with `hdr.rs` targets (`keep_world`/layered/composite) so “world so far” semantics remain coherent.

### 6.4 `frame_gen.rs`
- Frame generation interpolates between consecutive rendered worlds and manages its own present timing/thread and world copies. **Policy:** treat as **mutually exclusive/incompatible** with TFR initially (different temporal model: generated intermediate frames vs real/synthetic alternation). If TFR enabled, disable/pause frame_gen and invalidate its state; UI should prevent ambiguous combinations or define precedence (recommend: TFR off by default, frame_gen unaffected unless user enables TFR — then frame_gen disabled with clear setting state). Re-evaluate later.

### 6.5 Shader packs and post chains (`pack_post`, native packs, `post_chain.rs`)
- Packs may own render targets, depth access, and post ordering. **Vertical-slice policy:** support **vanilla (no active shader pack / no native pack post overriding world path)** first; when unsupported pack mode active, TFR must **safely fall back to real frames** (or be unavailable) without corrupting output/history. `post_chain` effects that run over the frame after world should apply to the final composited image as they do today.

### 6.6 `render_tick.rs` / `game.rs` / `Frame`
- Ensure `Frame` (or equivalent per-frame inputs) carries: stable world key/dimension identity, camera world position/orientation, matrices/projection params needed for history and cuts, active scaling mode flags/sizes, and TFR config snapshot.
- Fog/rain/particles/screen effects continue to tick/update per client tick/frame as today; live overlays must reflect current state on synthetic frames.

### 6.7 `graphics.rs` and UI
Add persisted options (names TBD, snake/camel consistent with file): `tfr`/`tfrEnabled`, `tfrReprojectionStrength`, `tfrHistoryStability`, `tfrCutDistance`, `tfrCutAngle`. Defaults per spec (off, 1.0, 0.06, 9.25, 14.0), clamped on load/save. Expose in Horizon graphics UI with safe presets; changing any TFR or scaling-related option triggers history invalidation. Consider preset interactions (performance presets may suggest on later; M2 keeps off by default).

## 7. Data flow summary

- **Real frame:** decide real → render world to scene(color,depth) → (prepare) → update history/meta atomically → composite real → live overlays → present/post/HUD-final as pipeline dictates.
- **Synthetic frame:** decide synthetic → (prepare dependency already satisfied from history) → motion field from history depth + matrices → composite synthetic to presentation → live overlays → present. No world scene render.
- **Any failure/invalid precondition:** invalidate as needed, fall back to real path this frame, keep pipeline consistent.

## 8. Invalidation and lifecycle checklist

- [ ] Disable TFR → invalidate + free TFR-exclusive GPU resources.
- [ ] Config change (any TFR param) → invalidate immediately.
- [ ] Scaling/source-mode/size change (`world_post` scale/upscale, resize, internal target changes) → invalidate/recreate.
- [ ] World change/unload/mismatch → invalidate.
- [ ] Missing/failed pipeline or pass error → fallback real + invalidate.
- [ ] Real pass abort/exceptional path → clear active-pass markers, restore output state, invalidate.

## 9. Testing and conformance hooks (for Agent #3)

- CPU core tests: alternation limit, cut distance/angle, world/size/deps gating, config clamp/change invalidation, deterministic decisions.
- Math/contract tests (once math helpers exist): matrix conventions, UV/NDC/depth mapping, sky no-translation path, validity/OOB/`w<=eps` rules.
- Integration/conformance scenarios from spec table (static camera, small pan/translate, cut thresholds, resize, world switch, shader failure fallback, LSR mode, TFR off) — driven via renderer test hooks/offscreen captures as feasible.

## 10. Open decisions for M2/M3

- Exact HDR compositing order for synthetic frames (before vs inside HDR resolve/layered path) given `hdr.wgsl` mark semantics.
- Precise TAA coexistence switch (hard disable vs bypass) and UI messaging.
- Frame-gen precedence UX.
- Shader-pack support level and detection of “unsupported pack state”.
- Motion-field/prepare dispatch sizes and performance budgets.

---
*M1 design authored for branch `tfr/impl` (base `origin/main` 441f49d57e).*
