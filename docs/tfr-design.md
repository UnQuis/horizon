# TFR (Temporal Frame Reprojection) — design (Horizon, clean room)

**Spec:** Clean-room behavioural specification (design follows Horizon's rendering contracts). No external implementation code or resources were consulted.

**Scope (v1):** architecture, core↔GPU interface, render integration points, TAA/scale/frame_gen/post/HDR policy, and invalidation. GPU pass implementations live in `crates/horizon-client-render/src/tfr/*.wgsl`, `tfr/passes.rs`, and `tfr/math.rs`.

**Review notes:** incorporated feedback on TAA/world_post reuse, camera-relative matrix/precision rules, scene sizing/rounding, real-mode blit semantics, `WorldKey`, frame_gen precedence, shader-pack fallback, HDR integration, cut/motion metrics, preconditions/invalidation/commit contracts, prepared-history rules, and Horizon reversed-Z math conventions.

## 1. Goals and non-goals

- **Goal:** alternate real (fully rendered world) and synthetic (reprojected from history) frames to reduce world-render cost while keeping HUD/crosshair/first-person view and other live overlays current every presented frame, per `TFR_DESCRIPTION.md`.
- **Default:** TFR **off** (no behavior change). All TFR-exclusive resources freed when off.
- **Clean-room:** behavior follows the spec + Horizon’s existing renderer contracts only.
- **Non-goals (initial):** no Luxium compatibility, no shader-pack-specific TFR path in the initial slice (see §6.5), no changes to Mojang assets/logic beyond integration hooks.

## 2. Module layout

All under `crates/horizon-client-render/src/tfr/`:

- `mod.rs` — public surface, re-exports, feature wiring into renderer.
- `core.rs` — **CPU-only, GPU-free, unit-testable** state machine and decisions.
- `history.rs` — CPU-side history metadata (camera, matrices, world id, sizes, validity flags) and invalidation bookkeeping; GPU resources owned by passes but described here.
- `config.rs` — config struct + validation/clamping + change detection.
- `passes.rs`, `math.rs`, `*.wgsl` — GPU passes, shaders, and math helpers for motion/prepare/composite.

Integration hooks (minimal where touching shared files):

- `crates/horizon-client-render/src/graphics.rs` — settings fields, load/save, preset/key plumbing as needed.
- `crates/horizon-client-render/src/gfx.rs` — renderer fields/resources, frame scheduling, real/synthetic branching, target management, present/composite ordering.
- `crates/horizon-client-render/src/gfx/world_post.rs` — coordination with scale/upscale and, if reuse is viable, TFR mode over its targets/history/reprojection machinery.
- `crates/horizon-client/src/game/render_tick.rs` (+ `game.rs` as needed) — plumb config/camera/world identity inputs to `Frame`/renderer.

## 3. Core API (CPU-only, no wgpu types)

Designed so core tests can write conformance-style tests without a device. Types live in `tfr::core`/`tfr::config` and use plain math/value types (`glam`/arrays) only.

```rust
pub struct TfrConfig {
    pub enabled: bool,
    pub reprojection_strength: f32,   // [0.0, 1.0], default 1.0
    pub history_stability: f32,       // [0.0, 0.35], default 0.06
    pub cut_distance: f32,            // [0.25, 16.0] world units, default 9.25
    pub cut_angle_deg: f32,           // [5.0, 90.0] degrees, default 14.0
}
impl TfrConfig {
    pub fn sanitized(self) -> Self; /* clamp to spec ranges */
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameKind { Real, Synthetic }

#[derive(Clone, Copy, Debug)]
pub struct CameraSample {
    /// Absolute world-space camera/eye position in f64 (cut-distance and reprojection translation use this).
    pub world_pos: [f64; 3],
    /// Camera orientation in degrees (same conventions as Horizon camera/yaw-pitch).
    pub yaw_deg: f32,
    pub pitch_deg: f32,
    /// Camera-relative transforms in Horizon's renderer convention (see §3.1, §8).
    /// Stored/uploaded as f32; CPU may compute with f64 intermediates where needed.
    pub view: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    pub view_proj: [[f32; 4]; 4],
    pub inv_view_proj: [[f32; 4]; 4],
}

/// Stable identity of the loaded world instance: (dimension + load/session generation).
/// Must change on world change or reload (including reloading the same dimension), so history from
/// a previous instance is never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WorldKey {
    pub dimension: i32,      // or equivalent stable dimension id
    pub generation: u64,     // incremented on world load/open/reload/session change
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameSize { pub width: u32, pub height: u32 }
impl FrameSize {
    pub fn valid(self) -> bool { self.width > 0 && self.height > 0 }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistorySourceMode {
    /// Scene color is the history source.
    DirectScene,
    /// Prepared history is the required source (internal scaling/LSR path where spec requires it).
    PreparedRequired,
}

#[derive(Clone, Copy, Debug)]
pub struct FrameContext {
    /// A world is active and renderable (level/session exists for this frame).
    pub world_active: bool,
    pub world: WorldKey,                 // meaningful only when world_active
    /// Isolated scene render size (world/internal resolution). Must be > 0 on both axes.
    pub scene_size: FrameSize,
    /// Presentation/output framebuffer size. Must be > 0 on both axes.
    pub present_size: FrameSize,
    pub camera: CameraSample,
    /// History exists, valid, same world, compatible sizes/mode/format.
    pub history_ready: bool,
    /// Prepared-history buffer valid for the current history when source mode requires it.
    pub prepared_history_ready: bool,
    /// Which source mode applies this frame given current scaling/path (see §4.4).
    pub history_source_mode: HistorySourceMode,
    /// Pipelines/bindings/resources available for the path required by this decision context.
    pub deps_ready: bool,
    /// Unsupported shader-pack/native-post state that prevents safe isolated TFR path (§6.5).
    pub unsupported_pack_state: bool,
}

pub struct TfrState { /* private */ }

pub enum Decision {
    Real { reason: RealReason },
    Synthetic,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RealReason {
    Disabled,
    NoActiveWorld,
    InvalidSize,
    UnsupportedPack,
    NoHistory,
    WorldMismatch,
    SizeMismatch,
    SourceModeMismatch,
    PreparedHistoryMissing,
    MissingDeps,
    LimitReached,
    CutDistance,
    CutAngle,
    Invalidated,
    ErrorFallback,
}

impl TfrState {
    pub fn new() -> Self;
    /// Invalidate history immediately (reason for diagnostics/logging).
    pub fn invalidate(&mut self, reason: &'static str);
    /// Notify of a TFR config change; invalidates history.
    pub fn on_config_changed(&mut self, new_cfg: TfrConfig);
    /// Notify of a change affecting history resolution/source mode (scale/upscale/resize/mode/format);
    /// invalidates history before the next decision.
    pub fn on_source_or_size_changed(&mut self);
    /// Pure decision for this frame given current (already-invalidated-if-needed) context.
    /// Does not mutate GPU state and does not assume success.
    pub fn begin_frame(&mut self, cfg: TfrConfig, ctx: FrameContext) -> Decision;
    /// Commit only after a fully successful real frame and atomic history update.
    /// Resets synthetic counter to 0 and stores history meta.
    pub fn commit_real(&mut self, meta: RealFrameMeta);
    /// Commit only after a fully successful synthetic frame presentation.
    /// Increments synthetic counter (max 1).
    pub fn commit_synthetic(&mut self);
    /// Record that a real path was taken as fallback/error recovery (no valid new history unless
    /// a successful real history update completed separately). Ensures synthetic is not allowed next.
    pub fn note_fallback_real(&mut self);
}
```

Notes:
- `begin_frame` is pure decision logic. It enforces: enabled, `world_active`, valid sizes (>0), no unsupported pack state, history/deps readiness, source-mode compatibility (including `prepared_history_ready` when `PreparedRequired`), world/size match, limit `synthetics_since_real < 1`, and cut thresholds.
- `synthetics_since_real` is reset **only** by `commit_real` on successful history update. It is never reset by synthetic or by failed/aborted real passes. `invalidate` / `note_fallback_real` must leave state such that synthetic is disallowed until a successful `commit_real`.
- **No commit on error/abort/incomplete pass.** If a real pass aborts or any required stage fails, caller invalidates and/or calls `note_fallback_real` and presents via safe fallback; history is not treated as updated.

## 3.1 Cut detection and meaningful motion (explicit)

All thresholds/metrics are fixed, deterministic, and applied consistently (CPU and, where relevant, GPU helpers use equivalent logic).

- **Distance cut:** let `d = world_pos_now - world_pos_ref` (f64 world units). Cut if `dot(d,d) > cut_distance^2`.
- **Angle cut:** compute wrapped deltas in degrees
  - `dyaw = wrap_degrees(yaw_now - yaw_ref)` to `(-180, 180]`
  - `dpitch = clamp_pitch(pitch_now) - clamp_pitch(pitch_ref)` (or wrapped consistently within valid pitch range)
  - `angle_metric = sqrt(dyaw*dyaw + dpitch*dpitch)` (L2 norm of components)
  Cut if `angle_metric > cut_angle_deg`.
- **Meaningful motion (for motion-field recompute only):** recompute motion field when camera changed beyond small fixed epsilons **below** cut thresholds, e.g.
  - position: `dot(d,d) > MOTION_POS_EPS2` (suggested `MOTION_POS_EPS = 0.001` world units)
  - or angle: `angle_metric > MOTION_ANGLE_EPS_DEG` (suggested `0.01` deg)
  Exact constants are implementation-defined but must be `<` cut ranges, documented in code, and stable/deterministic. **Absence of meaningful motion never forbids a synthetic frame**; it only allows skipping/reusing motion-field computation per §5.

## 3.2 Precision, matrices, and sizing rules

- **Camera/world space:** `world_pos` is absolute `[f64; 3]`. Distance/motion/cut math uses f64 on CPU where precision matters.
- **Transforms:** `view/proj/view_proj/inv_view_proj` are camera-relative transforms in Horizon’s renderer convention, consistent with `gfx/world_post.rs::reprojection(previous_view_proj, previous_camera, current_view_proj, current_camera)` and frame-generation camera motion. They are not mixed world-space+f32-position matrices. Storage/order follows `glam`/wgpu conventions used by existing shaders (column-major matrices; `matrix * vec4` multiplication order as in current WGSL). CPU/GPU transforms must be equivalent for identical inputs (testable).
- **Scene/present sizes (LSR/internal scaling):** `scene_size` follows `world_post` target sizing per axis: `max(1, round(present_axis * scale).clamp(1, 8192))` (see `gfx/world_post.rs::sizes_with_upscale`); same formula/rule must be used for core decisions, allocations, and tests. `present_size` is the framebuffer/output size.

## 4. History and resources (GPU, GPU pass layer implements; interface fixed here)

### 4.1 Logical artifacts (spec terms)
- **Scene target (real frames):** color + depth, isolated from presentation until blit/composite; linear-filterable color sampling.
- **Prepared history (when required):** color (+ optional edge metadata in alpha) at `scene_size`, linear.
- **Motion field:** lower-res target `max(1, floor(scene_size/2))` per axis, ≥16-bit float components for `delta_uv.xy` + validity + edge/disocclusion measure; linear sampling, clamp-to-edge.
- **Presentation/world-output target:** final world image for downstream HDR/post/UI stages after real blit or synthetic composite; live overlays drawn after.

### 4.2 Formats (align with `Renderer::frame_format`/HDR caps)
- Scene color/history color: match world/scene render format family (`Renderer::frame_format` / `WIDE_FRAME`/`LINEAR_FRAME` or swapchain-unorm path as applicable) so HDR marks and >1.0 values survive as today.
- Prepared history: same channel model as history color (alpha available for edge metadata).
- Depth history: sampleable depth-compatible representation (as `world_post` TAA uses a float-readable depth view/copy strategy; avoid depth-texture `textureLoad` pitfalls noted in `world_post.wgsl`).
- Motion field: `Rgba16Float` (or split) — RG = delta UV, B = validity/confidence, A = edge/depth-break measure.

### 4.3 Per-real-frame history snapshot
- `world: WorldKey`
- `camera`: `world_pos`, orientation, camera-relative matrices per §3.1/§8
- `scene_size`, `present_size`, `history_source_mode`
- references/indices to scene color, scene depth, prepared color and its validity
- `synthetics_since_real: u8` (0 or 1)

### 4.4 Prepared history / LSR modes (explicit)

- `HistorySourceMode::PreparedRequired` when the active internal-scaling/source mode requires a prepared history source per spec (world render/internal scaling path where reconstruction must read prepared/preprocessed history rather than raw scene history). Concretely, this is determined by the active `world_post` scaling/upscale/source configuration in the TFR-enabled path; the mode flag in `FrameContext` is the authoritative decision input.
- **Validity:** `prepared_history_ready` becomes true only after a successful prepare pass produces a prepared history consistent with the current valid scene/history, sizes, and mode; it is cleared on invalidation, size/source/mode change, or prepare failure.
- **If required but not ready/valid:** synthetic is **forbidden** (`PreparedHistoryMissing`) and the frame must take a real path (or real fallback). A real frame may generate/refresh prepared history; only after full success can history be committed as ready for subsequent synthetic use.

### 4.5 Invalidation triggers (must occur before `begin_frame` decision)

History is invalidated (and validity flags cleared) before deciding the affected frame on:
- TFR disable, or enable transition resets temporal state
- any TFR config parameter change (`on_config_changed`)
- any change affecting resolution/source mode/format (`on_source_or_size_changed`): `world_post` scale/upscale, window/output resize, `scene_size`/`present_size` change or mismatch vs resources, render format/path change, mode switch requiring different source
- no active world / `WorldKey` change (world switch or reload/generation change)
- unsupported pack state transition as needed to avoid stale reuse
- missing/failed pipeline or pass error, (re)alloc failure
- failed/aborted/incomplete real pass

Caller contract: apply `invalidate` / `on_config_changed` / `on_source_or_size_changed` (and update `FrameContext` flags) **before** `begin_frame` so the decision cannot use stale history.

## 5. GPU pass contract (GPU pass layer implements)

All passes deterministic for identical inputs/state/path.

1. **Real world pass (TFR active):** render world into isolated scene color+depth at `scene_size`. Preserve/restore global output state. Do not commit history on abort/error.
2. **Prepare history (conditional):** in: scene color, depth, texel size, mode params; out: prepared color (+ edge alpha). Run when `PreparedRequired` for the real-frame history update (and as specified by mode). On failure: invalidate prepared readiness and fall back safely.
3. **Motion field (synthetic, only when meaningful motion):** in: previous depth, geo reprojection matrix, sky/far reprojection matrix (no camera translation), sizes; out: `delta_uv`, validity, edge/break metric. If no meaningful motion (§3.1), recompute may be skipped and zero/previous-equivalent field used (must remain correct and deterministic). Compute target UV by reprojection + perspective divide; invalidate on bad `w`, OOB (with small tolerance), and invalid depth. Horizon uses **reversed-Z** (near=1, far=0): sky/far/infinite depth is detected near **0.0** (see §8), and sky/far reprojection uses the no-translation matrix.
4. **Final world output:**
   - **Real mode:** direct blit/copy/resolve of current scene image into the world-output target for downstream stages (**no temporal history sampling**). If composition/blit unavailable, perform fallback present (copy scene to output preserving geometry) and degrade safely.
   - **Synthetic mode:** reconstruct from valid history source (prepared if `PreparedRequired`, else direct scene history) using motion field (if applicable), reprojection strength, and history stability. Limited iterative refinement only while samples valid/in-bounds; clamp to safe interior; deterministic inward hole fill from valid neighborhood; edge-safe sampling to avoid depth leaks; normalize/clamp output to valid range; opaque write.
   - `history_stability` suppresses high-frequency instability only in reprojection-consistent neighborhood; must not sample non-reprojected history to create camera-lag ghosting; respect depth edges.

## 6. Integration points in Horizon

### 6.1 `gfx.rs` (`Renderer`, `Frame`, render loop)
- Add TFR resources/handles and `tfr::core::TfrState`/config snapshot to `Renderer`; integrate with recreate/reset/submit lifecycles; size/format changes invalidate.
- **Per-frame contract:** determine `WorldKey`/active world, sizes/mode flags, pack support state; apply config/source/size invalidations **before** `begin_frame`; build `FrameContext`; decide.
  - `Synthetic`: skip full world render; ensure prepared/history deps valid; motion-field (if meaningful motion) + synthetic composite → world-output; then live overlays.
  - `Real`: render world to isolated scene; (prepare if required and successful) → atomic history/meta update → **blit/resolve scene to world-output**; then live overlays; `commit_real` only on full success.
- **Live overlays every presented frame:** hand/held item, HUD, screen quads/camera overlays, debug/crosshair/outline per existing ordering — after world image is in the output target, without re-rendering world.
- **Readback/screenshot/capture/offscreen:** final composited/blitted frame is the normal output.

### 6.2 `gfx/world_post.rs` (scale, FXAA, TAA, upscale) — reuse-first, TAA/HDR concrete plan

**Existing relevant mechanisms (code refs):**
- `gfx/world_post.rs`: `Post::jitter` computes Halton sub-pixel jitter and `TaaUniform { reproject, jitter }`; tracks `taa_previous: Option<(DMat4 view_proj, DVec3 camera)>`; manages ping-pong history at output size (`History { history, resolve, show }`), depth binding, reset on invalidation/light-key changes (`baked_light`, rebuild). `reprojection(previous, previous_camera, current, current_camera)` is camera-relative. TAA resolve blends history with neighborhood clipping and jitter-aware weighting; `sizes_with_upscale` computes render/output sizes.
- `world_post.wgsl`: `fs_taa` reprojects history via depth and `taa.reproject`, clips to neighborhood, and blends; depth is sampled as float.

**Reuse-first investigation (required before duplicating history/reprojection):**
- Evaluate implementing TFR as a **mode over/alongside `world_post`**: real anchors render world into the world target and update a stable history source (reusing scene/depth/history resources and camera-relative reprojection math where compatible) **without** TAA accumulation/jitter cadence; synthetic frames reconstruct/resolve from that history for the current camera **without running the world pass**, then continue through normal present/post/UI stages.
- Preferred if it satisfies spec cadence (≤1 synthetic after successful real), deterministic reconstruction, and no TAA progressive-blend/jitter semantics leaking into TFR output.
- **Known constraints/conflicts (to resolve in M2):** Halton jitter sequence, accumulation+neighborhood clip tuned for AA/upscale, `taa_on`/reset coupling, history at output size with upscale/sharpen path, and TAA `previous` invalidation rules differ from TFR. Jitter must not perturb TFR anchor/cut/history semantics; synthetic must not depend on TAA accumulation state.

**Deterministic TAA/TFR policy:**
- **Precedence:** when TFR is enabled and the frame takes the TFR path, TFR owns temporal presentation for that frame.
- If reuse is viable: TFR uses explicitly controlled resources/state; TAA jitter is not applied to TFR decision/history camera samples, TAA accumulation resolve is not used as TFR synthetic reconstruction, and shared temporal state is reset/segregated so modes do not contaminate each other.
- If reuse is not viable: bypass TAA temporal accumulation/jitter/previous-history updates for TFR-presented frames (e.g. force TFR path independent of `taa_on` accumulation, do not advance TAA history for synthetic frames) and introduce dedicated TFR buffers. **Before end of M2**, record the concrete code-referenced rationale in this doc (or an appended M2 note committed with the implementation) — not left open.

**HDR concrete plan (validated/finalized in M2, integration point fixed now):**
- TFR produces the finalized world image (real blit or synthetic reconstruction) in the active world/frame format contract (`Renderer::frame_format`/HDR-capable path), preserving HDR alpha/mark semantics required by downstream passes.
- Integration point is **before HDR resolve/layered world compositing**: `gfx/hdr.rs`/`hdr.wgsl` (`resolve`/`keep_world`/coverage/layered HUD/composite/`fs_hdr*`) consume the finalized world image as they do for normal world frames. TFR does not create a parallel HDR path.
- M2 verifies mark propagation for real-blit and synthetic outputs (emissive/light marks, hand mark/coverage interactions, UI layering inputs); if marks are not preserved implicitly, add an explicit deterministic mark-preservation/emission step at this same boundary.

### 6.3 HDR (`gfx/hdr.rs`, `hdr.wgsl`)
- Real blit preserves scene color+alpha marks written during world rendering. Synthetic reconstruction must output values compatible with HDR resolve/layered expectations (including alpha-mark semantics) so brightness/lift/layered HUD/coverage remain correct; validated in M2 per §6.2.

### 6.4 `frame_gen.rs`
- **Precedence:** TFR has priority. When TFR is enabled/active, frame generation is stopped/paused and its state reset (`FrameGen::stop()` semantics: presenter stopped, pending/generated state cleared) so no generated frames are presented.
- **UI/config:** prevent ambiguous combinations (disable/hide frame_gen or auto-turn it off when enabling TFR, with explicit state). Re-enabling frame_gen requires TFR off.

### 6.5 Shader packs and post chains (`pack_post`, native packs, `post_chain.rs`)
- **Unsupported state (initial):** active shader pack / `pack_post` or native pack path that owns or alters world render targets/composition such that isolated scene/history/TFR composite cannot be guaranteed.
- **Behavior:** safe fallback to **real** frames only (or TFR reported unavailable), no synthetic decisions, no stale history reuse/corruption, output state restored. Vanilla (no unsupported pack/native-post world override) supported first.
- `post_chain` effects applied after world run on the final composited/blitted image as today.

### 6.6 `render_tick.rs` / `game.rs` / `Frame`
- Provide/propagate: `WorldKey` (dimension + load/generation), world-active flag, camera `world_pos`/orientation and camera-relative matrices, `scene_size`/`present_size`, scaling/source-mode indicators, unsupported-pack indicator, and TFR config snapshot.
- Per-tick/frame systems (fog, rain, particles, screen effects) continue updating as today; live overlays reflect current state on synthetic frames.

### 6.7 `graphics.rs` and UI
- Persisted options (names consistent with existing file): TFR enable + four spec parameters. Defaults per spec (off, 1.0, 0.06, 9.25, 14.0), clamped on load/save.
- Any TFR or source/scale/resolution-affecting change triggers invalidation before next decision (§4.5).
- UI enforces TFR-vs-frame_gen precedence (§6.4). TFR remains off by default in presets initially.

## 7. Data flow summary

- **Real frame:** invalidate-if-needed → `begin_frame`(Real) → render world to scene(color,depth) at `scene_size` → (prepare if `PreparedRequired`) → on full success update history/meta atomically + `commit_real` → **blit/resolve scene to world-output** → HDR resolve/layered compositing → live overlays → present/post.
- **Synthetic frame:** invalidate-if-needed → `begin_frame`(Synthetic) → (prepared ready if required) → motion field if meaningful motion → synthetic composite to world-output → HDR resolve/layered compositing → live overlays → `commit_synthetic` on full success → present/post. No world render.
- **Failure/abort/invalid precondition:** no commit; invalidate/note fallback; safe real/fallback present with consistent pipeline state.

## 8. Math and coordinate conventions (Horizon/wgpu, explicit)

**Horizon rendering facts (code refs):** `gfx.rs` uses reversed depth (near maps to 1, far to 0, depth compare `GreaterEqual`, e.g. around reversed-depth projection setup and pipeline depth states); TAA/depth reads in `world_post.wgsl` treat larger depth as nearer (`near = max(...)` over neighborhood) consistent with reversed-Z.

**Adaptation of `TFR_DESCRIPTION.md` to Horizon (reversed-Z):**
- Spec states far/sky depth ≈ 1.0 in a conventional [0,1] depth. In Horizon reversed-Z, **far/sky is depth ≈ 0.0** and **near is 1.0**.
- Sky/far (no camera translation) reprojection path is selected when sampled depth is within a small epsilon of **0.0** (not 1.0). Valid geometric depth is away from that far sentinel per tolerance.
- When reconstructing clip/NDC from stored depth for reprojection, use Horizon’s actual depth/NDC mapping (including reversed-Z). If forming NDC `z` from depth `d` in [0,1], account for reversed-Z (`z_ndc` mapping must match the projection used); CPU and GPU must implement the same convention.
- Depth validity: reject samples with invalid `w`/projection results, OOB (with small edge tolerance), and depth classified as invalid for the chosen path. Tolerances (e.g. `w > W_EPS`, screen/depth edge epsilons, far epsilon near 0.0) are fixed, documented, and shared by CPU/GPU logic.

**UV/screen conventions:**
- Follow existing Horizon/WGSL post conventions (e.g. `world_post.wgsl` fullscreen UV mapping/flip) for target sampling and output; document the exact UV origin and Y direction at implementation. Target/sample coordinates and delta-UV signs must be consistent end-to-end.

**Matrices/order:**
- Column-major storage, `mat4 * vec4` multiplication consistent with `glam`/wgsl usage in `world_post`/TAA paths. Geo reprojection maps previous-frame sample to current frame accounting for camera translation (camera-relative), sky/far reprojection omits translation.

**CPU/GPU equivalence:** reprojection, cut/meaningful-motion metrics (where applicable), validity classification, and size/rounding rules must be equivalent between CPU core/helpers and GPU passes for identical inputs.

## 9. Testing and conformance hooks (for core tests)

- CPU core tests: alternation limit, cut distance/angle (L2 metric + wrapping), meaningful-motion thresholds behavior (recompute-only, never blocks synthetic), world-active/size/deps/pack gating, config clamp/change invalidation, source/size invalidation ordering, prepared-required gating, deterministic decisions, `scene_size` rounding/max(1), `WorldKey` change invalidation, commit/reset rules (no commit on abort; reset only on successful `commit_real`).
- Math/contract tests: camera-relative matrix conventions, reversed-Z depth/far-sky handling (far≈0), UV/NDC mapping, sky no-translation path, validity/OOB/`w` epsilon rules, CPU/GPU-equivalent expectations.
- Integration/conformance scenarios from spec table (static camera, small pan/translate, cut thresholds, resize, world switch, shader failure fallback, LSR mode, TFR off) — via renderer hooks/offscreen captures as feasible.

## 10. M2 deliverables/decisions to close in implementation

- **TAA reuse outcome:** confirmed reuse/adaptation of `world_post` history/reprojection path for TFR with documented mapping, or dedicated buffers + explicit code-referenced rationale for bypassing TAA accumulation/jitter in TFR mode (recorded before end of M2).
- **HDR validation:** confirm mark propagation and finalize exact resolve/composite ordering per §6.2 against `gfx/hdr.rs`/`hdr.wgsl`.
- **Frame-gen/UI enforcement:** precedence and unambiguous settings behavior implemented.
- **Shader-pack detection:** concrete unsupported-state conditions wired to safe real fallback.
- **Constants finalized:** meaningful-motion epsilons, w/bounds/far tolerances, motion-field resolution/iteration limits as per spec/perf constraints.

