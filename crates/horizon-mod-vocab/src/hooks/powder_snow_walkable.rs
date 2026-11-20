//! The `powder_snow_walkable` hook: declarations the mod API generator reads (see `hooks/mod.rs`).
// catalog-order: 0352

pub trait Hooks {
    /// Whether a living entity can walk on the powder snow block at `pos`, with
    /// Horizon's vanilla answer (the walkable-type tag or leather boots on the
    /// feet) as the default: change `can_walk` to change that answer. Asked on the
    /// server where the collision is resolved and on a client predicting it, from
    /// the same decision point; a client's answer is prediction only, the server
    /// stays authoritative. With no listener `can_walk` is bit-for-bit vanilla's.
    #[hook(
        side = "both",
        contract = "listen",
        neoforge = "none (vanilla PowderSnowBlock.canEntityWalkOnPowderSnow; no NeoForge event)",
        cancellable = false,
        target = "horizon_client_render::collisions::native_powder_snow_walkable"
    )]
    fn powder_snow_walkable();
}
