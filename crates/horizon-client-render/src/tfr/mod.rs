//! Temporal Frame Reprojection (TFR): alternate real and synthetic world frames.
//!
//! Spec: `TFR_DESCRIPTION.md` (clean-room). See `docs/tfr-design.md`.

pub mod config;
pub mod core;

pub use config::TfrConfig;
