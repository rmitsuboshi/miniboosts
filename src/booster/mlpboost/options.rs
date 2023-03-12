//! This file defines some options of MLPBoost.

/// Secondary updates.
/// You can choose the heuristic updates from these options.
#[derive(Clone, Copy)]
pub enum Secondary {
    /// LPBoost update.
    LPB,
    // /// ERLPBoost update.
    // ERLPB,
    // /// No heuristic update.
    // Nothing,
}


