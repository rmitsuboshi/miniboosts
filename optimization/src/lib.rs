pub mod edge_minimization;
pub mod frank_wolfe;
pub mod objective_function;
pub mod soft_margin_optimization;

pub use edge_minimization::{
    DeformedEntropyRegularizedMaxEdge, EntropyRegularizedMaxEdge, RowGeneration,
    RowGenerationObjective, edge_minimization,
};
pub use frank_wolfe::{FrankWolfe, FwUpdateRule, StepSize};
pub use objective_function::{
    Entropy, ErlpSoftMarginObjective, ObjectiveFunction, SoftMarginObjective,
};
pub use soft_margin_optimization::{ColumnGeneration, soft_margin_optimization};
