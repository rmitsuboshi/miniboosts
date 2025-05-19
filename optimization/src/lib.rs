pub mod frank_wolfe;
pub mod objective_function;
pub mod soft_margin_optimization;
pub mod edge_minimization;

pub use objective_function::{
    Entropy,
    ErlpSoftMarginObjective,
    ObjectiveFunction,
    SoftMarginObjective,
};
pub use soft_margin_optimization::{
    soft_margin_optimization,
    ColumnGeneration,
};
pub use edge_minimization::{
    edge_minimization,
    RowGeneration,
    RowGenerationObjective,
    DeformedEntropyRegularizedMaxEdge,
    EntropyRegularizedMaxEdge,
};
pub use frank_wolfe::{
    FwUpdateRule,
    FrankWolfe,
    StepSize,
};

