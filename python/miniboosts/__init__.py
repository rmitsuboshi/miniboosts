"""Boosting algorithms for research and comparison, implemented in Rust, with scikit-learn-style estimators."""
from ._estimators import (
    AdaBoostClassifier, AdaBoostVClassifier, ArcGvClassifier, BoostByMajorityClassifier,
    BrownBoostClassifier, GradientBoostingClassifier, LogitBoostClassifier, MadaBoostClassifier,
    LPBoostClassifier, ERLPBoostClassifier, CorrectiveERLPBoostClassifier,
    MLPBoostClassifier, RobustBoostClassifier, SmoothBoostClassifier, SoftBoostClassifier,
    SparsiBoostClassifier, TotalBoostClassifier, GraphSeparationBoostingClassifier,
    DecisionTreeClassifier, BoundedLinearClassifier, LogisticRegressionClassifier,
    NaiveBayesClassifier, RBFClassifier, FiniteClassClassifier, RegressionTreeRegressor,
    PredictionAdapter, MajorityOfFiveClassifier, LarsenRitzertClassifier,
    BoostedLearnerClassifier, TwoLayerNetworkClassifier,
)

__version__ = '0.6.0'
__all__ = [
    'AdaBoostClassifier', 'AdaBoostVClassifier', 'ArcGvClassifier', 'BoostByMajorityClassifier',
    'BrownBoostClassifier', 'GradientBoostingClassifier', 'LogitBoostClassifier', 'MadaBoostClassifier',
    'LPBoostClassifier', 'ERLPBoostClassifier', 'CorrectiveERLPBoostClassifier',
    'MLPBoostClassifier', 'RobustBoostClassifier', 'SmoothBoostClassifier', 'SoftBoostClassifier',
    'SparsiBoostClassifier', 'TotalBoostClassifier', 'GraphSeparationBoostingClassifier',
    'DecisionTreeClassifier', 'BoundedLinearClassifier', 'LogisticRegressionClassifier',
    'NaiveBayesClassifier', 'RBFClassifier', 'FiniteClassClassifier', 'RegressionTreeRegressor',
    'PredictionAdapter', 'MajorityOfFiveClassifier', 'LarsenRitzertClassifier',
    'BoostedLearnerClassifier', 'TwoLayerNetworkClassifier',
]
