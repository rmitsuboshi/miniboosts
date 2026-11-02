"""Rust boosting algorithms through scikit-learn-style estimators."""
from ._estimators import (
    AdaBoostClassifier, AdaBoostVClassifier, ArcGvClassifier, BoostByMajorityClassifier,
    BrownBoostClassifier, GradientBoostingClassifier, LogitBoostClassifier, MadaBoostClassifier,
    LPBoostClassifier, ERLPBoostClassifier, CorrectiveERLPBoostClassifier,
    MLPBoostClassifier, RobustBoostClassifier, SmoothBoostClassifier, SoftBoostClassifier,
    SparsiBoostClassifier, TotalBoostClassifier, GraphSeparationBoostingClassifier,
    DecisionTreeClassifier, BoundedLinearClassifier, LogisticRegressionClassifier,
    NaiveBayesClassifier, RBFClassifier, FiniteClassClassifier, RegressionTreeRegressor,
    PredictionAdapter,
)

__version__ = '0.5.1'
__all__ = [
    'AdaBoostClassifier', 'AdaBoostVClassifier', 'ArcGvClassifier', 'BoostByMajorityClassifier',
    'BrownBoostClassifier', 'GradientBoostingClassifier', 'LogitBoostClassifier', 'MadaBoostClassifier',
    'LPBoostClassifier', 'ERLPBoostClassifier', 'CorrectiveERLPBoostClassifier',
    'MLPBoostClassifier', 'RobustBoostClassifier', 'SmoothBoostClassifier', 'SoftBoostClassifier',
    'SparsiBoostClassifier', 'TotalBoostClassifier', 'GraphSeparationBoostingClassifier',
    'DecisionTreeClassifier', 'BoundedLinearClassifier', 'LogisticRegressionClassifier',
    'NaiveBayesClassifier', 'RBFClassifier', 'FiniteClassClassifier', 'RegressionTreeRegressor',
    'PredictionAdapter',
]
