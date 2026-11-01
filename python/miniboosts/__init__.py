"""Rust boosting algorithms through scikit-learn-style estimators."""
from ._estimators import (
    AdaBoostClassifier, AdaBoostVClassifier, BrownBoostClassifier, MadaBoostClassifier,
    LPBoostClassifier, ERLPBoostClassifier, CorrectiveERLPBoostClassifier,
    MLPBoostClassifier, SmoothBoostClassifier, SoftBoostClassifier,
    SparsiBoostClassifier, TotalBoostClassifier, GraphSeparationBoostingClassifier,
    DecisionTreeClassifier, BoundedLinearClassifier, LogisticRegressionClassifier,
    NaiveBayesClassifier, RBFClassifier, FiniteClassClassifier, PredictionAdapter,
)

__version__ = '0.5.1'
__all__ = [
    'AdaBoostClassifier', 'AdaBoostVClassifier', 'BrownBoostClassifier', 'MadaBoostClassifier',
    'LPBoostClassifier', 'ERLPBoostClassifier', 'CorrectiveERLPBoostClassifier',
    'MLPBoostClassifier', 'SmoothBoostClassifier', 'SoftBoostClassifier',
    'SparsiBoostClassifier', 'TotalBoostClassifier', 'GraphSeparationBoostingClassifier',
    'DecisionTreeClassifier', 'BoundedLinearClassifier', 'LogisticRegressionClassifier',
    'NaiveBayesClassifier', 'RBFClassifier', 'FiniteClassClassifier', 'PredictionAdapter',
]
