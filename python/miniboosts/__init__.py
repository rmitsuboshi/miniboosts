"""Rust boosting algorithms through scikit-learn-style estimators."""
from ._estimators import (
    AdaBoostClassifier, AdaBoostVClassifier, MadaBoostClassifier,
    LPBoostClassifier, ERLPBoostClassifier, CorrectiveERLPBoostClassifier,
    MLPBoostClassifier, SmoothBoostClassifier, SoftBoostClassifier,
    TotalBoostClassifier, GraphSeparationBoostingClassifier,
    DecisionTreeClassifier, PredictionAdapter,
)

__version__ = '0.5.0'
__all__ = [
    'AdaBoostClassifier', 'AdaBoostVClassifier', 'MadaBoostClassifier',
    'LPBoostClassifier', 'ERLPBoostClassifier', 'CorrectiveERLPBoostClassifier',
    'MLPBoostClassifier', 'SmoothBoostClassifier', 'SoftBoostClassifier',
    'TotalBoostClassifier', 'GraphSeparationBoostingClassifier',
    'DecisionTreeClassifier', 'PredictionAdapter',
]
