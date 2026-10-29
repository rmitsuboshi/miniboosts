"""Estimator conventions and strict validation around the Rust implementation."""
import numbers
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_is_fitted, validate_data

from . import _native


def _reset(estimator):
    # Failed refits must not retain an old model or stale feature/label metadata.
    for name in list(vars(estimator)):
        if name.endswith('_') and not name.startswith('__'):
            delattr(estimator, name)


def _positive_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f'{name} must be a positive integer')


def _positive_finite(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and strictly positive')


def _training_data(estimator, X, y):
    # Preserve mixed Python values until validation; numpy may stringify NaN.
    raw_y = np.asarray(y, dtype=object)
    if raw_y.ndim != 1:
        raise ValueError('y must be one-dimensional')
    for label in raw_y:
        if isinstance(label, (str, bytes, bool, np.bool_)):
            continue
        if not isinstance(label, numbers.Real) or not np.isfinite(label):
            raise ValueError('y must contain finite numeric or string labels, without missing values')
    kinds = {str if isinstance(v, str) else bytes if isinstance(v, bytes) else numbers.Real
             for v in raw_y}
    if len(kinds) > 1:
        raise ValueError('class labels must be all numeric, all strings, or all bytes')
    X, y = validate_data(estimator, X, y, dtype=np.float64, ensure_min_samples=2)
    try:
        classes, inverse = np.unique(y, return_inverse=True)
    except TypeError as error:
        raise ValueError('class labels must have mutually comparable types') from error
    if len(classes) != 2:
        raise ValueError(f'exactly two classes are required; got {len(classes)}')
    encoded = np.where(inverse == 0, -1.0, 1.0)
    already_encoded = kinds == {numbers.Real} and np.array_equal(classes, [-1, 1])
    if not already_encoded:
        a, b = (v.item() if isinstance(v, np.generic) else v for v in classes)
        warnings.warn(
            f'Labels were encoded for training: {a!r} -> -1, {b!r} -> +1. '
            'predict() returns the original labels.', UserWarning, stacklevel=3)
    return X, encoded, classes


def _confidence(estimator, X, round_number):
    context = f'{type(estimator).__name__}, round {round_number}'
    raw = estimator.decision_function(X)
    if np.iscomplexobj(raw):
        raise ValueError(f'{context}: confidence must be real')
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{context}: confidence must be a numeric vector') from error
    if values.shape != (len(X),):
        raise ValueError(f'{context}: confidence shape {values.shape}; expected ({len(X)},)')
    invalid = ~np.isfinite(values) | (values < -1) | (values > 1)
    if np.any(invalid):
        row = int(np.flatnonzero(invalid)[0])
        raise ValueError(f'{context}, row {row}: invalid confidence {values[row]!r}; '
                         'expected a finite value in [-1, 1]')
    return values


class _BoostClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        """Fit a new ensemble, discarding any previous fitted state."""
        _reset(self)
        try:
            _positive_integer('max_iterations', self.max_iterations)
            X, encoded, classes = _training_data(self, X, y)
            template = self.estimator if self.estimator is not None else DecisionTreeClassifier()
            if not callable(getattr(template, 'fit', None)) or not callable(getattr(template, 'decision_function', None)):
                raise TypeError('estimator needs fit(X, y, sample_weight=...) and decision_function(X)')
            options = {}
            for name in ('tolerance', 'nu', 'gamma', 'kappa'):
                if hasattr(self, name):
                    value = getattr(self, name)
                    _positive_finite(name, value)
                    options[name] = float(value)
            if 'nu' in options and not 1 <= options['nu'] <= len(X):
                raise ValueError('nu must lie between 1 and the number of training examples')
            if 'gamma' in options and not 0 < options['gamma'] < .5:
                raise ValueError('gamma must lie strictly between 0 and 0.5')
            if 'kappa' in options and not 0 < options['kappa'] < 1:
                raise ValueError('kappa must lie strictly between 0 and 1')
            if hasattr(self, 'update_rule'):
                if self.update_rule not in ('classic', 'short_step', 'line_search'):
                    raise ValueError('update_rule must be classic, short_step, or line_search')
                options['update_rule'] = self.update_rule
            fitted, rounds = [], []
            # Own the training buffers: callbacks must not mutate shared X/y.
            training_X = np.array(X, copy=True)
            training_y = np.array(encoded, copy=True)

            def produce(weights, round_number):
                learner = clone(template)
                try:
                    learner.fit(training_X.copy(), training_y.copy(),
                                sample_weight=np.array(weights, dtype=np.float64))
                except Exception as error:
                    if hasattr(error, 'add_note'):
                        error.add_note(f'{type(learner).__name__}, round {round_number}: weak learner fit failed')
                    raise
                if hasattr(learner, 'classes_') and not np.array_equal(learner.classes_, [-1, 1]):
                    raise ValueError(f'{type(learner).__name__}, round {round_number}: '
                                     'classes_ must be ordered [-1, +1]')
                values = _confidence(learner, training_X.copy(), round_number)
                fitted.append(learner)
                rounds.append(round_number)
                return values.tolist()

            native = _native.train(self._algorithm, training_X.tolist(), training_y.tolist(),
                                   produce, int(self.max_iterations), **options)
            self.classes_ = classes
            self.estimators_ = fitted
            self.estimator_rounds_ = rounds
            self.n_iter_ = native.n_iterations
            self.stop_reason_ = native.stop_reason
            self._model_ = native
            return self
        except BaseException:
            _reset(self)
            raise

    def decision_function(self, X):
        """Return ensemble confidence, not a probability or bounded weak score."""
        check_is_fitted(self, '_model_')
        X = validate_data(self, X, reset=False, dtype=np.float64)
        predictions = [_confidence(estimator, np.array(X, copy=True), round_number).tolist()
                       for estimator, round_number in zip(self.estimators_, self.estimator_rounds_)]
        return np.asarray(self._model_.decision_function(predictions, len(X)))

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(np.intp)]


class _ToleranceClassifier(_BoostClassifier):
    def __init__(self, *, estimator=None, tolerance=0.01, max_iterations=100):
        self.estimator = estimator
        self.tolerance = tolerance
        self.max_iterations = max_iterations


class _CappedClassifier(_BoostClassifier):
    def __init__(self, *, estimator=None, nu=1.0, tolerance=0.01, max_iterations=100):
        self.estimator = estimator
        self.nu = nu
        self.tolerance = tolerance
        self.max_iterations = max_iterations


class AdaBoostClassifier(_ToleranceClassifier):
    """AdaBoost; tolerance controls the algorithm's iteration budget."""
    _algorithm = 'adaboost'


class AdaBoostVClassifier(_ToleranceClassifier):
    """Margin-maximizing AdaBoostV."""
    _algorithm = 'adaboostv'


class MadaBoostClassifier(_ToleranceClassifier):
    """MadaBoost with capped-product example weights."""
    _algorithm = 'madaboost'


class LPBoostClassifier(_CappedClassifier):
    """LPBoost; nu caps example weights at 1/nu."""
    _algorithm = 'lpboost'


class ERLPBoostClassifier(_CappedClassifier):
    """Entropy-regularized LPBoost."""
    _algorithm = 'erlpboost'


class CorrectiveERLPBoostClassifier(_CappedClassifier):
    """Corrective ERLPBoost with configurable Frank-Wolfe updates."""
    _algorithm = 'corrective_erlpboost'

    def __init__(self, *, estimator=None, nu=1.0, tolerance=0.01,
                 max_iterations=100, update_rule='short_step'):
        super().__init__(estimator=estimator, nu=nu, tolerance=tolerance, max_iterations=max_iterations)
        self.update_rule = update_rule


class MLPBoostClassifier(_CappedClassifier):
    """MLPBoost with configurable Frank-Wolfe updates."""
    _algorithm = 'mlpboost'

    def __init__(self, *, estimator=None, nu=1.0, tolerance=0.01,
                 max_iterations=100, update_rule='line_search'):
        super().__init__(estimator=estimator, nu=nu, tolerance=tolerance, max_iterations=max_iterations)
        self.update_rule = update_rule


class SmoothBoostClassifier(_BoostClassifier):
    """SmoothBoost; gamma is an assumed weak-learning advantage."""
    _algorithm = 'smoothboost'

    def __init__(self, *, estimator=None, gamma=0.25, kappa=0.1, max_iterations=100):
        self.estimator = estimator
        self.gamma = gamma
        self.kappa = kappa
        self.max_iterations = max_iterations


class SoftBoostClassifier(_CappedClassifier):
    """SoftBoost; numerical solver stops raise RuntimeError."""
    _algorithm = 'softboost'


class TotalBoostClassifier(_ToleranceClassifier):
    """Hard-margin specialization of SoftBoost."""
    _algorithm = 'totalboost'


class GraphSeparationBoostingClassifier(_BoostClassifier):
    """Graph-separation boosting with the Rust signature aggregation rule."""
    _algorithm = 'graph_separation'

    def __init__(self, *, estimator=None, max_iterations=100):
        self.estimator = estimator
        self.max_iterations = max_iterations


class DecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """Rust decision tree with entropy splits and bounded confidence."""
    def __init__(self, *, max_depth=1):
        self.max_depth = max_depth

    def fit(self, X, y, sample_weight=None):
        _reset(self)
        try:
            _positive_integer('max_depth', self.max_depth)
            X, encoded, classes = _training_data(self, X, y)
            weights = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
            if weights.shape != (len(X),) or not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError('sample_weight must be finite, nonnegative, and have one entry per row')
            # Scale before summing to avoid overflow in finite unnormalized weights.
            maximum = weights.max()
            if maximum <= 0:
                raise ValueError('sample_weight must have positive total mass')
            weights = weights / maximum
            weights = weights / weights.sum()
            self._model_ = _native.fit_tree(X.tolist(), encoded.tolist(), weights.tolist(), int(self.max_depth))
            self.classes_ = classes
            return self
        except BaseException:
            _reset(self)
            raise

    def decision_function(self, X):
        check_is_fitted(self, '_model_')
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.asarray(self._model_.decision_function(X.tolist()))

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(np.intp)]


class PredictionAdapter(ClassifierMixin, BaseEstimator):
    """Explicitly use a classifier's hard predictions as -1/+1 confidence.

    Unlike passing a decision_function directly, this discards confidence
    magnitude. The wrapped estimator must accept sample_weight in fit.
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y, sample_weight=None):
        _reset(self)
        try:
            X, encoded, classes = _training_data(self, X, y)
            fitted = clone(self.estimator)
            fitted.fit(X, encoded, sample_weight=sample_weight)
            self.estimator_ = fitted
            self.classes_ = classes
            return self
        except BaseException:
            _reset(self)
            raise

    def decision_function(self, X):
        check_is_fitted(self, 'estimator_')
        X = validate_data(self, X, reset=False, dtype=np.float64)
        raw = np.asarray(self.estimator_.predict(X))
        if raw.shape != (len(X),) or not np.isin(raw, [-1, 1]).all():
            raise ValueError('PredictionAdapter requires the wrapped estimator to predict -1/+1 labels')
        return raw.astype(np.float64)

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(np.intp)]
