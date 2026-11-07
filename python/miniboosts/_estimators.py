"""Estimator conventions and strict validation around the Rust implementation."""
import numbers
import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
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


def _nonnegative_integer(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


def _finite_real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real) or not np.isfinite(value):
        raise ValueError(f'{name} must be a finite real number')
    return float(value)


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


def _distribution(sample_weight, n_samples):
    weights = np.ones(n_samples) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    if weights.shape != (n_samples,) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError('sample_weight must be finite, nonnegative, and have one entry per row')
    # Scale before summing to avoid overflow in finite unnormalized weights.
    maximum = weights.max()
    if maximum <= 0:
        raise ValueError('sample_weight must have positive total mass')
    weights = weights / maximum
    return weights / weights.sum()


def _confidence(estimator, X, context):
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


def _prediction(estimator, X, context):
    # Regression weak learners: finite, unbounded real predictions.
    raw = estimator.predict(X)
    if np.iscomplexobj(raw):
        raise ValueError(f'{context}: prediction must be real')
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{context}: prediction must be a numeric vector') from error
    if values.shape != (len(X),):
        raise ValueError(f'{context}: prediction shape {values.shape}; expected ({len(X)},)')
    invalid = ~np.isfinite(values)
    if np.any(invalid):
        row = int(np.flatnonzero(invalid)[0])
        raise ValueError(f'{context}, row {row}: invalid prediction {values[row]!r}; '
                         'expected a finite value')
    return values


def _round_context(estimator, round_number):
    return f'{type(estimator).__name__}, round {round_number}'


class _BoostClassifier(ClassifierMixin, BaseEstimator):
    # Boosters whose weak learners fit real-valued working responses.
    _regression = False

    def _default_estimator(self):
        return DecisionTreeClassifier()

    def _weak_values(self, learner, X, round_number):
        context = _round_context(learner, round_number)
        if self._regression:
            return _prediction(learner, X, context)
        return _confidence(learner, X, context)

    def _max_iterations(self):
        _positive_integer('max_iterations', self.max_iterations)
        return int(self.max_iterations)

    def _options(self, n_samples):
        options = {}
        for name in ('tolerance', 'nu', 'gamma', 'kappa'):
            if hasattr(self, name):
                value = getattr(self, name)
                _positive_finite(name, value)
                options[name] = float(value)
        if 'nu' in options and not 1 <= options['nu'] <= n_samples:
            raise ValueError('nu must lie between 1 and the number of training examples')
        if 'gamma' in options and not 0 < options['gamma'] < .5:
            raise ValueError('gamma must lie strictly between 0 and 0.5')
        if 'kappa' in options and not 0 < options['kappa'] < 1:
            raise ValueError('kappa must lie strictly between 0 and 1')
        if hasattr(self, 'update_rule'):
            if self.update_rule not in ('classic', 'short_step', 'line_search'):
                raise ValueError('update_rule must be classic, short_step, or line_search')
            options['update_rule'] = self.update_rule
        return options

    def _set_diagnostics(self, native):
        """Copy algorithm-specific fitted attributes from the native model."""

    def fit(self, X, y):
        """Fit a new ensemble, discarding any previous fitted state."""
        _reset(self)
        try:
            max_iterations = self._max_iterations()
            X, encoded, classes = _training_data(self, X, y)
            template = self.estimator if self.estimator is not None else self._default_estimator()
            method = 'predict' if self._regression else 'decision_function'
            if not callable(getattr(template, 'fit', None)) or not callable(getattr(template, method, None)):
                raise TypeError(f'estimator needs fit(X, y, sample_weight=...) and {method}(X)')
            options = self._options(len(X))
            fitted, rounds = [], []
            # Own the training buffers: callbacks must not mutate shared X/y.
            training_X = np.array(X, copy=True)
            training_y = np.array(encoded, copy=True)

            def produce(weights, round_number, target):
                learner = clone(template)
                # Regression boosters pass their working response as target.
                fit_y = np.array(target, dtype=np.float64) if self._regression else training_y.copy()
                try:
                    learner.fit(training_X.copy(), fit_y,
                                sample_weight=np.array(weights, dtype=np.float64))
                except Exception as error:
                    if hasattr(error, 'add_note'):
                        error.add_note(f'{type(learner).__name__}, round {round_number}: weak learner fit failed')
                    raise
                if (not self._regression and hasattr(learner, 'classes_')
                        and not np.array_equal(learner.classes_, [-1, 1])):
                    raise ValueError(f'{type(learner).__name__}, round {round_number}: '
                                     'classes_ must be ordered [-1, +1]')
                values = self._weak_values(learner, training_X.copy(), round_number)
                fitted.append(learner)
                rounds.append(round_number)
                return values.tolist()

            native = _native.train(self._algorithm, training_X.tolist(), training_y.tolist(),
                                   produce, max_iterations, **options)
            self.classes_ = classes
            self.estimators_ = fitted
            self.estimator_rounds_ = rounds
            self.n_iter_ = native.n_iterations
            self.stop_reason_ = native.stop_reason
            self._set_diagnostics(native)
            self._model_ = native
            return self
        except BaseException:
            _reset(self)
            raise

    def decision_function(self, X):
        """Return ensemble confidence, not a probability or bounded weak score."""
        check_is_fitted(self, '_model_')
        X = validate_data(self, X, reset=False, dtype=np.float64)
        predictions = [self._weak_values(estimator, np.array(X, copy=True), round_number).tolist()
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


class ArcGvClassifier(_BoostClassifier):
    """Arc-gv (Breiman, 1999); its minimal margin converges to the optimum.

    The guarantee is asymptotic, so ``max_iterations`` is the round limit.
    ``stop_reason_ == "algorithm_stop"`` means a step size of 0 (the weak
    hypothesis does not beat the current minimal margin).
    """
    _algorithm = 'arcgv'

    def __init__(self, *, estimator=None, max_iterations=100):
        self.estimator = estimator
        self.max_iterations = max_iterations


class BoostByMajorityClassifier(_BoostClassifier):
    """Boost-by-majority (Freund, 1995): an unweighted vote of k hypotheses.

    ``gamma`` is the assumed advantage (weighted error at most 1/2 - gamma)
    and ``tolerance`` the target training error; k = ceil(ln(1/tolerance) /
    (2 gamma^2)). ``max_iterations=None`` runs all k rounds.
    """
    _algorithm = 'bbm'

    def __init__(self, *, estimator=None, gamma, tolerance=0.01, max_iterations=None):
        self.estimator = estimator
        self.gamma = gamma
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def _max_iterations(self):
        return None if self.max_iterations is None else super()._max_iterations()

    def _options(self, n_samples):
        gamma = _finite_real('gamma', self.gamma)
        if not 0 < gamma < .5:
            raise ValueError('gamma must lie strictly between 0 and 0.5')
        tolerance = _finite_real('tolerance', self.tolerance)
        if not 0 < tolerance < 1:
            raise ValueError('tolerance must lie strictly between 0 and 1')
        return {'gamma': gamma, 'tolerance': tolerance}

    def _set_diagnostics(self, native):
        self.n_rounds_ = native.n_rounds
        self.decided_early_ = native.decided_early


class BrownBoostClassifier(_BoostClassifier):
    """BrownBoost (Freund, 2001); tolerance is the target training error.

    After fitting, ``remaining_time_ <= 0`` means the loop exited, which is when
    Theorem 2 bounds the training error by ``tolerance``. ``stop_reason_`` is
    ``"edge_threshold"`` if a weak hypothesis had correlation at most
    ``edge_threshold``; solver failures raise ``RuntimeError``.
    """
    _algorithm = 'brownboost'

    def __init__(self, *, estimator=None, tolerance=0.1, edge_threshold=1e-3, max_iterations=100):
        self.estimator = estimator
        self.tolerance = tolerance
        self.edge_threshold = edge_threshold
        self.max_iterations = max_iterations

    def _options(self, n_samples):
        tolerance = _finite_real('tolerance', self.tolerance)
        if not 0 < tolerance < 1:
            raise ValueError('tolerance must lie strictly between 0 and 1')
        edge_threshold = _finite_real('edge_threshold', self.edge_threshold)
        if not 0 <= edge_threshold < 1:
            raise ValueError('edge_threshold must lie in [0, 1)')
        return {'tolerance': tolerance, 'edge_threshold': edge_threshold}

    def _set_diagnostics(self, native):
        self.remaining_time_ = native.remaining_time
        self.total_time_ = native.total_time


class GradientBoostingClassifier(_BoostClassifier):
    """Gradient boosting (Friedman, 2001) with a regression weak learner.

    Each round fits ``estimator`` (default ``RegressionTreeRegressor()``) to
    the pseudo-residuals with uniform weights, then line-searches the step.
    ``loss`` is ``"logistic"``, ``"exponential"``, or ``"squared"``.
    ``stop_reason_ == "no_minimizer"`` means a weak hypothesis with
    y h(x) >= 0 on every training row left the line search without a
    minimizer (logistic/exponential loss); that hypothesis is discarded.
    """
    _algorithm = 'gradient_boosting'
    _regression = True

    def __init__(self, *, estimator=None, loss='logistic', shrinkage=1.0, max_iterations=100):
        self.estimator = estimator
        self.loss = loss
        self.shrinkage = shrinkage
        self.max_iterations = max_iterations

    def _default_estimator(self):
        return RegressionTreeRegressor()

    def _options(self, n_samples):
        if self.loss not in ('logistic', 'exponential', 'squared'):
            raise ValueError('loss must be logistic, exponential, or squared')
        shrinkage = _finite_real('shrinkage', self.shrinkage)
        if not 0 < shrinkage <= 1:
            raise ValueError('shrinkage must lie in (0, 1]')
        return {'loss': self.loss, 'shrinkage': shrinkage}

    def _set_diagnostics(self, native):
        self.intercept_ = native.intercept
        self.coefficients_ = np.asarray(native.coefficients)


class LogitBoostClassifier(_BoostClassifier):
    """Two-class LogitBoost (Friedman, Hastie, and Tibshirani, 2000).

    Each round fits ``estimator`` (default ``RegressionTreeRegressor()``) by
    weighted least squares to the working response, clipped to
    [-max_response, max_response].
    """
    _algorithm = 'logitboost'
    _regression = True

    def __init__(self, *, estimator=None, max_response=4.0, max_iterations=100):
        self.estimator = estimator
        self.max_response = max_response
        self.max_iterations = max_iterations

    def _default_estimator(self):
        return RegressionTreeRegressor()

    def _options(self, n_samples):
        max_response = _finite_real('max_response', self.max_response)
        if max_response < 1:
            raise ValueError('max_response must be at least 1')
        return {'max_response': max_response}

    def _set_diagnostics(self, native):
        self.intercept_ = native.intercept
        self.coefficients_ = np.asarray(native.coefficients)


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


class RobustBoostClassifier(_BoostClassifier):
    """RobustBoost (Freund, 2009); tolerance is the error goal epsilon.

    ``remaining_time_ == 0`` means the clock reached 1, so at most a fraction
    2 * tolerance of the training scores y H(x) are at most ``theta``.
    ``stop_reason_`` is ``"nonpositive_edge"`` if a weak hypothesis had
    nonpositive weighted correlation; solver failures raise ``RuntimeError``.
    """
    _algorithm = 'robustboost'

    def __init__(self, *, estimator=None, tolerance=0.1, theta=0.0, sigma_f=0.1, max_iterations=100):
        self.estimator = estimator
        self.tolerance = tolerance
        self.theta = theta
        self.sigma_f = sigma_f
        self.max_iterations = max_iterations

    def _options(self, n_samples):
        tolerance = _finite_real('tolerance', self.tolerance)
        if not 0 < tolerance < 1:
            raise ValueError('tolerance must lie strictly between 0 and 1')
        theta = _finite_real('theta', self.theta)
        if theta < 0:
            raise ValueError('theta must be nonnegative')
        _positive_finite('sigma_f', self.sigma_f)
        return {'tolerance': tolerance, 'theta': theta, 'sigma_f': float(self.sigma_f)}

    def _set_diagnostics(self, native):
        self.remaining_time_ = native.remaining_time


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


class SparsiBoostClassifier(_BoostClassifier):
    """SparsiBoost: AdaBoostV followed by margin-preserving sparsification.

    Set ``n_hypotheses`` (at most T hypotheses), ``tolerance`` (target gap v),
    or both. ``max_iterations=None`` runs the AdaBoostV round count the
    algorithm derives from its target; a smaller cap voids its guarantee.
    """
    _algorithm = 'sparsiboost'

    def __init__(self, *, estimator=None, n_hypotheses=None, tolerance=None, seed=0,
                 max_iterations=None):
        self.estimator = estimator
        self.n_hypotheses = n_hypotheses
        self.tolerance = tolerance
        self.seed = seed
        self.max_iterations = max_iterations

    def _max_iterations(self):
        return None if self.max_iterations is None else super()._max_iterations()

    def _options(self, n_samples):
        if self.n_hypotheses is None and self.tolerance is None:
            raise ValueError('SparsiBoost needs n_hypotheses, tolerance, or both')
        if (isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, numbers.Integral)
                or not 0 <= self.seed < 2**64):
            raise ValueError('seed must be an integer in [0, 2**64)')
        options = {'seed': int(self.seed)}
        if self.n_hypotheses is not None:
            _positive_integer('n_hypotheses', self.n_hypotheses)
            options['n_hypotheses'] = int(self.n_hypotheses)
        if self.tolerance is not None:
            _positive_finite('tolerance', self.tolerance)
            options['tolerance'] = float(self.tolerance)
        return options

    def _set_diagnostics(self, native):
        self.n_selected_ = native.n_selected
        self.sparsification_error_ = native.sparsification_error
        self.sparsification_within_tolerance_ = native.sparsification_within_tolerance
        self.adaboostv_rounds_ = native.adaboostv_rounds
        self.adaboostv_tolerance_ = native.adaboostv_tolerance


class TotalBoostClassifier(_ToleranceClassifier):
    """Hard-margin specialization of SoftBoost."""
    _algorithm = 'totalboost'


class GraphSeparationBoostingClassifier(_BoostClassifier):
    """Graph-separation boosting with the Rust signature aggregation rule."""
    _algorithm = 'graph_separation'

    def __init__(self, *, estimator=None, max_iterations=100):
        self.estimator = estimator
        self.max_iterations = max_iterations


class _NativeWeakClassifier(ClassifierMixin, BaseEstimator):
    """A Rust weak learner fitted on (X, y, sample_weight)."""

    def _check_parameters(self):
        pass

    def _fit_native(self, X, encoded, weights):
        raise NotImplementedError

    def fit(self, X, y, sample_weight=None):
        _reset(self)
        try:
            self._check_parameters()
            X, encoded, classes = _training_data(self, X, y)
            weights = _distribution(sample_weight, len(X))
            self._model_ = self._fit_native(X, encoded, weights)
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


class DecisionTreeClassifier(_NativeWeakClassifier):
    """Rust decision tree with bounded confidence.

    ``split_by`` is ``"entropy"`` (default), ``"edge"``, ``"gini"``, or
    ``"twoing"``. Thresholds lie on equal-width bins unless
    ``exact_splits=True``, which tries every midpoint between distinct values.
    ``leaf_confidence="sign"`` makes nodes predict +-1 instead of the weighted
    purity. With ``max_depth=1``, ``split_by="edge"``, ``exact_splits=True``,
    and ``leaf_confidence="sign"``, the tree is an exact edge-maximizing stump.
    """
    def __init__(self, *, max_depth=1, split_by='entropy', exact_splits=False,
                 leaf_confidence='purity'):
        self.max_depth = max_depth
        self.split_by = split_by
        self.exact_splits = exact_splits
        self.leaf_confidence = leaf_confidence

    def _check_parameters(self):
        _positive_integer('max_depth', self.max_depth)
        if self.split_by not in ('entropy', 'edge', 'gini', 'twoing'):
            raise ValueError('split_by must be entropy, edge, gini, or twoing')
        if not isinstance(self.exact_splits, (bool, np.bool_)):
            raise ValueError('exact_splits must be a boolean')
        if self.leaf_confidence not in ('purity', 'sign'):
            raise ValueError('leaf_confidence must be purity or sign')

    def _fit_native(self, X, encoded, weights):
        return _native.fit_tree(X.tolist(), encoded.tolist(), weights.tolist(), int(self.max_depth),
                                self.split_by, bool(self.exact_splits), self.leaf_confidence)


def _check_selection(selection, gamma):
    """Validate a finite-class selection rule and return gamma as a float or None."""
    if selection not in ('largest', 'weakest', 'exactly'):
        raise ValueError('selection must be largest, weakest, or exactly')
    if selection == 'largest':
        return None
    if (isinstance(gamma, (bool, np.bool_)) or not isinstance(gamma, numbers.Real)
            or not 0 < gamma <= 1):
        raise ValueError('gamma must satisfy 0 < gamma <= 1 for this selection')
    return float(gamma)


class BoundedLinearClassifier(_NativeWeakClassifier):
    """Edge-maximizing linear function (w.x + b) / S with ||(w, b)||_2 <= 1.

    S = sqrt(R^2 + 1), where R is the largest training-row norm, so confidences
    lie in [-1, 1] on the training data; elsewhere they are clamped.
    """

    def _fit_native(self, X, encoded, weights):
        return _native.fit_bounded_linear(X.tolist(), encoded.tolist(), weights.tolist())


class LogisticRegressionClassifier(_NativeWeakClassifier):
    """Weighted, L2-regularized logistic regression (bias regularized too).

    ``output="sign"`` gives a halfspace; ``"tanh"`` gives tanh((w.x + b)/2),
    the difference of the two class probabilities. Non-convergence within
    100 Newton iterations raises ``RuntimeError``.
    """
    def __init__(self, *, regularization=1e-4, output='sign'):
        self.regularization = regularization
        self.output = output

    def _check_parameters(self):
        _positive_finite('regularization', self.regularization)
        if self.output not in ('sign', 'tanh'):
            raise ValueError('output must be sign or tanh')

    def _fit_native(self, X, encoded, weights):
        return _native.fit_logistic_regression(X.tolist(), encoded.tolist(), weights.tolist(),
                                               float(self.regularization), self.output)


class NaiveBayesClassifier(_NativeWeakClassifier):
    """Weighted Gaussian naive Bayes; confidence is P(+1 | x) - P(-1 | x)."""
    def __init__(self, *, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def _check_parameters(self):
        _positive_finite('var_smoothing', self.var_smoothing)

    def _fit_native(self, X, encoded, weights):
        return _native.fit_naive_bayes(X.tolist(), encoded.tolist(), weights.tolist(),
                                       float(self.var_smoothing))


class RBFClassifier(_NativeWeakClassifier):
    """Edge-maximizing Gaussian bump +-exp(-||x - c||^2 / (2 bandwidth^2)).

    Candidate centers are the training rows indexed by ``centers`` (all rows by
    default). After fitting, ``center_index_``, ``center_``, and ``sign_``
    describe the chosen bump. Each fit evaluates every bump on every row.
    ``selection`` chooses the bump as ``FiniteClassClassifier`` does.
    """
    def __init__(self, *, bandwidth, centers=None, selection='largest', gamma=None):
        self.bandwidth = bandwidth
        self.centers = centers
        self.selection = selection
        self.gamma = gamma

    def _check_parameters(self):
        _positive_finite('bandwidth', self.bandwidth)
        _check_selection(self.selection, self.gamma)

    def _fit_native(self, X, encoded, weights):
        centers = None
        if self.centers is not None:
            raw = np.asarray(self.centers)
            if (raw.ndim != 1 or len(raw) == 0 or raw.dtype == np.bool_
                    or not np.issubdtype(raw.dtype, np.integer)
                    or np.any(raw < 0) or np.any(raw >= len(X))):
                raise ValueError('centers must be a nonempty list of training row indices')
            centers = raw.astype(np.int64).tolist()
        model = _native.fit_rbf(X.tolist(), encoded.tolist(), weights.tolist(),
                                float(self.bandwidth), centers, self.selection,
                                _check_selection(self.selection, self.gamma))
        self.center_index_ = model.index if centers is None else centers[model.index]
        self.center_ = X[self.center_index_].copy()
        self.sign_ = model.sign
        self.assumption_violated_ = model.assumption_violated
        return model


class RegressionTreeRegressor(RegressorMixin, BaseEstimator):
    """Rust weighted least-squares regression tree, for regression boosters.

    A feature with at most ``n_bins`` (default 255) distinct values gets
    every split between consecutive values (threshold at the midpoint);
    otherwise its splits lie on ``n_bins`` equal-width bin boundaries.
    ``max_depth=0`` gives the weighted mean. Predictions are unbounded.
    """
    def __init__(self, *, max_depth=3, n_bins=None):
        self.max_depth = max_depth
        self.n_bins = n_bins

    def fit(self, X, y, sample_weight=None):
        _reset(self)
        try:
            _nonnegative_integer('max_depth', self.max_depth)
            if self.n_bins is not None:
                _positive_integer('n_bins', self.n_bins)
            X, y = validate_data(self, X, y, dtype=np.float64, y_numeric=True)
            weights = _distribution(sample_weight, len(X))
            n_bins = None if self.n_bins is None else int(self.n_bins)
            self._model_ = _native.fit_regression_tree(X.tolist(), np.asarray(y, dtype=np.float64).tolist(),
                                                       weights.tolist(), int(self.max_depth), n_bins)
            return self
        except BaseException:
            _reset(self)
            raise

    def predict(self, X):
        check_is_fitted(self, '_model_')
        X = validate_data(self, X, reset=False, dtype=np.float64)
        return np.asarray(self._model_.predict(X.tolist()))


class FiniteClassClassifier(ClassifierMixin, BaseEstimator):
    """Selects a member of a fixed list of hypotheses by its edge.

    ``hypotheses`` are already fitted objects with ``decision_function``
    returning values in [-1, 1]. With ``with_negations=True``, ``-h`` is also a
    candidate. Cloning keeps the same hypothesis objects (they are not
    refitted); do not mutate them. Ties go to the smaller index, then to +h.

    ``selection="largest"`` (default) picks the largest edge. The adversarial
    rules need ``gamma``: ``"weakest"`` picks the smallest edge that is at
    least ``gamma``, and ``"exactly"`` also scales that member by
    ``gamma / edge`` so its edge is exactly ``gamma`` (``sign_`` holds the
    signed scale). If no member reaches ``gamma``, the largest edge is used
    and ``assumption_violated_`` is True.
    """
    def __init__(self, hypotheses, *, with_negations=False, selection='largest', gamma=None):
        self.hypotheses = hypotheses
        self.with_negations = with_negations
        self.selection = selection
        self.gamma = gamma

    def __sklearn_clone__(self):
        # sklearn's default clone would refit-construct unfitted copies.
        return type(self)(list(self.hypotheses), with_negations=self.with_negations,
                          selection=self.selection, gamma=self.gamma)

    def fit(self, X, y, sample_weight=None):
        _reset(self)
        try:
            if not isinstance(self.hypotheses, (list, tuple)) or not self.hypotheses:
                raise ValueError('hypotheses must be a nonempty list')
            for h in self.hypotheses:
                if not callable(getattr(h, 'decision_function', None)):
                    raise TypeError('every hypothesis needs decision_function(X)')
            if not isinstance(self.with_negations, (bool, np.bool_)):
                raise ValueError('with_negations must be a boolean')
            gamma = _check_selection(self.selection, self.gamma)
            X, encoded, classes = _training_data(self, X, y)
            weights = _distribution(sample_weight, len(X))
            columns = [_confidence(h, X.copy(), f'{type(h).__name__}, hypothesis {j}').tolist()
                       for j, h in enumerate(self.hypotheses)]
            index, sign, violated = _native.select_finite_class(
                columns, encoded.tolist(), weights.tolist(), bool(self.with_negations),
                self.selection, gamma)
            self.index_ = index
            self.sign_ = sign
            self.assumption_violated_ = violated
            self.hypothesis_ = self.hypotheses[index]
            self.classes_ = classes
            return self
        except BaseException:
            _reset(self)
            raise

    def decision_function(self, X):
        check_is_fitted(self, 'hypothesis_')
        X = validate_data(self, X, reset=False, dtype=np.float64)
        context = f'{type(self.hypothesis_).__name__}, hypothesis {self.index_}'
        return self.sign_ * _confidence(self.hypothesis_, X, context)

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(np.intp)]


class _SubsampleMajority(ClassifierMixin, BaseEstimator):
    """Unweighted majority vote of a booster fitted on sub-samples (Rust picks the rows)."""

    def _check_parameters(self):
        _nonnegative_integer('seed', self.seed)

    def _subsamples(self, n_samples):
        raise NotImplementedError

    def fit(self, X, y):
        """Fit a clone of ``estimator`` on every sub-sample, discarding any previous fit."""
        _reset(self)
        try:
            self._check_parameters()
            X, encoded, classes = _training_data(self, X, y)
            template = self.estimator if self.estimator is not None else AdaBoostClassifier()
            if not callable(getattr(template, 'fit', None)) or not callable(getattr(template, 'predict', None)):
                raise TypeError('estimator needs fit(X, y) and predict(X)')
            fitted = []
            for k, rows in enumerate(self._subsamples(len(X))):
                rows = np.asarray(rows, dtype=np.intp)
                learner = clone(template)
                try:
                    # Labels are already -1/+1, so the voters do not re-encode them.
                    learner.fit(X[rows].copy(), encoded[rows].copy())
                except Exception as error:
                    if hasattr(error, 'add_note'):
                        error.add_note(f'{type(self).__name__}, sub-sample {k} ({len(rows)} rows): '
                                       'estimator fit failed')
                    raise
                fitted.append(learner)
            self.classes_ = classes
            self.estimators_ = fitted
            return self
        except BaseException:
            _reset(self)
            raise

    def decision_function(self, X):
        """Mean of the voters' -1/+1 predictions; its sign is the majority."""
        check_is_fitted(self, 'estimators_')
        X = validate_data(self, X, reset=False, dtype=np.float64)
        votes = []
        for k, learner in enumerate(self.estimators_):
            raw = np.asarray(learner.predict(np.array(X, copy=True)))
            if raw.shape != (len(X),) or not np.isin(raw, [-1, 1]).all():
                raise ValueError(f'{type(self).__name__}, voter {k}: predictions must be -1/+1')
            votes.append(raw.astype(np.float64))
        return np.mean(votes, axis=0)

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(np.intp)]


class MajorityOfFiveClassifier(_SubsampleMajority):
    """Majority-of-5 (Hoegsgaard, Larsen, and Mathiasen, NeurIPS 2024).

    Shuffles the rows with ``seed``, splits them into ``n_parts`` disjoint
    parts of almost equal size, fits a clone of ``estimator`` (default
    ``AdaBoostClassifier()``) on each, and predicts by unweighted majority.
    The paper's guarantee is for five parts with AdaBoost reaching margins
    Omega(gamma); other part counts follow its Majority-of-X experiments.
    """
    def __init__(self, *, estimator=None, n_parts=5, seed=0):
        self.estimator = estimator
        self.n_parts = n_parts
        self.seed = seed

    def _check_parameters(self):
        super()._check_parameters()
        _positive_integer('n_parts', self.n_parts)

    def _subsamples(self, n_samples):
        if self.n_parts > n_samples:
            raise ValueError('n_parts must not exceed the number of training examples')
        return _native.majority_of_five_parts(n_samples, int(self.n_parts), int(self.seed))


class LarsenRitzertClassifier(_SubsampleMajority):
    """Optimal weak-to-strong learner of Larsen and Ritzert (NeurIPS 2022).

    Fits a clone of ``estimator`` (default ``AdaBoostClassifier()``) on each of
    Hanneke's overlapping sub-samples of the shuffled rows (about m^0.79 of
    them) and predicts by unweighted majority. The guarantee needs voters with
    margins at least gamma/2 on their sub-samples. ``max_subsamples`` uses a
    random subset of the sub-samples, an approximation without the guarantee.
    """
    def __init__(self, *, estimator=None, seed=0, max_subsamples=None):
        self.estimator = estimator
        self.seed = seed
        self.max_subsamples = max_subsamples

    def _check_parameters(self):
        super()._check_parameters()
        if self.max_subsamples is not None:
            _positive_integer('max_subsamples', self.max_subsamples)

    def _subsamples(self, n_samples):
        count = None if self.max_subsamples is None else int(self.max_subsamples)
        return _native.larsen_ritzert_subsamples(n_samples, int(self.seed), count)


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
