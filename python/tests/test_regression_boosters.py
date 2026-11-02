import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.tree import DecisionTreeRegressor
from miniboosts import GradientBoostingClassifier, LogitBoostClassifier, RegressionTreeRegressor

from test_api import X, Y
from test_brownboost_sparsiboost import X2, Y2

DATA = np.arange(4).reshape(-1, 1)
SKEWED = np.array([-1, -1, -1, 1])


class RecordingRegressor(RegressorMixin, BaseEstimator):
    """Records the working response and weights; predicts a constant."""
    def __init__(self, value=0.0):
        self.value = value

    def fit(self, X, y, sample_weight=None):
        self.y_ = np.array(y, copy=True)
        self.weights_ = np.array(sample_weight, copy=True)
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


@pytest.mark.parametrize('loss,intercept,residual', [
    # F_0 = argmin_c sum L(y, c) for mean label -1/2; residual = -dL/dF at F_0.
    ('squared', -.5, [-.5, -.5, -.5, 1.5]),
    ('logistic', .5 * np.log(1 / 3), [-.5, -.5, -.5, 1.5]),
    ('exponential', .5 * np.log(1 / 3), [-3**-.5] * 3 + [3**.5]),
])
def test_gradient_boosting_intercept_and_pseudo_residuals(loss, intercept, residual):
    clf = GradientBoostingClassifier(estimator=RecordingRegressor(value=1.0), loss=loss,
                                     max_iterations=1).fit(DATA, SKEWED)
    assert clf.intercept_ == pytest.approx(intercept, abs=1e-15)
    np.testing.assert_allclose(clf.estimators_[0].y_, residual, atol=1e-14)
    np.testing.assert_allclose(clf.estimators_[0].weights_, [.25] * 4, atol=1e-15)


def test_gradient_boosting_squared_loss_steps_with_shrinkage():
    # Residual y is fitted exactly by a depth-1 tree, so the line search gives
    # rho = 1 and F_1 = nu y; then the residual (1 - nu) y gives F_2 = 0.75 y.
    clf = GradientBoostingClassifier(estimator=RegressionTreeRegressor(max_depth=1), loss='squared',
                                     shrinkage=.5, max_iterations=2).fit(DATA, Y)
    np.testing.assert_allclose(clf.coefficients_, [.5, .5], atol=1e-12)
    np.testing.assert_allclose(clf.decision_function(DATA), .75 * Y, atol=1e-12)
    assert (clf.n_iter_, clf.stop_reason_) == (2, 'max_iterations')


def test_gradient_boosting_zero_hypothesis_is_reported_not_raised():
    # rho is undefined for h = 0; also pins the Rust stop message.
    clf = GradientBoostingClassifier(estimator=RecordingRegressor(), max_iterations=5).fit(DATA, Y)
    assert (clf.n_iter_, clf.stop_reason_, len(clf.coefficients_)) == (1, 'zero_hypothesis', 0)
    np.testing.assert_array_equal(clf.decision_function(DATA), 0.0)


def test_logitboost_working_response_and_newton_step():
    # Algorithm 3 of Friedman, Hastie, and Tibshirani (2000): at F = 0,
    # p = 1/2, z = (y* - p) / (p (1 - p)) = 2y and w = 1/4. A depth-1 tree
    # fits z exactly, so F_1 = z/2 = y; then z = y (1 + e^{-2}) for p = sigma(2F).
    clf = LogitBoostClassifier(estimator=RegressionTreeRegressor(max_depth=1),
                               max_iterations=2).fit(DATA, Y)
    np.testing.assert_allclose(clf.coefficients_, [.5, .5])
    np.testing.assert_allclose(clf.decision_function(DATA), np.tanh(1 + (1 + np.exp(-2)) / 2) * Y,
                               atol=1e-12)
    recorder = LogitBoostClassifier(estimator=RecordingRegressor(), max_iterations=1).fit(DATA, Y)
    np.testing.assert_allclose(recorder.estimators_[0].y_, 2.0 * Y)
    np.testing.assert_allclose(recorder.estimators_[0].weights_, [.25] * 4)


def test_logitboost_weight_underflow_is_reported_not_raised():
    # On separable data |F| grows by about 1/2 per round until every weight
    # p (1 - p) underflows; also pins the Rust stop message.
    clf = LogitBoostClassifier(estimator=RegressionTreeRegressor(max_depth=1),
                               max_iterations=2000).fit(X, Y)
    assert clf.stop_reason_ == 'weights_underflow' and clf.n_iter_ < 2000
    np.testing.assert_array_equal(clf.predict(X), Y)


def test_gradient_boosting_separating_hypothesis_has_no_minimizer():
    # A depth-3 tree fits the round-1 residuals (multiples of y) with
    # y_i h_i > 0 on all 12 rows, so the logistic loss decreases for every
    # step size and argmin_rho does not exist; also pins the Rust message.
    clf = GradientBoostingClassifier(max_iterations=30).fit(X2, Y2)
    assert (clf.n_iter_, clf.stop_reason_, len(clf.coefficients_)) == (1, 'no_minimizer', 0)
    np.testing.assert_allclose(clf.decision_function(X2), np.tanh(clf.intercept_))
    # The squared loss is quadratic in rho, so the same tree is accepted with
    # rho = 1; it fits F = y exactly, so the next residual is 0.
    clf = GradientBoostingClassifier(loss='squared', max_iterations=3).fit(X2, Y2)
    np.testing.assert_allclose(clf.coefficients_, [1.0], atol=1e-12)
    assert clf.stop_reason_ == 'zero_hypothesis'
    np.testing.assert_allclose(clf.decision_function(X2), Y2, atol=1e-12)


@pytest.mark.parametrize('make', [
    lambda: GradientBoostingClassifier(estimator=RegressionTreeRegressor(max_depth=1),
                                       max_iterations=30),
    lambda: GradientBoostingClassifier(estimator=DecisionTreeRegressor(max_depth=2, random_state=0),
                                       shrinkage=.1, max_iterations=30),
    lambda: LogitBoostClassifier(max_iterations=30),
    lambda: LogitBoostClassifier(estimator=DecisionTreeRegressor(max_depth=2, random_state=0),
                                 max_iterations=30),
])
def test_regression_boosters_with_native_and_sklearn_regressors(make):
    clf = make()
    with pytest.warns(UserWarning, match='a.*-1.*b.*1'):
        clf.fit(X2, np.where(Y2 > 0, 'b', 'a'))
    np.testing.assert_array_equal(clf.predict(X2), np.where(Y2 > 0, 'b', 'a'))
    assert np.all(np.abs(clf.decision_function(X2)) <= 1)
    assert clone(clf).get_params().keys() == clf.get_params().keys()


def test_regression_weak_prediction_must_be_finite():
    with pytest.raises(ValueError, match=r'RecordingRegressor, round 1, row 0.*finite'):
        GradientBoostingClassifier(estimator=RecordingRegressor(value=np.nan)).fit(DATA, Y)
    clf = GradientBoostingClassifier(estimator=RecordingRegressor(value=1.0), max_iterations=1)
    clf.fit(DATA, Y)
    clf.estimators_[0].value = np.inf
    with pytest.raises(ValueError, match=r'round 1, row 0'):
        clf.decision_function(DATA)


def test_regression_tree_splits_between_unevenly_spaced_values():
    # Equal-width bins over [-3, 3] would merge 2 and 3; every split between
    # distinct values is a candidate, with the threshold at the midpoint.
    data = np.array([[-3.], [-2.], [-1.], [1.], [2.], [3.]])
    target = [0, 0, 0, 0, 0, 10]
    tree = RegressionTreeRegressor(max_depth=1).fit(data, target)
    np.testing.assert_array_equal(tree.predict(data), target)
    np.testing.assert_array_equal(tree.predict([[2.49], [2.5]]), [0, 10])


def test_regression_tree_matches_sklearn_on_distinct_values():
    # Both are greedy weighted least-squares trees over all splits between
    # distinct values; the data below has no ties in the split gains.
    z = np.sin(np.arange(12.))
    weights = np.arange(1, 13.)
    ours = RegressionTreeRegressor(max_depth=2).fit(X2, z, sample_weight=weights)
    theirs = DecisionTreeRegressor(max_depth=2).fit(X2, z, sample_weight=weights)
    np.testing.assert_allclose(ours.predict(X2), theirs.predict(X2), atol=1e-12)


def test_regression_tree_regressor():
    with pytest.raises(NotFittedError):
        RegressionTreeRegressor().predict(DATA)
    # Depth 0 is the weighted mean: (1 + 3) / 6.
    tree = RegressionTreeRegressor(max_depth=0).fit(DATA, [0, 0, 1, 1], sample_weight=[1, 1, 1, 3])
    np.testing.assert_allclose(tree.predict(DATA), [2 / 3] * 4)
    # A single bin admits no split.
    np.testing.assert_allclose(RegressionTreeRegressor(n_bins=1).fit(DATA, [0, 0, 1, 1]).predict(DATA),
                               [.5] * 4)
    step = RegressionTreeRegressor(max_depth=1).fit(DATA, [-.3, -.3, 2.5, 2.5])
    np.testing.assert_allclose(step.predict([[0.], [3.]]), [-.3, 2.5])


@pytest.mark.parametrize('make', [
    lambda: GradientBoostingClassifier(loss='hinge'), lambda: GradientBoostingClassifier(shrinkage=0),
    lambda: GradientBoostingClassifier(shrinkage=1.5), lambda: LogitBoostClassifier(max_response=.5),
    lambda: LogitBoostClassifier(max_iterations=0),
])
def test_parameter_validation(make):
    with pytest.raises(ValueError):
        make().fit(X, Y)


@pytest.mark.parametrize('params,y', [({'max_depth': -1}, [0, 1, 2, 3]), ({'n_bins': 0}, [0, 1, 2, 3]),
                                      ({}, [0, 1, np.nan, 3])])
def test_regression_tree_validation(params, y):
    with pytest.raises(ValueError):
        RegressionTreeRegressor(**params).fit(DATA, y)
