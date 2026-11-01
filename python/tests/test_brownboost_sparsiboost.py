import numpy as np
import pytest
from scipy.special import erfc
from sklearn.base import clone
from miniboosts import BrownBoostClassifier, DecisionTreeClassifier, SparsiBoostClassifier

from test_api import X, Y, ConfidenceLearner

# Labels follow the sign of x + y, so no single stump separates them (the
# data of the Rust BrownBoost/SparsiBoost examples).
X2 = np.array([[-2, 1], [-1, -2], [1, -3], [-3, 2], [0, -1], [-2, -1],
               [2, 1], [1, 2], [-1, 3], [3, -1], [0, 2], [2, 2]], dtype=float)
Y2 = np.array([-1] * 6 + [1] * 6)


@pytest.mark.parametrize('tolerance', [0.2, 0.1, 0.05])
def test_brownboost_total_time_and_theorem_2(tolerance):
    clf = BrownBoostClassifier(estimator=DecisionTreeClassifier(), tolerance=tolerance,
                               max_iterations=200).fit(X2, Y2)
    # Freund (2001): c = (erfc^-1(eps))^2, i.e. 1 - erf(sqrt(c)) = eps.
    assert erfc(np.sqrt(clf.total_time_)) == pytest.approx(tolerance, rel=1e-12)
    # The loop exited (s <= 0), so Theorem 2 bounds the training error by eps.
    assert clf.stop_reason_ == 'algorithm_stop' and clf.remaining_time_ <= 0
    assert np.mean(clf.predict(X2) != Y2) <= tolerance


def test_brownboost_constant_correlation_uses_all_time_in_one_round():
    # b_j = y_j h(x_j) = 0.8 for all j, so gamma(alpha, t) = 0.8 > nu along
    # Eq. (5) and the clock expires (t = s) in the first round.
    clf = BrownBoostClassifier(estimator=ConfidenceLearner(), max_iterations=10).fit(X, Y)
    assert (clf.n_iter_, clf.stop_reason_, clf.remaining_time_) == (1, 'algorithm_stop', 0.0)
    np.testing.assert_array_equal(clf.predict(X), Y)


def test_brownboost_edge_threshold_stop_is_reported_not_raised():
    # gamma = 0 <= nu: the hypothesis is discarded, no time is used, and the
    # ensemble is empty. This also pins the Rust stop message the binding matches.
    clf = BrownBoostClassifier(estimator=ConfidenceLearner(value=0.0), max_iterations=10).fit(X, Y)
    assert clf.stop_reason_ == 'edge_threshold'
    assert clf.n_iter_ == 1 and len(clf.estimators_) == 1
    assert clf.remaining_time_ == clf.total_time_
    np.testing.assert_array_equal(clf.decision_function(X), 0.0)


@pytest.mark.parametrize('params', [{'tolerance': 0}, {'tolerance': 1}, {'tolerance': np.nan},
                                    {'edge_threshold': -0.1}, {'edge_threshold': 1},
                                    {'max_iterations': 0}])
def test_brownboost_parameter_validation(params):
    with pytest.raises(ValueError):
        BrownBoostClassifier(**params).fit(X, Y)


def test_sparsiboost_target_size():
    clf = SparsiBoostClassifier(estimator=DecisionTreeClassifier(), n_hypotheses=3, seed=42).fit(X2, Y2)
    # Algorithm 1 with m = 12, T = 3: c = ceil(lg 12 / lg(2 + 12/3)) = 2, so
    # AdaBoostV runs cT = 6 rounds with nu = min(sqrt(2 ln 12 / 6), 1/2) = 1/2.
    assert (clf.adaboostv_rounds_, clf.adaboostv_tolerance_) == (6, 0.5)
    assert (clf.n_iter_, clf.stop_reason_) == (6, 'algorithm_stop')
    assert 1 <= clf.n_selected_ <= 3
    assert clf.sparsification_within_tolerance_ is None
    assert np.isfinite(clf.sparsification_error_) and clf.sparsification_error_ >= 0


def test_sparsiboost_target_gap_keeps_margins_within_half_gap():
    clf = SparsiBoostClassifier(estimator=DecisionTreeClassifier(), tolerance=0.5).fit(X2, Y2)
    # Gap v = 0.5: nu = v/2 for ceil(8 ln 12 / v^2) = 80 rounds.
    assert (clf.adaboostv_rounds_, clf.adaboostv_tolerance_) == (80, 0.25)
    assert clf.n_iter_ == 80
    assert clf.sparsification_error_ <= 0.25
    assert clf.sparsification_within_tolerance_ is True
    assert 1 <= clf.n_selected_ <= 80


def test_sparsiboost_size_cap_takes_precedence_over_gap():
    clf = SparsiBoostClassifier(estimator=DecisionTreeClassifier(), tolerance=0.5,
                                n_hypotheses=2).fit(X2, Y2)
    assert clf.adaboostv_rounds_ == 80 and 1 <= clf.n_selected_ <= 2
    assert clf.sparsification_within_tolerance_ == (clf.sparsification_error_ <= 0.25)


def test_sparsiboost_seed_is_reproducible_and_clonable():
    clf = SparsiBoostClassifier(estimator=DecisionTreeClassifier(), n_hypotheses=3, seed=7)
    first = clf.fit(X2, Y2).decision_function(X2)
    second = clone(clf).fit(X2, Y2).decision_function(X2)
    np.testing.assert_array_equal(first, second)


def test_sparsiboost_explicit_cap_truncates_adaboostv():
    clf = SparsiBoostClassifier(estimator=DecisionTreeClassifier(), tolerance=0.5,
                                max_iterations=3).fit(X2, Y2)
    assert (clf.n_iter_, clf.stop_reason_) == (3, 'max_iterations')


@pytest.mark.parametrize('params', [{}, {'n_hypotheses': 0}, {'n_hypotheses': 2.5},
                                    {'tolerance': 0}, {'n_hypotheses': 2, 'seed': -1},
                                    {'n_hypotheses': 2, 'seed': 2**64},
                                    {'n_hypotheses': 2, 'max_iterations': 0}])
def test_sparsiboost_parameter_validation(params):
    with pytest.raises(ValueError):
        SparsiBoostClassifier(**params).fit(X, Y)
