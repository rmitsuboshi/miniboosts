import numpy as np
import pytest
from miniboosts import (
    ArcGvClassifier, BoostByMajorityClassifier, DecisionTreeClassifier, RobustBoostClassifier,
)

from test_api import X, Y, ConfidenceLearner, WeightedChoiceLearner
from test_brownboost_sparsiboost import X2, Y2

DATA = np.arange(4).reshape(-1, 1)


def test_arcgv_follows_definition_5_1():
    # Breiman (1999), Def. 5.1, with l_h(z) = (1 - y h(x)) / 2. Round 1 uses
    # t_1 = 1/2: the first candidate has error 1/4, so Delta_1 =
    # min(1, 2 atanh(1/2)) = 1. Then er(z, b) = 1 on example 1 only, so
    # Q_2 is proportional to (1, e, 1, 1), and t_2 = top(c) = 1 gives Delta_2 = 1.
    clf = ArcGvClassifier(estimator=WeightedChoiceLearner(), max_iterations=2).fit(DATA, Y)
    np.testing.assert_allclose(clf.estimators_[0].weights_, [.25] * 4, atol=1e-15)
    np.testing.assert_allclose(clf.estimators_[1].weights_, np.array([1, np.e, 1, 1]) / (3 + np.e),
                               atol=1e-14)
    # c = b / |b| = (1/2, 1/2) on the two candidates.
    np.testing.assert_allclose(clf.decision_function(DATA), [-1, 0, 0, 1], atol=1e-14)
    assert clf.stop_reason_ == 'max_iterations'


def test_arcgv_zero_step_is_an_algorithm_stop():
    # A separating stump has margin 1 everywhere; the next (identical) stump
    # cannot beat it, so Delta_2 = 0 and the second candidate is discarded.
    clf = ArcGvClassifier(max_iterations=10).fit(X, Y)
    assert (clf.n_iter_, clf.stop_reason_) == (2, 'algorithm_stop')
    np.testing.assert_allclose(clf.decision_function(X), clf.estimators_[0].decision_function(X))


def test_bbm_rounds_majority_vote_and_early_decision():
    # k = ceil(ln(1/eps) / (2 gamma^2)) = ceil(ln 10 / 0.08) = 29. A
    # separating stump is always correct, so after 15 rounds every example
    # has a majority of correct votes and all weights are 0.
    clf = BoostByMajorityClassifier(gamma=0.2, tolerance=0.1).fit(X, Y)
    assert (clf.n_rounds_, clf.n_iter_, clf.decided_early_) == (29, 15, True)
    assert clf.stop_reason_ == 'algorithm_stop'
    np.testing.assert_array_equal(clf.predict(X), Y)


def test_bbm_votes_are_unweighted_signs():
    clf = BoostByMajorityClassifier(estimator=WeightedChoiceLearner(), gamma=0.2, tolerance=0.1,
                                    max_iterations=3).fit(DATA, Y)
    assert (clf.stop_reason_, clf.decided_early_) == ('max_iterations', False)
    votes = np.mean([e.values_[DATA[:, 0]] for e in clf.estimators_], axis=0)
    np.testing.assert_allclose(clf.decision_function(DATA), votes, atol=1e-15)
    # Round 1 weights are uniform: every example has r = 0 correct votes.
    np.testing.assert_allclose(clf.estimators_[0].weights_, [.25] * 4, atol=1e-15)


def test_robustboost_finished_run_bounds_training_error():
    clf = RobustBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), tolerance=0.2,
                                theta=0.1, max_iterations=200).fit(X2, Y2)
    # Reaching t = 1 keeps the average potential at eps; examples with
    # y H(x) <= theta (in particular misclassified ones) have potential >= 1/2.
    assert clf.remaining_time_ == 0.0 and clf.stop_reason_ == 'algorithm_stop'
    assert np.mean(clf.predict(X2) != Y2) <= 2 * 0.2


def test_robustboost_nonpositive_edge_is_reported_not_raised():
    # Also pins the Rust stop message the binding matches.
    clf = RobustBoostClassifier(estimator=ConfidenceLearner(value=0.0), max_iterations=5).fit(X, Y)
    assert (clf.n_iter_, clf.stop_reason_, clf.remaining_time_) == (1, 'nonpositive_edge', 1.0)
    np.testing.assert_array_equal(clf.decision_function(X), 0.0)


@pytest.mark.parametrize('make', [
    lambda: BoostByMajorityClassifier(gamma=0.5), lambda: BoostByMajorityClassifier(gamma=0),
    lambda: BoostByMajorityClassifier(gamma=0.2, tolerance=1),
    lambda: BoostByMajorityClassifier(gamma=0.2, max_iterations=0),
    lambda: RobustBoostClassifier(tolerance=0), lambda: RobustBoostClassifier(theta=-1),
    lambda: RobustBoostClassifier(sigma_f=0), lambda: ArcGvClassifier(max_iterations=None),
])
def test_parameter_validation(make):
    with pytest.raises(ValueError):
        make().fit(X, Y)
