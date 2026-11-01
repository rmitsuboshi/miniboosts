import numpy as np
import pytest
from scipy.optimize import brentq
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import GaussianNB
from miniboosts import (
    AdaBoostClassifier, BoundedLinearClassifier, DecisionTreeClassifier,
    FiniteClassClassifier, LogisticRegressionClassifier, NaiveBayesClassifier, RBFClassifier,
)

from test_api import X, Y, ConfidenceLearner
from test_brownboost_sparsiboost import X2, Y2

NATIVE = [DecisionTreeClassifier, BoundedLinearClassifier, LogisticRegressionClassifier,
          lambda: LogisticRegressionClassifier(output='tanh'), NaiveBayesClassifier,
          lambda: RBFClassifier(bandwidth=1.0)]


@pytest.mark.parametrize('make', NATIVE)
def test_native_learner_conventions(make):
    learner = make()
    with pytest.raises(NotFittedError):
        learner.predict(X)
    with pytest.warns(UserWarning, match='cat.*-1.*dog.*1'):
        learner.fit(X, ['cat', 'cat', 'dog', 'dog'])
    # A single weak hypothesis need not separate X (an RBF bump has no bias),
    # so only the label mapping is checked; accuracy is tested under boosting.
    np.testing.assert_array_equal(learner.classes_, ['cat', 'dog'])
    expected = np.where(learner.decision_function(X) >= 0, 'dog', 'cat')
    np.testing.assert_array_equal(learner.predict(X), expected)
    scores = learner.decision_function(X2[:, :1])
    assert scores.shape == (12,) and np.all(np.abs(scores) <= 1)
    assert clone(learner).get_params() == learner.get_params()
    with pytest.raises(ValueError, match='sample_weight'):
        make().fit(X, Y, sample_weight=[1, -1, 1, 1])
    with pytest.raises(ValueError):
        learner.decision_function(np.zeros((2, 2)))


@pytest.mark.parametrize('make', NATIVE)
def test_native_learners_inside_boosting(make):
    clf = AdaBoostClassifier(estimator=make(), max_iterations=30).fit(X2, Y2)
    np.testing.assert_array_equal(clf.predict(X2), Y2)


def test_bounded_linear_maximizes_the_edge():
    # g = 0.5 (3, 1) - 0.5 (-1, 1) = (2, 0) and S = sqrt(3^2 + 1): the best
    # member (w, b) = g/|g| has edge |g| / S = 2 / sqrt(10).
    h = BoundedLinearClassifier().fit([[3.], [-1.]], [1, -1])
    values = h.decision_function([[3.], [-1.]])
    assert 0.5 * (values[0] - values[1]) == pytest.approx(2 / np.sqrt(10), abs=1e-15)
    # Outside the training radius the confidence is clamped.
    assert h.decision_function([[100.]])[0] == 1.0


def test_logistic_regression_matches_the_regularized_minimizer():
    # With X = {-1, 1}, y = x, d = 1/2: by symmetry b = 0, and the gradient of
    # ln(1 + e^{-w}) + (lam/2) w^2 vanishes where w = sigma(-w) / lam.
    lam = 10.0
    w = brentq(lambda w: w - 1 / (lam * (1 + np.exp(w))), 0, 1)
    h = LogisticRegressionClassifier(regularization=lam, output='tanh').fit([[-1.], [1.]], [-1, 1])
    np.testing.assert_allclose(h.decision_function([[1.], [-1.]]),
                               [np.tanh(w / 2), -np.tanh(w / 2)], atol=1e-12)
    halfspace = LogisticRegressionClassifier(regularization=lam).fit([[-1.], [1.]], [-1, 1])
    np.testing.assert_array_equal(halfspace.decision_function([[.5], [-.5]]), [1., -1.])


def test_naive_bayes_matches_sklearn_with_uniform_weights():
    # With uniform weights the weighted variances, priors, and the smoothing
    # eps * max_k Var(x_k) equal GaussianNB's; h = P(+1|x) - P(-1|x).
    h = NaiveBayesClassifier().fit(X2, Y2)
    proba = GaussianNB().fit(X2, Y2).predict_proba(X2)
    np.testing.assert_allclose(h.decision_function(X2), proba[:, 1] - proba[:, 0], atol=1e-12)


def test_rbf_selects_the_best_signed_bump():
    data, labels = [[0.], [0.1], [5.]], [1, 1, -1]
    # Bumps at 0 and 0.1 tie on edge; the tie goes to the smaller index.
    h = RBFClassifier(bandwidth=1.0).fit(data, labels)
    assert (h.center_index_, h.sign_) == (0, 1.0)
    np.testing.assert_array_equal(h.center_, [0.])
    np.testing.assert_allclose(h.decision_function([[.5]]), [np.exp(-.5**2 / 2)], atol=1e-15)
    # Only the bump at 5 is a candidate: it enters negated.
    h = RBFClassifier(bandwidth=1.0, centers=[2]).fit(data, labels)
    assert (h.center_index_, h.sign_) == (2, -1.0)
    np.testing.assert_array_equal(h.decision_function([[5.]]), [-1.])


class Negated(ConfidenceLearner):
    def decision_function(self, X):
        return -super().decision_function(X)


def test_finite_class_selection_negation_and_clone():
    flat, good, bad = ConfidenceLearner(value=0.3), ConfidenceLearner(), Negated()
    # Uniform weights on X, Y: edges 0 (balanced labels), 0.8, and -0.8.
    h = FiniteClassClassifier([flat, good, bad]).fit(X, Y)
    assert (h.index_, h.sign_) == (1, 1.0) and h.hypothesis_ is good
    h = FiniteClassClassifier([flat, bad], with_negations=True).fit(X, Y)
    assert (h.index_, h.sign_) == (1, -1.0)
    np.testing.assert_allclose(h.decision_function(X), [-.8, -.8, .8, .8])
    copy = clone(h)
    assert copy.hypotheses[1] is bad and not hasattr(copy, 'index_')
    clf = AdaBoostClassifier(estimator=FiniteClassClassifier([flat, bad], with_negations=True),
                             max_iterations=5).fit(X, Y)
    np.testing.assert_array_equal(clf.predict(X), Y)


def test_finite_class_rejects_invalid_members():
    with pytest.raises(ValueError, match='nonempty'):
        FiniteClassClassifier([]).fit(X, Y)
    with pytest.raises(ValueError, match=r'ConfidenceLearner, hypothesis 0, row 0'):
        FiniteClassClassifier([ConfidenceLearner(value=2.0)]).fit(X, Y)


@pytest.mark.parametrize('learner', [
    LogisticRegressionClassifier(regularization=0), LogisticRegressionClassifier(output='clamp'),
    NaiveBayesClassifier(var_smoothing=-1), RBFClassifier(bandwidth=0),
    RBFClassifier(bandwidth=1, centers=[4]), RBFClassifier(bandwidth=1, centers=[]),
    RBFClassifier(bandwidth=1, centers=[0.5]), DecisionTreeClassifier(max_depth=0),
])
def test_native_parameter_validation(learner):
    with pytest.raises(ValueError):
        learner.fit(X, Y)
