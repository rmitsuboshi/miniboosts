import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as SkTree
from miniboosts import (
    AdaBoostClassifier, DecisionTreeClassifier, PredictionAdapter,
    AdaBoostVClassifier, MadaBoostClassifier, LPBoostClassifier,
    ERLPBoostClassifier, CorrectiveERLPBoostClassifier, MLPBoostClassifier,
    SmoothBoostClassifier, SoftBoostClassifier, TotalBoostClassifier,
    GraphSeparationBoostingClassifier,
)

X = np.array([[-2.], [-1.], [1.], [2.]])
Y = np.array([-1, -1, 1, 1])


class ConfidenceLearner(ClassifierMixin, BaseEstimator):
    def __init__(self, value=None):
        self.value = value

    def fit(self, X, y, sample_weight=None):
        self.weights_ = np.array(sample_weight, copy=True)
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, X):
        if self.value is not None:
            return np.full(len(X), self.value)
        return np.where(X[:, 0] < 0, -0.8, 0.8)


def test_native_tree_and_arbitrary_labels():
    clf = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), max_iterations=5)
    with pytest.raises(NotFittedError):
        clf.predict(X)
    with pytest.warns(UserWarning, match="cat.*-1.*dog.*1") as messages:
        assert clf.fit(X, np.array(['cat', 'cat', 'dog', 'dog'])) is clf
    assert len(messages) == 1
    np.testing.assert_array_equal(clf.classes_, ['cat', 'dog'])
    np.testing.assert_array_equal(clf.predict(X), ['cat', 'cat', 'dog', 'dog'])
    assert clf.decision_function(X).shape == (4,)
    assert clone(clf).get_params()['estimator__max_depth'] == 1


@pytest.mark.parametrize('value', [1.0000001, -1.0000001, np.nan, np.inf, -np.inf])
def test_invalid_confidence_is_an_error(value):
    clf = AdaBoostClassifier(estimator=ConfidenceLearner(value=value), max_iterations=3)
    with pytest.raises(ValueError, match=r'ConfidenceLearner.*round 1.*row 0'):
        clf.fit(X, Y)
    with pytest.raises(NotFittedError):
        clf.predict(X)


def test_callback_clone_weights_and_no_mapping_warning():
    import warnings
    template = ConfidenceLearner()
    clf = AdaBoostClassifier(estimator=template, max_iterations=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        clf.fit(X, Y)
    assert not caught
    assert not hasattr(template, 'weights_')
    np.testing.assert_allclose(clf.estimators_[0].weights_, np.full(4, .25))
    np.testing.assert_array_equal(clf.predict(X), Y)
    previous = clf.estimators_[0]
    clf.fit(X, Y)
    assert clf.estimators_[0] is not previous


@pytest.mark.parametrize('cls', [AdaBoostClassifier, AdaBoostVClassifier,
    MadaBoostClassifier, LPBoostClassifier, ERLPBoostClassifier,
    CorrectiveERLPBoostClassifier, MLPBoostClassifier, SmoothBoostClassifier,
    SoftBoostClassifier, TotalBoostClassifier, GraphSeparationBoostingClassifier])
def test_all_algorithms(cls):
    clf = cls(estimator=ConfidenceLearner(), max_iterations=10)
    clf.fit(X, Y)
    np.testing.assert_array_equal(clf.predict(X), Y)
    assert np.isfinite(clf.decision_function(X)).all()


def test_pipeline_and_explicit_sklearn_adapter():
    clf = make_pipeline(StandardScaler(), AdaBoostClassifier(
        estimator=PredictionAdapter(SkTree(max_depth=1, random_state=0)), max_iterations=5))
    clf.fit(X, Y)
    np.testing.assert_array_equal(clf.predict(X), Y)


@pytest.mark.parametrize('y', [[1, 1, 1, 1], [0, 1, 2, 0], [0, 0, 1, np.nan]])
def test_bad_labels(y):
    with pytest.raises(ValueError):
        AdaBoostClassifier().fit(X, y)


def test_failed_refit_discards_previous_model():
    clf = AdaBoostClassifier().fit(X, Y)
    with pytest.raises(ValueError):
        clf.fit(X, [1, 1, 1, 1])
    with pytest.raises(NotFittedError):
        clf.predict(X)


def test_inference_checks_confidence_and_feature_count():
    clf = AdaBoostClassifier(estimator=ConfidenceLearner()).fit(X, Y)
    clf.estimators_[0].value = 2.0
    with pytest.raises(ValueError, match=r'round 1.*row 0'):
        clf.decision_function(X)
    with pytest.raises(ValueError):
        clf.predict(np.zeros((4, 2)))


@pytest.mark.parametrize('cls', [DecisionTreeClassifier, lambda: PredictionAdapter(SkTree())])
def test_other_estimators_unfitted(cls):
    with pytest.raises(NotFittedError):
        cls().predict(X)


class BadShapeLearner(ConfidenceLearner):
    def decision_function(self, X):
        return np.zeros((len(X), 1))


class BrokenLearner(ConfidenceLearner):
    def fit(self, X, y, sample_weight=None):
        raise LookupError('original callback error')


class BrokenConfidenceLearner(ConfidenceLearner):
    def decision_function(self, X):
        raise AttributeError('internal learner bug')


@pytest.mark.parametrize('estimator,error,message', [
    (BadShapeLearner(), ValueError, 'shape'),
    (ConfidenceLearner(value=1j), ValueError, 'real'),
    (BrokenLearner(), LookupError, 'original callback error'),
    (BrokenConfidenceLearner(), AttributeError, 'internal learner bug'),
])
def test_callback_errors_preserve_type(estimator, error, message):
    with pytest.raises(error, match=message):
        AdaBoostClassifier(estimator=estimator).fit(X, Y)


class WeightedChoiceLearner(ClassifierMixin, BaseEstimator):
    """Select between two explicit hypotheses by weighted classification error."""
    def fit(self, X, y, sample_weight=None):
        self.weights_ = sample_weight.copy()
        candidates = np.asarray([[-1, 1, 1, 1], [-1, -1, -1, 1]])
        self.values_ = candidates[np.argmin((candidates != y) @ sample_weight)]
        return self

    def decision_function(self, X):
        return self.values_[X[:, 0].astype(int)]


def test_adaboost_reweights_and_combines_independent_hypotheses():
    # Freund/Schapire AdaBoost: alpha = 1/2 ln((1-e)/e),
    # D_next proportional to D * exp(-alpha*y*h). e1=1/4, e2=1/6.
    data = np.arange(4).reshape(-1, 1)
    clf = AdaBoostClassifier(estimator=WeightedChoiceLearner(), max_iterations=2).fit(data, Y)
    assert clf.n_iter_ == 2
    assert clf.stop_reason_ == 'max_iterations'
    assert len(clf.estimators_) == 2
    assert clf.estimators_[0] is not clf.estimators_[1]
    np.testing.assert_allclose(clf.estimators_[0].weights_, [.25]*4, atol=1e-14)
    np.testing.assert_allclose(clf.estimators_[1].weights_, [1/6, .5, 1/6, 1/6], atol=1e-14)
    a, b = .5*np.log(3), .5*np.log(5)
    expected = (a*np.array([-1, 1, 1, 1]) + b*np.array([-1, -1, -1, 1]))/(a+b)
    np.testing.assert_allclose(clf.decision_function(data), expected, atol=1e-14)
    # Replace cached scores with a different-size batch, then the original batch.
    np.testing.assert_allclose(clf.decision_function(data[[3, 1]]), expected[[3, 1]], atol=1e-14)
    np.testing.assert_allclose(clf.decision_function(data), expected, atol=1e-14)


class ZeroAtOriginLearner(ConfidenceLearner):
    def decision_function(self, X):
        return .8 * np.sign(X[:, 0])


def test_zero_confidence_tie_maps_to_positive_original_label():
    clf = AdaBoostClassifier(estimator=ZeroAtOriginLearner(), max_iterations=1)
    with pytest.warns(UserWarning, match='0.*-1.*1.*1'):
        clf.fit(X, [0, 0, 1, 1])
    np.testing.assert_array_equal(clf.decision_function([[0.]]), [0.])
    np.testing.assert_array_equal(clf.predict([[0.]]), [1])


@pytest.mark.parametrize('X_bad', [np.full((4, 1), np.nan), np.ones((4, 1), dtype=complex)*1j])
def test_bad_features(X_bad):
    with pytest.raises(ValueError):
        AdaBoostClassifier().fit(X_bad, Y)


@pytest.mark.parametrize('weights', [[0]*4, [-1, 1, 1, 1], [np.inf]*4, [1, 1]])
def test_tree_bad_weights(weights):
    with pytest.raises(ValueError, match='sample_weight'):
        DecisionTreeClassifier().fit(X, Y, sample_weight=weights)


def test_grid_search_clones_nested_estimator():
    from sklearn.model_selection import GridSearchCV
    X_more, y_more = np.tile(X, (3, 1)), np.tile(Y, 3)
    search = GridSearchCV(AdaBoostClassifier(estimator=DecisionTreeClassifier()),
                          {'estimator__max_depth': [1, 2], 'max_iterations': [1, 2]}, cv=2)
    search.fit(X_more, y_more)
    np.testing.assert_array_equal(search.predict(X), Y)


def test_boolean_labels_warn_and_round_trip():
    labels = np.array([False, False, True, True])
    with pytest.warns(UserWarning, match='False.*-1.*True.*1') as caught:
        clf = AdaBoostClassifier().fit(X, labels)
    assert len(caught) == 1
    assert clf.predict(X).dtype == np.bool_
    np.testing.assert_array_equal(clf.predict(X), labels)


def test_iteration_count_excludes_terminal_noop():
    clf = AdaBoostClassifier(estimator=ConfidenceLearner(), tolerance=1, max_iterations=100).fit(X, Y)
    assert clf.n_iter_ == 2  # ceil(ln(4) / tolerance**2)
    assert clf.estimator_rounds_ == [1, 2]
    assert clf.stop_reason_ == 'algorithm_stop'


def test_rust_failure_is_an_error_and_invalidates_model():
    clf = AdaBoostClassifier(estimator=ConfidenceLearner(value=0), max_iterations=1)
    # Zero-edge learners cannot yield a normalized, positive-weight Rust ensemble.
    with pytest.raises(RuntimeError, match='MiniBoosts failed'):
        clf.fit(X, Y)
    with pytest.raises(NotFittedError):
        clf.predict(X)


@pytest.mark.parametrize('labels', [[float('nan'), 'cat', 'cat', 'cat'], [1, 'cat', 1, 'cat']])
def test_label_validation_precedes_numpy_string_coercion(labels):
    with pytest.raises(ValueError):
        AdaBoostClassifier().fit(X, labels)


def test_object_numeric_signed_labels_need_no_warning():
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        AdaBoostClassifier().fit(X, np.array(Y, dtype=object))
    assert not caught
