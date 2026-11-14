"""scipy.sparse inputs: the Rust learners read them as sparse features."""
import numpy as np
import pytest
from scipy import sparse
from miniboosts import (
    AdaBoostClassifier, BoundedLinearClassifier, DecisionTreeClassifier,
    GradientBoostingClassifier, LogisticRegressionClassifier, MajorityOfFiveClassifier,
    NaiveBayesClassifier, RBFClassifier, RegressionTreeRegressor, _native,
)

# The sums over a sparse feature run over its stored entries only, so the
# results can differ from the dense ones in the last bits.
TOLERANCE = 1e-12

NATIVE = [lambda: DecisionTreeClassifier(max_depth=3), BoundedLinearClassifier,
          LogisticRegressionClassifier, lambda: LogisticRegressionClassifier(output='tanh'),
          NaiveBayesClassifier, lambda: RBFClassifier(bandwidth=2.0)]
FORMATS = [sparse.csc_matrix, sparse.csr_matrix, sparse.coo_matrix, sparse.csc_array]


def _data(n=60, d=8, density=0.25, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)) * (rng.random((n, d)) < density)
    X[:, 3] = 0.0  # an all-zero feature must be kept, not dropped
    y = np.where(X[:, 0] - X[:, 1] + 0.5 * X[:, 2] > 0, 1.0, -1.0)
    return X, y


X, Y = _data()
X_TEST, _ = _data(n=25, seed=1)


@pytest.mark.parametrize('make', NATIVE)
@pytest.mark.parametrize('fmt', FORMATS)
def test_native_learners_agree_with_dense_input(make, fmt):
    weights = np.linspace(1.0, 2.0, len(Y))
    dense = make().fit(X, Y, sample_weight=weights)
    learner = make().fit(fmt(X), Y, sample_weight=weights)
    assert learner.n_features_in_ == X.shape[1]
    expected = dense.decision_function(X_TEST)
    # A model fitted on sparse data predicts on sparse and on dense data.
    np.testing.assert_allclose(learner.decision_function(fmt(X_TEST)), expected,
                               rtol=0, atol=TOLERANCE)
    np.testing.assert_allclose(learner.decision_function(X_TEST), expected,
                               rtol=0, atol=TOLERANCE)
    np.testing.assert_array_equal(learner.predict(fmt(X_TEST)), dense.predict(X_TEST))


def test_rbf_center_is_a_dense_row():
    learner = RBFClassifier(bandwidth=2.0).fit(sparse.csr_matrix(X), Y)
    np.testing.assert_array_equal(learner.center_, X[learner.center_index_])


@pytest.mark.parametrize('fmt', FORMATS)
def test_regression_tree_agrees_with_dense_input(fmt):
    target = X[:, 0] - 2.0 * X[:, 2]
    dense = RegressionTreeRegressor(max_depth=3).fit(X, target)
    tree = RegressionTreeRegressor(max_depth=3).fit(fmt(X), target)
    np.testing.assert_allclose(tree.predict(fmt(X_TEST)), dense.predict(X_TEST),
                               rtol=0, atol=TOLERANCE)


@pytest.mark.parametrize('make', [
    lambda: AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2), max_iterations=10),
    lambda: AdaBoostClassifier(estimator=BoundedLinearClassifier(), max_iterations=10),
    lambda: GradientBoostingClassifier(max_iterations=10),
    # Not trees here: on a 20-row part several splits have exactly the same
    # score, and the last bits then decide which of them is taken.
    lambda: MajorityOfFiveClassifier(
        estimator=AdaBoostClassifier(estimator=BoundedLinearClassifier(), max_iterations=5),
        n_parts=3),
])
def test_boosters_pass_sparse_input_to_the_weak_learner(make):
    dense = make().fit(X, Y)
    clf = make().fit(sparse.csr_matrix(X), Y)
    # Weak hypotheses that differ in the last bits can give ensemble weights
    # that differ slightly more; the tolerance is still far below any margin.
    np.testing.assert_allclose(clf.decision_function(sparse.csr_matrix(X_TEST)),
                               dense.decision_function(X_TEST), rtol=0, atol=1e-9)
    np.testing.assert_array_equal(clf.predict(sparse.csc_matrix(X_TEST)), dense.predict(X_TEST))


def test_unsorted_duplicate_and_explicit_zero_entries():
    # Column 0 lists row 3 before row 1 and row 1 twice (2 + 3 = 5); column 1
    # stores an explicit zero.
    matrix = sparse.csc_matrix(
        (np.array([1.0, 2.0, 3.0, 0.0, -4.0]), np.array([3, 1, 1, 0, 2]), np.array([0, 3, 5])),
        shape=(4, 2))
    assert not matrix.has_canonical_format
    indices = matrix.indices.copy()
    dense = matrix.toarray()
    np.testing.assert_array_equal(dense, [[0, 0], [5, 0], [0, -4], [1, 0]])
    y = np.array([-1.0, 1.0, -1.0, 1.0])
    learner = BoundedLinearClassifier().fit(matrix, y)
    expected = BoundedLinearClassifier().fit(dense, y)
    np.testing.assert_allclose(learner.decision_function(matrix),
                               expected.decision_function(dense), rtol=0, atol=TOLERANCE)
    # The caller's matrix is not reordered in place.
    np.testing.assert_array_equal(matrix.indices, indices)


def test_invalid_sparse_input():
    learner = DecisionTreeClassifier().fit(sparse.csr_matrix(X), Y)
    with pytest.raises(ValueError):
        learner.decision_function(sparse.csr_matrix(X_TEST[:, :5]))
    bad = sparse.csr_matrix(X)
    bad.data[0] = np.nan
    with pytest.raises(ValueError):
        DecisionTreeClassifier().fit(bad, Y)


def test_native_boundary_validates_csc_arrays():
    y, w = [-1.0, 1.0, 1.0], [0.25, 0.25, 0.5]
    fit = _native.fit_bounded_linear
    fit((3, [0, 2], [0, 2], [1.0, -1.0]), y, w)
    for matrix, message in [
        ((3, [0, 2], [2, 0], [1.0, -1.0]), 'unsorted or duplicate'),
        ((3, [0, 2], [1, 1], [1.0, -1.0]), 'unsorted or duplicate'),
        ((3, [0, 2], [0, 3], [1.0, -1.0]), 'out of range'),
        ((3, [0, 1], [0, 2], [1.0, -1.0]), 'indptr'),
        ((3, [0, 2], [0, 2], [1.0, float('inf')]), 'non-finite'),
        ((2, [0, 2], [0, 1], [1.0, -1.0]), 'target'),
    ]:
        with pytest.raises(ValueError, match=message):
            fit(matrix, y, w)


def test_sparse_input_tag():
    for estimator in [DecisionTreeClassifier(), RegressionTreeRegressor()]:
        assert estimator.__sklearn_tags__().input_tags.sparse


def test_native_boundary_validates_array_bytes():
    y, w = [-1.0, 1.0, 1.0], [0.25, 0.25, 0.5]
    fit = _native.fit_bounded_linear
    f8 = lambda values: np.asarray(values, dtype=np.float64).tobytes()
    i8 = lambda values: np.asarray(values, dtype=np.int64).tobytes()
    dense = fit((3, 2, f8([1, 0, 0, 0, -1, 2])), y, w)
    from_csc = fit((3, i8([0, 2, 3]), i8([0, 2, 2]), f8([1, -1, 2])), y, w)
    rows = [[1.0, 0.0], [0.0, 0.0], [-1.0, 2.0]]
    expected = fit(rows, y, w).decision_function(rows)
    for model in [dense, from_csc]:
        np.testing.assert_allclose(model.decision_function((3, 2, f8(rows))), expected,
                                   rtol=0, atol=TOLERANCE)
    for matrix, message in [
        ((3, 2, f8([1, 0, 0, 0, -1])), 'shape'),
        ((3, 2, f8([1, 0, 0, 0, -1, 2])[:-1]), '8-byte'),
        ((3, 2, f8([1, 0, 0, 0, -1, np.nan])), 'non-finite'),
        ((3, 0, b''), 'at least one row and one feature'),
        ((3, i8([0, 2]), i8([-1, 2]), f8([1, -1])), 'nonnegative'),
        ((3, i8([0, 2]), i8([2, 0]), f8([1, -1])), 'unsorted or duplicate'),
        ((3, i8([0, 2]), i8([0, 2]), f8([1, -1])[:-3]), '8-byte'),
    ]:
        with pytest.raises(ValueError, match=message):
            fit(matrix, y, w)
