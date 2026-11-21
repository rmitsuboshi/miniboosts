"""TwoLayerNetworkClassifier: the Rust network as a weak learner."""
import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
import miniboosts as mb


def _disc():
    grid = np.arange(-4, 5) * 0.6
    X = np.array([[a, b] for a in grid for b in grid])
    y = np.where((X ** 2).sum(axis=1) < 2.25, 1.0, -1.0)
    return X, y


X, Y = _disc()


def test_fits_a_nonlinear_boundary_reproducibly():
    net = mb.TwoLayerNetworkClassifier(hidden_units=8, seed=3).fit(X, Y)
    assert (net.predict(X) == Y).mean() >= 0.9
    scores = net.decision_function(X)
    assert np.all(np.abs(scores) <= 1)
    assert net.edge_shortfall_ is False
    np.testing.assert_array_equal(clone(net).fit(X, Y).decision_function(X), scores)
    # Other weights give another stream; sparse input gives the same network.
    weights = np.linspace(1, 2, len(Y))
    assert not np.array_equal(
        mb.TwoLayerNetworkClassifier(seed=3).fit(X, Y, sample_weight=weights).decision_function(X),
        scores)
    np.testing.assert_array_equal(
        mb.TwoLayerNetworkClassifier(hidden_units=8, seed=3).fit(sparse.csr_matrix(X), Y)
        .decision_function(X), scores)


@pytest.mark.parametrize('params', [
    dict(optimizer='adam', learning_rate=0.03), dict(activation='relu'),
    dict(loss='squared'), dict(batch_size=16, iterations=600),
])
def test_options(params):
    net = mb.TwoLayerNetworkClassifier(**params).fit(X, Y)
    assert (net.predict(X) == Y).mean() >= 0.9


def test_sign_output_and_boosting():
    net = mb.TwoLayerNetworkClassifier(hidden_units=2, output='sign').fit(X, Y)
    assert set(net.decision_function(X)) <= {-1.0, 1.0}
    model = mb.AdaBoostClassifier(estimator=mb.TwoLayerNetworkClassifier(hidden_units=2),
                                  max_iterations=20).fit(X, Y)
    assert (model.predict(X) == Y).mean() >= 0.95


def test_shortfall_is_reported():
    # Opposite labels on identical rows: every edge is 0.
    net = mb.TwoLayerNetworkClassifier(iterations=3, max_retries=1).fit(
        np.ones((2, 2)), np.array([-1.0, 1.0]))
    assert net.edge_shortfall_ is True


@pytest.mark.parametrize('params', [
    dict(hidden_units=0), dict(iterations=0), dict(batch_size=0), dict(seed=-1), dict(seed=2**64),
    dict(learning_rate=0), dict(regularization=-1), dict(min_edge=2), dict(max_retries=-1),
    dict(activation='gelu'), dict(loss='hinge'), dict(output='clamp'), dict(optimizer='sgd'),
])
def test_invalid_parameters(params):
    net = mb.TwoLayerNetworkClassifier(**params)
    with pytest.raises(ValueError):
        net.fit(X, Y)
    assert not hasattr(net, '_model_')
