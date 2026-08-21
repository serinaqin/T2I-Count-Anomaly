import numpy as np
from src.probes import (fit_eval_classifier, fit_eval_magnitude,
                        decodability_map, count_direction, cv_r2)


def test_cv_r2_signal_vs_noise():
    rng = np.random.default_rng(0)
    n = 120
    X = rng.normal(size=(n, 3))
    y = 2 * X[:, 0] + rng.normal(0, 0.1, n)
    assert cv_r2(X, y) > 0.8                       # clear linear signal
    assert cv_r2(X, rng.normal(size=n)) < 0.3      # noise -> ~0, not wildly neg

def test_classifier_separable():
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 0.1, (50, 5))
    X1 = rng.normal(3, 0.1, (50, 5))
    X = np.vstack([X0, X1]); y = np.array([0] * 50 + [1] * 50)
    assert fit_eval_classifier(X, y)["bal_acc"] > 0.95

def test_magnitude_linear_signal():
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 8, 200)
    direction = np.array([1.0, 0, 0, 0, 0])
    X = counts[:, None] * direction + rng.normal(0, 0.05, (200, 5))
    assert fit_eval_magnitude(X, counts)["r2"] > 0.95

def test_decodability_map_ranks_sites():
    rng = np.random.default_rng(0)
    y = rng.integers(1, 6, 120).astype(float)
    good = y[:, None] * np.array([1.0, 0, 0]) + rng.normal(0, 0.05, (120, 3))
    bad = rng.normal(0, 1, (120, 3))
    m = decodability_map({"good": good, "bad": bad}, y, mode="magnitude")
    assert m["good"] > 0.9
    assert m["bad"] < 0.3

def test_count_direction_points_to_high():
    y = np.array([1, 1, 5, 5], float)
    X = np.zeros((4, 3)); X[y > 3, 0] = 5.0     # high-count samples shifted ch0
    d = count_direction(X, y)
    assert d[0] > 4 and abs(d[1]) < 1e-6
