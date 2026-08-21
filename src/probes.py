import numpy as np
from sklearn.linear_model import (LogisticRegression, LinearRegression, Ridge,
                                  RidgeCV)
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def cv_r2(X, y, k=5, n_pca=20) -> float:
    """Cross-validated R^2 of a standardized (PCA -> RidgeCV) regression. Robust
    for p >> n probes: PCA caps dimensionality and RidgeCV self-selects the
    penalty per fold, so an uninformative probe returns ~0, not a spurious large
    negative from overfitting."""
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y, float)
    k = min(k, len(y))
    steps = [StandardScaler()]
    npca = min(n_pca, X.shape[1], max(1, len(y) - len(y) // k - 1))
    if X.shape[1] > npca:
        steps.append(PCA(n_components=npca, random_state=0))
    steps.append(RidgeCV(alphas=np.logspace(-1, 4, 10)))
    model = make_pipeline(*steps)
    return float(cross_val_score(model, X, y, cv=k, scoring="r2").mean())


def fit_eval_classifier(X, y, seed: int = 0) -> dict:
    X, y = np.asarray(X), np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return {"bal_acc": float(balanced_accuracy_score(yte, clf.predict(Xte)))}


def fit_eval_magnitude(X, y, seed: int = 0) -> dict:
    X, y = np.asarray(X), np.asarray(y, float)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed)
    reg = LinearRegression().fit(Xtr, ytr)
    return {"r2": float(r2_score(yte, reg.predict(Xte)))}


def count_direction(X, y):
    """Difference-of-means 'more objects' direction: mean(high-count samples)
    - mean(low-count samples), split at the median count. A donor-free steering
    vector for the count."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    thr = np.median(y)
    hi, lo = y > thr, y <= thr
    if hi.sum() == 0 or lo.sum() == 0:
        return np.zeros(X.shape[1])
    return X[hi].mean(0) - X[lo].mean(0)


def decodability_map(features, y, mode="magnitude") -> dict:
    """Per site, how decodable is y from that site's (n x C) feature matrix.
    mode='magnitude' -> R2 (linear); mode='classify' -> balanced accuracy."""
    out = {}
    for site, X in features.items():
        if mode == "classify":
            out[site] = fit_eval_classifier(X, y)["bal_acc"]
        else:
            out[site] = fit_eval_magnitude(X, y)["r2"]
    return out
