import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split


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
