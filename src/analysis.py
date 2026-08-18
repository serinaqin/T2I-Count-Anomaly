import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def bootstrap_ci(x, ci=0.95, n=2000, seed=0):
    """Bootstrap confidence interval for the mean of x."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(axis=1)
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def sign_flip_pvalue(x, n=2000, seed=0):
    """Two-sided permutation (sign-flip) test that mean(x) != 0."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan")
    obs = abs(x.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n, len(x)))
    null = np.abs((x * signs).mean(axis=1))
    return float((null >= obs).mean())


def variance_explained(df, value: str, factor: str) -> float:
    grand = df[value].mean()
    ss_total = ((df[value] - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for _, group in df.groupby(factor):
        ss_between += len(group) * (group[value].mean() - grand) ** 2
    return float(ss_between / ss_total)


def noise_vs_text_decomposition(df, value="realized_count",
                                noise="seed", text="prompt_id") -> dict:
    return {
        "noise_var_explained": variance_explained(df, value, noise),
        "text_var_explained": variance_explained(df, value, text),
    }


def text_responsiveness(df, realized="realized_count", requested="count") -> dict:
    """OLS of realized count on requested count. slope~1 => text controls the
    count; slope~0 => text is ignored (count set elsewhere, e.g. the noise)."""
    x = df[[requested]].to_numpy(float)
    y = df[realized].to_numpy(float)
    reg = LinearRegression().fit(x, y)
    return {"slope": float(reg.coef_[0]),
            "r2": float(r2_score(y, reg.predict(x)))}


def count_variance_decomposition(df, value="realized_count",
                                 factors=("seed", "count", "obj")) -> dict:
    """Marginal eta^2 per factor: fraction of realized-count variance that
    aligns with each factor. Not orthogonal (factors can share variance) —
    read as relative importance, not an additive partition."""
    return {f: variance_explained(df, value, f) for f in factors}


def per_seed_summary(df, value="realized_count", seed="seed") -> pd.DataFrame:
    """Per-seed mean and std of realized count. A seed with low std across
    different prompts has a persistent 'preferred count' (noise-driven)."""
    g = df.groupby(seed)[value]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=0)})


def flag_degenerate(df, max_requested, factor=3, value="realized_count"):
    """Mark images whose realized count is implausibly large (e.g. collage
    blow-ups) as degenerate, without deleting them."""
    out = df.copy()
    out["degenerate"] = out[value] > factor * max_requested
    return out
