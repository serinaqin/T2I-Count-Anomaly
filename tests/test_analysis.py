import pandas as pd
from src.analysis import (variance_explained, noise_vs_text_decomposition,
                          text_responsiveness, count_variance_decomposition,
                          per_seed_summary, flag_degenerate,
                          bootstrap_ci, sign_flip_pvalue)


def test_bootstrap_ci_constant():
    lo, hi = bootstrap_ci([5.0] * 20)
    assert abs(lo - 5.0) < 1e-9 and abs(hi - 5.0) < 1e-9

def test_sign_flip_pvalue_strong_vs_null():
    assert sign_flip_pvalue([2.0] * 30, seed=0) < 0.01        # clearly nonzero mean
    assert sign_flip_pvalue([1, -1] * 15, seed=0) > 0.2       # zero-mean -> not sig.

def test_variance_explained_all_from_factor():
    # realized_count depends ONLY on seed
    rows = []
    for seed, val in {0: 2, 1: 5, 2: 7}.items():
        for prompt_id in range(4):
            rows.append({"realized_count": val, "seed": seed, "prompt_id": prompt_id})
    df = pd.DataFrame(rows)
    assert variance_explained(df, "realized_count", "seed") > 0.99
    assert variance_explained(df, "realized_count", "prompt_id") < 0.01

def test_decomposition_returns_both():
    df = pd.DataFrame({
        "realized_count": [2, 2, 5, 5],
        "seed": [0, 0, 1, 1],
        "prompt_id": [0, 1, 0, 1],
    })
    out = noise_vs_text_decomposition(df)
    assert out["noise_var_explained"] > 0.99
    assert out["text_var_explained"] < 0.01

def test_text_responsiveness_perfect():
    df = pd.DataFrame({"realized_count": [1, 2, 3, 4], "count": [1, 2, 3, 4]})
    out = text_responsiveness(df)
    assert abs(out["slope"] - 1.0) < 1e-6
    assert out["r2"] > 0.99

def test_text_responsiveness_ignored():
    df = pd.DataFrame({"realized_count": [3, 3, 3, 3, 3, 3],
                       "count": [1, 2, 3, 4, 5, 6]})
    out = text_responsiveness(df)
    assert abs(out["slope"]) < 1e-6

def test_count_variance_decomposition_keys_and_noise_dominant():
    rows = []
    for seed, val in {0: 2, 1: 5}.items():
        for c in [1, 2, 3]:
            rows.append({"realized_count": val, "seed": seed, "count": c, "obj": "cat"})
    df = pd.DataFrame(rows)
    out = count_variance_decomposition(df)
    assert set(out) == {"seed", "count", "obj"}
    assert out["seed"] > 0.99 and out["count"] < 0.01

def test_per_seed_summary():
    df = pd.DataFrame({"realized_count": [2, 2, 5, 5], "seed": [0, 0, 1, 1],
                       "count": [1, 2, 1, 2]})
    s = per_seed_summary(df)
    assert list(s.index) == [0, 1]
    assert s.loc[0, "mean"] == 2 and s.loc[0, "std"] == 0
    assert s.loc[1, "mean"] == 5

def test_flag_degenerate():
    df = pd.DataFrame({"realized_count": [1, 2, 49], "count": [1, 2, 1]})
    out = flag_degenerate(df, max_requested=5, factor=3)  # threshold = 15
    assert out["degenerate"].tolist() == [False, False, True]
