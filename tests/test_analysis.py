import pandas as pd
from src.analysis import variance_explained, noise_vs_text_decomposition

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
