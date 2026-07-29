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
