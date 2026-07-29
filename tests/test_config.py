from src.config import load_config, ExperimentConfig

def test_load_default_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "counts: [1, 2, 3]\n"
        "objects: [cat, dog]\n"
        "seeds: [0, 1]\n"
        "num_inference_steps: 30\n"
        "score_threshold: 0.3\n"
    )
    cfg = load_config(str(p))
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.counts == [1, 2, 3]
    assert cfg.objects == ["cat", "dog"]
    assert cfg.num_inference_steps == 30
