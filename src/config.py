from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    counts: list
    objects: list
    seeds: list
    num_inference_steps: int
    score_threshold: float
    capture_steps: list = None  # timesteps to snapshot activations (Phase 2+)


def load_config(path: str) -> ExperimentConfig:
    import yaml
    from dataclasses import fields
    with open(path) as f:
        d = yaml.safe_load(f)
    known = {f.name for f in fields(ExperimentConfig)}
    return ExperimentConfig(**{k: v for k, v in d.items() if k in known})
