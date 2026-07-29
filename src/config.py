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
    with open(path) as f:
        d = yaml.safe_load(f)
    return ExperimentConfig(**d)
