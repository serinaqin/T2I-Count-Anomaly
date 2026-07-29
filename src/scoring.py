import numpy as np


def exact_accuracy(pred, target) -> float:
    pred, target = np.asarray(pred), np.asarray(target)
    return float(np.mean(pred == target))


def mae(pred, target) -> float:
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    return float(np.mean(np.abs(pred - target)))


def tolerance_accuracy(pred, target, tol: int = 1) -> float:
    pred, target = np.asarray(pred), np.asarray(target)
    return float(np.mean(np.abs(pred - target) <= tol))


def count_from_detections(detections, target_label, score_threshold=0.3) -> int:
    return sum(
        1 for d in detections
        if d["label"] == target_label and d["score"] >= score_threshold
    )
