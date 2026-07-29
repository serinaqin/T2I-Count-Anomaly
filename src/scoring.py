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


def iou(box_a, box_b) -> float:
    """Intersection-over-union of two [x0, y0, x1, y1] boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def nms(detections, iou_threshold=0.5):
    """Greedy non-max suppression: drop lower-score boxes that overlap a kept
    box by more than iou_threshold. Removes GroundingDINO's duplicate boxes so
    counts reflect distinct instances, not overlapping detections."""
    kept = []
    for d in sorted(detections, key=lambda d: d["score"], reverse=True):
        if all(iou(d["box"], k["box"]) < iou_threshold for k in kept):
            kept.append(d)
    return kept
