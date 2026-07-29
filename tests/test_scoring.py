from src.scoring import (exact_accuracy, mae, tolerance_accuracy,
                         count_from_detections, iou, nms)

def test_exact_accuracy():
    assert exact_accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert exact_accuracy([1, 2, 3], [1, 2, 4]) == 2 / 3

def test_mae():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert mae([1, 2], [3, 2]) == 1.0

def test_tolerance_accuracy():
    assert tolerance_accuracy([1, 5], [2, 3], tol=1) == 0.5
    assert tolerance_accuracy([1, 5], [2, 3], tol=2) == 1.0

def test_count_from_detections_filters_label_and_threshold():
    dets = [
        {"label": "cat", "score": 0.9, "box": [0, 0, 1, 1]},
        {"label": "cat", "score": 0.2, "box": [1, 1, 2, 2]},
        {"label": "dog", "score": 0.9, "box": [2, 2, 3, 3]},
    ]
    assert count_from_detections(dets, "cat", score_threshold=0.3) == 1
    assert count_from_detections(dets, "cat", score_threshold=0.1) == 2
    assert count_from_detections(dets, "bird") == 0

def test_iou():
    assert iou([0, 0, 2, 2], [0, 0, 2, 2]) == 1.0        # identical
    assert iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0        # disjoint
    # half-overlap: [0,0,2,2] and [1,0,3,2] share a 1x2 area; union = 4+4-2=6
    assert abs(iou([0, 0, 2, 2], [1, 0, 3, 2]) - 2 / 6) < 1e-9

def test_nms_drops_overlapping_keeps_best():
    dets = [
        {"label": "dog", "score": 0.9, "box": [0, 0, 2, 2]},
        {"label": "dog", "score": 0.5, "box": [0, 0, 2, 2]},   # dup of the 0.9 box
        {"label": "dog", "score": 0.8, "box": [10, 10, 12, 12]},  # distinct
    ]
    kept = nms(dets, iou_threshold=0.5)
    assert len(kept) == 2
    assert {round(k["score"], 1) for k in kept} == {0.9, 0.8}  # kept higher-score dup
