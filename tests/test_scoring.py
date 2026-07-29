from src.scoring import exact_accuracy, mae, tolerance_accuracy, count_from_detections

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
