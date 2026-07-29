from src.detector import count_objects

class FakeDetector:
    def __init__(self, dets):
        self._dets = dets
    def detect(self, image, labels):
        return [d for d in self._dets if d["label"] in labels]

def test_count_objects_uses_detect_and_threshold():
    dets = [
        {"label": "cat", "score": 0.8, "box": [0, 0, 1, 1]},
        {"label": "cat", "score": 0.25, "box": [1, 1, 2, 2]},
    ]
    det = FakeDetector(dets)
    assert count_objects(det, image=None, object_label="cat", score_threshold=0.3) == 1
    assert count_objects(det, image=None, object_label="cat", score_threshold=0.2) == 2
