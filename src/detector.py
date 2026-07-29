from src.scoring import count_from_detections


def count_objects(detector, image, object_label, score_threshold=0.3) -> int:
    dets = detector.detect(image, labels=[object_label])
    return count_from_detections(dets, object_label, score_threshold)


class Detector:
    """GroundingDINO wrapper. Loads lazily; GPU only (used in Colab)."""

    def __init__(self, device="cuda", box_threshold=0.3, text_threshold=0.25):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import (AutoProcessor,
                                      AutoModelForZeroShotObjectDetection)
            model_id = "IDEA-Research/grounding-dino-tiny"
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id).to(self.device)

    def detect(self, image, labels):
        self._ensure_loaded()
        import torch
        text = ". ".join(labels) + "."
        inputs = self._processor(images=image, text=text,
                                 return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        # transformers renamed "labels" -> "text_labels" (>=4.51); support both.
        labels_out = results.get("text_labels", results.get("labels"))
        dets = []
        for score, label, box in zip(results["scores"], labels_out,
                                     results["boxes"]):
            dets.append({"label": str(label).strip().lower(),
                         "score": float(score),
                         "box": [float(x) for x in box]})
        return dets
