"""YuNet detect, MiniFASNet liveness, SFace embedding."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent.parent / "models"
YUNET = ROOT / "face_detection_yunet_2023mar.onnx"
SFACE = ROOT / "face_recognition_sface_2021dec.onnx"
LIVENESS = ROOT / "minifasnet_v2se.onnx"

MODELS = {
    "detector": "YuNet 2023mar",
    "embedding": "SFace MobileFaceNet 2021dec",
    "liveness": "MiniFASNetV2-SE",
}


class FaceEngine:
    def __init__(self) -> None:
        for path in (YUNET, SFACE, LIVENESS):
            if not path.exists():
                raise FileNotFoundError(path)
        self.detector = cv2.FaceDetectorYN.create(str(YUNET), "", (320, 320), 0.6, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE), "")
        self.liveness = ort.InferenceSession(str(LIVENESS), providers=["CPUExecutionProvider"])
        self.liveness_input = self.liveness.get_inputs()[0].name

    def analyze(self, raw: bytes) -> dict:
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("unreadable image")
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return {"ok": False, "reason": "no_face", "face_count": 0}
        if len(faces) > 1:
            return {"ok": False, "reason": "multiple_faces", "face_count": int(len(faces))}
        face = faces[0]
        liveness = self._liveness(image, face)
        aligned = self.recognizer.alignCrop(image, face)
        embedding = self.recognizer.feature(aligned).flatten().astype(np.float32)
        return {
            "ok": True,
            "reason": "ok",
            "face_count": 1,
            "liveness_score": liveness,
            "live": liveness >= 0.0,
            "embedding": embedding,
        }

    def _liveness(self, image: np.ndarray, face: np.ndarray) -> float:
        x, y, w, h = [int(v) for v in face[:4]]
        side = int(max(w, h) * 1.5)
        cx, cy = x + w // 2, y + h // 2
        x1, y1 = cx - side // 2, cy - side // 2
        crop = image[max(0, y1) : y1 + side, max(0, x1) : x1 + side]
        if crop.size == 0:
            return -1.0
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
        tensor = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        logits = self.liveness.run(None, {self.liveness_input: tensor[None]})[0][0]
        return float(logits[0] - logits[1])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)
