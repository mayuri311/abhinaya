"""Reference (Python) implementation of the hasta kNN pipeline.

Mirrors detect_and_crop / embed / kNN from the training notebook exactly, but
reads every constant from knn_bank_meta.json. The browser implementation in
lib/hasta/ is tested against this file, so keep the two in lockstep.

torch is not used here. If you ever add it, import it BEFORE mediapipe:
mediapipe-first makes the first DINOv2 forward crash with SIGFPE.
"""

import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hasta")

# Placeholders: TODO calibrate on a near-duplicate-grouped held-out set (see lib/hasta/config.ts).
MIN_TOP1 = 0.0
MIN_MARGIN = 0.0


class HastaPipeline:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, onnx_name="hasta_dinov2_vits14_embed.onnx"):
        with open(os.path.join(model_dir, "knn_bank_meta.json")) as f:
            self.meta = json.load(f)
        self.classes = self.meta["classes"]
        crop = self.meta["crop"]
        self.size, self.margin, self.detect_pad = crop["size"], crop["margin"], crop["detect_pad"]
        self.pad_fill = tuple(crop["pad_fill"])
        self.mean = np.array(self.meta["normalize"]["mean"], np.float32)
        self.std = np.array(self.meta["normalize"]["std"], np.float32)

        self.landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(model_dir, "hand_landmarker.task")),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        ))
        self.session = ort.InferenceSession(os.path.join(model_dir, onnx_name), providers=["CPUExecutionProvider"])
        self.bank = np.load(os.path.join(model_dir, "knn_bank_feats_fp16.npy")).astype(np.float32)
        self.labels = np.load(os.path.join(model_dir, "knn_bank_labels.npy")).astype(np.int64)

    def detect_and_crop(self, img):
        """PIL image -> (PIL crop | None, info). info carries the landmarks in padded-canvas pixels."""
        img = img.convert("RGB")
        W, H = img.size
        p = int(round(self.detect_pad * max(W, H)))  # Python round: half-to-even
        canvas = Image.new("RGB", (W + 2 * p, H + 2 * p), self.pad_fill)
        canvas.paste(img, (p, p))
        cw, ch = canvas.size
        res = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                              data=np.ascontiguousarray(np.asarray(canvas))))
        if not res.hand_landmarks:
            return None, {"detected": False, "pad": p}
        lm = np.array([(l.x * cw, l.y * ch) for l in res.hand_landmarks[0]])
        (x0, y0), (x1, y1) = lm.min(0), lm.max(0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(x1 - x0, y1 - y0) * (1 + 2 * self.margin)
        box = tuple(int(round(v)) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
        crop = canvas.crop(box).resize((self.size, self.size), Image.BICUBIC)
        h = res.handedness[0][0]
        return crop, {"detected": True, "pad": p, "landmarks_px": lm.tolist(), "box_padded": box,
                      "handedness": h.category_name, "detector_score": float(h.score)}

    def to_input(self, crop):
        x = (np.asarray(crop, np.float32) / 255.0 - self.mean) / self.std
        return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

    def embed(self, crop):
        return self.session.run(["embedding"], {"crop": self.to_input(crop)})[0][0]

    def class_scores(self, emb):
        sims = self.bank @ emb.astype(np.float32)
        scores = np.full(len(self.classes), -np.inf, np.float32)
        np.maximum.at(scores, self.labels, sims)
        return scores

    def classify(self, img, topn=5):
        t0 = time.perf_counter()
        crop, info = self.detect_and_crop(img)
        if crop is None:
            return {"status": "no_hand", "latencyMs": (time.perf_counter() - t0) * 1000}
        scores = self.class_scores(self.embed(crop))
        order = np.argsort(-scores)[:topn]
        top = [{"label": self.classes[i], "score": float(scores[i])} for i in order]
        margin = top[0]["score"] - top[1]["score"]
        x0, y0, x1, y1 = info["box_padded"]
        p = info["pad"]
        return {"status": "ok", "top5": top, "margin": margin,
                "lowConfidence": top[0]["score"] < MIN_TOP1 or margin < MIN_MARGIN,
                "box": {"x": x0 - p, "y": y0 - p, "w": x1 - x0, "h": y1 - y0},
                "handedness": info["handedness"], "detectorScore": info["detector_score"],
                "latencyMs": (time.perf_counter() - t0) * 1000}
