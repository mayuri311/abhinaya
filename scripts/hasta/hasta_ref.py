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

# Background fill (see lib/hasta/config.ts): every training image is shot on a green screen, so
# replacing the non-skin part of the crop with that green moves webcam crops toward the bank.
# scripts/hasta/bg_eval.py: on hands pasted onto real photos, top-1 0.31 -> 0.50.
SEGMENTER = "selfie_multiclass_256x256.tflite"  # storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/
BODY_SKIN = 2  # selfie_multiclass classes: 0 bg, 1 hair, 2 body-skin, 3 face-skin, 4 clothes, 5 others
BACKDROP_GREEN = (74, 198, 119)  # median green-screen pixel over 150 train images (bg_eval.backdrop_green)
# Lenient mask: skin confidence >= this counts as fully hand, so unsure palm/finger pixels aren't painted.
# bg_eval-style sweep (368 cluttered crops): 0.3 → 0.519, 0.4 → 0.557, 0.5 → 0.543, 0.6 → 0.546, raw conf → 0.500
SKIN_FULL_CONFIDENCE = 0.5


class HastaPipeline:
    def __init__(self, model_dir=DEFAULT_MODEL_DIR, onnx_name="hasta_dinov2_vits14_embed.onnx", background=None):
        """background: None (exact training preprocessing) or "green" (skin-segment + green fill)."""
        self.model_dir, self.background, self._segmenter = model_dir, background, None
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

    def skin_mask(self, crop):
        """[H, W, 1] body-skin confidence for the crop."""
        if self._segmenter is None:
            self._segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=os.path.join(self.model_dir, SEGMENTER)),
                running_mode=vision.RunningMode.IMAGE, output_confidence_masks=True, output_category_mask=False))
        res = self._segmenter.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(crop))))
        conf = res.confidence_masks[BODY_SKIN].numpy_view().astype(np.float32)
        return np.clip(conf / SKIN_FULL_CONFIDENCE, 0, 1).reshape(crop.size[1], crop.size[0], 1)

    def fill_background(self, crop, mask=None, fill=BACKDROP_GREEN):
        mask = self.skin_mask(crop) if mask is None else mask
        a = np.asarray(crop, np.float32)
        out = mask * a + (1 - mask) * np.array(fill, np.float32)
        return Image.fromarray(np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8))

    def model_input(self, crop):
        return self.fill_background(crop) if self.background == "green" else crop

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
        scores = self.class_scores(self.embed(self.model_input(crop)))
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
