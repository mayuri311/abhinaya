"""Old vs fine-tuned embedder on held-out hands, with honest settings.

Each held-out val hand (never in a bank, never trained on) is classified:
  green_screen   the original photo
  cluttered      pasted onto a photo from <eval backgrounds> (must be disjoint from training backgrounds)
  cluttered+fill same, with the app's green background fill
"hard" = val images with no near-duplicate in training (old-model cosine < 0.95 to every train crop),
because the dataset's near-duplicate frames inflate random-split accuracy.

    python scripts/hasta/ft_eval.py <dataset dir> <eval backgrounds dir> <old model dir> <new model dir>
"""

import argparse
import glob
import os
import random

import numpy as np
from PIL import Image

from bg_eval import composite
from hasta_ref import HastaPipeline
from parity_test import train_split

ONNX = "hasta_dinov2_vits14_embed_fp16.onnx"  # what the app ships


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("bg_dir")
    ap.add_argument("old_dir")
    ap.add_argument("new_dir")
    ap.add_argument("--n", type=int, default=0, help="limit val images (0 = all)")
    args = ap.parse_args()

    models = {"old": HastaPipeline(args.old_dir, ONNX), "new": HastaPipeline(args.new_dir, ONNX)}
    det = models["old"]
    classes = det.classes

    train = set(train_split(args.dataset_root, set(classes)).image_path)
    val = []
    for folder in sorted(os.listdir(args.dataset_root)):
        label = folder.split("(")[0].strip()
        if label in classes:
            val += [(p, label) for p in sorted(glob.glob(os.path.join(args.dataset_root, folder, "*"))) if p not in train]
    rng = random.Random(7)
    if args.n:
        val = rng.sample(val, args.n)
    bgs = sorted(glob.glob(os.path.join(args.bg_dir, "*.jpg")))

    settings = ["green_screen", "cluttered", "cluttered+fill"]
    rec = []  # (hard, setting, model, correct)
    detected = {s: 0 for s in ["green_screen", "cluttered"]}
    for i, (path, label) in enumerate(val):
        y = classes.index(label)
        img = Image.open(path).convert("RGB")
        g_crop, _ = det.detect_and_crop(img)
        hard = None
        if g_crop is not None:
            detected["green_screen"] += 1
            hard = float((det.bank @ models["old"].embed(g_crop)).max()) < 0.95
        c_crop, _ = det.detect_and_crop(composite(img, Image.open(rng.choice(bgs))))
        fill = det.fill_background(c_crop) if c_crop is not None else None
        if c_crop is not None:
            detected["cluttered"] += 1
        for name, m in models.items():
            for s, crop in [("green_screen", g_crop), ("cluttered", c_crop), ("cluttered+fill", fill)]:
                if crop is None:
                    continue
                rec.append((hard, s, name, int(np.argmax(m.class_scores(m.embed(crop)))) == y))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(val)}", flush=True)

    print(f"\n{len(val)} held-out val images · detected: green screen {detected['green_screen']}, "
          f"cluttered {detected['cluttered']} · hard (no near-dup in train): {sum(1 for r in rec if r[0] and r[1] == 'green_screen' and r[2] == 'old')}")
    print("top-1 over detected crops (29 classes):")
    print(f"  {'setting':16s} {'subset':6s} {'old':>7s} {'new':>7s}")
    for s in settings:
        for subset, keep in [("all", lambda h: True), ("hard", lambda h: h is True)]:
            row = []
            for name in models:
                hits = [c for h, ss, mm, c in rec if ss == s and mm == name and keep(h)]
                row.append(f"{np.mean(hits):.3f}" if hits else "   -  ")
            n = sum(1 for h, ss, mm, _ in rec if ss == s and mm == "old" and keep(h))
            print(f"  {s:16s} {subset:6s} {row[0]:>7s} {row[1]:>7s}   (n={n})")


if __name__ == "__main__":
    main()
