"""Does filling the background around the hand help on non-green-screen input?

Every training image is shot on a green screen, so the bank encodes "hand on
green". This simulates webcam input: held-out val hands are chroma-keyed off
the green and pasted onto real photos, then classified with

  raw            crop as is (exact training preprocessing)
  seg_green      non-skin pixels (MediaPipe selfie-multiclass segmenter) → backdrop green
  seg_dark_blur  non-skin pixels → blurred and darkened

Result on 400 val images (2026-10-13, 40 picsum backgrounds), top-1 over detected crops:
  green screen  raw 0.983  seg_green 0.686  seg_dark_blur 0.718
  cluttered     raw 0.313  seg_green 0.498  seg_dark_blur 0.459
(A mask drawn from the 21 landmarks instead of the segmenter only reached 0.34–0.37 cluttered.)

    python scripts/hasta/bg_eval.py <dataset dir> <dir of background .jpg photos> [--n 400]
"""

import argparse
import glob
import os
import random

import numpy as np
from PIL import Image, ImageFilter

from hasta_ref import HastaPipeline
from parity_test import train_split


def green_mask(img):
    """True where the pixel is green screen."""
    a = np.asarray(img).astype(np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    m = Image.fromarray(((g > r + 15) & (g > b + 15)).astype(np.uint8) * 255)
    return np.asarray(m.filter(ImageFilter.MedianFilter(5))) > 127


def backdrop_green(paths, n=150):
    """Median green-screen pixel; the source of hasta_ref.BACKDROP_GREEN."""
    px = []
    for p in random.Random(0).sample(paths, n):
        a = np.asarray(Image.open(p).convert("RGB").resize((166, 125)))
        px.append(a[green_mask(a)])
    return tuple(int(v) for v in np.median(np.concatenate(px), axis=0))


def dark_blur(pipe, crop, mask):
    a = np.asarray(crop, np.float32)
    bg = np.asarray(crop.filter(ImageFilter.GaussianBlur(8)), np.float32) * 0.4
    return Image.fromarray(np.clip(mask * a + (1 - mask) * bg + 0.5, 0, 255).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("bg_dir")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()

    pipe = HastaPipeline()
    train = set(train_split(args.dataset_root, set(pipe.classes)).image_path)
    val = []
    for folder in sorted(os.listdir(args.dataset_root)):
        label = folder.split("(")[0].strip()
        if label in pipe.classes:
            val += [(p, label) for p in sorted(glob.glob(os.path.join(args.dataset_root, folder, "*"))) if p not in train]
    rng = random.Random(0)
    val = rng.sample(val, min(args.n, len(val)))
    bgs = sorted(glob.glob(os.path.join(args.bg_dir, "*.jpg")))

    treatments = {
        "raw": lambda crop, mask: crop,
        "seg_green": lambda crop, mask: pipe.fill_background(crop, mask),
        "seg_dark_blur": lambda crop, mask: dark_blur(pipe, crop, mask),
    }
    hits = {(s, t): 0 for s in ["green_screen", "cluttered"] for t in treatments}
    detected = {"green_screen": 0, "cluttered": 0}
    for i, (path, label) in enumerate(val):
        img = Image.open(path).convert("RGB")
        soft = np.asarray(Image.fromarray((~green_mask(img)).astype(np.uint8) * 255)
                          .filter(ImageFilter.GaussianBlur(1.2)), np.float32)[..., None] / 255
        bg = np.asarray(Image.open(rng.choice(bgs)).convert("RGB").resize(img.size), np.float32)
        comp = Image.fromarray((soft * np.asarray(img) + (1 - soft) * bg).astype(np.uint8))
        for setting, im in [("green_screen", img), ("cluttered", comp)]:
            crop, _ = pipe.detect_and_crop(im)
            if crop is None:
                continue
            detected[setting] += 1
            mask = pipe.skin_mask(crop)
            for name, fn in treatments.items():
                pred = pipe.classes[int(np.argmax(pipe.class_scores(pipe.embed(fn(crop, mask)))))]
                hits[(setting, name)] += pred == label
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(val)}")

    n = len(val)
    print(f"\n{n} held-out val images · detection: green screen {detected['green_screen']}, cluttered {detected['cluttered']}")
    print("top-1 accuracy over detected crops (share of all images in brackets):")
    for setting in ["green_screen", "cluttered"]:
        row = "  ".join(f"{t} {hits[(setting, t)] / max(1, detected[setting]):.3f} ({hits[(setting, t)] / n:.3f})"
                        for t in treatments)
        print(f"  {setting:13s} {row}")


if __name__ == "__main__":
    main()
