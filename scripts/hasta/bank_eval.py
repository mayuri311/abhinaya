"""Compare kNN banks on held-out hands, with and without the green background fill.

Held-out val hands (never in any bank) are pasted onto backgrounds that were
NOT used to build the augmented bank, then classified by each bank with the
plain crop and with the green fill. Also reports the original green-screen
images, to catch regressions.

    python scripts/hasta/bank_eval.py <dataset dir> <dir of EVAL background .jpg> <bank dir> [<bank dir> ...] [--n 600]
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


def load_bank(d, n_classes):
    feats = np.load(os.path.join(d, "knn_bank_feats_fp16.npy")).astype(np.float32)
    labels = np.load(os.path.join(d, "knn_bank_labels.npy")).astype(np.int64)
    return feats, labels


def top1(bank, emb, n_classes):
    feats, labels = bank
    sims = feats @ emb
    scores = np.full(n_classes, -np.inf, np.float32)
    np.maximum.at(scores, labels, sims)
    return int(np.argmax(scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("bg_dir")
    ap.add_argument("banks", nargs="+")
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()

    pipe = HastaPipeline()
    C = len(pipe.classes)
    banks = {os.path.basename(os.path.normpath(d)): load_bank(d, C) for d in args.banks}
    for name, (f, _) in banks.items():
        print(f"bank {name}: {len(f)} entries")

    train = set(train_split(args.dataset_root, set(pipe.classes)).image_path)
    val = []
    for folder in sorted(os.listdir(args.dataset_root)):
        label = folder.split("(")[0].strip()
        if label in pipe.classes:
            val += [(p, label) for p in sorted(glob.glob(os.path.join(args.dataset_root, folder, "*"))) if p not in train]
    rng = random.Random(1)
    val = rng.sample(val, min(args.n, len(val)))
    bgs = sorted(glob.glob(os.path.join(args.bg_dir, "*.jpg")))

    settings = ["green_screen", "cluttered"]
    hits = {(s, b, m): 0 for s in settings for b in banks for m in ["crop", "green_fill"]}
    detected = {s: 0 for s in settings}
    for i, (path, label) in enumerate(val):
        img = Image.open(path).convert("RGB")
        y = pipe.classes.index(label)
        for setting, im in [("green_screen", img), ("cluttered", composite(img, Image.open(rng.choice(bgs))))]:
            crop, _ = pipe.detect_and_crop(im)
            if crop is None:
                continue
            detected[setting] += 1
            embs = {"crop": pipe.embed(crop), "green_fill": pipe.embed(pipe.fill_background(crop))}
            for b, bank in banks.items():
                for m, e in embs.items():
                    hits[(setting, b, m)] += top1(bank, e, C) == y
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(val)}", flush=True)

    print(f"\n{len(val)} held-out images · detected: " + ", ".join(f"{s} {detected[s]}" for s in settings))
    print("top-1 over detected crops:")
    for s in settings:
        for b in banks:
            print(f"  {s:13s} bank={b:10s} " + "  ".join(
                f"{m} {hits[(s, b, m)] / max(1, detected[s]):.3f}" for m in ["crop", "green_fill"]))


if __name__ == "__main__":
    main()
