"""Parity test: preprocessing + ONNX embedder must reproduce the training embeddings.

The bank holds the embedding of every TRAIN crop, so a train image run through
this pipeline must find itself in the bank at cosine >= 0.999. The train split
is rebuilt exactly as in the notebook (sorted listing, stratified, seed 42).

    python scripts/hasta/parity_test.py <Bharatanatyam-Mudra-Dataset dir> [--onnx NAME] [--n 5]
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from hasta_ref import HastaPipeline

THRESHOLD = 0.999


def train_split(dataset_root, classes):
    rows = []
    for folder in sorted(os.listdir(dataset_root)):
        d = os.path.join(dataset_root, folder)
        if folder.startswith(".") or not os.path.isdir(d):
            continue
        label = re.sub(r"\(\d+\)$", "", folder).strip()
        if label not in classes:
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append((os.path.join(d, f), label))
    df = pd.DataFrame(rows, columns=["image_path", "label"])
    train_df, _ = train_test_split(df, test_size=0.1, stratify=df.label, random_state=42)
    return train_df


def pick_samples(train_df, n, seed=0):
    """n train images, each from a different class."""
    labels = train_df.label.drop_duplicates().sample(n, random_state=seed)
    return [train_df[train_df.label == l].sample(1, random_state=seed).iloc[0] for l in labels]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("--onnx", default="hasta_dinov2_vits14_embed.onnx")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    pipe = HastaPipeline(onnx_name=args.onnx)
    samples = pick_samples(train_split(args.dataset_root, set(pipe.classes)), args.n)

    failures = 0
    for row in samples:
        crop, info = pipe.detect_and_crop(Image.open(row.image_path))
        if crop is None:
            print(f"SKIP {row.label:14s} no hand detected (was detected at training time?)")
            failures += 1
            continue
        sims = pipe.bank @ pipe.embed(crop)
        j = int(sims.argmax())
        ok = sims[j] >= THRESHOLD and pipe.classes[pipe.labels[j]] == row.label
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'} {row.label:14s} self-match cos={sims[j]:.5f} "
              f"bank label={pipe.classes[pipe.labels[j]]}  {os.path.basename(row.image_path)}")

    print(f"\n{len(samples) - failures}/{len(samples)} passed (threshold {THRESHOLD})")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
