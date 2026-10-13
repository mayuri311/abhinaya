"""Add background-augmented entries to the kNN bank (no training).

Every training image is shot on a green screen, so the bank only knows "hand
on green". For each TRAIN image this pastes the hand onto a real photo,
crops it exactly as at inference (detect_and_crop), embeds the crop and its
mirror, and appends them to the original bank. Val images are never used.

    python scripts/hasta/augment_bank.py <dataset dir> <dir of bank background .jpg> <out dir> [--per-image 1]

RESULT (2026-10-13, 213 bank / 44 disjoint eval backgrounds, 600 held-out val images):
this does NOT help with the current frozen DINOv2 embedder, so don't ship it.
  cluttered, plain crop:  original bank 0.296  augmented 0.182
  cluttered, green fill:  original bank 0.538  augmented 0.453
  green screen:           unchanged (0.977)
110/111 cluttered queries matched a composite entry, but only 15% of those had the right label:
the frozen features are dominated by the background, so a webcam crop matches whichever
composite has a similar-looking photo behind it. Background invariance needs fine-tuning the
embedder (with this compositing as augmentation); rerun this on the fine-tuned model then.

Writes knn_bank_feats_fp16.npy, knn_bank_labels.npy and knn_bank_meta.json to
<out dir>; upload those three in place of the originals. Keep the backgrounds
used here out of any evaluation (bank_eval.py takes a separate set).
"""

import argparse
import glob
import json
import os
import random

import numpy as np
from PIL import Image

from bg_eval import composite
from hasta_ref import DEFAULT_MODEL_DIR, HastaPipeline
from parity_test import train_split


def embed_batch(pipe, crops):
    x = np.concatenate([pipe.to_input(c) for c in crops])
    return pipe.session.run(["embedding"], {"crop": x})[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("bg_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--per-image", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    pipe = HastaPipeline()
    c2i = {c: i for i, c in enumerate(pipe.classes)}
    train_df = train_split(args.dataset_root, set(pipe.classes))
    bgs = sorted(glob.glob(os.path.join(args.bg_dir, "*.jpg")))
    rng = random.Random(0)

    feats, labels, crops, crop_labels = [], [], [], []
    tried = detected = 0

    def flush():
        if crops:
            feats.append(embed_batch(pipe, crops))
            labels.extend(crop_labels)
            crops.clear(); crop_labels.clear()

    for i, (path, label) in enumerate(zip(train_df.image_path, train_df.label)):
        img = Image.open(path).convert("RGB")
        for _ in range(args.per_image):
            tried += 1
            crop, _ = pipe.detect_and_crop(composite(img, Image.open(rng.choice(bgs))))
            if crop is None:
                continue
            detected += 1
            crops += [crop, crop.transpose(Image.FLIP_LEFT_RIGHT)]
            crop_labels += [c2i[label]] * 2
        if len(crops) >= args.batch:
            flush()
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(train_df)}  detected {detected}/{tried}", flush=True)
    flush()

    new = np.concatenate(feats).astype(np.float16)
    old = np.load(os.path.join(DEFAULT_MODEL_DIR, "knn_bank_feats_fp16.npy"))
    old_y = np.load(os.path.join(DEFAULT_MODEL_DIR, "knn_bank_labels.npy"))
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "knn_bank_feats_fp16.npy"), np.concatenate([old, new]))
    np.save(os.path.join(args.out_dir, "knn_bank_labels.npy"), np.concatenate([old_y, np.array(labels, np.int16)]))
    meta = dict(pipe.meta)
    meta["bank_augmentation"] = {
        "original_entries": int(len(old)), "added_entries": int(len(new)),
        "how": "train hands chroma-keyed onto real photos (despilled), detect_and_crop, embedded + mirrored",
        "backgrounds": len(bgs), "per_image": args.per_image, "detected": f"{detected}/{tried}",
    }
    with open(os.path.join(args.out_dir, "knn_bank_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"\nbank {len(old)} + {len(new)} = {len(old) + len(new)} entries → {args.out_dir}")


if __name__ == "__main__":
    main()
