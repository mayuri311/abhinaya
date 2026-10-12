"""Write test fixtures from the Python reference pipeline (tests/fixtures/hasta/, gitignored).

For each sample: <name>.png (PIL-decoded, so JS and Python see identical pixels),
<name>.rgb (raw RGB), <name>.crop.rgb (the PIL crop) and <name>.json (Python
result, landmarks, box, embedding). Samples:
  clear_*    train images (their bank entry is themselves)
  ambiguous  the held-out val image with the smallest top1−top2 margin
  no_hand    a background-only patch cut from a dataset image

    python scripts/hasta/make_fixtures.py <Bharatanatyam-Mudra-Dataset dir>
"""

import argparse
import json
import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from hasta_ref import HastaPipeline
from parity_test import pick_samples, train_split

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "fixtures", "hasta")


def write(pipe, name, img, kind):
    img = img.convert("RGB")
    img.save(os.path.join(OUT, f"{name}.png"))
    img = Image.open(os.path.join(OUT, f"{name}.png")).convert("RGB")
    np.asarray(img).tofile(os.path.join(OUT, f"{name}.rgb"))
    crop, info = pipe.detect_and_crop(img)
    rec = {"name": name, "kind": kind, "width": img.width, "height": img.height, **pipe.classify(img)}
    rec.pop("latencyMs")
    if crop is not None:
        np.asarray(crop).tofile(os.path.join(OUT, f"{name}.crop.rgb"))
        rec.update(pad=info["pad"], landmarksPx=info["landmarks_px"], boxPadded=list(info["box_padded"]),
                   embedding=pipe.embed(crop).tolist())
    with open(os.path.join(OUT, f"{name}.json"), "w") as f:
        json.dump(rec, f)
    top = rec["top5"][0] if rec["status"] == "ok" else None
    print(f"{name:12s} {rec['status']:8s} {top and top['label']} {top and round(top['score'], 4)} "
          f"margin={rec.get('margin', 0):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset_root")
    ap.add_argument("--n-clear", type=int, default=5)
    ap.add_argument("--n-val-scan", type=int, default=150)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    pipe = HastaPipeline()
    train_df = train_split(args.dataset_root, set(pipe.classes))
    for i, row in enumerate(pick_samples(train_df, args.n_clear)):
        write(pipe, f"clear_{i}", Image.open(row.image_path), "clear")

    # rebuild val = everything not in train, scan a sample for the smallest margin
    all_paths = {os.path.join(r, f) for r, _, fs in os.walk(args.dataset_root) for f in fs}
    val = [p for p in sorted(all_paths - set(train_df.image_path))
           if os.path.basename(os.path.dirname(p)).split("(")[0] in pipe.classes]
    rng = np.random.default_rng(0)
    best = None
    for p in rng.choice(val, size=min(args.n_val_scan, len(val)), replace=False):
        r = pipe.classify(Image.open(p))
        if r["status"] == "ok" and (best is None or r["margin"] < best[1]):
            best = (p, r["margin"])
    write(pipe, "ambiguous", Image.open(best[0]), "ambiguous")

    # background strip from a dataset image (studio wall, no hand)
    src = Image.open(train_df.image_path.iloc[0]).convert("RGB")
    bg = src.crop((0, 0, src.width // 5, src.height // 3)).resize((src.width, src.height), Image.BICUBIC)
    write(pipe, "no_hand", bg, "no_hand")


if __name__ == "__main__":
    main()
