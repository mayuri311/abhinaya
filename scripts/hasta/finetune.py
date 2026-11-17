"""Fine-tune the DINOv2 ViT-S/14 hasta embedder to ignore the background.

Every training image is shot on a green screen, so the frozen embedder learned
"hand on green" and matches webcam crops by their background. This trains it
on the same hands pasted onto random real photos, with webcam-style damage,
so hands of the same mudra embed close together whatever is behind them.

The output contract is unchanged: ONNX input "crop" [B,3,224,224] ImageNet-
normalised, output "embedding" [B,768] = L2norm([L2norm(cls); L2norm(mean patch)]).
So the app only needs the new .onnx + rebuilt bank (same file names).

Runs on CUDA (Colab), Apple MPS or CPU. Steps, each resumable:

    python scripts/hasta/finetune.py prep   --dataset <Bharatanatyam-Mudra-Dataset> --cache <dir>
    python scripts/hasta/finetune.py train  --cache <dir> --bgs <train backgrounds dir> --out <dir>
    python scripts/hasta/finetune.py export --out <dir>
    python scripts/hasta/finetune.py bank   --cache <dir> --out <dir>

Honest check (near-duplicate frames inflate random-split accuracy): group near-duplicates
and hold whole groups out, then train/evaluate on that split instead:

    python scripts/hasta/finetune.py groups --cache <dir>            # writes index_grouped.json
    python scripts/hasta/finetune.py train  --cache <dir> --index index_grouped.json \
        --bgs <train bgs> --eval-bgs <held-out bgs> --out <scratch dir> --eval-first

<out> ends up holding the four app files plus the fp16 ONNX. Evaluate with
scripts/hasta/ft_eval.py on backgrounds that were NOT used for training.
"""

# torch MUST be imported before mediapipe (mediapipe-first → SIGFPE on the first DINOv2 forward)
import torch  # noqa: I001
import torch.nn as nn
import torch.nn.functional as F

import argparse
import glob
import io
import json
import math
import os
import random
import shutil
import time

import numpy as np
from PIL import Image, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "..", "models", "hasta")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── prep: cache crops (exactly as the app cuts them) + hand masks ────────────

def _prep_worker(args):
    rows, cache = args
    from bg_eval import green_mask
    from hasta_ref import HastaPipeline
    pipe = HastaPipeline(load_embedder=False)
    out = []
    for split, path, label in rows:
        crop, _ = pipe.detect_and_crop(Image.open(path))
        if crop is None:
            continue
        stem = f"{split}/{label}/{os.path.splitext(os.path.basename(path))[0].replace(' ', '_')}"
        os.makedirs(os.path.join(cache, os.path.dirname(stem)), exist_ok=True)
        crop.save(os.path.join(cache, stem + ".png"))
        a = np.asarray(crop)
        bg = green_mask(crop) | (a.max(axis=2) <= 3)  # green screen, or black padding past the frame
        Image.fromarray((~bg).astype(np.uint8) * 255).save(os.path.join(cache, stem + ".mask.png"))
        out.append({"split": split, "label": label, "image_path": path, "crop": stem + ".png", "mask": stem + ".mask.png"})
    return out


def prep(a):
    from multiprocessing import Pool
    from parity_test import train_split
    meta = json.load(open(os.path.join(MODELS, "knn_bank_meta.json")))
    classes = meta["classes"]
    train_df = train_split(a.dataset, set(classes))
    train = set(train_df.image_path)
    rows = []
    for folder in sorted(os.listdir(a.dataset)):
        label = folder.split("(")[0].strip()
        if label not in classes:
            continue
        for p in sorted(glob.glob(os.path.join(a.dataset, folder, "*"))):
            if p.lower().endswith((".jpg", ".jpeg", ".png")):
                rows.append(("train" if p in train else "val", p, label))
    chunks = [(rows[i::a.workers], a.cache) for i in range(a.workers)]
    with Pool(a.workers) as pool:
        index = [r for part in pool.map(_prep_worker, chunks) for r in part]
    index.sort(key=lambda r: (r["split"], r["label"], r["crop"]))
    json.dump({"classes": classes, "items": index}, open(os.path.join(a.cache, "index.json"), "w"))
    n = {s: sum(r["split"] == s for r in index) for s in ("train", "val")}
    print(f"cached {len(index)}/{len(rows)} detected crops: train {n['train']}, val {n['val']}")


# ─── data: hands pasted onto random photos, with webcam-style damage ──────────

class Composites(torch.utils.data.Dataset):
    def __init__(self, cache, items, classes, bgs, train=True, seed=0):
        self.cache, self.items, self.bgs, self.train = cache, items, bgs, train
        self.c2i = {c: i for i, c in enumerate(classes)}
        self.seed = seed

    def __len__(self):
        return len(self.items)

    def _bg(self, rng, size):
        r = rng.random()
        if r < 0.08:  # flat colour
            return Image.new("RGB", (size, size), tuple(rng.randrange(256) for _ in range(3)))
        if r < 0.14:  # the original green backdrop: keep green-screen inputs working
            return None
        bg = Image.open(rng.choice(self.bgs)).convert("RGB")
        w, h = bg.size
        s = int(min(w, h) * rng.uniform(0.25, 1.0))
        x, y = rng.randrange(w - s + 1), rng.randrange(h - s + 1)
        return bg.crop((x, y, x + s, y + s)).resize((size, size), Image.BICUBIC)

    def __getitem__(self, i):
        it = self.items[i]
        rng = random.Random() if self.train else random.Random(self.seed * 100003 + i)
        crop = Image.open(os.path.join(self.cache, it["crop"])).convert("RGB")
        mask = Image.open(os.path.join(self.cache, it["mask"])).convert("L")
        size = crop.size[0]

        if self.train:  # geometry on hand + mask together, so revealed area becomes background
            ang = rng.uniform(-15, 15)
            sc = rng.uniform(0.85, 1.15)
            tx, ty = (rng.uniform(-0.08, 0.08) * size for _ in range(2))
            crop = crop.rotate(ang, Image.BICUBIC, translate=(tx, ty), fillcolor=(0, 0, 0)).resize(
                (int(size * sc),) * 2, Image.BICUBIC)
            mask = mask.rotate(ang, Image.BILINEAR, translate=(tx, ty), fillcolor=0).resize(
                (int(size * sc),) * 2, Image.BILINEAR)
            # back to size: centre crop or pad
            if crop.size[0] >= size:
                o = (crop.size[0] - size) // 2
                crop, mask = crop.crop((o, o, o + size, o + size)), mask.crop((o, o, o + size, o + size))
            else:
                o = (size - crop.size[0]) // 2
                c2, m2 = Image.new("RGB", (size, size)), Image.new("L", (size, size))
                c2.paste(crop, (o, o)); m2.paste(mask, (o, o))
                crop, mask = c2, m2

        bg = self._bg(rng, size)
        if bg is None:
            img = crop
        else:
            soft = np.asarray(mask.filter(ImageFilter.GaussianBlur(rng.uniform(0.8, 2.0))), np.float32)[..., None] / 255
            fg = np.asarray(crop, np.float32).copy()
            fg[..., 1] = np.minimum(fg[..., 1], fg[..., 0])  # despill green fringe
            img = Image.fromarray(np.clip(soft * fg + (1 - soft) * np.asarray(bg, np.float32), 0, 255).astype(np.uint8))

        if self.train:
            img = self._damage(img, rng)
        x = (np.asarray(img, np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), self.c2i[it["label"]]

    @staticmethod
    def _damage(img, rng):
        from torchvision.transforms import functional as TF
        size = img.size[0]
        if rng.random() < 0.15:  # crop ran past the frame edge → black band
            band = int(size * rng.uniform(0.05, 0.3))
            side = rng.randrange(4)
            box = [(0, 0, band, size), (size - band, 0, size, size), (0, 0, size, band), (0, size - band, size, size)][side]
            img = img.copy(); img.paste((0, 0, 0), box)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = TF.adjust_brightness(img, rng.uniform(0.65, 1.35))
        img = TF.adjust_contrast(img, rng.uniform(0.7, 1.3))
        img = TF.adjust_saturation(img, rng.uniform(0.6, 1.4))
        img = TF.adjust_hue(img, rng.uniform(-0.04, 0.04))
        if rng.random() < 0.05:
            img = img.convert("L").convert("RGB")
        if rng.random() < 0.3:  # low-res webcam
            s = rng.randint(96, 180)
            img = img.resize((s, s), Image.BILINEAR).resize((size, size), Image.BICUBIC)
        if rng.random() < 0.2:
            img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
        if rng.random() < 0.4:  # compression
            buf = io.BytesIO(); img.save(buf, "JPEG", quality=rng.randint(30, 85)); buf.seek(0)
            img = Image.open(buf).convert("RGB")
        if rng.random() < 0.3:  # sensor noise
            a = np.asarray(img, np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, rng.uniform(2, 8), (size, size, 3))
            img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        return img


# ─── model: same embedding as the exported ONNX ───────────────────────────────

class HastaEmbedder(nn.Module):
    """[B,3,224,224] → [B,768] L2-normalised [cls ; patch-mean] (== knn_bank_meta 'cls+patch')."""

    def __init__(self, bb):
        super().__init__()
        self.bb = bb

    def forward(self, x):
        out = self.bb.forward_features(x)
        cls = F.normalize(out["x_norm_clstoken"], dim=-1)
        pm = F.normalize(out["x_norm_patchtokens"].mean(1), dim=-1)
        return F.normalize(torch.cat([cls, pm], dim=1), dim=-1)


class CosFace(nn.Module):
    def __init__(self, dim, n, s=30.0, m=0.25):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n, dim) * 0.01)
        self.s, self.m = s, m

    def forward(self, emb, y=None):
        cos = emb @ F.normalize(self.w, dim=-1).T
        if y is not None:
            cos = cos - self.m * F.one_hot(y, cos.shape[1])
        return self.s * cos


def load_backbone():
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", verbose=False)


@torch.no_grad()
def embed_loader(model, loader, dev):
    model.eval()
    E, Y = [], []
    for x, y in loader:
        E.append(model(x.to(dev)).float().cpu()); Y.append(y)
    return torch.cat(E), torch.cat(Y)


def knn_top1(bank_e, bank_y, q_e, q_y, n_classes):
    sims = q_e @ bank_e.T
    scores = torch.full((len(q_e), n_classes), -2.0)
    scores = scores.scatter_reduce(1, bank_y.expand(len(q_e), -1), sims, reduce="amax")
    return (scores.argmax(1) == q_y).float().mean().item()


def train(a):
    dev = device()
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    idx = json.load(open(os.path.join(a.cache, a.index)))
    classes, items = idx["classes"], idx["items"]
    tr = [r for r in items if r["split"] == "train"]
    va = [r for r in items if r["split"] == "val"]
    bgs = sorted(glob.glob(os.path.join(a.bgs, "*.jpg")))
    # monitoring backgrounds: --eval-bgs (held out) if given, else a slice of the training ones
    dev_bgs = sorted(glob.glob(os.path.join(a.eval_bgs, "*.jpg"))) if a.eval_bgs else bgs[: max(1, len(bgs) // 10)]

    def loader(items_, bgs_, train_, bs, shuffle):
        return torch.utils.data.DataLoader(Composites(a.cache, items_, classes, bgs_, train=train_), batch_size=bs,
                                           shuffle=shuffle, num_workers=a.workers, persistent_workers=a.workers > 0,
                                           drop_last=train_ and shuffle)

    train_dl = loader(tr, bgs, True, a.batch, True)
    bank_dl = torch.utils.data.DataLoader(_Clean(a.cache, tr, classes), batch_size=128, num_workers=a.workers)
    green_dl = torch.utils.data.DataLoader(_Clean(a.cache, va, classes), batch_size=128, num_workers=a.workers)
    clutter_dl = loader(va, dev_bgs, False, 128, False)

    bb = load_backbone()
    for p in bb.parameters():
        p.requires_grad = False
    tune = list(bb.blocks[-a.tune_blocks:]) + [bb.norm]
    for m in tune:
        for p in m.parameters():
            p.requires_grad = True
    model = HastaEmbedder(bb).to(dev)
    head = CosFace(768, len(classes)).to(dev)

    opt = torch.optim.AdamW([
        {"params": [p for m in tune for p in m.parameters()], "lr": a.lr},
        {"params": head.parameters(), "lr": a.head_lr},
    ], weight_decay=0.05)
    steps = a.epochs * len(train_dl)
    warm = len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.txt"), "a")

    def evaluate(tag):
        be, by = embed_loader(model, bank_dl, dev)
        # bank = train crops + mirrors, exactly like knn_bank_feats
        bf, _ = embed_loader(model, _Clean(a.cache, tr, classes, flip=True).loader(a.workers), dev)
        be, by = torch.cat([be, bf]), torch.cat([by, by])
        ge, gy = embed_loader(model, green_dl, dev)
        ce, cy = embed_loader(model, clutter_dl, dev)
        msg = (f"[{tag}] val kNN top-1: green screen {knn_top1(be, by, ge, gy, len(classes)):.3f}  "
               f"pasted on photos ({'held-out' if a.eval_bgs else 'train'} bgs) {knn_top1(be, by, ce, cy, len(classes)):.3f}")
        print(msg, flush=True); log.write(msg + "\n"); log.flush()

    if a.eval_first:
        evaluate("frozen")
    step = 0
    for ep in range(a.epochs):
        model.train(); head.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            loss = F.cross_entropy(head(model(x), y), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); step += 1
            tot += loss.item() * len(y); n += len(y)
        msg = f"epoch {ep + 1}/{a.epochs} loss {tot / n:.3f} ({time.time() - t0:.0f}s)"
        print(msg, flush=True); log.write(msg + "\n"); log.flush()
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            evaluate(f"epoch {ep + 1}")
        torch.save({"backbone": bb.state_dict(), "head": head.state_dict(), "classes": classes, "args": vars(a)},
                   os.path.join(a.out, "finetuned.pt"))


class _Clean(torch.utils.data.Dataset):
    """Cached crops as-is (optionally mirrored): what the bank is built from."""

    def __init__(self, cache, items, classes, flip=False):
        self.cache, self.items, self.flip = cache, items, flip
        self.c2i = {c: i for i, c in enumerate(classes)}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.cache, self.items[i]["crop"])).convert("RGB")
        if self.flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        x = (np.asarray(img, np.float32) / 255 - MEAN) / STD
        return torch.from_numpy(x.transpose(2, 0, 1).copy()), self.c2i[self.items[i]["label"]]

    def loader(self, workers):
        return torch.utils.data.DataLoader(self, batch_size=128, num_workers=workers)


# ─── groups: hold out whole near-duplicate groups ─────────────────────────────

def groups(a):
    """Frozen-model embeddings of every cached crop → connected components at cosine > thr →
    hold out ~20% of groups (stratified loosely by shuffling groups) as 'val'."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    dev = device()
    idx = json.load(open(os.path.join(a.cache, "index.json")))
    classes, items = idx["classes"], idx["items"]
    model = HastaEmbedder(load_backbone()).to(dev)
    E, _ = embed_loader(model, _Clean(a.cache, items, classes).loader(a.workers), dev)
    rows, cols = [], []
    for i in range(0, len(E), 2048):
        r, c = ((E[i:i + 2048] @ E.T) > a.group_thr).nonzero(as_tuple=True)
        rows.append(r + i); cols.append(c)
    rows, cols = torch.cat(rows).numpy(), torch.cat(cols).numpy()
    n_groups, g = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(E), len(E))), directed=False)
    order = list(range(n_groups))
    random.Random(0).shuffle(order)
    held, count = set(), 0
    for gid in order:  # take whole groups until ~20% of crops are held out
        if count >= 0.2 * len(items):
            break
        held.add(gid); count += int((g == gid).sum())
    out = [dict(r, split="val" if g[i] in held else "train", group=int(g[i])) for i, r in enumerate(items)]
    json.dump({"classes": classes, "items": out}, open(os.path.join(a.cache, "index_grouped.json"), "w"))
    sizes = np.bincount(g)
    present = len({r["label"] for r in out if r["split"] == "val"})
    print(f"{n_groups} groups at cos>{a.group_thr} (largest {sizes.max()}, median {int(np.median(sizes))}); "
          f"held out {count} crops in {len(held)} groups, {present}/{len(classes)} classes present")


# ─── export + bank ────────────────────────────────────────────────────────────

def export(a):
    import onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16
    import onnx
    ck = torch.load(os.path.join(a.out, "finetuned.pt"), map_location="cpu")
    bb = load_backbone(); bb.load_state_dict(ck["backbone"])
    emb = HastaEmbedder(bb).float().eval()
    x = torch.randn(4, 3, 224, 224)
    path = os.path.join(a.out, "hasta_dinov2_vits14_embed.onnx")
    torch.onnx.export(emb, x[:1], path, input_names=["crop"], output_names=["embedding"],
                      dynamic_axes={"crop": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, dynamo=False)
    with torch.no_grad():
        t = emb(x).numpy()
    o = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"crop": x.numpy()})[0]
    print(f"onnx vs torch max |Δ| {np.abs(o - t).max():.2e}")
    fp16 = path.replace(".onnx", "_fp16.onnx")
    onnx.save(convert_float_to_float16(onnx.load(path), keep_io_types=True), fp16)
    o16 = ort.InferenceSession(fp16, providers=["CPUExecutionProvider"]).run(None, {"crop": x.numpy()})[0]
    cos = (o16 * o).sum(1) / np.linalg.norm(o16, axis=1)
    print(f"fp16 vs fp32 min cos {cos.min():.5f} (must be >= 0.999 to ship)")
    for f in ("hand_landmarker.task", "selfie_multiclass_256x256.tflite"):
        if os.path.exists(os.path.join(MODELS, f)):
            shutil.copy(os.path.join(MODELS, f), os.path.join(a.out, f))


def bank(a):
    import onnxruntime as ort
    idx = json.load(open(os.path.join(a.cache, "index.json")))
    classes = idx["classes"]
    tr = [r for r in idx["items"] if r["split"] == "train"]
    sess = ort.InferenceSession(os.path.join(a.out, "hasta_dinov2_vits14_embed.onnx"), providers=["CPUExecutionProvider"])
    feats, labels = [], []
    for flip in (False, True):  # train crops, then their mirrors (same layout as the original bank)
        ds = _Clean(a.cache, tr, classes, flip=flip)
        for i in range(0, len(ds), 64):
            xs, ys = zip(*(ds[j] for j in range(i, min(i + 64, len(ds)))))
            feats.append(sess.run(["embedding"], {"crop": torch.stack(xs).numpy()})[0]); labels += ys
    f = np.concatenate(feats)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    np.save(os.path.join(a.out, "knn_bank_feats_fp16.npy"), f.astype(np.float16))
    np.save(os.path.join(a.out, "knn_bank_labels.npy"), np.array(labels, np.int16))
    meta = json.load(open(os.path.join(MODELS, "knn_bank_meta.json")))
    ck_args = torch.load(os.path.join(a.out, "finetuned.pt"), map_location="cpu")["args"]
    meta["background_fill"] = "none"  # trained on real backgrounds: the app skips the green fill
    meta["embedder"] = {"base": "dinov2_vits14", "finetuned": "CosFace on train hands pasted onto random photos "
                        "(scripts/hasta/finetune.py)", "tune_blocks": ck_args["tune_blocks"], "epochs": ck_args["epochs"]}
    json.dump(meta, open(os.path.join(a.out, "knn_bank_meta.json"), "w"), indent=1)
    print(f"bank: {len(f)} entries → {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "groups", "train", "export", "bank"])
    ap.add_argument("--dataset")
    ap.add_argument("--cache")
    ap.add_argument("--bgs")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "models", "hasta_ft"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=3e-3)
    ap.add_argument("--tune-blocks", type=int, default=6)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--eval-first", action="store_true")
    ap.add_argument("--index", default="index.json")
    ap.add_argument("--eval-bgs", default=None)
    ap.add_argument("--group-thr", type=float, default=0.95)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    {"prep": prep, "groups": groups, "train": train, "export": export, "bank": bank}[a.step](a)


if __name__ == "__main__":
    main()
