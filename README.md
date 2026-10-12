# CMU Abhinaya

Website for CMU's Bharatanatyam competitive dance team.

Built with Next.js (App Router), TypeScript, and Tailwind CSS. Deployed on Vercel.

## Local development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Deploy to Vercel

1. Push the repo to GitHub.
2. Import the repo at [vercel.com/new](https://vercel.com/new).
3. Vercel auto-detects Next.js — no additional configuration needed.
4. Every push to `main` triggers a production deploy; other branches get preview deploys.

## Mudra model (kNN engine)

The playground has two engines, picked at build time by `NEXT_PUBLIC_HASTA_ENGINE`:
`rules` (default) or `knn` (MediaPipe hand crop → DINOv2 ViT-S/14 ONNX → cosine kNN).

The kNN model files are not in git. They live in a Hugging Face model repo, and the
app loads them from `NEXT_PUBLIC_HASTA_ASSET_BASE`, e.g.
`https://huggingface.co/<user>/<repo>/resolve/<commit-sha>` (pin the commit so
browser caches never go stale). Files:

| File | |
|---|---|
| `hand_landmarker.task` | MediaPipe hand landmarker |
| `hasta_dinov2_vits14_embed_fp16.onnx` | embedder (fp16 export, see below) |
| `knn_bank_feats_fp16.npy`, `knn_bank_labels.npy` | reference bank |
| `knn_bank_meta.json` | class names + every preprocessing constant |

Locally, put them in `models/hasta/` and run `ln -sfn ../models/hasta public/hasta-assets`;
the default asset base is `/hasta-assets`. Then `NEXT_PUBLIC_HASTA_ENGINE=knn npm run dev`.

Other settings (`lib/hasta/config.ts`): `NEXT_PUBLIC_HASTA_MIN_TOP1` /
`NEXT_PUBLIC_HASTA_MIN_MARGIN` (low-confidence gate; placeholders until calibrated),
`NEXT_PUBLIC_HASTA_SMOOTHING_N`, `NEXT_PUBLIC_HASTA_MIN_INTERVAL_MS`, `NEXT_PUBLIC_HASTA_ONNX_FILE`.

**Swapping in a new embedder / bank:** drop the new files in `models/hasta/`, then

```bash
python -m venv scripts/hasta/.venv && scripts/hasta/.venv/bin/pip install -r scripts/hasta/requirements.txt
scripts/hasta/.venv/bin/python scripts/hasta/quantize.py                       # writes _fp16 / _int8 variants
scripts/hasta/.venv/bin/python scripts/hasta/parity_test.py <dataset> --onnx hasta_dinov2_vits14_embed_fp16.onnx
scripts/hasta/.venv/bin/python scripts/hasta/make_fixtures.py <dataset>
npm test && npm run test:e2e
```

`<dataset>` is a clone of [Bharatanatyam-Mudra-Dataset](https://github.com/jisharajr/Bharatanatyam-Mudra-Dataset).
Only ship a variant that passes parity (cosine ≥ 0.999). Upload to Hugging Face and
point `NEXT_PUBLIC_HASTA_ASSET_BASE` at the new commit.

## Project structure

| Path | Purpose |
|---|---|
| `app/` | Next.js App Router pages and root layout |
| `components/` | Shared React components (NavBar, etc.) |
| `data/` | JSON content files (`team.json`, `mudras.json`, `hastaLabels.json`) |
| `lib/hasta/` | kNN mudra classifier (browser) |
| `scripts/hasta/` | Python reference pipeline, parity test, quantisation |
| `tests/` | `vitest` unit tests, browser e2e (`tests/e2e`) |
| `public/` | Static assets (images, fonts) |

## Pages

| Route | Description |
|---|---|
| `/` | Home — scrapbook aesthetic, photo scroll + embedded videos |
| `/about` | Team history & description for potential sponsors |
| `/team` | Exec board and member roster (from `data/team.json`) |
| `/playground` | Mudra recognition via webcam + MediaPipe Hands |
| `/contact` | Mailing list, TartanConnect, socials |
