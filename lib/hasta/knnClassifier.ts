/**
 * DINOv2 + kNN hasta classifier (browser). Pixels in, HastaResult out:
 *
 *   pad frame → HandLandmarker (IMAGE) → square crop → PIL-bicubic 224 →
 *   [optional: skin-segment, fill background green] → ImageNet-normalise →
 *   [worker] ONNX embed → cosine kNN (per-class max)
 *
 * Same contract as scripts/hasta/hasta_ref.py; see lib/hasta/preprocess.ts.
 */

import { FilesetResolver, HandLandmarker, ImageSegmenter } from "@mediapipe/tasks-vision";
import { fillBackground } from "./background";
import { HASTA_CONFIG, MEDIAPIPE_WASM } from "./config";
import { cropPixels, padAmount, resizeBicubic, toTensor, cropBox, type PixelBuffer, type PixelBox } from "./preprocess";
import { rankClasses } from "./rank";
import type { HastaMeta, HastaResult } from "./types";
import type { WorkerIn, WorkerOut } from "./embed.worker";

export type FrameSource = HTMLVideoElement | HTMLImageElement | HTMLCanvasElement | ImageBitmap;

export type KnnDebug = {
  pad: number;
  landmarksPx: [number, number][];
  boxPadded: PixelBox;
  /** Crop exactly as at training time. */
  crop: PixelBuffer;
  /** What the embedder actually saw (crop, or crop with the background filled). */
  modelInput: PixelBuffer;
  embedding: Float32Array;
  embedMs: number;
  knnMs: number;
};

export type KnnClassifier = {
  classify(source: FrameSource, opts?: { debug?: boolean }): Promise<HastaResult & { debug?: KnnDebug }>;
  executionProvider: string;
  meta: HastaMeta;
  close(): void;
};

export type KnnOptions = {
  assetBase?: string;
  files?: Partial<typeof HASTA_CONFIG.files>;
  thresholds?: { minTop1: number; minMargin: number };
  executionProviders?: string[];
  landmarkerDelegate?: "CPU" | "GPU";
  background?: "green" | "none";
  onProgress?: (loaded: number, total: number) => void;
  /** Override for non-Next bundlers (the e2e harness). */
  createWorker?: () => Worker;
};

function sourceSize(s: FrameSource): [number, number] {
  if (s instanceof HTMLVideoElement) return [s.videoWidth, s.videoHeight];
  if (s instanceof HTMLImageElement) return [s.naturalWidth, s.naturalHeight];
  return [s.width, s.height];
}

export async function createKnnClassifier(opts: KnnOptions = {}): Promise<KnnClassifier> {
  const base = (opts.assetBase ?? HASTA_CONFIG.assetBase).replace(/\/$/, "");
  const files = { ...HASTA_CONFIG.files, ...opts.files };
  const url = (f: string) => `${base}/${f}`;
  const thresholds = opts.thresholds ?? { minTop1: HASTA_CONFIG.minTop1, minMargin: HASTA_CONFIG.minMargin };

  const metaRes = await fetch(url(files.meta));
  if (!metaRes.ok) throw new Error(`${metaRes.status} fetching ${url(files.meta)}`);
  const meta = (await metaRes.json()) as HastaMeta;

  const worker = opts.createWorker?.() ??
    new Worker(new URL("./embed.worker.ts", import.meta.url), { type: "module" });

  let nextId = 0;
  const pending = new Map<number, { resolve: (m: Extract<WorkerOut, { type: "scores" }>) => void; reject: (e: Error) => void }>();
  let readyResolve!: (ep: string) => void;
  let readyReject!: (e: Error) => void;
  const ready = new Promise<string>((res, rej) => { readyResolve = res; readyReject = rej; });

  worker.onmessage = (e: MessageEvent<WorkerOut>) => {
    const m = e.data;
    if (m.type === "progress") opts.onProgress?.(m.loaded, m.total);
    else if (m.type === "ready") readyResolve(m.executionProvider);
    else if (m.type === "scores") { pending.get(m.id)?.resolve(m); pending.delete(m.id); }
    else if (m.type === "error") {
      if (m.id === undefined) readyReject(new Error(m.message));
      else { pending.get(m.id)?.reject(new Error(m.message)); pending.delete(m.id); }
    }
  };
  worker.onerror = (e) => readyReject(new Error(e.message || "embedder worker failed to start"));

  const init: WorkerIn = {
    type: "init",
    onnxUrl: url(files.embedder),
    featsUrl: url(files.bankFeats),
    labelsUrl: url(files.bankLabels),
    numClasses: meta.classes.length,
    cropSize: meta.crop.size,
    executionProviders: opts.executionProviders ?? ["webgpu", "wasm"],
  };
  worker.postMessage(init);

  const background = opts.background ?? HASTA_CONFIG.background;
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  const [landmarker, segmenter, executionProvider] = await Promise.all([
    HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: url(files.landmarker), delegate: opts.landmarkerDelegate ?? "CPU" },
      runningMode: "IMAGE",
      numHands: 1,
      minHandDetectionConfidence: 0.3,
      minHandPresenceConfidence: 0.3,
    }),
    background === "green"
      ? ImageSegmenter.createFromOptions(vision, {
          baseOptions: { modelAssetPath: HASTA_CONFIG.segmenterUrl, delegate: "CPU" },
          runningMode: "IMAGE",
          outputConfidenceMasks: true,
          outputCategoryMask: false,
        })
      : null,
    ready,
  ]);

  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d", { willReadFrequently: true })!;
  const [fr, fg, fb] = meta.crop.pad_fill;
  const segCanvas = document.createElement("canvas");
  const segCtx = segCanvas.getContext("2d", { willReadFrequently: true })!;

  /** Skin-segment the crop and paint everything else the training backdrop green. */
  function greenBackground(crop: PixelBuffer): PixelBuffer {
    segCanvas.width = crop.width;
    segCanvas.height = crop.height;
    const img = segCtx.createImageData(crop.width, crop.height);
    for (let i = 0; i < crop.width * crop.height; i++) {
      img.data[i * 4] = crop.data[i * 3];
      img.data[i * 4 + 1] = crop.data[i * 3 + 1];
      img.data[i * 4 + 2] = crop.data[i * 3 + 2];
      img.data[i * 4 + 3] = 255;
    }
    segCtx.putImageData(img, 0, 0);
    const res = segmenter!.segment(segCanvas);
    try {
      const mask = res.confidenceMasks?.[HASTA_CONFIG.skinClass]?.getAsFloat32Array();
      if (!mask) throw new Error("segmenter returned no skin mask");
      const full = HASTA_CONFIG.skinFullConfidence;
      for (let i = 0; i < mask.length; i++) mask[i] = Math.min(1, mask[i] / full);
      return fillBackground(crop, mask, HASTA_CONFIG.backdropGreen);
    } finally {
      res.close();
    }
  }

  function embed(tensor: Float32Array) {
    const id = nextId++;
    return new Promise<Extract<WorkerOut, { type: "scores" }>>((resolve, reject) => {
      pending.set(id, { resolve, reject });
      worker.postMessage({ type: "embed", id, tensor } satisfies WorkerIn, [tensor.buffer]);
    });
  }

  async function classify(source: FrameSource, { debug = false } = {}) {
    const t0 = performance.now();
    const [W, H] = sourceSize(source);
    const p = padAmount(W, H, meta.crop.detect_pad);
    const cw = W + 2 * p;
    const ch = H + 2 * p;
    if (canvas.width !== cw || canvas.height !== ch) { canvas.width = cw; canvas.height = ch; }
    ctx.fillStyle = `rgb(${fr},${fg},${fb})`;
    ctx.fillRect(0, 0, cw, ch);
    ctx.drawImage(source, p, p);

    const det = landmarker.detect(canvas);
    if (!det.landmarks[0]) return { status: "no_hand" as const, latencyMs: performance.now() - t0 };

    const landmarksPx = det.landmarks[0].map((l) => [l.x * cw, l.y * ch] as [number, number]);
    const boxPadded = cropBox(landmarksPx, meta.crop.margin);
    const [x0, y0, x1, y1] = boxPadded;
    // getImageData returns transparent black outside the canvas = PIL crop's zero fill
    const region = ctx.getImageData(x0, y0, x1 - x0, y1 - y0);
    const rgb = cropPixels({ data: region.data, width: region.width, height: region.height, channels: 4 },
      [0, 0, region.width, region.height]);
    const crop = resizeBicubic(rgb, meta.crop.size, meta.crop.size);

    const modelInput = segmenter ? greenBackground(crop) : crop;
    const out = await embed(toTensor(modelInput, meta));
    const ranked = rankClasses(out.scores, meta.classes, thresholds);
    const hand = det.handedness[0]?.[0];
    const result: HastaResult & { debug?: KnnDebug } = {
      status: "ok",
      ...ranked,
      box: { x: x0 - p, y: y0 - p, w: x1 - x0, h: y1 - y0 },
      handedness: hand?.categoryName === "Left" ? "Left" : "Right",
      detectorScore: hand?.score ?? 0,
      latencyMs: performance.now() - t0,
    };
    if (debug) {
      result.debug = { pad: p, landmarksPx, boxPadded, crop, modelInput, embedding: out.embedding,
        embedMs: out.embedMs, knnMs: out.knnMs };
    }
    return result;
  }

  return {
    classify,
    executionProvider,
    meta,
    close() {
      landmarker.close();
      segmenter?.close();
      worker.terminate();
      for (const { reject } of pending.values()) reject(new Error("classifier closed"));
      pending.clear();
    },
  };
}
