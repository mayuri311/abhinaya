/**
 * Runtime config for the hasta classifiers. NEXT_PUBLIC_* values are inlined
 * at build time, so each Vercel environment (Production / Preview) can pick
 * its own engine and model revision without code changes.
 *
 * Swapping in a fine-tuned embedder + rebuilt bank = upload the new files to
 * the asset location and point NEXT_PUBLIC_HASTA_ASSET_BASE at the new
 * revision. Every preprocessing constant is read from knn_bank_meta.json.
 */

export type HastaEngine = "rules" | "knn";

/** MediaPipe runtime; keep the version in step with @mediapipe/tasks-vision in package.json. */
export const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

function num(v: string | undefined, fallback: number): number {
  const n = v === undefined || v === "" ? NaN : Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export const HASTA_CONFIG = {
  /** "rules" (default, what production runs) or "knn" (DINOv2 + kNN bank). */
  engine: (process.env.NEXT_PUBLIC_HASTA_ENGINE === "knn" ? "knn" : "rules") as HastaEngine,

  /** Folder holding the model files, no trailing slash. */
  assetBase: (process.env.NEXT_PUBLIC_HASTA_ASSET_BASE ?? "/hasta-assets").replace(/\/$/, ""),
  files: {
    landmarker: "hand_landmarker.task",
    // fp16 export of hasta_dinov2_vits14_embed.onnx (scripts/hasta/quantize.py).
    // It passed parity_test.py; the int8 one did not. Re-run both after any swap.
    embedder: process.env.NEXT_PUBLIC_HASTA_ONNX_FILE ?? "hasta_dinov2_vits14_embed_fp16.onnx",
    bankFeats: "knn_bank_feats_fp16.npy",
    bankLabels: "knn_bank_labels.npy",
    meta: "knn_bank_meta.json",
  },

  /**
   * Low-confidence gate: fires when top1 score < minTop1 OR top1 − top2 < minMargin.
   * TODO: these are PLACEHOLDERS, not tuned values. Calibrate both on a held-out
   * set whose near-duplicate groups are held out together (random-split val is
   * inflated), then set them via NEXT_PUBLIC_HASTA_MIN_TOP1 / _MIN_MARGIN.
   */
  minTop1: num(process.env.NEXT_PUBLIC_HASTA_MIN_TOP1, 0.0),
  minMargin: num(process.env.NEXT_PUBLIC_HASTA_MIN_MARGIN, 0.0),

  /** Live video: majority vote over the last N predictions. */
  smoothingWindow: num(process.env.NEXT_PUBLIC_HASTA_SMOOTHING_N, 5),
  /** Live video: minimum gap between inferences. A new one never starts before the last finishes. */
  minIntervalMs: num(process.env.NEXT_PUBLIC_HASTA_MIN_INTERVAL_MS, 100),
} as const;
