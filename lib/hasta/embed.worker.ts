/// <reference lib="webworker" />
/**
 * Off-main-thread embedder + kNN: ONNX (onnxruntime-web) and the bank matmul.
 * Model and bank are fetched once, with progress, and kept in Cache Storage
 * (asset URLs should be revision-pinned, so cached copies never go stale).
 */

import * as ort from "onnxruntime-web/webgpu";
import { classScores, makeBank, parseNpy, type Bank } from "./bank";

export type WorkerInit = {
  type: "init";
  onnxUrl: string;
  featsUrl: string;
  labelsUrl: string;
  numClasses: number;
  cropSize: number;
  executionProviders: string[];
};
export type WorkerEmbed = { type: "embed"; id: number; tensor: Float32Array };
export type WorkerIn = WorkerInit | WorkerEmbed;

export type WorkerOut =
  | { type: "progress"; loaded: number; total: number }
  | { type: "ready"; executionProvider: string }
  | { type: "error"; message: string; id?: number }
  | { type: "scores"; id: number; scores: Float32Array; embedding: Float32Array; embedMs: number; knnMs: number };

declare const self: DedicatedWorkerGlobalScope;
const post = (m: WorkerOut, transfer: Transferable[] = []) => self.postMessage(m, transfer);

const CACHE_NAME = "hasta-assets";

let session: ort.InferenceSession | null = null;
let bank: Bank | null = null;
let cropSize = 0;

async function fetchAll(urls: string[]): Promise<ArrayBuffer[]> {
  const cache = typeof caches !== "undefined" ? await caches.open(CACHE_NAME).catch(() => null) : null;
  const responses = await Promise.all(urls.map(async (url) => {
    const hit = await cache?.match(url);
    if (hit) return { url, res: hit, cached: true };
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${res.status} fetching ${url}`);
    return { url, res, cached: false };
  }));

  const totals = responses.map(({ res }) => Number(res.headers.get("content-length")) || 0);
  const loaded = totals.map(() => 0);
  const total = totals.reduce((a, b) => a + b, 0);
  const report = () => post({ type: "progress", loaded: loaded.reduce((a, b) => a + b, 0), total });

  return Promise.all(responses.map(async ({ url, res, cached }, i) => {
    const reader = res.body!.getReader();
    const chunks: Uint8Array[] = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded[i] += value.length;
      report();
    }
    const out = new Uint8Array(chunks.reduce((n, c) => n + c.length, 0));
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    if (cache && !cached) {
      cache.put(url, new Response(out.slice(), { headers: { "content-length": String(out.length) } })).catch(() => {});
    }
    return out.buffer;
  }));
}

async function init(msg: WorkerInit) {
  ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ort.env.versions.web}/dist/`;
  cropSize = msg.cropSize;
  const [model, feats, labels] = await fetchAll([msg.onnxUrl, msg.featsUrl, msg.labelsUrl]);
  bank = makeBank(parseNpy(feats), parseNpy(labels), msg.numClasses);

  let lastErr: unknown;
  for (const ep of msg.executionProviders) {
    try {
      session = await ort.InferenceSession.create(model, { executionProviders: [ep] });
      // warm-up, so the first real frame doesn't pay for shader / kernel compilation
      const z = new ort.Tensor("float32", new Float32Array(3 * cropSize * cropSize), [1, 3, cropSize, cropSize]);
      await session.run({ crop: z });
      post({ type: "ready", executionProvider: ep });
      return;
    } catch (err) {
      lastErr = err;
      session = null;
    }
  }
  throw lastErr ?? new Error("no execution provider available");
}

async function embed({ id, tensor }: WorkerEmbed) {
  if (!session || !bank) throw new Error("embedder not ready");
  const t0 = performance.now();
  const out = await session.run({ crop: new ort.Tensor("float32", tensor, [1, 3, cropSize, cropSize]) });
  const embedding = new Float32Array(out.embedding.data as Float32Array);
  // The export already L2-normalises; re-normalise so the fp16 variant's tiny drift can't push scores past 1.
  let n = 0;
  for (let i = 0; i < embedding.length; i++) n += embedding[i] * embedding[i];
  n = Math.sqrt(n) || 1;
  for (let i = 0; i < embedding.length; i++) embedding[i] /= n;
  const t1 = performance.now();
  const scores = classScores(bank, embedding);
  post({ type: "scores", id, scores, embedding, embedMs: t1 - t0, knnMs: performance.now() - t1 },
    [scores.buffer, embedding.buffer]);
}

self.onmessage = (e: MessageEvent<WorkerIn>) => {
  const msg = e.data;
  const task = msg.type === "init" ? init(msg) : embed(msg);
  task.catch((err) => post({ type: "error", message: String(err), id: msg.type === "embed" ? msg.id : undefined }));
};
