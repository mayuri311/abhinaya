/**
 * Pixel-exact port of the training-time crop (scripts/hasta/hasta_ref.py):
 *
 *   p = round(detect_pad * max(W, H))                 Python round → half-to-even
 *   canvas = black (W+2p, H+2p), image pasted at (p, p)
 *   box = round(centre ± side/2), side = max(bw, bh) * (1 + 2 * margin)
 *   crop = canvas.crop(box).resize((size, size), Image.BICUBIC)
 *
 * PIL's crop fills out-of-canvas pixels with 0, and its BICUBIC resize is a
 * separable, antialiased (support widens when downscaling) a = −0.5 filter run
 * in 22-bit fixed point with 8-bit rounding between the two passes. All of
 * that is reproduced here; the canvas' own drawImage scaling is NOT bicubic.
 *
 * Pure functions on pixel buffers, so they run (and are tested) in Node too.
 */

import type { HastaMeta } from "./types";

/** Python's round(): ties go to the even neighbour. */
export function roundHalfEven(x: number): number {
  if (Math.abs(x % 1) === 0.5) return 2 * Math.round(x / 2) + 0; // + 0 turns -0 into 0
  return Math.round(x);
}

export function padAmount(width: number, height: number, detectPad: number): number {
  return roundHalfEven(detectPad * Math.max(width, height));
}

/** [x0, y0, x1, y1] integer box in padded-canvas pixels. */
export type PixelBox = [number, number, number, number];

export function cropBox(landmarksPx: ReadonlyArray<readonly [number, number]>, margin: number): PixelBox {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of landmarksPx) {
    if (x < x0) x0 = x;
    if (x > x1) x1 = x;
    if (y < y0) y0 = y;
    if (y > y1) y1 = y;
  }
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const side = Math.max(x1 - x0, y1 - y0) * (1 + 2 * margin);
  return [
    roundHalfEven(cx - side / 2),
    roundHalfEven(cy - side / 2),
    roundHalfEven(cx + side / 2),
    roundHalfEven(cy + side / 2),
  ];
}

/** Interleaved 8-bit pixels; channels is 3 (RGB) or 4 (RGBA, alpha ignored). */
export type PixelBuffer = { data: Uint8Array | Uint8ClampedArray; width: number; height: number; channels: 3 | 4 };

/** PIL Image.crop: integer box, area outside the source is black. Returns RGB. */
export function cropPixels(src: PixelBuffer, [x0, y0, x1, y1]: PixelBox): PixelBuffer {
  const w = x1 - x0;
  const h = y1 - y0;
  const out = new Uint8Array(w * h * 3);
  const c = src.channels;
  for (let y = 0; y < h; y++) {
    const sy = y + y0;
    if (sy < 0 || sy >= src.height) continue;
    for (let x = 0; x < w; x++) {
      const sx = x + x0;
      if (sx < 0 || sx >= src.width) continue;
      const si = (sy * src.width + sx) * c;
      const di = (y * w + x) * 3;
      out[di] = src.data[si];
      out[di + 1] = src.data[si + 1];
      out[di + 2] = src.data[si + 2];
    }
  }
  return { data: out, width: w, height: h, channels: 3 };
}

// ─── PIL bicubic (libImaging/Resample.c) ────────────────────────────────────

const PRECISION_BITS = 32 - 8 - 2;
const ONE = 1 << PRECISION_BITS;
const HALF = 1 << (PRECISION_BITS - 1);
const BICUBIC_SUPPORT = 2.0;

function bicubicFilter(x: number): number {
  const a = -0.5;
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
  if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
  return 0.0;
}

type Coeffs = { ksize: number; bounds: Int32Array; kk: Int32Array };

function precomputeCoeffs(inSize: number, outSize: number): Coeffs {
  const scale = inSize / outSize;
  const filterscale = scale < 1.0 ? 1.0 : scale;
  const support = BICUBIC_SUPPORT * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const k = new Float64Array(ksize);
  const ss = 1.0 / filterscale;

  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0.0;
    for (let x = 0; x < xmax; x++) {
      const w = bicubicFilter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      const w = ww !== 0.0 ? k[x] / ww : k[x];
      // normalize_coeffs_8bpc
      kk[xx * ksize + x] = w < 0 ? Math.trunc(-0.5 + w * ONE) : Math.trunc(0.5 + w * ONE);
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, kk };
}

function clip8(v: number): number {
  if (v >= ONE * 256) return 255;
  if (v <= 0) return 0;
  return Math.floor(v / ONE);
}

function resampleHorizontal(src: PixelBuffer, outW: number): PixelBuffer {
  const { ksize, bounds, kk } = precomputeCoeffs(src.width, outW);
  const out = new Uint8Array(outW * src.height * 3);
  for (let y = 0; y < src.height; y++) {
    const row = y * src.width;
    for (let xx = 0; xx < outW; xx++) {
      const xmin = bounds[xx * 2];
      const xmax = bounds[xx * 2 + 1];
      const kOff = xx * ksize;
      let s0 = HALF, s1 = HALF, s2 = HALF;
      for (let x = 0; x < xmax; x++) {
        const k = kk[kOff + x];
        const si = (row + x + xmin) * 3;
        s0 += src.data[si] * k;
        s1 += src.data[si + 1] * k;
        s2 += src.data[si + 2] * k;
      }
      const di = (y * outW + xx) * 3;
      out[di] = clip8(s0);
      out[di + 1] = clip8(s1);
      out[di + 2] = clip8(s2);
    }
  }
  return { data: out, width: outW, height: src.height, channels: 3 };
}

function resampleVertical(src: PixelBuffer, outH: number): PixelBuffer {
  const { ksize, bounds, kk } = precomputeCoeffs(src.height, outH);
  const w = src.width;
  const out = new Uint8Array(w * outH * 3);
  for (let yy = 0; yy < outH; yy++) {
    const ymin = bounds[yy * 2];
    const ymax = bounds[yy * 2 + 1];
    const kOff = yy * ksize;
    for (let x = 0; x < w; x++) {
      let s0 = HALF, s1 = HALF, s2 = HALF;
      for (let y = 0; y < ymax; y++) {
        const k = kk[kOff + y];
        const si = ((y + ymin) * w + x) * 3;
        s0 += src.data[si] * k;
        s1 += src.data[si + 1] * k;
        s2 += src.data[si + 2] * k;
      }
      const di = (yy * w + x) * 3;
      out[di] = clip8(s0);
      out[di + 1] = clip8(s1);
      out[di + 2] = clip8(s2);
    }
  }
  return { data: out, width: w, height: outH, channels: 3 };
}

/** PIL Image.resize((size, size), Image.BICUBIC) on an RGB buffer. */
export function resizeBicubic(src: PixelBuffer, outW: number, outH: number): PixelBuffer {
  let img = src;
  if (outW !== img.width) img = resampleHorizontal(img, outW);
  if (outH !== img.height) img = resampleVertical(img, outH);
  return img === src ? { ...src, data: src.data.slice() } : img;
}

/** RGB crop → NCHW float32, scaled to [0,1] and ImageNet-normalised. */
export function toTensor(crop: PixelBuffer, meta: HastaMeta): Float32Array {
  const { mean, std } = meta.normalize;
  const n = crop.width * crop.height;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      out[c * n + i] = (crop.data[i * 3 + c] / 255 - mean[c]) / std[c];
    }
  }
  return out;
}

/** Padded-canvas landmarks → square 224 crop, exactly as at training time. */
export function cropFromLandmarks(
  padded: PixelBuffer,
  landmarksPx: ReadonlyArray<readonly [number, number]>,
  meta: HastaMeta,
): { box: PixelBox; crop: PixelBuffer } {
  const box = cropBox(landmarksPx, meta.crop.margin);
  const crop = resizeBicubic(cropPixels(padded, box), meta.crop.size, meta.crop.size);
  return { box, crop };
}
