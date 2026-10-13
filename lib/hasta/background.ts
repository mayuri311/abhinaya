import type { PixelBuffer } from "./preprocess";

/**
 * out = mask · crop + (1 − mask) · fill, rounded half up; mask is the per-pixel
 * skin confidence in [0, 1]. Mirrors HastaPipeline.fill_background.
 */
export function fillBackground(crop: PixelBuffer, mask: Float32Array, fill: readonly [number, number, number]): PixelBuffer {
  const n = crop.width * crop.height;
  if (mask.length !== n) throw new Error(`mask ${mask.length} vs crop ${n}`);
  const out = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    const m = Math.min(1, Math.max(0, mask[i]));
    for (let c = 0; c < 3; c++) {
      const v = Math.floor(m * crop.data[i * crop.channels + c] + (1 - m) * fill[c] + 0.5);
      out[i * 3 + c] = v < 0 ? 0 : v > 255 ? 255 : v;
    }
  }
  return { data: out, width: crop.width, height: crop.height, channels: 3 };
}
