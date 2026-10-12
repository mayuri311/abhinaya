/** kNN reference bank: .npy loading and per-class cosine scoring. */

export type NpyArray = { shape: number[]; data: Float32Array | Int16Array | Uint16Array };

/** Minimal .npy reader for little-endian <f2 (returned as raw Uint16), <f4 and <i2, C order. */
export function parseNpy(buf: ArrayBuffer): NpyArray {
  const bytes = new Uint8Array(buf);
  if (bytes[0] !== 0x93 || String.fromCharCode(...bytes.slice(1, 6)) !== "NUMPY") {
    throw new Error("not a .npy file");
  }
  const major = bytes[6];
  const view = new DataView(buf);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(bytes.subarray(headerStart, headerStart + headerLen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  if (/'fortran_order':\s*True/.test(header)) throw new Error(".npy: fortran_order not supported");
  const shape = (/'shape':\s*\(([^)]*)\)/.exec(header)?.[1] ?? "")
    .split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  const count = shape.reduce((a, b) => a * b, 1);
  const body = buf.slice(headerStart + headerLen);
  switch (descr) {
    case "<f2": return { shape, data: new Uint16Array(body, 0, count) };
    case "<f4": return { shape, data: new Float32Array(body, 0, count) };
    case "<i2": return { shape, data: new Int16Array(body, 0, count) };
    default: throw new Error(`.npy dtype ${descr} not supported`);
  }
}

/** IEEE half → float32, exact (incl. subnormals / inf / nan). */
export function halfToFloat32(src: Uint16Array): Float32Array {
  const table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    table[h] =
      exp === 0 ? sign * frac * 2 ** -24
      : exp === 31 ? (frac ? NaN : sign * Infinity)
      : sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = table[src[i]];
  return out;
}

export type Bank = { feats: Float32Array; labels: Int16Array; dim: number; size: number; numClasses: number };

export function makeBank(feats: NpyArray, labels: NpyArray, numClasses: number): Bank {
  const [size, dim] = feats.shape;
  if (labels.shape[0] !== size) throw new Error(`bank: ${size} feats vs ${labels.shape[0]} labels`);
  const f = feats.data instanceof Uint16Array ? halfToFloat32(feats.data) : (feats.data as Float32Array);
  return { feats: f, labels: labels.data as Int16Array, dim, size, numClasses };
}

/** Per-class score = max cosine similarity over that class's bank entries (float32 math). */
export function classScores(bank: Bank, embedding: Float32Array): Float32Array {
  const { feats, labels, dim, size } = bank;
  const scores = new Float32Array(bank.numClasses).fill(-Infinity);
  for (let i = 0; i < size; i++) {
    let s = 0;
    const off = i * dim;
    for (let d = 0; d < dim; d++) s += feats[off + d] * embedding[d];
    const c = labels[i];
    if (s > scores[c]) scores[c] = s;
  }
  return scores;
}
