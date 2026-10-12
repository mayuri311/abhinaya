/**
 * JS crop vs Python (PIL) crop, pixel for pixel, using the landmarks the
 * Python detector found. Fixtures: python scripts/hasta/make_fixtures.py.
 */
import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { cropFromLandmarks, cropPixels, padAmount, roundHalfEven, type PixelBuffer } from "@/lib/hasta/preprocess";
import type { HastaMeta } from "@/lib/hasta/types";

const ROOT = path.resolve(__dirname, "../..");
const FIXTURES = path.join(ROOT, "tests/fixtures/hasta");
const META_PATH = path.join(ROOT, "models/hasta/knn_bank_meta.json");
const haveFixtures = existsSync(FIXTURES) && existsSync(META_PATH);

describe("roundHalfEven (Python round)", () => {
  it("rounds ties to even", () => {
    expect([0.5, 1.5, 2.5, 166.5, 167.5, -0.5, -1.5, -2.5].map(roundHalfEven))
      .toEqual([0, 2, 2, 166, 168, 0, -2, -2]);
    expect([2.4, 2.6, -2.4, -2.6, 3].map(roundHalfEven)).toEqual([2, 3, -2, -3, 3]);
  });

  it("pads a 666x500 frame by 166, not 167", () => {
    expect(padAmount(666, 500, 0.25)).toBe(166);
  });
});

describe("cropPixels (PIL crop)", () => {
  it("fills outside-image area with black", () => {
    const src: PixelBuffer = { data: new Uint8Array([10, 20, 30, 40, 50, 60]), width: 2, height: 1, channels: 3 };
    expect(Array.from(cropPixels(src, [-1, 0, 2, 2]).data))
      .toEqual([0, 0, 0, 10, 20, 30, 40, 50, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
  });
});

describe.skipIf(!haveFixtures)("JS crop == PIL crop on fixtures", () => {
  const meta = haveFixtures ? (JSON.parse(readFileSync(META_PATH, "utf8")) as HastaMeta) : null!;
  const fixtures = haveFixtures
    ? readdirSync(FIXTURES).filter((f) => f.endsWith(".json"))
        .map((f) => JSON.parse(readFileSync(path.join(FIXTURES, f), "utf8")))
        .filter((r) => r.status === "ok")
    : [];

  it("has fixtures with a detected hand", () => expect(fixtures.length).toBeGreaterThanOrEqual(3));

  for (const fx of fixtures) {
    it(`${fx.name}: identical box and bytes`, () => {
      const raw = readFileSync(path.join(FIXTURES, `${fx.name}.rgb`));
      const p = padAmount(fx.width, fx.height, meta.crop.detect_pad);
      expect(p).toBe(fx.pad);

      // pad exactly as hasta_ref: pad_fill canvas, image pasted at (p, p)
      const cw = fx.width + 2 * p, ch = fx.height + 2 * p;
      const padded = new Uint8Array(cw * ch * 3);
      for (let i = 0; i < cw * ch; i++) padded.set(meta.crop.pad_fill, i * 3);
      for (let y = 0; y < fx.height; y++) {
        padded.set(raw.subarray(y * fx.width * 3, (y + 1) * fx.width * 3), ((y + p) * cw + p) * 3);
      }

      const { box, crop } = cropFromLandmarks({ data: padded, width: cw, height: ch, channels: 3 }, fx.landmarksPx, meta);
      expect(box).toEqual(fx.boxPadded);
      const expected = readFileSync(path.join(FIXTURES, `${fx.name}.crop.rgb`));
      expect(crop.data.length).toBe(expected.length);
      let diff = 0;
      for (let i = 0; i < expected.length; i++) if (crop.data[i] !== expected[i]) diff++;
      expect(diff).toBe(0);
    });
  }
});
