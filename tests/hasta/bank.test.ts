import { existsSync, readdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";
import { classScores, halfToFloat32, makeBank, parseNpy } from "@/lib/hasta/bank";
import { majorityVote, rankClasses } from "@/lib/hasta/rank";
import type { HastaMeta, HastaResult } from "@/lib/hasta/types";

const ROOT = path.resolve(__dirname, "../..");
const MODELS = path.join(ROOT, "models/hasta");
const FIXTURES = path.join(ROOT, "tests/fixtures/hasta");
const haveModels = existsSync(path.join(MODELS, "knn_bank_feats_fp16.npy"));

const toAB = (b: Buffer) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;

describe("halfToFloat32", () => {
  it("decodes normal, subnormal and special values", () => {
    const out = halfToFloat32(new Uint16Array([0x3c00, 0xc000, 0x0001, 0x7c00, 0x0000, 0x3555]));
    expect(out[0]).toBe(1);
    expect(out[1]).toBe(-2);
    expect(out[2]).toBe(2 ** -24);
    expect(out[3]).toBe(Infinity);
    expect(out[4]).toBe(0);
    expect(out[5]).toBeCloseTo(1 / 3, 3);
  });
});

describe("rankClasses / majorityVote", () => {
  const ok = (label: string, lowConfidence = false): HastaResult => ({
    status: "ok", top5: [{ label, score: 0.9 }, { label: "x", score: 0.8 }], margin: 0.1, lowConfidence,
    box: { x: 0, y: 0, w: 1, h: 1 }, handedness: "Right", detectorScore: 1, latencyMs: 1,
  });
  const none: HastaResult = { status: "no_hand", latencyMs: 1 };

  it("ranks and applies both thresholds", () => {
    const s = new Float32Array([0.5, 0.9, 0.85]);
    const r = rankClasses(s, ["a", "b", "c"], { minTop1: 0, minMargin: 0.1 });
    expect(r.top5.map((t) => t.label)).toEqual(["b", "c", "a"]);
    expect(r.margin).toBeCloseTo(0.05, 6);
    expect(r.lowConfidence).toBe(true);
    expect(rankClasses(s, ["a", "b", "c"], { minTop1: 0.95, minMargin: 0 }).lowConfidence).toBe(true);
    expect(rankClasses(s, ["a", "b", "c"], { minTop1: 0.5, minMargin: 0.01 }).lowConfidence).toBe(false);
  });

  it("votes, counting no_hand and unsure, ties to the most recent", () => {
    expect(majorityVote([])).toBeNull();
    expect(majorityVote([ok("a"), ok("b"), ok("a")])!.status).toBe("ok");
    expect((majorityVote([ok("a"), ok("b"), ok("a")]) as Extract<HastaResult, { status: "ok" }>).top5[0].label).toBe("a");
    expect(majorityVote([ok("a"), none, none])!.status).toBe("no_hand");
    expect((majorityVote([ok("a"), ok("b")]) as Extract<HastaResult, { status: "ok" }>).top5[0].label).toBe("b");
    expect((majorityVote([ok("a", true), ok("b", true), ok("c")]) as Extract<HastaResult, { status: "ok" }>).lowConfidence).toBe(true);
  });
});

describe.skipIf(!haveModels)("bank on real artifacts", () => {
  const meta = haveModels ? (JSON.parse(readFileSync(path.join(MODELS, "knn_bank_meta.json"), "utf8")) as HastaMeta) : null!;
  const bank = haveModels
    ? makeBank(parseNpy(toAB(readFileSync(path.join(MODELS, "knn_bank_feats_fp16.npy")))),
        parseNpy(toAB(readFileSync(path.join(MODELS, "knn_bank_labels.npy")))), meta.classes.length)
    : null!;

  it("parses shapes and labels", () => {
    expect([bank.size, bank.dim]).toEqual([23832, 768]);
    expect(Math.max(...bank.labels)).toBe(meta.classes.length - 1);
    let n = 0;
    for (let d = 0; d < bank.dim; d++) n += bank.feats[d] ** 2;
    expect(Math.sqrt(n)).toBeCloseTo(1, 2);
  });

  const fixtures = existsSync(FIXTURES)
    ? readdirSync(FIXTURES).filter((f) => f.endsWith(".json"))
        .map((f) => JSON.parse(readFileSync(path.join(FIXTURES, f), "utf8"))).filter((r) => r.status === "ok")
    : [];
  for (const fx of fixtures) {
    it(`${fx.name}: JS kNN ranking matches Python`, () => {
      const r = rankClasses(classScores(bank, new Float32Array(fx.embedding)), meta.classes, { minTop1: 0, minMargin: 0 });
      expect(r.top5.map((t) => t.label)).toEqual(fx.top5.map((t: { label: string }) => t.label));
      r.top5.forEach((t, i) => expect(t.score).toBeCloseTo(fx.top5[i].score, 4));
    });
  }
});
