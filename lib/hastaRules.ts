/**
 * hastaRules.ts
 * Ported from hasta_rules.js. Self-contained — only imports the shared
 * Landmark type. Add new hastas to HASTA_SCORERS to extend the classifier.
 *
 * Public interface (the ONLY thing the playground should call):
 *   classifyHasta(landmarks, handedness) → ClassifyResult
 */

import type { Landmark } from "@/lib/handGeometry";

// ─── public types ────────────────────────────────────────────────────────────

export type ClassifyResult = {
  hasta:     string | null; // null = no confident match
  score:     number;        // 0..1
  allScores: Record<string, number>;
};

// ─── landmark index constants ────────────────────────────────────────────────

const WRIST  = 0;
const THUMB  = { CMC: 1, MCP: 2, IP:  3, TIP: 4  } as const;
const INDEX  = { MCP: 5, PIP: 6, DIP: 7, TIP: 8  } as const;
const MIDDLE = { MCP: 9, PIP:10, DIP:11, TIP:12  } as const;
const RING   = { MCP:13, PIP:14, DIP:15, TIP:16  } as const;
const PINKY  = { MCP:17, PIP:18, DIP:19, TIP:20  } as const;

type FingerIdx = { MCP: number; PIP: number; DIP: number; TIP: number };

// ─── geometry primitives ─────────────────────────────────────────────────────

function sub(a: Landmark, b: Landmark) {
  return { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) };
}

function dist(a: Landmark, b: Landmark): number {
  const d = sub(a, b);
  return Math.sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
}

/** Angle (radians) at vertex b in the chain a–b–c. */
function angleAt(a: Landmark, b: Landmark, c: Landmark): number {
  const ba  = sub(a, b);
  const bc  = sub(c, b);
  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const mag = Math.sqrt(
    (ba.x * ba.x + ba.y * ba.y + ba.z * ba.z) *
    (bc.x * bc.x + bc.y * bc.y + bc.z * bc.z),
  );
  return mag === 0 ? 0 : Math.acos(Math.max(-1, Math.min(1, dot / mag)));
}

const deg     = (rad: number) => (rad * 180) / Math.PI;
const clamp01 = (v: number)   => Math.max(0, Math.min(1, v));

/** Wrist → middle-MCP length; normalises every distance to hand size. */
function handScale(lm: Landmark[]): number {
  return dist(lm[WRIST], lm[MIDDLE.MCP]) || 1e-6;
}

/**
 * Soft 0..1 extension for a non-thumb finger.
 * Uses MCP→PIP→TIP so it captures overall openness across the whole finger.
 * Maps [100°, 170°] → [0, 1].
 */
function fingerExtension(lm: Landmark[], f: FingerIdx): number {
  const a = angleAt(lm[f.MCP], lm[f.PIP], lm[f.TIP]);
  return clamp01((deg(a) - 100) / (170 - 100));
}

/**
 * Soft 0..1 extension for the thumb, using the IP joint angle.
 * Maps [120°, 175°] → [0, 1].
 */
function thumbExtension(lm: Landmark[]): number {
  const a = angleAt(lm[THUMB.MCP], lm[THUMB.IP], lm[THUMB.TIP]);
  return clamp01((deg(a) - 120) / (175 - 120));
}

/**
 * 0..1 togetherness of the four fingertips.
 * 1 = tightly together, 0 = widely spread.
 * Maps average normalised gap [0.15, 0.45] → [1, 0].
 */
function fingersTogetherness(lm: Landmark[]): number {
  const s   = handScale(lm);
  const avg = (
    dist(lm[INDEX.TIP], lm[MIDDLE.TIP]) +
    dist(lm[MIDDLE.TIP], lm[RING.TIP])  +
    dist(lm[RING.TIP],   lm[PINKY.TIP])
  ) / (3 * s);
  return clamp01((0.45 - avg) / (0.45 - 0.15));
}

function fourFingerExtension(lm: Landmark[]): number {
  return (
    fingerExtension(lm, INDEX)  +
    fingerExtension(lm, MIDDLE) +
    fingerExtension(lm, RING)   +
    fingerExtension(lm, PINKY)
  ) / 4;
}

function fourFingerCurl(lm: Landmark[]): number {
  return 1 - fourFingerExtension(lm);
}

// ─── hasta scorers (0..1) ────────────────────────────────────────────────────

/** Pataka: all four extended + together, thumb tucked. */
function scorePataka(lm: Landmark[]): number {
  return (
    fourFingerExtension(lm) * 0.45 +
    fingersTogetherness(lm) * 0.35 +
    (1 - thumbExtension(lm)) * 0.20
  );
}

/** Mushti: all four curled, thumb folded over. */
function scoreMushti(lm: Landmark[]): number {
  return (
    fourFingerCurl(lm)       * 0.75 +
    (1 - thumbExtension(lm)) * 0.25
  );
}

/**
 * Tripataka: index/middle/pinky extended + together; ring half-bent.
 * Key signal: ring lags clearly behind the other three (ringLagging).
 */
function scoreTripataka(lm: Landmark[]): number {
  const indExt = fingerExtension(lm, INDEX);
  const midExt = fingerExtension(lm, MIDDLE);
  const pinExt = fingerExtension(lm, PINKY);
  const ringExt = fingerExtension(lm, RING);
  const threeExt   = (indExt + midExt + pinExt) / 3;
  const ringBent   = 1 - ringExt;                        // reward ring being bent
  const ringLagging = clamp01(threeExt - ringExt);        // ring behind the others
  return (
    threeExt    * 0.35 +
    ringBent    * 0.25 +
    ringLagging * 0.20 +
    fingersTogetherness(lm)  * 0.12 +
    (1 - thumbExtension(lm)) * 0.08
  );
}

/**
 * Ardhachandra: all four extended + SPREAD; thumb extended laterally
 * (perpendicular, same plane as fingers — not strictly raised).
 */
function scoreArdhachandra(lm: Landmark[]): number {
  return (
    fourFingerExtension(lm)          * 0.45 +
    (1 - fingersTogetherness(lm))    * 0.35 + // spread = low togetherness
    thumbExtension(lm)               * 0.20
  );
}

/**
 * Shikhara: all four curled; thumb strictly raised UP
 * (tip clearly above IP joint in image space).
 */
function scoreShikhara(lm: Landmark[]): number {
  const s = handScale(lm);
  const thumbRaised = clamp01(
    (lm[THUMB.IP].y - lm[THUMB.TIP].y) / (s * 0.5),
  );
  return (
    fourFingerCurl(lm)  * 0.50 +
    thumbRaised         * 0.30 +
    thumbExtension(lm)  * 0.20
  );
}

/**
 * Kapittha: all four curled; thumb TIP pressing near the bent index TIP/DIP.
 * Disambiguates from Mushti by the thumb-near-index proximity check.
 */
function scoreKapittha(lm: Landmark[]): number {
  const s           = handScale(lm);
  const nearTip     = dist(lm[THUMB.TIP], lm[INDEX.TIP]) / s;
  const nearDIP     = dist(lm[THUMB.TIP], lm[INDEX.DIP]) / s;
  const thumbNearIndex = clamp01(1 - Math.min(nearTip, nearDIP) / 0.6);
  return (
    fourFingerCurl(lm) * 0.55 +
    thumbNearIndex     * 0.45
  );
}

// ─── registry & public classifier ────────────────────────────────────────────

const HASTA_SCORERS: Record<string, (lm: Landmark[]) => number> = {
  pataka:       scorePataka,
  tripataka:    scoreTripataka,
  ardhachandra: scoreArdhachandra,
  mushti:       scoreMushti,
  shikhara:     scoreShikhara,
  kapittha:     scoreKapittha,
};

const CONFIDENCE_FLOOR = 0.60;

/**
 * Classify a single hand.
 * @param landmarks  21 MediaPipe landmarks for ONE hand.
 * @param handedness "Left" | "Right" from MediaPipe (most rules are mirror-invariant).
 */
export function classifyHasta(
  landmarks: Landmark[],
  handedness?: string,
): ClassifyResult {
  void handedness; // reserved for future mirror-dependent rules
  if (!landmarks || landmarks.length < 21) {
    return { hasta: null, score: 0, allScores: {} };
  }

  const allScores: Record<string, number> = {};
  let best: { hasta: string | null; score: number } = { hasta: null, score: 0 };

  for (const [name, scorer] of Object.entries(HASTA_SCORERS)) {
    const s = clamp01(scorer(landmarks));
    allScores[name] = s;
    if (s > best.score) best = { hasta: name, score: s };
  }

  return best.score >= CONFIDENCE_FLOOR
    ? { hasta: best.hasta, score: best.score, allScores }
    : { hasta: null, score: best.score, allScores };
}

// Re-exported for debug overlay + future dataset validation
export {
  fingerExtension,
  thumbExtension,
  fingersTogetherness,
  fourFingerExtension,
  fourFingerCurl,
  handScale,
  HASTA_SCORERS,
};
