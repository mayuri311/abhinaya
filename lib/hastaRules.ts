/**
 * hastaRules.ts
 * Rule-based classifier for the 28 single-hand (asamyukta) hastas of
 * Bharatanatyam. Self-contained — only imports the shared Landmark type.
 * Add new hastas to HASTA_SCORERS to extend the classifier.
 *
 * Public interface (the ONLY thing the playground should call):
 *   classifyHasta(landmarks, handedness) → ClassifyResult
 *
 * Some hastas are inherently hard to disambiguate from 2D landmarks alone
 * (e.g. Pataka vs Sarpasirsha differ only by palm cupping; Alapadma vs
 * Ardhachandra differ mainly in degree of spread). Their scorers share
 * features but lean on whichever cue is most distinguishing.
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
const ALL_FINGERS: FingerIdx[] = [INDEX, MIDDLE, RING, PINKY];

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

/** Normalised distance between two landmarks. */
function ndist(lm: Landmark[], a: number, b: number): number {
  return dist(lm[a], lm[b]) / handScale(lm);
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

/**
 * z-depth boost for a curled finger. When the tip is behind the MCP in depth
 * (TIP.z > MCP.z in MediaPipe coords where larger z = further from camera),
 * the finger is likely truly curled even if 2D angle is ambiguous (occlusion).
 * Returns 0..1 extra curl evidence, capped conservatively so it doesn't
 * overwhelm the angle signal.
 */
function fingerCurlZ(lm: Landmark[], f: FingerIdx): number {
  const s    = handScale(lm);
  const zDiff = ((lm[f.TIP].z ?? 0) - (lm[f.MCP].z ?? 0)) / s;
  // Positive zDiff = tip is deeper = stronger curl evidence
  return clamp01(zDiff / 0.25);
}

/**
 * fourFingerCurl enhanced with z-depth: if fingertips are behind their MCPs
 * in 3D space, we trust the curl even when the 2D angle looks ambiguous.
 * The z signal is blended lightly (max +0.15 boost) to avoid over-reliance
 * on noisy depth estimates.
 */
function fourFingerCurl(lm: Landmark[]): number {
  const angleCurl = 1 - fourFingerExtension(lm);
  const zBoost = (
    fingerCurlZ(lm, INDEX)  +
    fingerCurlZ(lm, MIDDLE) +
    fingerCurlZ(lm, RING)   +
    fingerCurlZ(lm, PINKY)
  ) / 4;
  return clamp01(angleCurl + zBoost * 0.15);
}

// ─── extended geometry helpers (added for the full 28-hasta set) ─────────────

/**
 * 0..1 pinch score between two fingertips/landmarks.
 * Close (≤ near) = 1, far (≥ far) = 0. Both bounds are normalised distances.
 */
function pinch(
  lm: Landmark[],
  a: number,
  b: number,
  near = 0.12,
  far  = 0.50,
): number {
  const d = ndist(lm, a, b);
  return clamp01((far - d) / (far - near));
}

/**
 * 0..1 spread score between two fingertips: far apart = 1, touching = 0.
 * Useful for Kartarimukha (wide spread between index & middle) and similar.
 */
function pairSpread(
  lm: Landmark[],
  tipA: number,
  tipB: number,
  near = 0.15,
  far  = 0.45,
): number {
  const d = ndist(lm, tipA, tipB);
  return clamp01((d - near) / (far - near));
}

/**
 * 0..1 score for how perpendicular the thumb is to the index finger
 * (thumb CMC→TIP vs index MCP→TIP). 90° = 1, parallel/opposite = 0.
 * Distinguishes Chandrakala (thumb at right angle) from Suchi (thumb tucked).
 */
function thumbPerpendicular(lm: Landmark[]): number {
  const v1 = sub(lm[THUMB.TIP], lm[THUMB.CMC]);
  const v2 = sub(lm[INDEX.TIP], lm[INDEX.MCP]);
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const mag = Math.sqrt(
    (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z) *
    (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z),
  );
  if (mag === 0) return 0;
  const angleDeg = deg(Math.acos(Math.max(-1, Math.min(1, dot / mag))));
  return clamp01(1 - Math.abs(angleDeg - 90) / 90);
}

/**
 * 0..1 score for how raised the thumb tip is above the IP joint (image y
 * decreases upward in MediaPipe normalised coordinates).
 */
function thumbRaised(lm: Landmark[]): number {
  const s = handScale(lm);
  return clamp01((lm[THUMB.IP].y - lm[THUMB.TIP].y) / (s * 0.5));
}

/**
 * 0..1 score for all five fingertips bunched together at one point (Mukula).
 * Averages the pairwise normalised distance across all 10 tip pairs.
 */
function fingertipBunching(lm: Landmark[]): number {
  const s = handScale(lm);
  const tips = [THUMB.TIP, INDEX.TIP, MIDDLE.TIP, RING.TIP, PINKY.TIP];
  let sum = 0, n = 0;
  for (let i = 0; i < tips.length; i++) {
    for (let j = i + 1; j < tips.length; j++) {
      sum += dist(lm[tips[i]], lm[tips[j]]) / s;
      n++;
    }
  }
  const avg = sum / n;
  return clamp01((0.50 - avg) / (0.50 - 0.15));
}

/**
 * 0..1 "half-bent" peak: returns 1 when the finger sits at the target
 * extension level, falling off linearly toward 0 at fully extended/curled.
 * Used by hastas where fingers are loosely curved (Padmakosha, Aralam).
 */
function fingerHalfBent(lm: Landmark[], f: FingerIdx, target = 0.5): number {
  return clamp01(1 - Math.abs(fingerExtension(lm, f) - target) * 2);
}

// ─── hasta scorers (0..1) ────────────────────────────────────────────────────

// Each scorer returns a 0..1 score. Weights inside each function sum to ~1.0
// so the score reads as a confidence. The classifier picks the highest score
// above CONFIDENCE_FLOOR.

/** Pataka: all four extended + together, thumb tucked. */
function scorePataka(lm: Landmark[]): number {
  return (
    fourFingerExtension(lm)  * 0.45 +
    fingersTogetherness(lm)  * 0.35 +
    (1 - thumbExtension(lm)) * 0.20
  );
}

/**
 * Tripataka: index/middle/pinky extended + together; ring folded; thumb TUCKED.
 * Key discriminator from Kangula: thumb must be tucked (not out to the side).
 */
function scoreTripataka(lm: Landmark[]): number {
  const indExt  = fingerExtension(lm, INDEX);
  const midExt  = fingerExtension(lm, MIDDLE);
  const pinExt  = fingerExtension(lm, PINKY);
  const ringExt = fingerExtension(lm, RING);
  const threeExt    = (indExt + midExt + pinExt) / 3;
  const ringBent    = 1 - ringExt;
  const ringLagging = clamp01(threeExt - ringExt);
  return (
    threeExt    * 0.26 +
    ringBent    * 0.22 +
    ringLagging * 0.15 +
    fingersTogetherness(lm)  * 0.15 +
    (1 - thumbExtension(lm)) * 0.22   // thumb tucked is the main Kangula separator
  );
}

/** Ardhapataka: only index + middle extended, held together; ring + pinky curled. */
function scoreArdhapataka(lm: Landmark[]): number {
  const indExt    = fingerExtension(lm, INDEX);
  const midExt    = fingerExtension(lm, MIDDLE);
  const ringCurl  = 1 - fingerExtension(lm, RING);
  const pinkyCurl = 1 - fingerExtension(lm, PINKY);
  const imClose   = 1 - pairSpread(lm, INDEX.TIP, MIDDLE.TIP);
  return (
    indExt    * 0.20 +
    midExt    * 0.20 +
    ringCurl  * 0.20 +
    pinkyCurl * 0.20 +
    imClose                  * 0.12 +
    (1 - thumbExtension(lm)) * 0.08
  );
}

/** Kartarimukha: index + middle extended and SPREAD WIDE; ring + pinky curled. */
function scoreKartarimukha(lm: Landmark[]): number {
  const indExt    = fingerExtension(lm, INDEX);
  const midExt    = fingerExtension(lm, MIDDLE);
  const ringCurl  = 1 - fingerExtension(lm, RING);
  const pinkyCurl = 1 - fingerExtension(lm, PINKY);
  const imSpread  = pairSpread(lm, INDEX.TIP, MIDDLE.TIP, 0.20, 0.55);
  return (
    indExt    * 0.17 +
    midExt    * 0.17 +
    ringCurl  * 0.17 +
    pinkyCurl * 0.17 +
    imSpread                 * 0.25 +
    (1 - thumbExtension(lm)) * 0.07
  );
}

/** Mayura: ring tip touches thumb tip; index/middle/pinky extended. */
function scoreMayura(lm: Landmark[]): number {
  return (
    pinch(lm, RING.TIP, THUMB.TIP) * 0.40 +
    fingerExtension(lm, INDEX)     * 0.20 +
    fingerExtension(lm, MIDDLE)    * 0.20 +
    fingerExtension(lm, PINKY)     * 0.20
  );
}

/**
 * Ardhachandra: all four extended + SPREAD; thumb extended laterally.
 * Gate: spread and thumb only score if all four fingers are near-fully
 * extended (≥ ~0.65 avg). Below that, the score can't clear the floor.
 */
function scoreArdhachandra(lm: Landmark[]): number {
  const ext     = fourFingerExtension(lm);
  // Ramps 0→1 over [0.60, 0.85] average extension — loose fingers kill the score
  const extGate = clamp01((ext - 0.60) / 0.25);
  const spread  = 1 - fingersTogetherness(lm);
  return (
    extGate * (
      spread             * 0.55 +
      thumbExtension(lm) * 0.25
    ) +
    ext * 0.20
  );
}

/** Arala: index curved at the tip; middle + ring + pinky extended. */
function scoreArala(lm: Landmark[]): number {
  const indHalf  = fingerHalfBent(lm, INDEX, 0.5);
  const midExt   = fingerExtension(lm, MIDDLE);
  const ringExt  = fingerExtension(lm, RING);
  const pinkyExt = fingerExtension(lm, PINKY);
  return (
    indHalf  * 0.30 +
    midExt   * 0.23 +
    ringExt  * 0.23 +
    pinkyExt * 0.24
  );
}

/** Shukatunda: Arala + ring also folded. Index curved, mid + pinky extended. */
function scoreShukatunda(lm: Landmark[]): number {
  const indHalf  = fingerHalfBent(lm, INDEX, 0.4);
  const midExt   = fingerExtension(lm, MIDDLE);
  const ringCurl = 1 - fingerExtension(lm, RING);
  const pinkyExt = fingerExtension(lm, PINKY);
  return (
    indHalf  * 0.25 +
    midExt   * 0.25 +
    ringCurl * 0.25 +
    pinkyExt * 0.25
  );
}

/** Mushti: all four curled, thumb folded over. */
function scoreMushti(lm: Landmark[]): number {
  return (
    fourFingerCurl(lm)       * 0.75 +
    (1 - thumbExtension(lm)) * 0.25
  );
}

/** Shikhara: all four curled; thumb raised straight UP. */
function scoreShikhara(lm: Landmark[]): number {
  return (
    fourFingerCurl(lm) * 0.50 +
    thumbRaised(lm)    * 0.30 +
    thumbExtension(lm) * 0.20
  );
}

/** Kapittha: all four curled; thumb tip pressing the bent index tip/DIP. */
function scoreKapittha(lm: Landmark[]): number {
  const s              = handScale(lm);
  const nearTip        = dist(lm[THUMB.TIP], lm[INDEX.TIP]) / s;
  const nearDIP        = dist(lm[THUMB.TIP], lm[INDEX.DIP]) / s;
  const thumbNearIndex = clamp01(1 - Math.min(nearTip, nearDIP) / 0.6);
  return (
    fourFingerCurl(lm) * 0.55 +
    thumbNearIndex     * 0.45
  );
}

/** Katakamukha: index + middle tips meet thumb tip; ring + pinky curled. */
function scoreKatakamukha(lm: Landmark[]): number {
  return (
    pinch(lm, INDEX.TIP,  THUMB.TIP) * 0.30 +
    pinch(lm, MIDDLE.TIP, THUMB.TIP) * 0.30 +
    (1 - fingerExtension(lm, RING))  * 0.20 +
    (1 - fingerExtension(lm, PINKY)) * 0.20
  );
}

/** Suchi: index extended; mid + ring + pinky curled; thumb tucked. */
function scoreSuchi(lm: Landmark[]): number {
  return (
    fingerExtension(lm, INDEX)        * 0.30 +
    (1 - fingerExtension(lm, MIDDLE)) * 0.22 +
    (1 - fingerExtension(lm, RING))   * 0.22 +
    (1 - fingerExtension(lm, PINKY))  * 0.18 +
    (1 - thumbExtension(lm))          * 0.08
  );
}

/** Chandrakala: index extended + thumb extended PERPENDICULAR; others curled. */
function scoreChandrakala(lm: Landmark[]): number {
  return (
    fingerExtension(lm, INDEX)        * 0.22 +
    (1 - fingerExtension(lm, MIDDLE)) * 0.15 +
    (1 - fingerExtension(lm, RING))   * 0.15 +
    (1 - fingerExtension(lm, PINKY))  * 0.13 +
    thumbExtension(lm)                * 0.15 +
    thumbPerpendicular(lm)            * 0.20
  );
}

/** Padmakosha: all five fingers half-bent and spread. */
function scorePadmakosha(lm: Landmark[]): number {
  const halfAvg = (
    fingerHalfBent(lm, INDEX,  0.55) +
    fingerHalfBent(lm, MIDDLE, 0.55) +
    fingerHalfBent(lm, RING,   0.55) +
    fingerHalfBent(lm, PINKY,  0.55)
  ) / 4;
  return (
    halfAvg                       * 0.50 +
    (1 - fingersTogetherness(lm)) * 0.30 +
    thumbExtension(lm)            * 0.20
  );
}

/**
 * Sarpasirsha: Pataka with palm slightly cupped (fingers mostly extended but
 * NOT all the way), held together, thumb tucked. Differentiator from Pataka
 * is the partial curl that registers below "fully extended".
 */
function scoreSarpasirsha(lm: Landmark[]): number {
  const slightCurl = (
    fingerHalfBent(lm, INDEX,  0.75) +
    fingerHalfBent(lm, MIDDLE, 0.75) +
    fingerHalfBent(lm, RING,   0.75) +
    fingerHalfBent(lm, PINKY,  0.75)
  ) / 4;
  return (
    slightCurl               * 0.35 +
    fingersTogetherness(lm)  * 0.30 +
    (1 - thumbExtension(lm)) * 0.20 +
    fourFingerExtension(lm)  * 0.15
  );
}

/** Mrigasirsha: thumb + pinky stretched out, middle three softly bent. */
function scoreMrigasirsha(lm: Landmark[]): number {
  const indHalf   = fingerHalfBent(lm, INDEX,  0.55);
  const midHalf   = fingerHalfBent(lm, MIDDLE, 0.55);
  const ringHalf  = fingerHalfBent(lm, RING,   0.55);
  const pinkyExt  = fingerExtension(lm, PINKY);
  const thumbExt  = thumbExtension(lm);
  return (
    indHalf  * 0.18 +
    midHalf  * 0.18 +
    ringHalf * 0.18 +
    pinkyExt * 0.22 +
    thumbExt * 0.24
  );
}

/** Simhamukha: middle + ring tips meet thumb tip; index + pinky extended. */
function scoreSimhamukha(lm: Landmark[]): number {
  return (
    pinch(lm, MIDDLE.TIP, THUMB.TIP) * 0.28 +
    pinch(lm, RING.TIP,   THUMB.TIP) * 0.28 +
    fingerExtension(lm, INDEX)       * 0.22 +
    fingerExtension(lm, PINKY)       * 0.22
  );
}

/**
 * Kangula: ring curled; index + middle + pinky extended; thumb OUT to the side.
 * Key discriminator from Tripataka: thumb must be extended (not tucked).
 * Adding a thumbTucked penalty means a tucked thumb tanks the score.
 */
function scoreKangula(lm: Landmark[]): number {
  const indExt   = fingerExtension(lm, INDEX);
  const midExt   = fingerExtension(lm, MIDDLE);
  const pinkyExt = fingerExtension(lm, PINKY);
  const ringExt  = fingerExtension(lm, RING);
  const ringCurl     = 1 - ringExt;
  const ringIsolated = clamp01(((indExt + midExt + pinkyExt) / 3) - ringExt);
  const thumbOut     = thumbExtension(lm);
  const thumbTucked  = 1 - thumbOut; // penalty: if thumb is tucked it's Tripataka, not Kangula
  return (
    ringCurl       * 0.20 +
    ringIsolated   * 0.15 +
    indExt         * 0.12 +
    midExt         * 0.12 +
    pinkyExt       * 0.12 +
    thumbOut       * 0.25 +               // thumb out is the main Tripataka separator
    (1 - thumbTucked * 0.5) * 0.04       // soft penalty when thumb is tucked
  );
}

/**
 * Alapadma: all five extended + WIDELY spread, more open than Ardhachandra.
 * The "veryWide" term gives extra weight only past a high spread threshold.
 */
function scoreAlapadma(lm: Landmark[]): number {
  const spread   = 1 - fingersTogetherness(lm);
  const veryWide = clamp01((spread - 0.6) / 0.4);
  return (
    fourFingerExtension(lm) * 0.35 +
    spread                  * 0.20 +
    veryWide                * 0.20 +
    thumbExtension(lm)      * 0.25
  );
}

/**
 * Chatura: four fingers extended together, thumb folded tightly across the
 * MIDDLE of the palm (toward middle/ring MCP base area).
 * Differentiator from Pataka: the thumb crosses noticeably past the index base.
 * We measure thumb proximity to both the middle and ring MCPs and take the max.
 */
function scoreChatura(lm: Landmark[]): number {
  // Thumb crosses the palm — it lands near the middle or ring finger base
  const thumbToMid  = pinch(lm, THUMB.TIP, MIDDLE.MCP, 0.08, 0.55);
  const thumbToRing = pinch(lm, THUMB.TIP, RING.MCP,   0.08, 0.55);
  const thumbAcross = Math.max(thumbToMid, thumbToRing);
  return (
    fourFingerExtension(lm)  * 0.30 +
    fingersTogetherness(lm)  * 0.20 +
    (1 - thumbExtension(lm)) * 0.20 +
    thumbAcross              * 0.30
  );
}

/** Bhramara: middle tip meets thumb tip; index bent down; ring + pinky out. */
function scoreBhramara(lm: Landmark[]): number {
  return (
    pinch(lm, MIDDLE.TIP, THUMB.TIP) * 0.30 +
    (1 - fingerExtension(lm, INDEX)) * 0.20 +
    fingerExtension(lm, RING)        * 0.25 +
    fingerExtension(lm, PINKY)       * 0.25
  );
}

/** Hamsasya: thumb + index tip pinch; middle + ring + pinky extended. */
function scoreHamsasya(lm: Landmark[]): number {
  return (
    pinch(lm, INDEX.TIP, THUMB.TIP) * 0.35 +
    fingerExtension(lm, MIDDLE)     * 0.22 +
    fingerExtension(lm, RING)       * 0.22 +
    fingerExtension(lm, PINKY)      * 0.21
  );
}

/** Hamsapaksha: Pataka shape but with thumb extended sideways (not tucked). */
function scoreHamsapaksha(lm: Landmark[]): number {
  return (
    fourFingerExtension(lm) * 0.40 +
    fingersTogetherness(lm) * 0.25 +
    thumbExtension(lm)      * 0.35
  );
}

/** Mukula: all five fingertips drawn together at one point. */
function scoreMukula(lm: Landmark[]): number {
  return (
    fingertipBunching(lm)            * 0.55 +
    (1 - fourFingerExtension(lm))    * 0.20 +
    pinch(lm, INDEX.TIP, THUMB.TIP)  * 0.15 +
    (1 - thumbExtension(lm))         * 0.10
  );
}

/**
 * Tamrachuda: Mushti-like with the thumb pressed against the side of the
 * bent index and slightly raised. Differentiator from Mushti: thumb tip
 * sits near the index PIP/DIP (not tucked across all four fingers).
 */
function scoreTamrachuda(lm: Landmark[]): number {
  const otherCurl = (
    (1 - fingerExtension(lm, MIDDLE)) +
    (1 - fingerExtension(lm, RING))   +
    (1 - fingerExtension(lm, PINKY))
  ) / 3;
  return (
    (1 - fingerExtension(lm, INDEX))    * 0.20 +
    otherCurl                           * 0.30 +
    thumbRaised(lm)                     * 0.25 +
    pinch(lm, THUMB.TIP, INDEX.PIP, 0.10, 0.45) * 0.25
  );
}

/** Trishula: index + middle + ring extended together; pinky curled; thumb tucked. */
function scoreTrishula(lm: Landmark[]): number {
  const tripleTogether = 1 - (
    pairSpread(lm, INDEX.TIP, MIDDLE.TIP) +
    pairSpread(lm, MIDDLE.TIP, RING.TIP)
  ) / 2;
  return (
    fingerExtension(lm, INDEX)       * 0.20 +
    fingerExtension(lm, MIDDLE)      * 0.20 +
    fingerExtension(lm, RING)        * 0.20 +
    (1 - fingerExtension(lm, PINKY)) * 0.25 +
    tripleTogether                   * 0.10 +
    (1 - thumbExtension(lm))         * 0.05
  );
}

// ─── registry & public classifier ────────────────────────────────────────────

const HASTA_SCORERS: Record<string, (lm: Landmark[]) => number> = {
  pataka:       scorePataka,
  tripataka:    scoreTripataka,
  ardhapataka:  scoreArdhapataka,
  kartarimukha: scoreKartarimukha,
  mayura:       scoreMayura,
  ardhachandra: scoreArdhachandra,
  arala:        scoreArala,
  shukatunda:   scoreShukatunda,
  mushti:       scoreMushti,
  shikhara:     scoreShikhara,
  kapittha:     scoreKapittha,
  katakamukha:  scoreKatakamukha,
  suchi:        scoreSuchi,
  chandrakala:  scoreChandrakala,
  padmakosha:   scorePadmakosha,
  sarpasirsha:  scoreSarpasirsha,
  mrigasirsha:  scoreMrigasirsha,
  simhamukha:   scoreSimhamukha,
  kangula:      scoreKangula,
  alapadma:     scoreAlapadma,
  chatura:      scoreChatura,
  bhramara:     scoreBhramara,
  hamsasya:     scoreHamsasya,
  hamsapaksha:  scoreHamsapaksha,
  mukula:       scoreMukula,
  tamrachuda:   scoreTamrachuda,
  trishula:     scoreTrishula,
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
  pinch,
  pairSpread,
  thumbPerpendicular,
  thumbRaised,
  fingertipBunching,
  fingerHalfBent,
  HASTA_SCORERS,
};

// ALL_FINGERS is exported for reference; suppresses unused-binding lint until used.
void ALL_FINGERS;
