export type Landmark  = { x: number; y: number; z: number };
export type FingerState = "extended" | "half" | "curled";
export type ThumbState  = "raised" | "extended" | "tucked";

export type HandGeometry = {
  handedness: "Left" | "Right";
  fingers: {
    thumb:  FingerState;
    index:  FingerState;
    middle: FingerState;
    ring:   FingerState;
    pinky:  FingerState;
  };
  thumbState: ThumbState;
  spreads: {
    indexMiddle: number; // × 100, 1 dp
    middleRing:  number;
    ringPinky:   number;
  };
};

// ─── helpers ────────────────────────────────────────────────────────────────

function dist2D(a: Landmark, b: Landmark): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

// Angle at `vertex` in the chain a–vertex–c, using all three x/y/z axes.
// Returns radians: π = straight, π/2 = 90° bend, 0 = fully folded back.
function jointAngle(a: Landmark, vertex: Landmark, c: Landmark): number {
  const v1 = { x: a.x - vertex.x, y: a.y - vertex.y, z: a.z - vertex.z };
  const v2 = { x: c.x - vertex.x, y: c.y - vertex.y, z: c.z - vertex.z };
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const mag = Math.sqrt(
    (v1.x ** 2 + v1.y ** 2 + v1.z ** 2) *
    (v2.x ** 2 + v2.y ** 2 + v2.z ** 2),
  );
  return mag === 0 ? Math.PI : Math.acos(Math.min(1, Math.max(-1, dot / mag)));
}

// Classify a finger using the minimum of its PIP and DIP joint angles.
// Taking the minimum means if EITHER joint bends clearly, we catch it —
// this is especially important for the pinky where short segments make the
// PIP angle noisy and the DIP angle often gives a stronger curl signal.
//
// Thresholds (tune here if readings feel off):
//   > 2.5 rad (≈143°) → extended
//   > 1.7 rad (≈ 97°) → half
//   ≤ 1.7 rad          → curled
const EXT_THRESHOLD  = 2.5;
const HALF_THRESHOLD = 1.7;

function fingerState(
  lm: Landmark[],
  mcpIdx: number,
  pipIdx: number,
  dipIdx: number,
  tipIdx: number,
): FingerState {
  const pipAngle = jointAngle(lm[mcpIdx], lm[pipIdx], lm[dipIdx]);
  const dipAngle = jointAngle(lm[pipIdx], lm[dipIdx], lm[tipIdx]);
  const angle    = Math.min(pipAngle, dipAngle); // most-bent joint wins
  if (angle > EXT_THRESHOLD)  return "extended";
  if (angle > HALF_THRESHOLD) return "half";
  return "curled";
}

// Finger data: [mcp, pip, dip, tip]
const FINGER_JOINTS: [number, number, number, number][] = [
  [5,  6,  7,  8],   // index
  [9,  10, 11, 12],  // middle
  [13, 14, 15, 16],  // ring
  [17, 18, 19, 20],  // pinky
];

function getThumbState(lm: Landmark[]): ThumbState {
  if (lm[4].y < lm[3].y) return "raised"; // tip above IP joint
  const palmWidth   = dist2D(lm[5], lm[17]);
  const thumbSpread = dist2D(lm[4], lm[5]);
  if (thumbSpread > palmWidth * 0.5) return "extended";
  return "tucked";
}

function thumbStateToFingerState(ts: ThumbState): FingerState {
  return ts === "tucked" ? "curled" : "extended";
}

// ─── public API ─────────────────────────────────────────────────────────────

export function analyzeHand(
  landmarks: Landmark[],
  handedness: "Left" | "Right",
): HandGeometry {
  const thumbState = getThumbState(landmarks);
  const [index, middle, ring, pinky] = FINGER_JOINTS.map(
    ([mcp, pip, dip, tip]) => fingerState(landmarks, mcp, pip, dip, tip),
  );
  const round1 = (n: number) => Math.round(n * 1000) / 10;

  return {
    handedness,
    fingers: {
      thumb: thumbStateToFingerState(thumbState),
      index:  index!,
      middle: middle!,
      ring:   ring!,
      pinky:  pinky!,
    },
    thumbState,
    spreads: {
      indexMiddle: round1(dist2D(landmarks[8],  landmarks[12])),
      middleRing:  round1(dist2D(landmarks[12], landmarks[16])),
      ringPinky:   round1(dist2D(landmarks[16], landmarks[20])),
    },
  };
}
