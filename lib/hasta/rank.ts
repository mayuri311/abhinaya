import type { HastaResult, HastaScore } from "./types";

export function rankClasses(
  scores: Float32Array,
  classes: string[],
  thresholds: { minTop1: number; minMargin: number },
  topn = 5,
): { top5: HastaScore[]; margin: number; lowConfidence: boolean } {
  const top5 = classes
    .map((label, i) => ({ label, score: scores[i] }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topn);
  const margin = top5[0].score - top5[1].score;
  const lowConfidence = top5[0].score < thresholds.minTop1 || margin < thresholds.minMargin;
  return { top5, margin, lowConfidence };
}

/** What the UI shows for one result: a committed label, "not sure", or no hand. */
export function voteKey(r: HastaResult): string {
  if (r.status === "no_hand") return "no_hand";
  return r.lowConfidence ? "unsure" : r.top5[0].label;
}

/**
 * Majority vote over the last N results (no_hand and unsure count as votes,
 * so dropping the hand clears the label within ~N/2 frames). Ties go to the
 * most recent. Returns the newest result carrying the winning vote.
 */
export function majorityVote(window: HastaResult[]): HastaResult | null {
  if (!window.length) return null;
  const counts = new Map<string, number>();
  for (const r of window) counts.set(voteKey(r), (counts.get(voteKey(r)) ?? 0) + 1);
  let best: HastaResult | null = null;
  let bestCount = 0;
  for (let i = window.length - 1; i >= 0; i--) {
    const c = counts.get(voteKey(window[i]))!;
    if (c > bestCount) {
      best = window[i];
      bestCount = c;
    }
  }
  return best;
}
