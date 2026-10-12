export type HastaScore = { label: string; score: number };

/** Crop box in original-image pixels (may extend past the image edges). */
export type HastaBox = { x: number; y: number; w: number; h: number };

export type HastaResult =
  | { status: "no_hand"; latencyMs: number }
  | {
      status: "ok";
      /** Classes ranked by max cosine similarity to their bank entries. */
      top5: HastaScore[];
      /** top1.score − top2.score */
      margin: number;
      lowConfidence: boolean;
      box: HastaBox;
      handedness: "Left" | "Right";
      detectorScore: number;
      latencyMs: number;
    };

/** knn_bank_meta.json */
export type HastaMeta = {
  classes: string[];
  k: number;
  temp: number;
  embedding: string;
  crop: {
    size: number;
    margin: number;
    detect_pad: number;
    pad_fill: [number, number, number];
    resize: string;
    detector: string;
  };
  normalize: { mean: [number, number, number]; std: [number, number, number] };
  input: string;
};
