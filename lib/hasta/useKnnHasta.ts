"use client";

import { useEffect, useRef, useState, type RefObject } from "react";
import { HASTA_CONFIG } from "./config";
import { createKnnClassifier, type KnnClassifier } from "./knnClassifier";
import { majorityVote } from "./rank";
import type { HastaResult } from "./types";

export type KnnPhase = "idle" | "loading" | "ready" | "error";

/**
 * Lazily loads the kNN classifier once `enabled` turns on, then classifies the
 * video in a self-paced loop: the next frame is only taken after the previous
 * result is back (and at least minIntervalMs later), so it never queues up.
 */
export function useKnnHasta(videoRef: RefObject<HTMLVideoElement | null>, enabled: boolean) {
  const [phase, setPhase] = useState<KnnPhase>("idle");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const [backend, setBackend] = useState("");
  const [latest, setLatest] = useState<HastaResult | null>(null);
  const [smoothed, setSmoothed] = useState<HastaResult | null>(null);
  /** Latest raw result, for drawing the crop box at frame rate. */
  const latestRef = useRef<HastaResult | null>(null);

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let clf: KnnClassifier | null = null;
    const recent: HastaResult[] = [];

    (async () => {
      setPhase("loading");
      try {
        clf = await createKnnClassifier({ onProgress: (l, t) => setProgress(t ? l / t : 0) });
        if (cancelled) return clf.close();
        setBackend(clf.executionProvider);
        setPhase("ready");
        while (!cancelled) {
          const t0 = performance.now();
          const video = videoRef.current;
          if (video && video.readyState >= 2 && video.videoWidth > 0) {
            const r = await clf.classify(video);
            if (cancelled) break;
            recent.push(r);
            if (recent.length > HASTA_CONFIG.smoothingWindow) recent.shift();
            latestRef.current = r;
            setLatest(r);
            setSmoothed(majorityVote(recent));
          }
          const wait = HASTA_CONFIG.minIntervalMs - (performance.now() - t0);
          await new Promise((res) => setTimeout(res, Math.max(0, wait)));
        }
      } catch (err) {
        if (!cancelled) {
          setPhase("error");
          setError(String(err));
        }
      }
    })();

    return () => {
      cancelled = true;
      latestRef.current = null;
      clf?.close();
    };
  }, [enabled, videoRef]);

  return { phase, progress, error, backend, latest, smoothed, latestRef };
}
