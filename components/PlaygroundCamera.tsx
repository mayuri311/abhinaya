"use client";

import { useEffect, useRef, useState } from "react";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import { analyzeHand, type FingerState, type HandGeometry, type Landmark } from "@/lib/handGeometry";
import { classifyHasta, type ClassifyResult } from "@/lib/hastaRules";
import { HASTA_CONFIG, MEDIAPIPE_WASM } from "@/lib/hasta/config";
import { useKnnHasta } from "@/lib/hasta/useKnnHasta";
import type { HastaResult } from "@/lib/hasta/types";
import MUDRAS_RAW from "@/data/mudras.json";
import HASTA_LABELS from "@/data/hastaLabels.json";

const MUDRAS = MUDRAS_RAW as Record<string, {
  name: string;
  translation: string;
  description: string;
  viniyoga: string[];
}>;

type Mudra = (typeof MUDRAS)[string];

const IS_KNN = HASTA_CONFIG.engine === "knn";

/** kNN model class ("Katakamukha_2") → mudras.json entry + display name. */
function mudraForClass(cls: string): { mudra: Mudra | null; title: string } {
  const mudra = MUDRAS[(HASTA_LABELS as Record<string, string>)[cls]] ?? null;
  const variant = /_(\d+)$/.exec(cls)?.[1];
  const base = mudra?.name ?? cls;
  return { mudra, title: variant ? `${base} ${variant}` : base };
}
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Connections grouped per finger for extension-based coloring
const FINGER_GROUPS: {
  key: keyof HandGeometry["fingers"];
  connections: [number, number][];
}[] = [
  { key: "thumb",  connections: [[0,1],[1,2],[2,3],[3,4]] },
  { key: "index",  connections: [[0,5],[5,6],[6,7],[7,8]] },
  { key: "middle", connections: [[0,9],[9,10],[10,11],[11,12]] },
  { key: "ring",   connections: [[0,13],[13,14],[14,15],[15,16]] },
  { key: "pinky",  connections: [[0,17],[17,18],[18,19],[19,20]] },
];
const PALM_CONN: [number, number][] = [[5,9],[9,13],[13,17],[0,17]];

const FINGERTIP_KEYS: [number, keyof HandGeometry["fingers"]][] = [
  [4, "thumb"], [8, "index"], [12, "middle"], [16, "ring"], [20, "pinky"],
];

// ─── canvas drawing ─────────────────────────────────────────────────────────

function skeletonColor(state: FingerState): string {
  if (state === "extended") return "rgba(196,154,42,0.85)";
  if (state === "half")     return "rgba(196,154,42,0.50)";
  return "rgba(196,154,42,0.18)";
}

function tipFill(state: FingerState): string {
  if (state === "extended") return "#FEFCF7";
  if (state === "half")     return "#C49A2A";
  return "#7B4A2A";
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  lm: Landmark[],
  w: number,
  h: number,
  fingers: HandGeometry["fingers"],
) {
  ctx.lineWidth = 1.5;
  for (const { key, connections } of FINGER_GROUPS) {
    ctx.strokeStyle = skeletonColor(fingers[key]);
    for (const [a, b] of connections) {
      ctx.beginPath();
      ctx.moveTo(lm[a].x * w, lm[a].y * h);
      ctx.lineTo(lm[b].x * w, lm[b].y * h);
      ctx.stroke();
    }
  }
  ctx.strokeStyle = "rgba(196,154,42,0.28)";
  ctx.lineWidth = 1;
  for (const [a, b] of PALM_CONN) {
    ctx.beginPath();
    ctx.moveTo(lm[a].x * w, lm[a].y * h);
    ctx.lineTo(lm[b].x * w, lm[b].y * h);
    ctx.stroke();
  }
}

function drawDots(
  ctx: CanvasRenderingContext2D,
  lm: Landmark[],
  w: number,
  h: number,
  fingers: HandGeometry["fingers"],
) {
  const tipMap = new Map(FINGERTIP_KEYS);

  lm.forEach((pt, i) => {
    const tipKey = tipMap.get(i);
    const isTip  = tipKey !== undefined;
    const state  = isTip ? fingers[tipKey!] : null;

    ctx.beginPath();
    ctx.arc(pt.x * w, pt.y * h, isTip ? 6 : 3.5, 0, Math.PI * 2);
    ctx.fillStyle = state ? tipFill(state) : "#7B4A2A";
    ctx.fill();
    if (isTip) {
      ctx.strokeStyle = "#7B4A2A";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  });
}

/** Square crop the kNN classifier embedded, in video pixels. */
function drawCropBox(ctx: CanvasRenderingContext2D, r: HastaResult | null) {
  if (r?.status !== "ok") return;
  const { x, y, w, h } = r.box;
  ctx.save();
  ctx.setLineDash([8, 6]);
  ctx.lineWidth = 2;
  ctx.strokeStyle = r.lowConfidence ? "rgba(254,252,247,0.55)" : "rgba(196,154,42,0.85)";
  ctx.strokeRect(x, y, w, h);
  ctx.restore();
}

// ─── component ──────────────────────────────────────────────────────────────

type Status = "loading" | "ready" | "requesting" | "running" | "denied" | "error";

const FINGER_LABELS: { key: keyof HandGeometry["fingers"]; label: string }[] = [
  { key: "thumb",  label: "Thumb"  },
  { key: "index",  label: "Index"  },
  { key: "middle", label: "Middle" },
  { key: "ring",   label: "Ring"   },
  { key: "pinky",  label: "Pinky"  },
];

export default function PlaygroundCamera() {
  const videoRef      = useRef<HTMLVideoElement>(null);
  const canvasRef     = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<HandLandmarker | null>(null);
  const rafRef        = useRef<number>(0);
  const frameCount    = useRef(0);

  const [status,         setStatus]         = useState<Status>("loading");
  const [errorMsg,       setErrorMsg]       = useState("");
  const [geometry,       setGeometry]       = useState<HandGeometry | null>(null);
  const [classification, setClassification] = useState<ClassifyResult | null>(null);
  const [showDebug,      setShowDebug]      = useState(false);

  const { latestRef: knnBoxRef, ...knn } = useKnnHasta(videoRef, IS_KNN && status === "running");

  // Load MediaPipe model
  useEffect(() => {
    let cancelled = false;
    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 1,
        });
        if (!cancelled) { landmarkerRef.current = landmarker; setStatus("ready"); }
      } catch (err) {
        if (!cancelled) { setStatus("error"); setErrorMsg(String(err)); }
      }
    }
    init();
    return () => { cancelled = true; };
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      const video = videoRef.current;
      if (video?.srcObject)
        (video.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      landmarkerRef.current?.close();
    };
  }, []);

  async function startCamera() {
    setStatus("requesting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      const video = videoRef.current!;
      video.srcObject = stream;
      await video.play();
      setStatus("running");
      runLoop();
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "NotAllowedError")
        setStatus("denied");
      else { setStatus("error"); setErrorMsg(String(err)); }
    }
  }

  function runLoop() {
    const video      = videoRef.current!;
    const canvas     = canvasRef.current!;
    const ctx        = canvas.getContext("2d")!;
    const landmarker = landmarkerRef.current!;

    function loop() {
      if (video.readyState >= 2 && video.videoWidth > 0) {
        const w = video.videoWidth;
        const h = video.videoHeight;
        canvas.width  = w;
        canvas.height = h;
        ctx.clearRect(0, 0, w, h);

        const result = landmarker.detectForVideo(video, performance.now());

        if (result.landmarks[0]) {
          const handedness =
            (result.handednesses[0]?.[0]?.categoryName as "Left" | "Right") ?? "Right";
          const geo    = analyzeHand(result.landmarks[0] as Landmark[], handedness);
          const classif = IS_KNN ? null : classifyHasta(result.landmarks[0] as Landmark[], handedness);

          drawSkeleton(ctx, result.landmarks[0] as Landmark[], w, h, geo.fingers);
          drawDots(ctx, result.landmarks[0] as Landmark[], w, h, geo.fingers);

          frameCount.current++;
          if (frameCount.current % 4 === 0) {
            setGeometry(geo);
            setClassification(classif);
          }
        } else {
          frameCount.current++;
          if (frameCount.current % 4 === 0) {
            setGeometry(null);
            setClassification(null);
          }
        }
        drawCropBox(ctx, knnBoxRef.current);
      }
      rafRef.current = requestAnimationFrame(loop);
    }

    rafRef.current = requestAnimationFrame(loop);
  }

  // What the overlay commits to: rules → its top hasta; kNN → the smoothed
  // top-1, unless the smoothed result is "no hand" or "not sure".
  const knnShown = knn.smoothed;
  const knnTop = knnShown?.status === "ok" && !knnShown.lowConfidence ? mudraForClass(knnShown.top5[0].label) : null;
  const activeMudra = IS_KNN ? knnTop?.mudra ?? null : classification?.hasta ? MUDRAS[classification.hasta] : null;
  const activeTitle = IS_KNN ? knnTop?.title : activeMudra?.name;
  const activeKey   = IS_KNN ? (knnShown?.status === "ok" ? knnShown.top5[0].label : "") : classification?.hasta ?? "";
  const activeScore = IS_KNN ? (knnShown?.status === "ok" ? knnShown.top5[0].score : 0) : classification?.score ?? 0;
  const knnUnsure   = IS_KNN && knnShown?.status === "ok" && knnShown.lowConfidence ? knnShown : null;
  const knnNoHand   = IS_KNN && knn.phase === "ready" && knnShown?.status === "no_hand";

  return (
    <div className="flex flex-col items-center gap-6">

      {/* ── Camera viewport ── */}
      <div
        className="relative w-full max-w-2xl rounded-sm overflow-hidden bg-ink/5"
        style={{ aspectRatio: "16/9" }}
      >
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover"
          style={{ transform: "scaleX(-1)" }}
          playsInline
          muted
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
          style={{ transform: "scaleX(-1)" }}
        />

        {/* Status screens */}
        {status !== "running" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-5 bg-cream/90">
            {status === "loading" && (
              <>
                <div className="w-1.5 h-1.5 rounded-full bg-henna/50 animate-pulse" />
                <p className="text-xs tracking-[0.22em] uppercase text-ink/40">Loading model…</p>
              </>
            )}
            {status === "ready" && (
              <button
                onClick={startCamera}
                className="text-xs tracking-[0.22em] uppercase border border-henna/30 text-henna px-10 py-3.5 hover:bg-henna hover:text-cream transition-all duration-700"
              >
                Start Camera
              </button>
            )}
            {status === "requesting" && (
              <p className="text-xs tracking-[0.22em] uppercase text-ink/40">Waiting for permission…</p>
            )}
            {status === "denied" && (
              <div className="text-center space-y-3">
                <p className="text-sm text-ink/60">Camera access was denied.</p>
                <p className="text-xs text-ink/35">Allow camera access in your browser settings, then reload.</p>
              </div>
            )}
            {status === "error" && (
              <div className="text-center space-y-3 px-8">
                <p className="text-sm text-ink/60">Something went wrong.</p>
                <p className="text-xs text-ink/35 font-mono break-all">{errorMsg}</p>
              </div>
            )}
          </div>
        )}

        {/* ── Google Lens overlay ── */}
        {status === "running" && activeMudra && (
          <div
            key={activeKey}
            className="absolute inset-0 pointer-events-none"
          >
            {/* Viniyoga bubbles — float in from top */}
            <div className="absolute top-4 left-4 right-4 flex flex-wrap gap-2">
              {activeMudra.viniyoga.map((use, i) => (
                <span
                  key={use}
                  className="px-3 py-1.5 text-[10px] tracking-[0.18em] uppercase bg-ink/50 backdrop-blur-sm border border-cream/[0.12] text-cream/70 animate-bubble-in"
                  style={{ animationDelay: `${i * 65}ms` }}
                >
                  {use}
                </span>
              ))}
            </div>

            {/* Bottom info panel */}
            <div className="absolute bottom-0 inset-x-0 animate-panel-in">
              {/* Gradient fade into panel */}
              <div className="h-10 bg-gradient-to-t from-ink/65 to-transparent" />
              <div className="bg-ink/62 backdrop-blur-md px-6 py-5 border-t border-gold/20 space-y-2">
                <div className="flex items-baseline justify-between">
                  <span className="font-heading text-[2rem] leading-none text-cream tracking-wide">
                    {activeTitle}
                  </span>
                  <span className="text-[10px] tracking-[0.26em] uppercase text-gold/75 ml-4 shrink-0">
                    {activeMudra.translation}
                  </span>
                </div>
                <p className="text-[11px] text-cream/50 leading-relaxed">
                  {activeMudra.description}
                </p>
                {/* Confidence bar */}
                <div className="flex items-center gap-3 pt-1">
                  <div className="flex-1 h-px bg-cream/10 relative overflow-hidden">
                    <div
                      className="absolute inset-y-0 left-0 bg-gold/55 transition-all duration-300"
                      style={{ width: `${Math.max(0, activeScore) * 100}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-gold/55 tabular-nums shrink-0">
                    {IS_KNN ? `sim ${activeScore.toFixed(2)}` : `${Math.round(activeScore * 100)}%`}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* kNN: not sure → show the top two instead of committing */}
        {status === "running" && knnUnsure && (
          <div key="unsure" className="absolute bottom-0 inset-x-0 pointer-events-none animate-panel-in">
            <div className="h-10 bg-gradient-to-t from-ink/55 to-transparent" />
            <div className="bg-ink/55 backdrop-blur-md px-6 py-4 border-t border-cream/10 space-y-2">
              <span className="text-[10px] tracking-[0.26em] uppercase text-cream/50">Not sure — one of</span>
              <div className="flex items-baseline gap-4">
                {knnUnsure.top5.slice(0, 2).map((c, i) => (
                  <span key={c.label} className="flex items-baseline gap-2">
                    {i > 0 && <span className="font-heading italic text-cream/35 text-lg">or</span>}
                    <span className="font-heading text-2xl leading-none text-cream/85">{mudraForClass(c.label).title}</span>
                    <span className="text-[10px] text-gold/55 tabular-nums">{c.score.toFixed(2)}</span>
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* kNN: no hand — a normal, frequent state, so say what to do */}
        {status === "running" && knnNoHand && (
          <div className="absolute bottom-5 inset-x-0 flex justify-center pointer-events-none">
            <span className="px-4 py-2 text-[10px] tracking-[0.22em] uppercase bg-ink/45 backdrop-blur-sm border border-cream/[0.12] text-cream/75">
              No hand detected — keep one hand in frame
            </span>
          </div>
        )}

        {/* kNN: model download / failure */}
        {status === "running" && IS_KNN && knn.phase !== "ready" && knn.phase !== "idle" && (
          <div className="absolute bottom-5 inset-x-0 flex justify-center pointer-events-none px-6">
            <span className="px-4 py-2 text-[10px] tracking-[0.22em] uppercase bg-ink/45 backdrop-blur-sm text-cream/70 break-all text-center">
              {knn.phase === "loading"
                ? `Loading mudra model · ${Math.round(knn.progress * 100)}%`
                : `Mudra model failed to load · ${knn.error}`}
            </span>
          </div>
        )}

        {/* Idle hint when running but no gesture */}
        {status === "running" && !activeMudra && (!IS_KNN || (knn.phase === "ready" && !knnUnsure && !knnNoHand)) && (
          <div className="absolute bottom-4 inset-x-0 flex justify-center pointer-events-none">
            <span className="text-[10px] tracking-[0.24em] uppercase text-cream/20">
              show a mudra
            </span>
          </div>
        )}
      </div>

      {/* ── Debug toggle ── */}
      {status === "running" && (
        <button
          onClick={() => setShowDebug((v) => !v)}
          className="text-[10px] tracking-[0.2em] uppercase text-ink/25 hover:text-ink/50 transition-colors duration-300 border-b border-ink/10 pb-px"
        >
          {showDebug ? "hide debug" : "debug"}
        </button>
      )}

      {/* ── Debug panels (hidden by default) ── */}
      {status === "running" && showDebug && (
        <div className="w-full max-w-2xl space-y-6 px-1">

          {/* kNN: top-5 + raw numbers */}
          {IS_KNN && knn.latest?.status === "ok" && (
            <div className="space-y-2">
              {knn.latest.top5.map(({ label, score }) => (
                <div key={label} className="flex items-center gap-3">
                  <span className="text-xs tracking-[0.1em] uppercase text-ink/30 w-28 flex-shrink-0">
                    {mudraForClass(label).title}
                  </span>
                  <div className="flex-1 h-px bg-ink/8 relative overflow-hidden">
                    <div
                      className="absolute inset-y-0 left-0 bg-henna/40 transition-all duration-300"
                      style={{ width: `${Math.max(0, score) * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-ink/30 w-9 text-right tabular-nums">
                    {score.toFixed(3)}
                  </span>
                </div>
              ))}
              <p className="text-[10px] tracking-[0.14em] uppercase text-ink/30 pt-1">
                margin {knn.latest.margin.toFixed(3)} · {knn.latest.handedness} {knn.latest.detectorScore.toFixed(2)} ·{" "}
                {Math.round(knn.latest.latencyMs)} ms · {knn.backend}
                {knn.latest.lowConfidence && " · low confidence"}
              </p>
            </div>
          )}
          {IS_KNN && knn.latest?.status === "no_hand" && (
            <p className="text-[10px] tracking-[0.14em] uppercase text-ink/30">
              no hand · {Math.round(knn.latest.latencyMs)} ms · {knn.backend}
            </p>
          )}

          {/* All scores */}
          {classification?.allScores && (
            <div className="space-y-2">
              {Object.entries(classification.allScores)
                .sort(([, a], [, b]) => b - a)
                .map(([name, score]) => (
                  <div key={name} className="flex items-center gap-3">
                    <span className="text-xs tracking-[0.1em] uppercase text-ink/30 w-28 flex-shrink-0">
                      {name}
                    </span>
                    <div className="flex-1 h-px bg-ink/8 relative overflow-hidden">
                      <div
                        className="absolute inset-y-0 left-0 bg-henna/40 transition-all duration-300"
                        style={{ width: `${score * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-ink/30 w-7 text-right tabular-nums">
                      {Math.round(score * 100)}
                    </span>
                  </div>
                ))}
            </div>
          )}

          {/* Finger state chips */}
          <div className="space-y-4">
            <div className="flex items-center gap-2 flex-wrap">
              {FINGER_LABELS.map(({ key, label }) => {
                const state = geometry?.fingers[key] ?? "curled";
                const sub   = key === "thumb" && geometry ? ` · ${geometry.thumbState}` : "";
                return (
                  <span
                    key={key}
                    className={`px-3 py-1.5 text-xs tracking-[0.14em] uppercase transition-all duration-200 ${
                      state === "extended"
                        ? "bg-henna text-cream"
                        : state === "half"
                          ? "border border-henna/50 text-gold"
                          : "border border-henna/15 text-ink/25"
                    }`}
                  >
                    {label}{sub}
                  </span>
                );
              })}
              <span className="ml-auto text-xs tracking-[0.14em] uppercase text-ink/25">
                {geometry?.handedness ?? "—"} hand
              </span>
            </div>

            <div className="flex items-center gap-6 text-xs text-ink/35">
              <span className="tracking-[0.16em] uppercase">Spread</span>
              {geometry ? (
                <>
                  <span>i↔m <span className="text-henna">{geometry.spreads.indexMiddle}</span></span>
                  <span>m↔r <span className="text-henna">{geometry.spreads.middleRing}</span></span>
                  <span>r↔p <span className="text-henna">{geometry.spreads.ringPinky}</span></span>
                </>
              ) : (
                <span className="text-ink/20">— no hand detected —</span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
