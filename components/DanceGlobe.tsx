"use client";

import { useEffect, useRef, useState } from "react";
import { geoOrthographic, geoPath, geoGraticule10, geoDistance } from "d3-geo";
import { feature } from "topojson-client";
import type { FeatureCollection } from "geojson";
import worldData from "world-atlas/countries-110m.json";
import originsData from "@/data/danceOrigins.json";
import beatsData from "@/data/aboutBeats.json";

// Theme palette (mirrors globals.css @theme tokens)
const C = {
  cream: "#FEFCF7",
  ink: "#1E1208",
  henna: "#7B4A2A",
  gold: "#C49A2A",
  sand: "#EDE0CC",
};

type Origin = {
  name: string;
  state: string;
  region: string;
  lat: number;
  lon: number;
};
type Beat = {
  eyebrow: string;
  title: string;
  body?: string[];
  list?: string[];
  stats?: { count: string; label: string }[];
  cta?: { text: string; href: string };
};

const origins = originsData as Origin[];
const beats = beatsData as Beat[];

type WorldTopo = Parameters<typeof feature>[0];
const topo = worldData as unknown as WorldTopo;
const land = feature(topo, topo.objects.countries) as unknown as FeatureCollection;

const N = origins.length;

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const ease = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

export default function DanceGlobe() {
  const sectionRef = useRef<HTMLElement | null>(null);
  const stageRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const labelRef = useRef<HTMLDivElement | null>(null);
  const [active, setActive] = useState(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    const section = sectionRef.current;
    if (!canvas || !stage || !section) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const projection = geoOrthographic().clipAngle(90).precision(0.4);
    const graticule = geoGraticule10();

    let w = 0;
    let h = 0;
    let baseR = 0;

    function resize() {
      if (!stage || !canvas || !ctx) return;
      const rect = stage.getBoundingClientRect();
      w = rect.width;
      h = rect.height;
      baseR = (Math.min(w, h) / 2) * 0.78;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      projection.translate([w / 2, h / 2]);
    }

    function scrollProgress() {
      if (!section) return 0;
      const rect = section.getBoundingClientRect();
      const scrollable = rect.height - window.innerHeight;
      if (scrollable <= 0) return 0;
      return clamp(-rect.top / scrollable, 0, 1);
    }

    let lastActive = -1;

    function draw(time: number) {
      if (!ctx) return;
      const p = scrollProgress();
      const t = clamp(p * (N - 1), 0, N - 1);
      const i0 = Math.floor(t);
      const i1 = Math.min(i0 + 1, N - 1);
      const frac = t - i0;
      const f = ease(frac);

      const lon = lerp(origins[i0].lon, origins[i1].lon, f);
      const lat = lerp(origins[i0].lat, origins[i1].lat, f);
      projection.rotate([-lon, -lat]);
      const center: [number, number] = [lon, lat];

      const R = baseR;
      projection.scale(R);

      const path = geoPath(projection, ctx);
      ctx.clearRect(0, 0, w, h);
      const cx = w / 2;
      const cy = h / 2;

      // Atmosphere glow
      const glow = ctx.createRadialGradient(cx, cy, R * 0.92, cx, cy, R * 1.22);
      glow.addColorStop(0, "rgba(196,154,42,0.28)");
      glow.addColorStop(1, "rgba(196,154,42,0)");
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(cx, cy, R * 1.22, 0, Math.PI * 2);
      ctx.fill();

      // Ocean sphere — warm color-graded gradient
      const ocean = ctx.createRadialGradient(
        cx - R * 0.35,
        cy - R * 0.35,
        R * 0.1,
        cx,
        cy,
        R
      );
      ocean.addColorStop(0, "#FBF3E2");
      ocean.addColorStop(0.6, C.sand);
      ocean.addColorStop(1, "#D8C29C");
      ctx.beginPath();
      path({ type: "Sphere" });
      ctx.fillStyle = ocean;
      ctx.fill();

      // Graticule
      ctx.beginPath();
      path(graticule);
      ctx.strokeStyle = "rgba(123,74,42,0.16)";
      ctx.lineWidth = 0.5;
      ctx.stroke();

      // Land
      ctx.beginPath();
      path(land);
      ctx.fillStyle = C.henna;
      ctx.fill();
      ctx.strokeStyle = "rgba(196,154,42,0.55)";
      ctx.lineWidth = 0.6;
      ctx.stroke();

      // Terminator shading for depth
      const shade = ctx.createRadialGradient(
        cx - R * 0.4,
        cy - R * 0.4,
        R * 0.2,
        cx,
        cy,
        R
      );
      shade.addColorStop(0, "rgba(254,252,247,0.18)");
      shade.addColorStop(0.7, "rgba(30,18,8,0)");
      shade.addColorStop(1, "rgba(30,18,8,0.30)");
      ctx.beginPath();
      ctx.arc(cx, cy, R, 0, Math.PI * 2);
      ctx.fillStyle = shade;
      ctx.fill();

      const activeIdx = Math.round(t);

      // Markers
      for (let k = 0; k < N; k++) {
        const o = origins[k];
        if (geoDistance(center, [o.lon, o.lat]) >= Math.PI / 2) continue;
        const xy = projection([o.lon, o.lat]);
        if (!xy) continue;
        const [mx, my] = xy;
        const isActive = k === activeIdx;
        ctx.beginPath();
        ctx.arc(mx, my, isActive ? 3 : 2, 0, Math.PI * 2);
        ctx.fillStyle = isActive ? C.gold : "rgba(123,74,42,0.7)";
        ctx.fill();
        ctx.strokeStyle = C.ink;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }

      // Pop logic: label peaks at each point, recedes between points
      const d = Math.abs(t - activeIdx);
      const op = clamp(1 - d / 0.35, 0, 1);

      const ao = origins[activeIdx];
      const visible = geoDistance(center, [ao.lon, ao.lat]) < Math.PI / 2;
      const axy = projection([ao.lon, ao.lat]);

      if (visible && axy && op > 0.01) {
        const [mx, my] = axy;
        const pulse = reduced ? 0 : Math.sin(time / 600) * 0.5 + 0.5;
        ctx.beginPath();
        ctx.arc(mx, my, 4 + pulse * 7, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(196,154,42,${0.5 * op * (1 - pulse)})`;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(mx, my);
        ctx.lineTo(mx + 16, my);
        ctx.strokeStyle = `rgba(196,154,42,${0.7 * op})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();

        const label = labelRef.current;
        if (label) {
          label.style.opacity = String(op);
          label.style.transform = `translate(${mx}px, ${my}px)`;
        }
      } else {
        const label = labelRef.current;
        if (label) label.style.opacity = "0";
      }

      if (activeIdx !== lastActive) {
        lastActive = activeIdx;
        setActive(activeIdx);
      }
    }

    let raf = 0;
    let running = false;
    function loop(time: number) {
      draw(time);
      raf = requestAnimationFrame(loop);
    }
    function start() {
      if (running) return;
      running = true;
      raf = requestAnimationFrame(loop);
    }
    function stop() {
      running = false;
      cancelAnimationFrame(raf);
    }

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(stage);
    const io = new IntersectionObserver(
      ([entry]) => (entry.isIntersecting ? start() : stop()),
      { threshold: 0 }
    );
    io.observe(section);

    return () => {
      stop();
      ro.disconnect();
      io.disconnect();
    };
  }, []);

  return (
    <section
      ref={sectionRef}
      className="relative mx-auto grid max-w-6xl grid-cols-1 px-6 md:grid-cols-2 md:px-10"
      aria-label="About CMU Abhinaya — origins of the classical dance forms"
    >
      {/* About narrative — one beat per dance form */}
      <div className="order-2 md:order-1">
        {beats.map((beat, k) => (
          <div
            key={beat.eyebrow}
            className="flex min-h-screen w-full flex-col justify-center"
          >
            <div
              className="transition-opacity duration-700"
              style={{ opacity: k === active ? 1 : 0.32 }}
            >
              <p className="mb-3 text-[11px] uppercase tracking-[0.3em] text-gold/70">
                {beat.eyebrow}
              </p>
              <h2 className="font-heading text-4xl leading-tight text-ink md:text-6xl">
                {beat.title}
              </h2>

              {beat.body?.map((para) => (
                <p
                  key={para.slice(0, 24)}
                  className="mt-6 max-w-md text-base font-light leading-8 text-ink/65"
                >
                  {para}
                </p>
              ))}

              {beat.list && (
                <ul className="mt-6 flex flex-col gap-4">
                  {beat.list.map((item) => (
                    <li
                      key={item}
                      className="flex items-center gap-4 font-light text-ink/65"
                    >
                      <span className="h-1.5 w-1.5 flex-shrink-0 rounded-full bg-henna/50" />
                      {item}
                    </li>
                  ))}
                </ul>
              )}

              {beat.stats && (
                <div className="mt-8 flex gap-16">
                  {beat.stats.map(({ count, label }) => (
                    <div key={label}>
                      <p className="font-heading text-5xl text-henna">{count}</p>
                      <p className="mt-1 text-[11px] uppercase tracking-[0.22em] text-ink/40">
                        {label}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {beat.cta && (
                <a
                  href={beat.cta.href}
                  className="mt-6 inline-block text-henna underline underline-offset-4 transition-colors duration-500 hover:text-gold"
                >
                  {beat.cta.text}
                </a>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Globe column — sticky cell stretches to match the narrative's height,
          so the inner sticky stage pins through the whole scroll range */}
      <div className="order-1 md:order-2">
        <div className="relative flex flex-col items-center justify-center py-16 md:sticky md:top-0 md:h-screen md:py-0">
          <div
            ref={stageRef}
            className="relative aspect-square w-full max-w-[32rem]"
          >
            <canvas ref={canvasRef} className="absolute inset-0" />
            {/* Floating point popup — name + state · region */}
            <div
              ref={labelRef}
              className="pointer-events-none absolute left-0 top-0 opacity-0 will-change-transform"
            >
              <div className="absolute -translate-y-1/2 translate-x-5 whitespace-nowrap rounded-sm border border-gold/40 bg-white/90 px-3 py-2 text-left shadow-[0_4px_18px_rgba(30,18,8,0.12)] backdrop-blur-sm">
                <p className="font-heading text-xl italic leading-none text-ink">
                  {origins[active].name}
                </p>
                <p className="mt-1 text-[10px] uppercase tracking-[0.18em] text-henna/70">
                  {origins[active].state} · {origins[active].region}
                </p>
              </div>
            </div>
          </div>

          {/* Progress bar — the 8 classical forms */}
          <div className="mt-8 w-full max-w-[20rem]">
            <div className="flex items-center gap-1.5">
              {origins.map((o, k) => (
                <span
                  key={o.name}
                  className={`h-[3px] flex-1 rounded-full transition-all duration-500 ${
                    k === active ? "bg-gold" : k < active ? "bg-henna/40" : "bg-henna/15"
                  }`}
                />
              ))}
            </div>
            <p className="mt-3 text-center text-[10px] uppercase tracking-[0.28em] text-henna/55">
              {String(active + 1).padStart(2, "0")} / {String(N).padStart(2, "0")} · {origins[active].name}
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
