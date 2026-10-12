"use client";

import { useState } from "react";

export type VideoProvider = "youtube" | "instagram";

type Props = {
  provider: VideoProvider;
  /** YouTube video id, or Instagram post shortcode. */
  id: string;
  title: string;
  /** Tailwind rotation class, e.g. "rotate-[-1.4deg]". */
  rotation?: string;
  className?: string;
};

function embedSrc(provider: VideoProvider, id: string): string {
  if (provider === "instagram") {
    return `https://www.instagram.com/p/${id}/embed`;
  }
  return `https://www.youtube-nocookie.com/embed/${id}?autoplay=1&rel=0`;
}

/** Small henna flourish that echoes the polaroid corner marks. */
function CornerMark({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 28 28" className={`w-5 h-5 ${className}`} fill="none" aria-hidden="true">
      <circle cx="4.5" cy="4.5" r="2.8" stroke="currentColor" strokeWidth="0.7" opacity="0.4" />
      <circle cx="4.5" cy="4.5" r="1.1" fill="currentColor" opacity="0.55" />
      <line x1="8.2" y1="4.5" x2="15" y2="4.5" stroke="currentColor" strokeWidth="0.6" opacity="0.28" />
      <line x1="4.5" y1="8.2" x2="4.5" y2="15" stroke="currentColor" strokeWidth="0.6" opacity="0.28" />
    </svg>
  );
}

/**
 * A video embed dressed as a scrapbook artifact: liquid-glass frame, henna
 * corner marks, a slight rotation that straightens on hover. Click-to-play
 * facade keeps the heavy provider iframe out of the initial load.
 */
export default function LiquidVideo({
  provider,
  id,
  title,
  rotation = "rotate-0",
  className = "",
}: Props) {
  const [playing, setPlaying] = useState(false);

  return (
    <figure
      className={`liquid-glass liquid-glass-sheen group relative overflow-hidden rounded-sm p-2.5 ${rotation} transition-transform duration-500 ease-out hover:rotate-0 ${className}`}
    >
      <span className="absolute left-3 top-3 z-10 text-henna/40"><CornerMark /></span>
      <span className="absolute right-3 top-3 z-10 rotate-90 text-henna/40"><CornerMark /></span>
      <span className="absolute bottom-3 left-3 z-10 -rotate-90 text-henna/40"><CornerMark /></span>
      <span className="absolute bottom-3 right-3 z-10 rotate-180 text-henna/40"><CornerMark /></span>

      <div
        className="relative w-full overflow-hidden rounded-sm bg-ink/5"
        style={{ aspectRatio: provider === "instagram" ? "4 / 5" : "16 / 9" }}
      >
        {playing ? (
          <iframe
            src={embedSrc(provider, id)}
            title={title}
            loading="lazy"
            allow="autoplay; encrypted-media; picture-in-picture"
            allowFullScreen
            className="absolute inset-0 h-full w-full border-0"
          />
        ) : (
          <button
            type="button"
            onClick={() => setPlaying(true)}
            aria-label={`Play ${title}`}
            className="group/btn absolute inset-0 flex flex-col items-center justify-center gap-4 bg-gradient-to-br from-sand/60 to-henna/15 transition-colors duration-500 hover:from-sand/40"
          >
            <span className="flex h-16 w-16 items-center justify-center rounded-full border border-gold/40 bg-cream/70 text-henna backdrop-blur-sm transition-transform duration-500 group-hover/btn:scale-110">
              <svg viewBox="0 0 24 24" className="ml-0.5 h-6 w-6" fill="currentColor" aria-hidden="true">
                <path d="M8 5.14v13.72a1 1 0 0 0 1.5.86l11-6.86a1 1 0 0 0 0-1.72l-11-6.86A1 1 0 0 0 8 5.14Z" />
              </svg>
            </span>
            <span className="text-[10px] uppercase tracking-[0.28em] text-henna/60">
              Play
            </span>
          </button>
        )}
      </div>

      <figcaption className="sr-only">{title}</figcaption>
    </figure>
  );
}
