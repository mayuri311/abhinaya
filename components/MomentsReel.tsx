import HennaDivider from "@/components/HennaDivider";
import LiquidText from "@/components/LiquidText";
import LiquidVideo, { type VideoProvider } from "@/components/LiquidVideo";
import reel from "@/data/reel.json";

type Moment = {
  provider: VideoProvider;
  id: string;
  eyebrow: string;
  title: string;
  body: string;
  rotation?: string;
};

const moments = reel as Moment[];

/**
 * Home "Moments" reel — video and text woven together rather than stacked in
 * bare blocks. Each beat alternates sides; the media is framed as a liquid-glass
 * scrapbook artifact so it belongs beside the polaroids and the globe.
 */
export default function MomentsReel() {
  return (
    <section
      className="mx-auto w-full max-w-5xl px-6 py-24 md:px-10"
      aria-label="Moments — performances and behind the scenes"
    >
      <header className="mb-20 text-center">
        <p className="mb-4 text-xs uppercase tracking-[0.28em] text-ink/35">
          In Motion
        </p>
        <h2 className="font-heading text-5xl md:text-6xl">
          <LiquidText>Moments</LiquidText>
        </h2>
        <div className="mx-auto mt-8 w-52 text-gold/60">
          <HennaDivider />
        </div>
      </header>

      <div className="flex flex-col gap-24">
        {moments.map((m, i) => {
          const textLeft = i % 2 === 0;
          return (
            <article
              key={`${m.title}-${i}`}
              className="grid grid-cols-1 items-center gap-10 md:grid-cols-2 md:gap-16"
            >
              {/* Text beat */}
              <div className={textLeft ? "md:order-1" : "md:order-2"}>
                <p className="mb-3 text-[11px] uppercase tracking-[0.3em] text-gold/70">
                  {m.eyebrow}
                </p>
                <h3 className="font-heading text-3xl leading-tight text-ink md:text-4xl">
                  {m.title}
                </h3>
                <p className="mt-5 max-w-md text-base font-light leading-8 text-ink/65">
                  {m.body}
                </p>
              </div>

              {/* Video, framed */}
              <div className={textLeft ? "md:order-2" : "md:order-1"}>
                <LiquidVideo
                  provider={m.provider}
                  id={m.id}
                  title={m.title}
                  rotation={m.rotation}
                />
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
