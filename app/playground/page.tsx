import HennaDivider from "@/components/HennaDivider";
import LiquidText from "@/components/LiquidText";
import PlaygroundWrapper from "@/components/PlaygroundWrapper";

export default function Playground() {
  return (
    <main className="flex-1 w-full max-w-3xl mx-auto px-8 py-20">
      <header className="text-center mb-16 animate-fade-up">
        <p className="text-xs tracking-[0.28em] uppercase text-ink/35 mb-4">
          Interactive
        </p>
        <h1 className="font-heading text-6xl"><LiquidText>Playground</LiquidText></h1>
        <div className="mt-6 w-52 text-gold/60 mx-auto">
          <HennaDivider />
        </div>
        <p className="mt-8 max-w-md mx-auto text-base font-light leading-8 text-ink/60">
          A living page from our notebook: hold a hand to the camera and it reads
          the <span className="italic font-heading text-ink/75">asamyukta hastas</span> —
          single-hand mudras — naming each one and what it speaks.
        </p>
      </header>

      {/* Specimen card — the camera framed as a scrapbook artifact so it sits
          on the same wall as the polaroids, not off on its own. */}
      <div
        className="liquid-glass liquid-glass-sheen relative rounded-sm p-4 md:p-6 rotate-[-0.6deg] animate-fade-up"
        style={{ animationDelay: "180ms" }}
      >
        <div className="flex items-center justify-between px-1 pb-4">
          <span className="text-[10px] tracking-[0.26em] uppercase text-henna/60">
            Mudra Specimen
          </span>
          <span className="text-[10px] tracking-[0.26em] uppercase text-ink/30">
            No. 01 · Live
          </span>
        </div>

        <PlaygroundWrapper />

        <p className="mt-6 px-1 text-center text-[11px] leading-6 text-ink/40 font-heading italic">
          MediaPipe tracks 21 landmarks in real time — fingertips in cream,
          knuckles in henna. Nothing leaves your device.
        </p>
      </div>
    </main>
  );
}
