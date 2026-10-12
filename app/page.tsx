import HennaDivider from "@/components/HennaDivider";
import DanceGlobe from "@/components/DanceGlobe";
import LiquidText from "@/components/LiquidText";
import MomentsReel from "@/components/MomentsReel";

function LotusIcon() {
  return (
    <svg
      viewBox="0 0 100 100"
      fill="none"
      stroke="currentColor"
      strokeWidth="0.8"
      strokeLinecap="round"
      className="w-16 h-16"
      aria-hidden="true"
    >
      {/* Four petals radiating from center */}
      <path d="M50,50 C42,40 42,28 50,22 C58,28 58,40 50,50Z" />
      <path d="M50,50 C60,42 72,42 78,50 C72,58 60,58 50,50Z" />
      <path d="M50,50 C58,60 58,72 50,78 C42,72 42,60 50,50Z" />
      <path d="M50,50 C40,58 28,58 22,50 C28,42 40,42 50,50Z" />
      {/* Center bindi */}
      <circle cx="50" cy="50" r="3.5" fill="currentColor" stroke="none" />
    </svg>
  );
}

export default function Home() {
  return (
    <main className="flex flex-1 flex-col">
      <section className="flex flex-1 flex-col items-center justify-center px-8 py-28 text-center">
      <div className="text-henna/50 mb-10 animate-fade-in">
        <LotusIcon />
      </div>

      <h1
        className="font-heading text-7xl md:text-8xl tracking-widest animate-fade-up"
        style={{ animationDelay: "120ms" }}
      >
        <LiquidText goo>CMU Abhinaya</LiquidText>
      </h1>

      <p
        className="mt-5 text-xs tracking-[0.25em] uppercase text-ink/40 animate-fade-up"
        style={{ animationDelay: "260ms" }}
      >
        Carnegie Mellon&rsquo;s premier competitive Indian Classical Dance Team
      </p>

      <div
        className="mt-12 w-52 text-gold/60 animate-fade-in"
        style={{ animationDelay: "420ms" }}
      >
        <HennaDivider />
      </div>

      <p
        className="mt-12 max-w-lg text-base leading-9 text-ink/60 font-light animate-fade-up"
        style={{ animationDelay: "540ms" }}
      >
        We convey stories, experiences, and emotions through Bharatanatyam,
        Kuchipudi, Odissi, and Mohiniattam — connecting with our Indian
        heritage and building community at Carnegie Mellon.
      </p>

      <a
        href="https://tartanconnect.cmu.edu/abhinaya/home/"
        target="_blank"
        rel="noopener noreferrer"
        className="mt-14 text-xs tracking-[0.22em] uppercase border border-henna/30 text-henna px-10 py-3.5 hover:bg-henna hover:text-cream transition-all duration-700 animate-fade-in"
        style={{ animationDelay: "740ms" }}
      >
        Show Interest
      </a>
      </section>

      <MomentsReel />

      <DanceGlobe />
    </main>
  );
}
