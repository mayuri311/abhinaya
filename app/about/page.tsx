import HennaDivider from "@/components/HennaDivider";

const performances = ["Diwali on the Mall", "Holi", "Dancer's Symposium"];

const stats = [
  { count: "5", label: "Officers" },
  { count: "19", label: "Members" },
];

export default function About() {
  return (
    <main className="flex-1 max-w-2xl mx-auto px-8 py-24">
      <header className="text-center mb-20 animate-fade-up">
        <p className="text-xs tracking-[0.28em] uppercase text-ink/35 mb-4">
          Our Story
        </p>
        <h1 className="font-heading text-6xl text-ink">About</h1>
        <div className="mt-10 w-52 text-gold/60 mx-auto">
          <HennaDivider />
        </div>
      </header>

      <section
        className="space-y-8 text-base leading-9 text-ink/65 font-light animate-fade-up"
        style={{ animationDelay: "180ms" }}
      >
        <p>
          CMU Abhinaya is Carnegie Mellon University&rsquo;s premier competitive
          Indian classical dance team. Our members train in Kuchipudi,
          Bharatanatyam, Odissi, Mohiniattam, and other classical styles —
          conveying stories, experiences, and emotions through movement.
        </p>
        <p>
          We serve as a space to connect with Indian heritage, build community,
          and practice wellness. Abhinaya performs at some of CMU&rsquo;s most
          celebrated events, reaching audiences across campus and beyond.
        </p>
      </section>

      <div
        className="my-16 w-52 text-gold/60 mx-auto animate-fade-in"
        style={{ animationDelay: "320ms" }}
      >
        <HennaDivider />
      </div>

      <section
        className="text-center animate-fade-up"
        style={{ animationDelay: "400ms" }}
      >
        <p className="text-xs tracking-[0.28em] uppercase text-ink/35 mb-10">
          We Perform At
        </p>
        <ul className="flex flex-col items-center gap-5">
          {performances.map((event) => (
            <li key={event} className="flex items-center gap-4 text-ink/60 font-light">
              <span className="w-1.5 h-1.5 rounded-full bg-henna/50 flex-shrink-0" />
              {event}
            </li>
          ))}
        </ul>
      </section>

      <div
        className="my-16 w-52 text-gold/60 mx-auto animate-fade-in"
        style={{ animationDelay: "520ms" }}
      >
        <HennaDivider />
      </div>

      <section
        className="flex justify-center gap-20 animate-fade-up"
        style={{ animationDelay: "620ms" }}
      >
        {stats.map(({ count, label }) => (
          <div key={label} className="text-center">
            <p className="font-heading text-6xl text-henna">{count}</p>
            <p className="mt-2 text-xs tracking-[0.22em] uppercase text-ink/35">
              {label}
            </p>
          </div>
        ))}
      </section>

      <div
        className="my-16 w-52 text-gold/60 mx-auto animate-fade-in"
        style={{ animationDelay: "720ms" }}
      >
        <HennaDivider />
      </div>

      <section
        className="text-center space-y-6 animate-fade-up"
        style={{ animationDelay: "820ms" }}
      >
        <p className="text-xs tracking-[0.28em] uppercase text-ink/35 mb-8">
          For Sponsors
        </p>
        <p className="text-base leading-9 text-ink/60 font-light">
          Abhinaya is committed to sharing the beauty and depth of Indian
          classical dance with the CMU community and beyond. We welcome
          partnerships that support our mission of cultural connection and
          artistic excellence.
        </p>
        <p className="text-base leading-9 text-ink/60 font-light">
          Reach us at{" "}
          <a
            href="mailto:cmu.abhinaya@gmail.com"
            className="text-henna underline underline-offset-4 hover:text-gold transition-colors duration-500"
          >
            cmu.abhinaya@gmail.com
          </a>
          .
        </p>
      </section>
    </main>
  );
}
