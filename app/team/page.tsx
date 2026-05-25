import HennaDivider from "@/components/HennaDivider";
import PolaroidCard from "@/components/PolaroidCard";
import teamData from "@/data/team.json";

const CAPTAIN_ROTATIONS = ["rotate-[-1.8deg]", "rotate-[0.6deg]", "rotate-[-1.2deg]"] as const;
const OTHER_ROTATIONS   = ["rotate-[1.5deg]",  "rotate-[-0.9deg]"]                   as const;

type Member = { name: string; role: string; image?: string };

export default function Team() {
  const captains = (teamData as Member[]).filter((m) => m.role === "Captain");
  const others   = (teamData as Member[]).filter((m) => m.role !== "Captain");

  return (
    <main className="flex-1 max-w-4xl mx-auto px-8 py-24">

      <header className="text-center mb-20 animate-fade-up">
        <p className="text-xs tracking-[0.28em] uppercase text-ink/35 mb-4">
          Leadership
        </p>
        <h1 className="font-heading text-6xl text-ink">Team</h1>
        <div className="mt-10 w-52 text-gold/60 mx-auto">
          <HennaDivider />
        </div>
      </header>

      {/* ── Captains ── */}
      <section className="animate-fade-up" style={{ animationDelay: "180ms" }}>
        <p className="text-xs tracking-[0.28em] uppercase text-ink/30 text-center mb-14">
          Captains
        </p>
        <div className="grid grid-cols-3 gap-10">
          {captains.map((m, i) => (
            <PolaroidCard
              key={m.name}
              name={m.name}
              role={m.role}
              image={m.image}
              rotation={CAPTAIN_ROTATIONS[i] ?? "rotate-0"}
              size="lg"
            />
          ))}
        </div>
      </section>

      <div
        className="my-20 w-52 text-gold/60 mx-auto animate-fade-in"
        style={{ animationDelay: "360ms" }}
      >
        <HennaDivider />
      </div>

      {/* ── Other roles ── */}
      <section
        className="animate-fade-up flex justify-center gap-12"
        style={{ animationDelay: "460ms" }}
      >
        {others.map((m, i) => (
          <div key={m.name} className="w-44">
            <PolaroidCard
              name={m.name}
              role={m.role}
              image={m.image}
              rotation={OTHER_ROTATIONS[i] ?? "rotate-0"}
              size="sm"
            />
          </div>
        ))}
      </section>

    </main>
  );
}
