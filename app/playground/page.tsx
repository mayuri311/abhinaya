import HennaDivider from "@/components/HennaDivider";
import PlaygroundWrapper from "@/components/PlaygroundWrapper";

export default function Playground() {
  return (
    <main className="flex-1 max-w-3xl mx-auto w-full px-8 py-20">
      <header className="text-center mb-16 animate-fade-up">
        <p className="text-xs tracking-[0.28em] uppercase text-ink/35 mb-4">
          Interactive
        </p>
        <h1 className="font-heading text-6xl text-ink">Playground</h1>
        <div className="mt-10 w-52 text-gold/60 mx-auto">
          <HennaDivider />
        </div>
      </header>

      <PlaygroundWrapper />

      <p
        className="mt-10 text-center text-xs leading-7 text-ink/35 max-w-md mx-auto animate-fade-in"
        style={{ animationDelay: "400ms" }}
      >
        Hold your hand up to the camera. MediaPipe tracks 21 landmarks in
        real time — fingertip joints are shown in cream, knuckles in henna.
      </p>
    </main>
  );
}
