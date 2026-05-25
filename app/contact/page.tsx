import HennaDivider from "@/components/HennaDivider";

const contacts = [
  {
    label: "Email",
    value: "cmu.abhinaya@gmail.com",
    href: "mailto:cmu.abhinaya@gmail.com",
  },
  {
    label: "Instagram",
    value: "@cmu.abhinaya",
    href: "https://www.instagram.com/cmu.abhinaya/",
  },
  {
    label: "YouTube",
    // Update this URL to your channel once confirmed
    value: "CMU Abhinaya",
    href: "https://www.youtube.com/@cmuabhinaya",
  },
  {
    label: "TartanConnect",
    value: "Show Interest",
    href: "https://tartanconnect.cmu.edu/abhinaya/home/",
  },
  {
    label: "Address",
    value: "5000 Forbes Ave, Pittsburgh PA 15213",
    href: undefined,
  },
];

export default function Contact() {
  return (
    <main className="flex-1 max-w-2xl mx-auto px-8 py-24">
      <header className="text-center mb-20 animate-fade-up">
        <p className="text-xs tracking-[0.28em] uppercase text-ink/35 mb-4">
          Get in Touch
        </p>
        <h1 className="font-heading text-6xl text-ink">Contact</h1>
        <div className="mt-10 w-52 text-gold/60 mx-auto">
          <HennaDivider />
        </div>
      </header>

      <section
        className="animate-fade-up"
        style={{ animationDelay: "180ms" }}
      >
        {contacts.map(({ label, value, href }) => (
          <div
            key={label}
            className="flex items-baseline justify-between py-7 border-b border-henna/10"
          >
            <span className="text-xs tracking-[0.22em] uppercase text-ink/35">
              {label}
            </span>
            {href ? (
              <a
                href={href}
                target={href.startsWith("http") ? "_blank" : undefined}
                rel="noopener noreferrer"
                className="font-heading text-2xl text-ink hover:text-henna transition-colors duration-500"
              >
                {value}
              </a>
            ) : (
              <span className="font-heading text-xl text-ink/70">{value}</span>
            )}
          </div>
        ))}
      </section>
    </main>
  );
}
