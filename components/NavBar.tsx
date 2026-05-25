"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/about", label: "About" },
  { href: "/team", label: "Team" },
  { href: "/playground", label: "Playground" },
  { href: "/contact", label: "Contact" },
];

export default function NavBar() {
  const pathname = usePathname();

  return (
    <nav className="w-full flex items-center gap-10 px-10 py-5 bg-cream border-b border-gold/20">
      <Link
        href="/"
        className="font-heading italic text-xl text-ink tracking-widest mr-auto"
      >
        Abhinaya
      </Link>
      {links.map(({ href, label }) => (
        <Link
          key={href}
          href={href}
          className={`text-xs tracking-[0.18em] uppercase transition-colors duration-700 ${
            pathname === href
              ? "text-gold"
              : "text-ink/40 hover:text-ink"
          }`}
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}
