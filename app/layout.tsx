import type { Metadata } from "next";
import { Cormorant_Garamond, DM_Sans } from "next/font/google";
import NavBar from "@/components/NavBar";
import { LiquidDefs } from "@/components/LiquidText";
import "./globals.css";

const heading = Cormorant_Garamond({
  subsets: ["latin"],
  weight: ["300", "400", "600"],
  style: ["normal", "italic"],
  variable: "--font-heading",
  display: "swap",
});

const body = DM_Sans({
  subsets: ["latin"],
  weight: ["300", "400", "500"],
  variable: "--font-body",
  display: "swap",
});

export const metadata: Metadata = {
  title: "CMU Abhinaya",
  description:
    "Carnegie Mellon University's premier competitive Indian Classical Dance Team.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${heading.variable} ${body.variable} h-full`}>
      <body className="min-h-full flex flex-col bg-cream text-ink">
        <LiquidDefs />
        <NavBar />
        {children}
      </body>
    </html>
  );
}
