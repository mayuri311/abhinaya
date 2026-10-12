import type { ElementType, ReactNode } from "react";

type Props = {
  children: ReactNode;
  /** Element to render as. Defaults to <span>. */
  as?: ElementType;
  className?: string;
  /**
   * Adds a subtle watery wobble via the shared SVG turbulence filter.
   * Reserve for large display headings — it's disabled under reduced-motion.
   */
  goo?: boolean;
};

/**
 * Flowing gold→henna→ink gradient text — a self-contained adaptation of the
 * 21st.dev "liquid text" effect (no external deps, works with the project's
 * Tailwind v4 setup). The gradient drift and optional goo both honor
 * prefers-reduced-motion via globals.css.
 *
 * Requires <LiquidDefs /> mounted once (see app/layout.tsx) when goo is used.
 */
export default function LiquidText({
  children,
  as: Tag = "span",
  className = "",
  goo = false,
}: Props) {
  return (
    <Tag className={`liquid-text ${goo ? "liquid-goo" : ""} ${className}`}>
      {children}
    </Tag>
  );
}

/**
 * Shared SVG filter for the liquid wobble. Rendered once at the layout root.
 * The animated baseFrequency gives the letters a slow, living ripple.
 */
export function LiquidDefs() {
  return (
    <svg
      aria-hidden="true"
      focusable="false"
      width="0"
      height="0"
      style={{ position: "absolute" }}
    >
      <defs>
        <filter id="liquid-goo" x="-8%" y="-8%" width="116%" height="116%">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.008 0.012"
            numOctaves={2}
            seed={7}
            result="noise"
          >
            <animate
              attributeName="baseFrequency"
              dur="14s"
              values="0.008 0.012; 0.012 0.008; 0.008 0.012"
              repeatCount="indefinite"
            />
          </feTurbulence>
          <feDisplacementMap
            in="SourceGraphic"
            in2="noise"
            scale="5"
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
      </defs>
    </svg>
  );
}
