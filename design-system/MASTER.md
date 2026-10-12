# CMU Abhinaya — Design System (MASTER)

> Source of truth for the site. Page-specific deviations live in `design-system/pages/*.md`
> and override these rules. If no page file exists, use this file exclusively.

**Aesthetic:** *Liquid graceful scrapbook* — a warm, handmade editorial feel
(polaroids, henna flourishes, slight rotations) elevated with fluid "liquid glass"
motion and flowing gradient text. Handmade warmth, but never cluttered or childish.

---

## 1. Palette (warm brand — do not swap for the generator's red defaults)

| Token        | Hex       | Use                                              |
|--------------|-----------|--------------------------------------------------|
| `--color-cream` | `#FEFCF7` | Page background, on-dark text                  |
| `--color-ink`   | `#1E1208` | Primary text                                   |
| `--color-henna` | `#7B4A2A` | Secondary text, links, hand-drawn marks, land  |
| `--color-gold`  | `#C49A2A` | Accent, active state, dividers, highlights     |
| `--color-sand`  | `#EDE0CC` | Muted surfaces, photo placeholders             |

Derived opacity ladder used across the site: text `ink/75 · ink/65 · ink/40 · ink/35`,
lines/marks `henna/50 · henna/30 · gold/60 · gold/20`.

**Contrast:** body text is `ink` / `ink/65` on `cream` (passes AA). Gold is an *accent*,
never body-text color on cream (fails 4.5:1). On the dark `ink` scrims (video/camera
overlays), use `cream` and `gold/75`.

CLAUDE.md names this year's colors as *dark red / white / gold*. Henna reads as the warm
brown-red that anchors the scrapbook feel; treat a deeper red as an optional seasonal
accent, not a palette rewrite.

## 2. Typography

- **Heading:** Cormorant Garamond (`--font-heading`) — 300/400/600, italic enabled.
  Large, airy, `tracking-wide`. This carries the editorial elegance.
- **Body:** DM Sans (`--font-body`) — 300/400/500, `leading-8`/`leading-9`, `font-light`.
- **Eyebrows / labels:** DM Sans, `text-xs uppercase tracking-[0.22em–0.30em] text-ink/35`.
- Scale: Hero 7xl–8xl · H1 6xl · H2 4xl–6xl · body 16px · label 11–12px.

## 3. Style: Liquid Glass (over scrapbook)

- **Glass surfaces:** `.liquid-glass` — translucent cream/white, `backdrop-blur`, hairline
  gold border, soft warm shadow. Used for framed media, overlays, contact panel.
- **Liquid text:** `<LiquidText>` — flowing gold→henna→ink gradient sheen with an optional
  gentle gooey wobble. Use for the hero title and page H1s. Legibility first: sheen only
  on large headings; body stays solid `ink`.
- **Effects match style:** warm shadows `rgba(30,18,8,0.10)`, radius kept small
  (`rounded-sm`) to preserve the handmade/print feel — glass adds blur, not big radii.
- **Scrapbook artifacts:** polaroid frames, henna corner marks, `HennaDivider`, slight
  rotations (`rotate-[-1.8deg]`…`rotate-[1.5deg]`) that straighten on hover.

## 4. Motion (dial: 6/10 — standard, graceful)

- Micro-interactions 150–300ms; expressive fades 700ms–1.6s (existing `fade-up`,
  `fade-in`, `bubble-in`, `panel-in`). Keep the slow, floaty entrances.
- Liquid flow: 8–14s gradient drift, `ease` loop. Gooey wobble ≤ subtle.
- **All liquid/flow animation must honor `prefers-reduced-motion`** → freeze to a static
  gradient, no displacement, no drift.
- Easing: `cubic-bezier(0.16,1,0.3,1)` for entrances (already the house curve).

## 5. Video ↔ text integration

- Videos never sit in bare rectangles. Wrap every embed in `<LiquidVideo>`: liquid-glass
  frame, henna corner marks, slight rotation, gold hairline.
- On Home, videos are woven *between* text beats ("Moments" reel) — alternating
  text-left/video-right, driven by `data/reel.json`. Aspect-ratio reserved to avoid CLS.
- Support YouTube and Instagram embeds behind one `provider` field.

## 6. Per-surface intent

- **Home:** hero (liquid title) → Moments reel (video+text) → scroll globe. Immersive.
- **Team:** polaroid wall. Unchanged rhythm; H1 becomes liquid.
- **About:** editorial long-form; liquid H1; keep dividers.
- **Playground:** frame the camera as a "specimen card" (liquid-glass + corner marks +
  caption) so it reads as a scrapbook artifact, not a raw tech demo. It must feel like it
  belongs on the same wall as the polaroids.
- **Contact:** professional + on-theme. One calm liquid-glass panel, generous whitespace,
  clear rows. Restraint over decoration; still warm.

## 7. Anti-patterns (avoid)

- Emojis as icons (use inline SVG — lotus/henna marks are the icon language).
- Gold as body text on cream (contrast fail).
- Big border-radii or drop shadows that break the print/handmade feel.
- Liquid displacement strong enough to hurt heading legibility.
- Vibrant/block/playful color — stays warm and muted.
- Bare `<iframe>` or unframed video rectangles.
- Ignoring reduced-motion on any flowing effect.
