# CMU Abhinaya

Website for CMU's Bharatanatyam competitive dance team.

Built with Next.js (App Router), TypeScript, and Tailwind CSS. Deployed on Vercel.

## Local development

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Deploy to Vercel

1. Push the repo to GitHub.
2. Import the repo at [vercel.com/new](https://vercel.com/new).
3. Vercel auto-detects Next.js — no additional configuration needed.
4. Every push to `main` triggers a production deploy.

## Project structure

| Path | Purpose |
|---|---|
| `app/` | Next.js App Router pages and root layout |
| `components/` | Shared React components (NavBar, etc.) |
| `data/` | JSON content files (`team.json`, `mudras.json`) |
| `public/` | Static assets (images, fonts) |

## Pages

| Route | Description |
|---|---|
| `/` | Home — scrapbook aesthetic, photo scroll + embedded videos |
| `/about` | Team history & description for potential sponsors |
| `/team` | Exec board and member roster (from `data/team.json`) |
| `/playground` | Mudra recognition via webcam + MediaPipe Hands |
| `/contact` | Mailing list, TartanConnect, socials |
