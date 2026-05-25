# CMU Abhinaya Website

Bharatanatyam competitive dance team site. Built with Next.js (App Router),
React, Tailwind CSS. Deployed on Vercel.

## Pages
- / : Home — scrapbook aesthetic, photo scroll + embedded videos (from Instagram or YouTube)
- /team : Exec board featured on top, members below (data from team.json)
- /about : Team history & description, aimed at potential sponsors
- /playground : Mudra recognition — webcam + MediaPipe Hands, classifies
  asamyukta hastas, shows name + meaning from mudras.json
- /contact : Mailing list, TartanConnect, socials

## Conventions
- Content lives in JSON data files, not hardcoded in components
- Tailwind for all styling; no CSS-in-JS
- Components in /components, pages in /app, data in /data
- Keep the playground classifier modular and swappable (rule-based now,
  trained model later) behind a single classifyHasta(landmarks) interface

## Design language
- Scrapbook / handmade feel: polaroid frames, slight rotations, warm tones
- Team colors change every year, keep the this year's colors [dark red, white, gold]