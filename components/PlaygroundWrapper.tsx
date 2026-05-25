"use client";

import dynamic from "next/dynamic";

// PlaygroundCamera uses browser globals — must be client-only
const PlaygroundCamera = dynamic(
  () => import("@/components/PlaygroundCamera"),
  { ssr: false }
);

export default function PlaygroundWrapper() {
  return <PlaygroundCamera />;
}
