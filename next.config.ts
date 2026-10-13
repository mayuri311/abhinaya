import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async redirects() {
    return [
      // playground.<any domain>/anything → <that domain>/playground. Add the
      // subdomain to the Vercel project (Settings → Domains) for this to fire.
      {
        source: "/:path*",
        has: [{ type: "host", value: "playground\\.(?<domain>.+)" }],
        destination: "https://:domain/playground",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
