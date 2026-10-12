// Browser side of hasta.e2e.mjs: runs the real lib/hasta classifier on fixture PNGs.
import { createKnnClassifier } from "@/lib/hasta/knnClassifier";

type Opts = { executionProvider: string; names: string[]; thresholds: { minTop1: number; minMargin: number } };

async function loadImage(src: string) {
  const img = new Image();
  img.src = src;
  await img.decode();
  return img;
}

(window as unknown as { runHasta: (o: Opts) => Promise<unknown> }).runHasta = async (o: Opts) => {
  const clf = await createKnnClassifier({
    assetBase: "/models/hasta",
    thresholds: o.thresholds,
    executionProviders: [o.executionProvider],
    createWorker: () => new Worker("/out/embed.worker.js", { type: "module" }),
  });
  const results = [];
  for (const name of o.names) {
    const img = await loadImage(`/fixtures/${name}.png`);
    await clf.classify(img); // first call pays one-off costs; time the second
    const r = await clf.classify(img, { debug: true });
    const { debug, ...rest } = r;
    results.push({
      name, ...rest,
      debug: debug && { ...debug, crop: Array.from(debug.crop.data), embedding: Array.from(debug.embedding) },
    });
  }
  const ep = clf.executionProvider;
  clf.close();
  return { executionProvider: ep, results };
};
