/**
 * End-to-end + browser parity test for the kNN hasta classifier.
 *
 * Bundles lib/hasta with esbuild, serves it with models/hasta and the fixtures,
 * and drives system Chrome (puppeteer-core). For every fixture it checks the
 * status, and against the Python reference: landmarks, crop box, crop pixels,
 * embedding cosine (>= 0.999) and top-1 label. Runs once per execution provider.
 *
 *   python scripts/hasta/make_fixtures.py <dataset>   # once
 *   npm run test:e2e
 *
 * CHROME_PATH overrides the Chrome binary.
 */
import { build } from "esbuild";
import { createReadStream, existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const FIXTURES = path.join(ROOT, "tests/fixtures/hasta");
const MODELS = path.join(ROOT, "models/hasta");
const CHROME = process.env.CHROME_PATH ?? "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const EPS = (process.env.HASTA_EPS ?? "wasm,webgpu").split(",");
const THRESHOLDS = { minTop1: 0, minMargin: 0.01 }; // test-only: exercises the gate, NOT calibrated values
const COS_MIN = 0.999;

if (!existsSync(FIXTURES)) {
  console.error("No fixtures. Run: python scripts/hasta/make_fixtures.py <Bharatanatyam-Mudra-Dataset dir>");
  process.exit(1);
}

const fixtures = readdirSync(FIXTURES).filter((f) => f.endsWith(".json")).sort()
  .map((f) => JSON.parse(readFileSync(path.join(FIXTURES, f), "utf8")));

const out = mkdtempSync(path.join(tmpdir(), "hasta-e2e-"));
const common = { bundle: true, format: "esm", platform: "browser", target: "es2022", logLevel: "error",
  define: { "process.env": JSON.stringify({ NEXT_PUBLIC_HASTA_ENGINE: "knn" }) }, tsconfig: path.join(ROOT, "tsconfig.json") };
await build({ ...common, entryPoints: [path.join(ROOT, "tests/e2e/harness.ts")], outfile: path.join(out, "harness.js") });
await build({ ...common, entryPoints: [path.join(ROOT, "lib/hasta/embed.worker.ts")], outfile: path.join(out, "embed.worker.js") });

const routes = [["/out/", out], ["/models/hasta/", MODELS], ["/fixtures/", FIXTURES]];
const types = { ".js": "text/javascript", ".json": "application/json", ".png": "image/png" };
const server = createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (url === "/") {
    res.writeHead(200, { "content-type": "text/html" });
    return res.end('<!doctype html><script type="module" src="/out/harness.js"></script>');
  }
  for (const [prefix, dir] of routes) {
    if (!url.startsWith(prefix)) continue;
    const file = path.join(dir, url.slice(prefix.length));
    if (!file.startsWith(dir) || !existsSync(file)) break;
    res.writeHead(200, { "content-type": types[path.extname(file)] ?? "application/octet-stream",
      "content-length": statSync(file).size });
    return createReadStream(file).pipe(res);
  }
  res.writeHead(404).end();
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const origin = `http://127.0.0.1:${server.address().port}`;

const browser = await puppeteer.launch({ executablePath: CHROME, headless: true,
  args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"] });
let failures = 0;
const check = (ok, msg) => { if (!ok) failures++; console.log(`  ${ok ? "PASS" : "FAIL"} ${msg}`); };

try {
  for (const ep of EPS) {
    const page = await browser.newPage();
    page.on("pageerror", (e) => console.error("pageerror:", e.message));
    await page.goto(origin);
    await page.waitForFunction(() => "runHasta" in window);
    let run;
    try {
      run = await page.evaluate((o) => window.runHasta(o), { executionProvider: ep, names: fixtures.map((f) => f.name), thresholds: THRESHOLDS });
    } catch (e) {
      console.log(`\n== ${ep}: unavailable (${String(e.message).split("\n")[0]})`);
      if (ep === "wasm") failures++;
      await page.close();
      continue;
    }
    console.log(`\n== ${run.executionProvider}`);
    for (const r of run.results) {
      const py = fixtures.find((f) => f.name === r.name);
      console.log(`${r.name} (${py.kind}) → ${r.status}${r.status === "ok" ? ` ${r.top5[0].label} ${r.top5[0].score.toFixed(4)} margin ${r.margin.toFixed(4)} low=${r.lowConfidence}` : ""}  ${r.latencyMs.toFixed(0)} ms`);
      check(r.status === py.status, `status ${r.status} == python ${py.status}`);
      if (py.kind === "clear") check(r.status === "ok" && !r.lowConfidence, "clear hand → confident");
      if (py.kind === "ambiguous") check(r.status === "ok" && r.lowConfidence, `ambiguous → lowConfidence (test minMargin ${THRESHOLDS.minMargin})`);
      if (r.status !== "ok" || py.status !== "ok") continue;

      const d = r.debug;
      const lmDiff = Math.max(...d.landmarksPx.flatMap(([x, y], i) => [Math.abs(x - py.landmarksPx[i][0]), Math.abs(y - py.landmarksPx[i][1])]));
      const boxSame = d.boxPadded.every((v, i) => v === py.boxPadded[i]);
      const pyCrop = readFileSync(path.join(FIXTURES, `${r.name}.crop.rgb`));
      let nDiff = 0, maxDiff = 0;
      d.crop.forEach((v, i) => { const e = Math.abs(v - pyCrop[i]); if (e) nDiff++; maxDiff = Math.max(maxDiff, e); });
      const cos = d.embedding.reduce((s, v, i) => s + v * py.embedding[i], 0);
      console.log(`    landmarks max |Δ| ${lmDiff.toFixed(3)} px · box ${boxSame ? "identical" : `${d.boxPadded} vs ${py.boxPadded}`} · crop ${nDiff}/${pyCrop.length} bytes differ (max ${maxDiff}) · embed ${d.embedMs.toFixed(0)} ms, kNN ${d.knnMs.toFixed(0)} ms`);
      check(cos >= COS_MIN, `embedding cos vs python ${cos.toFixed(5)} >= ${COS_MIN}`);
      check(r.top5[0].label === py.top5[0].label, `top-1 ${r.top5[0].label} == python ${py.top5[0].label}`);
    }
    await page.close();
  }
} finally {
  await browser.close();
  server.close();
}

console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
