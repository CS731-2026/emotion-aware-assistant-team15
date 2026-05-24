import { spawn } from "node:child_process";
import { once } from "node:events";
import { access } from "node:fs/promises";
import { chromium } from "playwright-core";

const ROOT = new URL("..", import.meta.url).pathname;
const CHROME = process.env.CHROME_PATH || "/usr/bin/google-chrome";
const DEFAULT_DEBUG_PDF =
  "/home/rli/下载/collins-et-al-2026-identifying-hearing-difficulty-moments-in-conversational-audio.pdf";

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

async function existingDebugPdfPath() {
  if (process.env.PDF_DEBUG_PATH) return process.env.PDF_DEBUG_PATH;
  try {
    await access(DEFAULT_DEBUG_PDF);
    return DEFAULT_DEBUG_PDF;
  } catch {
    return "";
  }
}

async function startServer() {
  const debugPdfPath = await existingDebugPdfPath();
  const child = spawn("python", ["-u", "main.py", "--mode", "web"], {
    cwd: ROOT,
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
      ...(debugPdfPath ? { PDF_DEBUG_PATH: debugPdfPath } : {}),
    },
  });
  let output = "";
  const ready = new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      child.kill("SIGINT");
      reject(new Error(`server did not start:\n${output}`));
    }, 10000);
    const onData = (chunk) => {
      output += chunk.toString();
      const match = output.match(/Web app running at (http:\/\/[^\s]+)/);
      if (match) {
        clearTimeout(timer);
        resolve(match[1]);
      }
    };
    child.stdout.on("data", onData);
    child.stderr.on("data", onData);
    child.on("exit", (code) => reject(new Error(`server exited with ${code}:\n${output}`)));
  });
  const baseUrl = await ready;
  return { child, baseUrl };
}

async function fetchText(url, expectedType) {
  const response = await fetch(url);
  requireCondition(response.ok, `${url} returned ${response.status}`);
  const contentType = response.headers.get("content-type") || "";
  requireCondition(contentType.includes(expectedType), `${url} content-type was ${contentType}`);
  return response.text();
}

async function fetchBinary(url, options = {}) {
  const response = await fetch(url, options);
  const bytes = new Uint8Array(await response.arrayBuffer());
  return { response, bytes };
}

function importedChunks(source) {
  return [...source.matchAll(/from"\.\/(assets\/[^"]+\.js)"/g)].map((match) => match[1]);
}

async function findWorkerAsset(baseUrl, entry) {
  const sources = [entry];
  for (const chunk of importedChunks(entry)) {
    sources.push(await fetchText(`${baseUrl}/pdf-workspace/${chunk}`, "javascript"));
  }
  const workerMatch = sources
    .join("\n")
    .match(/\/pdf-workspace\/assets\/pdf\.worker\.min-[^"']+\.(?:mjs|js)/);
  requireCondition(workerMatch, "PDF test bundle did not reference an emitted PDF.js worker asset");
  return workerMatch[0];
}

async function main() {
  const { child, baseUrl } = await startServer();
  const requestedUrls = [];
  const consoleMessages = [];
  let browser;
  try {
    const pageHtml = await fetchText(`${baseUrl}/pdf-test`, "text/html");
    for (const forbidden of ["Load Sample", "Selected Passage", "Chat", "Camera", "Emotion", "Context Preview"]) {
      requireCondition(!pageHtml.includes(forbidden), `/pdf-test contains old app UI text: ${forbidden}`);
    }
    const entry = await fetchText(`${baseUrl}/pdf-workspace/pdf-test.js`, "javascript");
    const css = await fetchText(`${baseUrl}/pdf-workspace/pdf-workspace.css`, "css");
    requireCondition(css.includes(".textLayer{position:absolute"), "PDF.js textLayer positioning CSS is missing");
    requireCondition(css.includes(".textLayer :is(span,br)") && css.includes("cursor:text"), "PDF.js text span CSS is missing");
    requireCondition(css.includes(".pdfViewer .page{") && css.includes("position:relative"), "PDF.js page positioning CSS is missing");
    requireCondition(css.includes(".pdf-test-app{") && css.includes(".pdf-test-viewer{"), "/pdf-test outer layout CSS is missing");
    requireCondition(css.includes(".pdf-test-debug-panel{") && css.includes(".pdf-test-copy-button{"), "/pdf-test debug panel CSS is missing");
    requireCondition(!css.includes(".pdf-test-viewer{transform"), "/pdf-test must not manually transform the PDF viewer");
    for (const required of [
      "Browser selection text",
      "Library selection text",
      "Final highlight text",
      "Copy Captured Text",
      "Crop preview",
      "Matched Markdown Blocks",
      "Matched block id",
      "Markdown content",
      "Area Select",
      "Highlight Area",
      "LLM Input Preview",
      "Recommended LLM mode",
      "text_context",
      "table_context",
      "formula_context",
      "image_multimodal",
      "image_plus_context",
      "fallback_image_only",
    ]) {
      requireCondition(entry.includes(required), `PDF test bundle missing debug label: ${required}`);
    }
    requireCondition(entry.includes("pdf-test-crop-preview"), "PDF test bundle missing crop preview image UI");
    requireCondition(entry.includes("pdf-test-area-button"), "PDF test bundle missing area selection toggle UI");
    requireCondition(entry.includes("/api/debug/parse"), "PDF test bundle should parse the debug PDF before matching");
    requireCondition(entry.includes("/api/document/match-blocks"), "PDF test bundle should call the coordinate block matcher");
    requireCondition(!entry.includes("/api/document/highlight"), "PDF test bundle should not save highlights to the backend");
    requireCondition(!entry.includes("/api/chat"), "PDF test bundle should not call chat");
    const workerPath = await findWorkerAsset(baseUrl, entry);
    await fetchText(`${baseUrl}${workerPath}`, "javascript");
    const rootWorker = await fetch(`${baseUrl}/pdf-workspace.mjs`);
    requireCondition(rootWorker.status === 404, "/pdf-workspace.mjs unexpectedly exists");

    const fullPdf = await fetchBinary(`${baseUrl}/api/debug/pdf`);
    requireCondition([200, 206].includes(fullPdf.response.status), `/api/debug/pdf returned ${fullPdf.response.status}`);
    requireCondition(fullPdf.response.headers.get("accept-ranges") === "bytes", "debug PDF missing Accept-Ranges header");
    requireCondition(fullPdf.bytes[0] === 0x25 && fullPdf.bytes[1] === 0x50, "debug PDF did not start with %P");
    const rangePdf = await fetchBinary(`${baseUrl}/api/debug/pdf`, { headers: { Range: "bytes=0-3" } });
    requireCondition(rangePdf.response.status === 206, `debug PDF range returned ${rangePdf.response.status}`);
    requireCondition(rangePdf.response.headers.get("content-range")?.startsWith("bytes 0-3/"), "debug PDF missing Content-Range");
    requireCondition(new TextDecoder().decode(rangePdf.bytes) === "%PDF", "debug PDF range body was not %PDF");
    console.log(`PDF test static checks OK; worker=${workerPath}`);

    browser = await chromium.launch({
      executablePath: CHROME,
      headless: true,
      args: [
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-dev-shm-usage",
        "--disable-background-networking",
        "--disable-features=NetworkServiceSandbox",
        "--no-default-browser-check",
        "--no-first-run",
      ],
    });
    const page = await browser.newPage();
    page.on("request", (request) => requestedUrls.push(request.url()));
    page.on("console", (message) => consoleMessages.push(`${message.type()}: ${message.text()}`));
    page.on("pageerror", (error) => consoleMessages.push(`pageerror: ${error.message}`));

    await page.goto(`${baseUrl}/pdf-test`, { waitUntil: "networkidle" });
    await page.waitForFunction(() => {
      const canvas = document.querySelector("#pdf-test-root canvas");
      return canvas && canvas.width > 0 && canvas.height > 0;
    }, { timeout: 20000 });
    const uiState = await page.evaluate(() => ({
      text: document.body.innerText,
      canvas: {
        width: document.querySelector("#pdf-test-root canvas")?.width || 0,
        height: document.querySelector("#pdf-test-root canvas")?.height || 0,
      },
    }));
    for (const forbidden of ["Load Sample", "Selected Passage", "Chat", "Camera", "Emotion", "Context Preview"]) {
      requireCondition(!uiState.text.includes(forbidden), `/pdf-test rendered old app UI text: ${forbidden}`);
    }
    requireCondition(uiState.canvas.width > 0 && uiState.canvas.height > 0, "PDF page 1 did not render to canvas");
    requireCondition(
      requestedUrls.some((url) => url.includes("/api/debug/pdf")),
      "browser did not request the debug PDF endpoint",
    );
    requireCondition(
      !requestedUrls.some((url) => url.endsWith("/pdf-workspace.mjs")),
      "browser requested /pdf-workspace.mjs",
    );
    requireCondition(
      !consoleMessages.some((message) => message.includes("Setting up fake worker failed")),
      `fake worker failure occurred:\n${consoleMessages.join("\n")}`,
    );
    console.log("PDF TEST SMOKE CHECK PASSED");
  } finally {
    if (browser) await browser.close();
    child.kill("SIGINT");
    await once(child, "exit").catch(() => {});
  }
}

main().catch((error) => {
  console.error(error.stack || error.message || String(error));
  process.exit(1);
});
