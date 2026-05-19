import { spawn } from "node:child_process";
import { once } from "node:events";
import { chromium } from "playwright-core";

const ROOT = new URL("..", import.meta.url).pathname;
const CHROME = process.env.CHROME_PATH || "/usr/bin/google-chrome";

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

function createSmokePdf() {
  const text = "React PDF Workspace Smoke";
  const stream = `BT /F1 24 Tf 72 720 Td (${text}) Tj ET`;
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    `<< /Length ${stream.length} >>\nstream\n${stream}\nendstream`,
  ];
  let pdf = "%PDF-1.4\n";
  const offsets = [0];
  objects.forEach((object, index) => {
    offsets.push(Buffer.byteLength(pdf, "utf8"));
    pdf += `${index + 1} 0 obj\n${object}\nendobj\n`;
  });
  const xrefOffset = Buffer.byteLength(pdf, "utf8");
  pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  offsets.slice(1).forEach((offset) => {
    pdf += `${String(offset).padStart(10, "0")} 00000 n \n`;
  });
  pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;
  return Buffer.from(pdf, "utf8");
}

async function startServer() {
  const child = spawn("python", ["-u", "main.py", "--mode", "web"], {
    cwd: ROOT,
    stdio: ["ignore", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
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
    child.on("exit", (code) => {
      reject(new Error(`server exited with ${code}:\n${output}`));
    });
  });
  const baseUrl = await ready;
  return { child, baseUrl };
}

async function fetchOk(url, expectedType) {
  const response = await fetch(url);
  requireCondition(response.ok, `${url} returned ${response.status}`);
  const contentType = response.headers.get("content-type") || "";
  requireCondition(contentType.includes(expectedType), `${url} content-type was ${contentType}`);
  return response.text();
}

async function main() {
  const { child, baseUrl } = await startServer();
  const requestedUrls = [];
  const consoleMessages = [];
  let browser;
  try {
    const entry = await fetchOk(`${baseUrl}/pdf-workspace/pdf-workspace.js`, "javascript");
    await fetchOk(`${baseUrl}/pdf-workspace/pdf-workspace.css`, "text/css");
    const workerMatch = entry.match(/\/pdf-workspace\/assets\/pdf\.worker\.min-[^"']+\.mjs/);
    requireCondition(workerMatch, "entry bundle did not reference an emitted PDF.js worker asset");
    await fetchOk(`${baseUrl}${workerMatch[0]}`, "javascript");
    const rootWorker = await fetch(`${baseUrl}/pdf-workspace.mjs`);
    requireCondition(rootWorker.status === 404, "/pdf-workspace.mjs unexpectedly exists; worker should use the emitted asset path");
    console.log(`PDF workspace static assets OK; worker=${workerMatch[0]}`);

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

    await page.goto(baseUrl, { waitUntil: "networkidle" });
    await page.setInputFiles("#uploadInput", {
      name: "workspace-smoke.pdf",
      mimeType: "application/pdf",
      buffer: createSmokePdf(),
    });
    await page.waitForSelector("#pdfWorkspaceRoot:not(.hidden) .page canvas", { timeout: 20000 });
    await page.waitForTimeout(1000);

    const uiState = await page.evaluate(() => {
      const visible = (id) => {
        const element = document.getElementById(id);
        if (!element) return false;
        const style = window.getComputedStyle(element);
        return !element.classList.contains("hidden") && style.display !== "none" && style.visibility !== "hidden";
      };
      const canvas = document.querySelector("#pdfWorkspaceRoot .page canvas");
      return {
        workspaceVisible: visible("pdfWorkspaceRoot"),
        legacyControlsVisible: visible("legacyReaderControls"),
        legacySelectionVisible: visible("legacySelectionPanel"),
        oldPdfVisible: visible("pdfViewer"),
        paperTextVisible: visible("paperText"),
        canvasWidth: canvas ? canvas.width : 0,
        canvasHeight: canvas ? canvas.height : 0,
      };
    });

    requireCondition(uiState.workspaceVisible, "React PDF workspace was not visible");
    requireCondition(!uiState.legacyControlsVisible, "legacy paper reader controls remained visible in PDF mode");
    requireCondition(!uiState.legacySelectionVisible, "legacy selected passage panel remained visible in PDF mode");
    requireCondition(!uiState.oldPdfVisible, "old PDF fallback viewer remained visible in React PDF mode");
    requireCondition(!uiState.paperTextVisible, "text fallback reader remained visible in React PDF mode");
    requireCondition(uiState.canvasWidth > 0 && uiState.canvasHeight > 0, "PDF canvas did not render");
    requireCondition(
      !requestedUrls.some((url) => url.endsWith("/pdf-workspace.mjs")),
      "browser requested /pdf-workspace.mjs instead of emitted worker asset",
    );
    requireCondition(
      requestedUrls.some((url) => /\/pdf-workspace\/assets\/pdf\.worker\.min-.*\.mjs$/.test(url)),
      "browser did not request the emitted PDF.js worker module",
    );
    requireCondition(
      !consoleMessages.some((message) => message.includes("Setting up fake worker failed")),
      `fake worker failure occurred:\n${consoleMessages.join("\n")}`,
    );
    console.log("PDF WORKSPACE SMOKE CHECK PASSED");
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
