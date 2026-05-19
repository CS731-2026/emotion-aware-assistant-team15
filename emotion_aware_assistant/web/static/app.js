const PDFJS_SOURCES = [
  {
    label: "local PDF.js assets",
    module: "/vendor/pdfjs/pdf.min.mjs",
    worker: "/vendor/pdfjs/pdf.worker.min.mjs",
  },
  {
    label: "PDF.js CDN",
    module: "https://cdn.jsdelivr.net/npm/pdfjs-dist@5.7.284/build/pdf.min.mjs",
    worker: "https://cdn.jsdelivr.net/npm/pdfjs-dist@5.7.284/build/pdf.worker.min.mjs",
  },
];

const state = {
  documentId: null,
  documentType: "txt",
  currentPage: 1,
  pageCount: 0,
  selectedText: "",
  selectionRects: [],
  highlights: [],
  lastHighlightId: null,
  lastContext: null,
  lastQuestion: "",
  pdfUrl: null,
  pdfDoc: null,
  pdfjs: null,
  pdfJsSource: null,
  pdfScale: 1.25,
  cameraStream: null,
  cameraTimer: null,
};

const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function setText(id, value) {
  $(id).textContent = value == null || value === "" ? "-" : String(value);
}

async function updateDocument(payload) {
  state.documentId = payload.document_id || null;
  state.documentType = payload.type || "txt";
  state.currentPage = payload.current_page || 1;
  state.pageCount = payload.page_count || 0;
  state.pdfUrl = payload.pdf_url || null;
  state.selectedText = "";
  state.selectionRects = [];
  state.lastHighlightId = null;
  state.lastContext = null;
  state.highlights = [];
  setText("docTitle", payload.title);
  updatePageInfo();
  resetContextPreview();

  if (state.documentType === "pdf" && state.pdfUrl) {
    setText("paperText", payload.current_page_text || "");
    await loadStoredHighlights();
    const mounted = await showPdfWorkspace(payload);
    if (!mounted) {
      await showPdfReader(state.pdfUrl);
    }
  } else {
    showTextReader(payload.current_page_text || "No text extracted.");
  }
  syncReaderControls();
}

function showTextReader(text) {
  state.pdfDoc = null;
  if (window.PdfWorkspaceIsland) {
    window.PdfWorkspaceIsland.unmount($("pdfWorkspaceRoot"));
  }
  setLegacyReaderVisible(true);
  $("pdfWorkspaceRoot").classList.add("hidden");
  $("pdfViewer").classList.add("hidden");
  $("paperText").classList.remove("hidden");
  setText("paperText", text);
  setText("readerStatus", "Text reader ready.");
  renderHighlights();
}

async function showPdfWorkspace(payload) {
  if (!window.PdfWorkspaceIsland || !state.pdfUrl) {
    return false;
  }
  state.pdfDoc = null;
  setLegacyReaderVisible(false);
  $("paperText").classList.add("hidden");
  $("pdfViewer").classList.add("hidden");
  $("pdfWorkspaceRoot").classList.remove("hidden");
  setText("readerStatus", "React PDF workspace active.");
  return window.PdfWorkspaceIsland.mount($("pdfWorkspaceRoot"), {
    documentId: state.documentId,
    documentType: state.documentType,
    pdfUrl: state.pdfUrl,
    pageCount: state.pageCount,
    currentPage: state.currentPage,
    title: payload.title,
    currentPageText: payload.current_page_text || "",
    highlights: state.highlights,
  });
}

async function showPdfReader(url) {
  setLegacyReaderVisible(true);
  $("pdfWorkspaceRoot").classList.add("hidden");
  $("paperText").classList.add("hidden");
  $("pdfViewer").classList.remove("hidden");
  setText("readerStatus", "Loading PDF viewer...");
  try {
    const pdfjs = await loadPdfJs();
    state.pdfjs = pdfjs;
    const loadingTask = pdfjs.getDocument({ url });
    state.pdfDoc = await loadingTask.promise;
    state.pageCount = state.pdfDoc.numPages;
    state.currentPage = Math.min(Math.max(1, state.currentPage), state.pageCount);
    updatePageInfo();
    await renderPdfPage(state.currentPage);
  } catch (error) {
    console.warn("PDF.js failed; using text fallback.", error);
    showTextReader($("paperText").textContent || "PDF text fallback is unavailable.");
    setText("readerStatus", `PDF.js unavailable; using extracted text fallback. ${error.message || error}`);
  }
}

function setLegacyReaderVisible(visible) {
  ["legacyReaderControls", "legacySelectionPanel", "readerStatus"].forEach((id) => {
    const element = $(id);
    if (element) element.classList.toggle("hidden", !visible);
  });
}

async function loadPdfJs() {
  if (state.pdfjs) return state.pdfjs;
  let lastError = null;
  for (const source of PDFJS_SOURCES) {
    try {
      const module = await import(source.module);
      module.GlobalWorkerOptions.workerSrc = source.worker;
      state.pdfJsSource = source.label;
      setText("readerStatus", `PDF viewer ready (${source.label}).`);
      return module;
    } catch (error) {
      lastError = error;
    }
  }
  throw new Error(`Unable to load PDF.js assets. ${lastError ? lastError.message : ""}`.trim());
}

async function renderPdfPage(pageNumber) {
  if (!state.pdfDoc || !state.pdfjs) return;
  const page = await state.pdfDoc.getPage(pageNumber);
  const viewport = page.getViewport({ scale: state.pdfScale });
  const outputScale = window.devicePixelRatio || 1;
  const canvas = $("pdfCanvas");
  const context = canvas.getContext("2d");
  canvas.width = Math.floor(viewport.width * outputScale);
  canvas.height = Math.floor(viewport.height * outputScale);
  canvas.style.width = `${Math.floor(viewport.width)}px`;
  canvas.style.height = `${Math.floor(viewport.height)}px`;

  const pageLayer = $("pdfPageLayer");
  pageLayer.style.width = `${Math.floor(viewport.width)}px`;
  pageLayer.style.height = `${Math.floor(viewport.height)}px`;
  $("pdfTextLayer").style.width = pageLayer.style.width;
  $("pdfTextLayer").style.height = pageLayer.style.height;
  $("highlightLayer").style.width = pageLayer.style.width;
  $("highlightLayer").style.height = pageLayer.style.height;

  const transform = outputScale !== 1 ? [outputScale, 0, 0, outputScale, 0, 0] : null;
  await page.render({ canvasContext: context, transform, viewport }).promise;
  await renderTextLayer(page, viewport);
  renderHighlights();
  updatePageInfo();
  setText("readerStatus", `PDF page ${state.currentPage} rendered at ${Math.round(state.pdfScale * 100)}% (${state.pdfJsSource}).`);
}

async function renderTextLayer(page, viewport) {
  const textLayer = $("pdfTextLayer");
  textLayer.innerHTML = "";
  textLayer.dataset.pageNumber = String(state.currentPage);
  const textContent = await page.getTextContent();
  textContent.items.forEach((item) => {
    if (!item.str) return;
    const span = document.createElement("span");
    span.textContent = item.str + (item.hasEOL ? "\n" : " ");
    const tx = state.pdfjs.Util.transform(viewport.transform, item.transform);
    span.style.transform = `matrix(${tx.map((value) => Number(value).toFixed(4)).join(",")})`;
    span.style.fontSize = "1px";
    span.dataset.pageNumber = String(state.currentPage);
    textLayer.appendChild(span);
  });
}

async function loadStoredHighlights() {
  if (!state.documentId) return;
  try {
    const payload = await api(`/api/document/highlights/${state.documentId}`);
    state.highlights = payload.highlights || [];
  } catch (error) {
    console.warn("Stored highlight load failed", error);
    state.highlights = [];
  }
}

function updatePageInfo() {
  setText("pageInfo", `Page ${state.currentPage || "-"} / ${state.pageCount || "-"}`);
}

function resetContextPreview() {
  setText("selectedPreview", "No selected passage yet.");
  setText("passageBadge", "Passage type: -");
  setText("difficultyBadge", "Difficulty: -");
  setText("surroundingPreview", "No context yet.");
  $("retrievedChunks").textContent = "No chunks yet.";
  $("contextDebug").textContent = "No context yet.";
}

function syncReaderControls() {
  const pdfActive = state.documentType === "pdf" && Boolean(state.pdfDoc);
  $("zoomInBtn").disabled = !pdfActive;
  $("zoomOutBtn").disabled = !pdfActive;
  $("highlightSelectionBtn").disabled = !state.documentId;
}

function updateEmotion(payload) {
  setText("rawEmotion", payload.raw_emotion);
  setText("smoothedEmotion", payload.smoothed_emotion);
  setText("learningState", payload.state);
  setText("trend", payload.trend);
  const confidence = Number(payload.confidence || 0);
  setText("confidence", confidence.toFixed(2));
  $("confidenceMeter").value = confidence;
  setText("duration", `${Number(payload.duration_sec || 0).toFixed(1)}s`);
  setText("strategy", payload.strategy);
  setText("sourceMode", payload.source || payload.source_mode);
  setText("dominantState", payload.dominant_state);
  renderTimeline(payload.history || []);
}

function appendMessage(role, text) {
  const node = document.createElement("div");
  node.className = `message ${role}`;
  node.textContent = `${role === "user" ? "User" : "Assistant"}:\n${text}`;
  $("chatHistory").appendChild(node);
  $("chatHistory").scrollTop = $("chatHistory").scrollHeight;
}

async function refreshStatus() {
  const payload = await api("/api/status");
  setText("backendStatus", "Backend connected");
  setText("apiStatus", `API: ${payload.api_key_status}`);
  setText("modelStatus", `LLM: ${payload.llm_client}`);
  setText("apiKeyStatus", payload.api_key_status);
  setText("emotionModelStatus", payload.emotion_model_status);
  setText("faceDetectorStatus", payload.face_detector_status);
  setText("logPath", payload.log_path);
  $("promptPreview").textContent = payload.last_prompt_preview || "No prompt yet.";
  $("requestSummary").textContent = JSON.stringify(payload.last_request_summary || {}, null, 2);
  const selector = $("modelSelector");
  if (!selector.options.length) {
    Object.keys(payload.models || { dummy: "dummy" }).forEach((alias) => {
      const option = document.createElement("option");
      option.value = alias;
      option.textContent = alias;
      selector.appendChild(option);
    });
    if ([...selector.options].some((option) => option.value === "dummy")) {
      selector.value = "dummy";
    }
  }
  syncReaderControls();
}

async function loadSample() {
  await updateDocument(await api("/api/document/load-sample", { method: "POST" }));
  await refreshStatus();
}

async function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  await updateDocument(await api("/api/document/upload", { method: "POST", body: form }));
  await refreshStatus();
}

async function gotoPage(delta) {
  if (!state.pageCount) return;
  const target = Math.max(1, Math.min(state.pageCount, state.currentPage + delta));
  const payload = await api(`/api/document/page/${target}`);
  state.currentPage = payload.page_number;
  updatePageInfo();
  if (state.documentType === "pdf" && state.pdfDoc) {
    await renderPdfPage(target);
  } else {
    setText("paperText", payload.text);
  }
}

async function changeZoom(delta) {
  if (!state.pdfDoc) return;
  state.pdfScale = Math.max(0.7, Math.min(2.5, Number((state.pdfScale + delta).toFixed(2))));
  await renderPdfPage(state.currentPage);
}

function captureSelection() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) return;
  const text = selection.toString().trim();
  if (!text) return;

  if (state.documentType === "pdf" && selectionInside($("pdfTextLayer"), selection)) {
    state.selectionRects = selectionRectsRelativeTo($("pdfPageLayer"), selection);
    state.currentPage = Number($("pdfTextLayer").dataset.pageNumber || state.currentPage);
    state.selectedText = text;
    setText("selectedPreview", text);
    renderHighlights();
    return;
  }

  if (selectionInside($("paperText"), selection)) {
    state.selectionRects = [];
    state.selectedText = text;
    setText("selectedPreview", text);
  }
}

function selectionInside(element, selection) {
  if (!element) return false;
  const anchor = selection.anchorNode;
  const focus = selection.focusNode;
  return (anchor && element.contains(anchor)) || (focus && element.contains(focus));
}

function selectionRectsRelativeTo(pageLayer, selection) {
  if (!pageLayer || selection.rangeCount === 0) return [];
  const base = pageLayer.getBoundingClientRect();
  const pageWidth = pageLayer.clientWidth;
  const pageHeight = pageLayer.clientHeight;
  return [...selection.getRangeAt(0).getClientRects()]
    .map((rect) => ({
      left: Math.max(0, rect.left - base.left),
      top: Math.max(0, rect.top - base.top),
      width: Math.min(rect.width, pageWidth),
      height: Math.min(rect.height, pageHeight),
    }))
    .filter((rect) => rect.width > 1 && rect.height > 1 && rect.left < pageWidth && rect.top < pageHeight)
    .slice(0, 80);
}

function approximateSelectionRects() {
  const pageLayer = $("pdfPageLayer");
  if (!pageLayer || !pageLayer.clientWidth) return [];
  return [
    {
      left: 24,
      top: 36,
      width: Math.max(60, Math.min(pageLayer.clientWidth - 48, 520)),
      height: 24,
    },
  ];
}

async function useSelected() {
  captureSelection();
  const payload = await api("/api/document/context", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_id: state.documentId,
      selected_text: state.selectedText,
      page_number: state.currentPage,
      user_question: $("questionInput").value,
    }),
  });
  state.selectedText = payload.selected_text || state.selectedText;
  state.lastContext = payload;
  setText("selectedPreview", state.selectedText || "No selected passage yet.");
  renderContext(payload);
  return payload;
}

async function saveHighlight() {
  captureSelection();
  if (!state.selectedText) {
    alert("Select text in the paper before highlighting.");
    return null;
  }
  const rects = state.documentType === "pdf"
    ? (state.selectionRects.length ? state.selectionRects : approximateSelectionRects())
    : [];
  const payload = await api("/api/document/highlight", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_id: state.documentId,
      page_number: state.currentPage,
      selected_text: state.selectedText,
      rects,
      color: "yellow",
      user_question: $("questionInput").value,
    }),
  });
  state.lastHighlightId = payload.highlight_id;
  state.lastContext = payload.context;
  state.highlights.push(payload.highlight);
  state.selectionRects = [];
  renderContext(payload.context);
  renderHighlights();
  return payload;
}

async function explainSelected() {
  const highlight = await saveHighlight();
  if (!highlight) return;
  const question = $("questionInput").value.trim() || "Can you explain the highlighted passage?";
  await ask(question, null, { preserveSelection: true });
}

async function setManualOverride(value) {
  const payload = await api("/api/emotion/manual", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ emotion: value }),
  });
  updateEmotion(payload);
}

async function refreshEmotionState() {
  updateEmotion(await api("/api/emotion/state"));
}

async function ask(question, followupAction = null, options = {}) {
  if (!options.preserveSelection) captureSelection();
  const q = question || $("questionInput").value.trim() || state.lastQuestion;
  if (!q && !followupAction) {
    alert("Enter a question, click a follow-up after a previous question, or use Summarize Page/Section.");
    return;
  }
  const selectedText = state.selectedText || (state.lastContext && state.lastContext.selected_text) || "";
  state.lastQuestion = q;
  appendMessage("user", followupAction ? `${followupAction}: ${q}` : q);
  $("chatStatus").textContent = "Waiting for answer...";
  const start = performance.now();
  const payload = await api("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      document_id: state.documentId,
      document_type: state.documentType,
      highlight_id: state.lastHighlightId,
      selected_text: selectedText,
      page_number: state.currentPage,
      user_question: q,
      model_alias: $("modelSelector").value || "dummy",
      followup_action: followupAction,
    }),
  });
  appendMessage("assistant", payload.answer);
  updateEmotion(payload.learning_state);
  state.lastContext = payload.context_debug || state.lastContext;
  $("promptPreview").textContent = payload.prompt_preview || "No prompt preview returned.";
  $("requestSummary").textContent = JSON.stringify(payload.request_summary || {}, null, 2);
  $("contextDebug").textContent = JSON.stringify(payload.context_debug || {}, null, 2);
  $("chatStatus").textContent = `Latency: ${Number(payload.latency || 0).toFixed(2)}s (${Math.round(performance.now() - start)} ms browser round trip)`;
  await refreshStatus();
}

function renderContext(payload) {
  state.lastContext = payload;
  if (payload.selected_text) {
    state.selectedText = payload.selected_text;
    setText("selectedPreview", payload.selected_text);
  }
  setText("passageBadge", `Passage type: ${payload.passage_type || "-"}`);
  setText("difficultyBadge", `Difficulty: ${payload.difficulty_hint || "-"}`);
  setText("surroundingPreview", payload.surrounding_text || "No surrounding context.");
  $("contextDebug").textContent = JSON.stringify(payload, null, 2);
  const chunks = payload.retrieved_chunks || [];
  const list = $("retrievedChunks");
  list.innerHTML = "";
  if (!chunks.length) {
    list.textContent = "No chunks yet.";
    return;
  }
  chunks.forEach((chunk, index) => {
    const item = document.createElement("div");
    item.className = "chunk-card";
    item.textContent = `Chunk ${index + 1}: ${chunk}`;
    list.appendChild(item);
  });
}

function renderHighlights() {
  const layer = $("highlightLayer");
  if (!layer) return;
  layer.innerHTML = "";
  if (state.documentType !== "pdf") return;
  const saved = state.highlights.filter((highlight) => Number(highlight.page_number) === Number(state.currentPage));
  saved.forEach((highlight) => {
    (highlight.rects || []).forEach((rect) => addHighlightRect(rect, highlight.color || "yellow", false));
  });
  state.selectionRects.forEach((rect) => addHighlightRect(rect, "yellow", true));
}

function addHighlightRect(rect, color, pending) {
  const node = document.createElement("div");
  node.className = `highlight-rect${pending ? " pending" : ""}`;
  node.style.left = `${rect.left}px`;
  node.style.top = `${rect.top}px`;
  node.style.width = `${rect.width}px`;
  node.style.height = `${rect.height}px`;
  node.dataset.color = color;
  $("highlightLayer").appendChild(node);
}

function renderTimeline(history) {
  const timeline = $("stateTimeline");
  timeline.innerHTML = "";
  history.slice(-10).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.state} · ${item.trend} · ${Number(item.confidence || 0).toFixed(2)}`;
    timeline.appendChild(li);
  });
  if (!timeline.children.length) {
    const li = document.createElement("li");
    li.textContent = "No state history yet.";
    timeline.appendChild(li);
  }
}

async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Browser webcam API is not available.");
    return;
  }
  state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  $("webcamVideo").srcObject = state.cameraStream;
  state.cameraTimer = window.setInterval(sendFrame, 800);
}

function stopCamera() {
  if (state.cameraTimer) {
    window.clearInterval(state.cameraTimer);
    state.cameraTimer = null;
  }
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }
  $("webcamVideo").srcObject = null;
}

async function sendFrame() {
  const video = $("webcamVideo");
  if (!video.videoWidth) return;
  const canvas = $("frameCanvas");
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const image = canvas.toDataURL("image/jpeg", 0.7);
  try {
    updateEmotion(await api("/api/emotion/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    }));
  } catch (error) {
    console.warn("Frame processing failed", error);
  }
}

function bindEvents() {
  $("loadSampleBtn").addEventListener("click", () => loadSample().catch(alertError));
  $("uploadInput").addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) uploadDocument(file).catch(alertError);
  });
  $("prevPageBtn").addEventListener("click", () => gotoPage(-1).catch(alertError));
  $("nextPageBtn").addEventListener("click", () => gotoPage(1).catch(alertError));
  $("zoomOutBtn").addEventListener("click", () => changeZoom(-0.15).catch(alertError));
  $("zoomInBtn").addEventListener("click", () => changeZoom(0.15).catch(alertError));
  $("paperText").addEventListener("mouseup", captureSelection);
  $("paperText").addEventListener("keyup", captureSelection);
  $("pdfViewer").addEventListener("mouseup", captureSelection);
  $("pdfViewer").addEventListener("keyup", captureSelection);
  $("useSelectedBtn").addEventListener("click", () => useSelected().catch(alertError));
  $("highlightSelectionBtn").addEventListener("click", () => saveHighlight().catch(alertError));
  $("explainSelectedBtn").addEventListener("click", () => explainSelected().catch(alertError));
  $("summarizePageBtn").addEventListener("click", () => ask("Can you summarize the current page or section?").catch(alertError));
  $("askBtn").addEventListener("click", () => ask(null).catch(alertError));
  document.querySelectorAll(".followupBtn").forEach((button) => {
    button.addEventListener("click", () => ask(null, button.dataset.action).catch(alertError));
  });
  $("manualOverride").addEventListener("change", (event) => setManualOverride(event.target.value).catch(alertError));
  $("startCameraBtn").addEventListener("click", () => startCamera().catch(alertError));
  $("stopCameraBtn").addEventListener("click", stopCamera);
  window.addEventListener("pdf-workspace:highlight-created", (event) => {
    const payload = event.detail || {};
    const highlight = payload.highlight || {};
    state.lastHighlightId = payload.highlight_id || highlight.highlight_id || state.lastHighlightId;
    state.selectedText = payload.context?.selected_text || highlight.selected_text || state.selectedText;
    state.lastContext = payload.context || state.lastContext;
    if (state.selectedText) setText("selectedPreview", state.selectedText);
    if (payload.context) renderContext(payload.context);
  });
  window.addEventListener("pdf-workspace:selection-context", (event) => {
    if (event.detail) {
      $("contextDebug").textContent = JSON.stringify(event.detail, null, 2);
    }
  });
}

function alertError(error) {
  console.error(error);
  alert(error.message || String(error));
}

bindEvents();
syncReaderControls();
refreshStatus().then(refreshEmotionState).catch((error) => {
  setText("backendStatus", `Backend error: ${error.message}`);
});
