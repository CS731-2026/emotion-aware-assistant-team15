import React, { useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { GlobalWorkerOptions } from "pdfjs-dist";
import pdfWorkerUrl from "pdfjs-dist/build/pdf.worker.min.mjs?url";
import {
  AreaHighlight,
  MonitoredHighlightContainer,
  PdfHighlighter,
  PdfLoader,
  TextHighlight,
  useHighlightContainerContext,
} from "react-pdf-highlighter-plus";
import "react-pdf-highlighter-plus/style/style.css";
import "./workspace.css";

GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

const roots = new Map();

function api(path, options = {}) {
  return fetch(path, options).then(async (response) => {
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  });
}

function PdfWorkspace({ document }) {
  const [highlights, setHighlights] = useState(document.highlights || []);
  const [threads, setThreads] = useState([]);
  const [areaMode, setAreaMode] = useState(false);
  const [zoom, setZoom] = useState("page-width");
  const [page, setPage] = useState(1);
  const [status, setStatus] = useState("PDF workspace ready.");
  const highlighterRef = useRef(null);

  const normalizedHighlights = useMemo(
    () => highlights.map((highlight) => toViewerHighlight(highlight)).filter(Boolean),
    [highlights],
  );

  const onSelection = async (selection) => {
    const selectedText = (selection.content?.text || "").trim();
    const confidence = selection.type === "area" ? 0 : scoreSelectedText(selectedText, document.currentPageText || "");
    const fallbackNeeded = selection.type === "area" || confidence < 0.5;
    const croppedImage = fallbackNeeded
      ? selection.content?.image || cropSelectionFromCanvas(selection.position)
      : "";
    const columnSide = columnSideFromPosition(selection.position);
    const highlightType = selection.type === "area" ? "area" : fallbackNeeded ? "screenshot_fallback" : "text";

    setStatus(highlightType === "text" ? "Saving text highlight..." : "Saving screenshot-backed highlight...");
    const highlightPayload = await api("/api/document/highlight", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_id: document.documentId,
        page_number: pageNumberFromPosition(selection.position),
        highlight_type: highlightType,
        selected_text: fallbackNeeded ? "" : selectedText,
        rects: [],
        scaled_rects: selection.position?.rects || [],
        position: selection.position,
        cropped_image: croppedImage,
        column_side: columnSide,
        color: "yellow",
        user_question: "Can you explain this highlighted PDF selection?",
      }),
    });
    const savedHighlight = highlightPayload.highlight;
    setHighlights((current) => [...current, savedHighlight]);
    emitWorkspaceEvent("pdf-workspace:highlight-created", highlightPayload);
    const thread = await explainHighlight(savedHighlight, fallbackNeeded ? "" : selectedText);
    setThreads((current) => [...current, thread]);
    setStatus("Highlight saved with linked explanation.");
  };

  const explainHighlight = async (highlight, selectedText) => {
    const fallbackNote = highlight.highlight_type === "text"
      ? ""
      : " The selection is screenshot-backed because it is an area/figure/formula/table or text extraction was unreliable.";
    try {
      const payload = await api("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: document.documentId,
          highlight_id: highlight.highlight_id,
          selected_text: selectedText,
          page_number: highlight.page_number,
          user_question: `Can you explain this highlighted PDF selection?${fallbackNote}`,
          model_alias: "dummy",
        }),
      });
      emitWorkspaceEvent("pdf-workspace:selection-context", payload.context_debug || {});
      return {
        threadId: highlight.explanation_thread?.thread_id || `thread-${highlight.highlight_id}`,
        highlightId: highlight.highlight_id,
        railSide: highlight.explanation_thread?.rail_side || oppositeRail(highlight.column_side),
        title: highlight.highlight_type === "text" ? "Text highlight" : "Area highlight",
        body: payload.answer,
        metadata: {
          highlightType: highlight.highlight_type,
          textConfidence: highlight.text_confidence,
          croppedImagePath: highlight.cropped_image_path,
        },
      };
    } catch (error) {
      return {
        threadId: highlight.explanation_thread?.thread_id || `thread-${highlight.highlight_id}`,
        highlightId: highlight.highlight_id,
        railSide: highlight.explanation_thread?.rail_side || oppositeRail(highlight.column_side),
        title: "Explanation unavailable",
        body: error.message || String(error),
        metadata: {
          highlightType: highlight.highlight_type,
          textConfidence: highlight.text_confidence,
          croppedImagePath: highlight.cropped_image_path,
        },
      };
    }
  };

  const goToPage = (delta) => {
    const next = Math.max(1, Math.min(document.pageCount || 999, page + delta));
    setPage(next);
    highlighterRef.current?.goToPage(next);
  };

  return (
    <div className="pdf-workspace-shell">
      <aside className="pdf-explanation-rail pdf-explanation-rail-left">
        <Rail title="Left rail" threads={threads.filter((thread) => thread.railSide === "left")} />
      </aside>
      <section className="pdf-workspace-main">
        <div className="pdf-workspace-toolbar">
          <button type="button" onClick={() => goToPage(-1)}>Previous page</button>
          <span>Page {page}</span>
          <button type="button" onClick={() => goToPage(1)}>Next page</button>
          <button type="button" onClick={() => setZoom((current) => numericZoom(current) - 0.15)}>Zoom -</button>
          <button type="button" onClick={() => setZoom((current) => numericZoom(current) + 0.15)}>Zoom +</button>
          <button type="button" className={areaMode ? "active" : ""} onClick={() => setAreaMode((value) => !value)}>
            Area select
          </button>
        </div>
        <p className="pdf-workspace-status">{status}</p>
        <PdfLoader
          document={document.pdfUrl}
          workerSrc={pdfWorkerUrl}
          beforeLoad={<div className="pdf-workspace-loading">Loading PDF...</div>}
        >
          {(pdfDocument) => (
            <PdfHighlighter
              pdfDocument={pdfDocument}
              highlights={normalizedHighlights}
              pdfScaleValue={zoom}
              onSelection={onSelection}
              enableAreaSelection={() => areaMode}
              areaSelectionMode={areaMode}
              textSelectionColor="rgba(255, 226, 143, 0.35)"
              utilsRef={(utils) => {
                highlighterRef.current = utils;
              }}
            >
              <HighlightRenderer />
            </PdfHighlighter>
          )}
        </PdfLoader>
      </section>
      <aside className="pdf-explanation-rail pdf-explanation-rail-right">
        <Rail title="Right rail" threads={threads.filter((thread) => thread.railSide !== "left")} />
      </aside>
    </div>
  );
}

function HighlightRenderer() {
  const { highlight, isScrolledTo } = useHighlightContainerContext();
  const highlightTip = {
    position: highlight.position,
    content: (
      <div className="pdf-highlight-tip">
        <strong>{highlight.highlight_type || highlight.type}</strong>
        <span>{highlight.selected_text_preview || "Screenshot-backed area"}</span>
      </div>
    ),
  };
  return (
    <MonitoredHighlightContainer highlightTip={highlightTip}>
      {(highlight.type || highlight.highlight_type) === "area" || highlight.highlight_type === "screenshot_fallback" ? (
        <AreaHighlight highlight={highlight} isScrolledTo={isScrolledTo} />
      ) : (
        <TextHighlight highlight={highlight} isScrolledTo={isScrolledTo} />
      )}
    </MonitoredHighlightContainer>
  );
}

function Rail({ title, threads }) {
  return (
    <>
      <h3>{title}</h3>
      {threads.length === 0 ? <p className="pdf-rail-empty">No linked explanations yet.</p> : null}
      {threads.map((thread) => (
        <article className="pdf-explanation-card" key={thread.threadId}>
          <h4>{thread.title}</h4>
          <p>{thread.body}</p>
          <dl>
            <dt>Type</dt><dd>{thread.metadata.highlightType}</dd>
            <dt>Confidence</dt><dd>{Number(thread.metadata.textConfidence || 0).toFixed(2)}</dd>
            <dt>Crop</dt><dd>{thread.metadata.croppedImagePath ? "stored" : "not needed"}</dd>
          </dl>
        </article>
      ))}
    </>
  );
}

function toViewerHighlight(highlight) {
  const position = highlight.position || {
    boundingRect: highlight.scaled_rects?.[0],
    rects: highlight.scaled_rects || [],
  };
  if (!position?.boundingRect) return null;
  return {
    ...highlight,
    id: highlight.highlight_id || highlight.id,
    type: highlight.highlight_type === "screenshot_fallback" ? "area" : highlight.highlight_type || "text",
    content: {
      text: highlight.selected_text || "",
    },
    position,
  };
}

function scoreSelectedText(text, pageText) {
  const value = (text || "").trim();
  if (!value) return 0;
  if (value.includes("\ufffd")) return 0.15;
  const printable = [...value].filter((char) => char.trim() && char >= " " && char !== "\u007f").length;
  const alnum = [...value].filter((char) => /[A-Za-z0-9]/.test(char)).length;
  if (printable / Math.max(value.length, 1) < 0.75 || alnum / Math.max(printable, 1) < 0.45) return 0.25;
  const normalized = value.replace(/\s+/g, " ").toLowerCase();
  const normalizedPage = (pageText || "").replace(/\s+/g, " ").toLowerCase();
  if (normalizedPage && normalizedPage.includes(normalized)) return 0.98;
  if (normalized.split(" ").length <= 2) return 0.45;
  return 0.65;
}

function cropSelectionFromCanvas(position) {
  const rect = position?.boundingRect;
  if (!rect) return "";
  const pageNumber = pageNumberFromPosition(position);
  const pageNode = document.querySelector(`#pdfWorkspaceRoot .page[data-page-number="${pageNumber}"]`);
  const canvas = pageNode?.querySelector("canvas");
  if (!canvas) return "";
  const scaleX = canvas.width / Math.max(rect.width || canvas.clientWidth || 1, 1);
  const scaleY = canvas.height / Math.max(rect.height || canvas.clientHeight || 1, 1);
  const width = Math.max(1, (rect.x2 - rect.x1) * scaleX);
  const height = Math.max(1, (rect.y2 - rect.y1) * scaleY);
  const crop = document.createElement("canvas");
  crop.width = width;
  crop.height = height;
  const ctx = crop.getContext("2d");
  ctx.drawImage(canvas, rect.x1 * scaleX, rect.y1 * scaleY, width, height, 0, 0, width, height);
  return crop.toDataURL("image/png");
}

function pageNumberFromPosition(position) {
  return Number(position?.boundingRect?.pageNumber || position?.rects?.[0]?.pageNumber || 1);
}

function columnSideFromPosition(position) {
  const rect = position?.boundingRect;
  if (!rect) return "full_width";
  const width = Math.max(Number(rect.width || 1), 1);
  const x1 = Number(rect.x1 || 0) / width;
  const x2 = Number(rect.x2 || 0) / width;
  if (x2 - x1 > 0.55 || (x1 < 0.4 && x2 > 0.6)) return "full_width";
  return (x1 + x2) / 2 < 0.5 ? "left" : "right";
}

function oppositeRail(columnSide) {
  if (columnSide === "right") return "left";
  return "right";
}

function numericZoom(value) {
  const numeric = typeof value === "number" ? value : 1.15;
  return Math.max(0.7, Math.min(2.5, Number(numeric.toFixed(2))));
}

function emitWorkspaceEvent(name, detail) {
  window.dispatchEvent(new CustomEvent(name, { detail }));
}

window.PdfWorkspaceIsland = {
  mount(target, document) {
    if (!target) return false;
    let root = roots.get(target);
    if (!root) {
      root = createRoot(target);
      roots.set(target, root);
    }
    root.render(<PdfWorkspace document={document} />);
    return true;
  },
  unmount(target) {
    const root = roots.get(target);
    if (!root) return;
    root.unmount();
    roots.delete(target);
  },
};
