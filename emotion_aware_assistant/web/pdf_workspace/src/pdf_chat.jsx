import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  AreaHighlight,
  PdfHighlighter,
  PdfLoader,
  TextHighlight,
  useHighlightContainerContext,
} from "react-pdf-highlighter-plus";
import "pdfjs-dist/web/pdf_viewer.css";
import "react-pdf-highlighter-plus/style/style.css";
import "./pdf_chat.css";

const REACTION_WINDOW_MAX_MS = 60000;
const REACTION_WINDOW_SAMPLE_MS = 2000;
const REACTION_WINDOW_MIN_SAMPLES = 2;
const REACTION_WINDOW_SENTINEL_DELAY_MS = 1000;
const REACTION_PROGRAMMATIC_SCROLL_MS = 800;
const REACTION_WINDOW_LAYOUT_SETTLE_MS = 400;
const REACTION_WINDOW_MIN_READ_SEC_FLOOR = 8;
const REACTION_WINDOW_MIN_READ_SEC_CEIL = 25;
const REACTION_WINDOW_READING_WORDS_PER_SEC = 12;
const SPEECH_RECORDING_MAX_MS = 15000;

function HighlightContainer({ onActivate }) {
  const { highlight, isScrolledTo } = useHighlightContainerContext();

  function handleClick(event) {
    event.stopPropagation();
    onActivate?.(highlight);
  }

  if (highlight.type === "area") {
    return (
      <div className="pdf-chat-highlight-shell" onClickCapture={handleClick}>
        <AreaHighlight highlight={highlight} isScrolledTo={isScrolledTo} />
      </div>
    );
  }

  return (
    <span className="pdf-chat-highlight-shell" onClickCapture={handleClick}>
      <TextHighlight highlight={highlight} isScrolledTo={isScrolledTo} />
    </span>
  );
}

function PdfDocumentView({
  pdfDocument,
  highlights,
  areaMode,
  onPdfDocumentLoaded,
  onSelection,
  onHighlightClick,
  highlighterUtilsRef,
}) {
  useEffect(() => {
    onPdfDocumentLoaded(pdfDocument);
  }, [pdfDocument, onPdfDocumentLoaded]);

  return (
    <PdfHighlighter
      pdfDocument={pdfDocument}
      highlights={highlights}
      onSelection={onSelection}
      enableAreaSelection={(event) => areaMode || event.altKey}
      areaSelectionMode={areaMode}
      utilsRef={(utils) => {
        highlighterUtilsRef.current = utils;
      }}
    >
      <HighlightContainer onActivate={onHighlightClick} />
    </PdfHighlighter>
  );
}

function PdfChatApp() {
  const [activeDocument, setActiveDocument] = useState(null);
  const [libraryKey, setLibraryKey] = useState(0);

  function handleBackToLibrary() {
    setActiveDocument(null);
    setLibraryKey((value) => value + 1);
  }

  if (activeDocument) {
    return (
      <PdfChatWorkspace
        documentSummary={activeDocument}
        onBack={handleBackToLibrary}
        onDocumentChanged={setActiveDocument}
      />
    );
  }

  return <PaperLibrary key={libraryKey} onOpenDocument={setActiveDocument} />;
}

function PaperLibrary({ onOpenDocument }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const [uploadStatus, setUploadStatus] = useState("");
  const [preparingDocument, setPreparingDocument] = useState(null);
  const [prepareStatus, setPrepareStatus] = useState(null);

  async function loadDocuments() {
    setLoading(true);
    setError("");
    try {
      const payload = await fetchJson("/api/documents?library_only=1");
      setDocuments(payload.documents || []);
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadDocuments();
  }, []);

  async function handleUpload(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setUploading(true);
    setError("");
    setUploadStatus("Uploading PDF");
    const formData = new FormData();
    formData.append("file", file);
    try {
      setUploadStatus("Extracting text and layout");
      const response = await fetch("/api/documents/upload", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }
      setPreparingDocument(payload.meta || { document_id: payload.document_id, file_name: file.name });
      setPrepareStatus(payload.prepare_status || null);
      const prepared = await pollPreparationStatus(payload.document_id, setPrepareStatus);
      const refreshed = await fetchJson(`/api/documents/${payload.document_id}`);
      setUploadStatus("Ready");
      await loadDocuments();
      setPreparingDocument(refreshed.meta || prepared.meta || payload.meta || { document_id: payload.document_id, file_name: file.name });
    } catch (err) {
      setError(err?.message || String(err));
      setUploadStatus("Upload failed");
    } finally {
      setUploading(false);
    }
  }

  async function handleArchiveDocument(document) {
    if (!document?.document_id) {
      setError("Cannot remove this paper because its document id is missing.");
      return;
    }
    if (!window.confirm("Remove this paper from the library?")) return;
    setError("");
    try {
      await postJson(`/api/documents/${document.document_id}/archive`, {});
      await loadDocuments();
    } catch (err) {
      setError(err?.message || String(err));
    }
  }

  return (
    <main className="pdf-chat-library">
      <header className="pdf-chat-library-header">
        <div>
          <h1>Paper Reading Assistant</h1>
          <p>Upload a paper once, then reopen it from your archive with saved selections and threads.</p>
        </div>
        <div className="pdf-chat-library-tools" aria-label="Library tools">
          <a
            href="/settings"
            target="_blank"
            rel="noreferrer"
            title="Open settings. Users can configure API keys and model roles before uploading."
          >
            Settings
          </a>
          <label className="pdf-chat-primary-action">
            Upload PDF
            <input type="file" accept="application/pdf" onChange={handleUpload} disabled={uploading} hidden />
          </label>
        </div>
      </header>

      <section className="pdf-chat-library-grid">
        <UploadPaperCard
          uploading={uploading}
          uploadStatus={uploadStatus}
          prepareStatus={prepareStatus}
          preparingDocument={preparingDocument}
          onUpload={handleUpload}
        />
        {loading ? <p className="pdf-chat-muted">Loading papers...</p> : null}
        {error ? (
          <div className="pdf-chat-error">
            <p>{error}</p>
            <button type="button" onClick={loadDocuments}>Retry</button>
          </div>
        ) : null}
        {!loading && !documents.length ? (
          <div className="pdf-chat-empty-state">
            <h2>No saved papers yet</h2>
            <p>Upload a PDF to prepare it for grounded selection explanations.</p>
          </div>
        ) : null}
        {documents.map((document) => (
          <PaperCard
            key={document.document_id}
            document={document}
            onOpen={() => onOpenDocument(document)}
            onArchiveDocument={handleArchiveDocument}
          />
        ))}
      </section>
    </main>
  );
}

function UploadPaperCard({ uploading, uploadStatus, prepareStatus, preparingDocument, onUpload }) {
  return (
    <article className="pdf-chat-upload-card">
      <div className="pdf-chat-file-icon" aria-hidden="true">PDF</div>
      <h2>Upload a paper</h2>
      <p>Preparation runs in the background so later explanations can use profile, local context, and global retrieval.</p>
      <label className="pdf-chat-secondary-action">
        Choose PDF
        <input type="file" accept="application/pdf" onChange={onUpload} disabled={uploading} hidden />
      </label>
      <PreparationStatus status={uploadStatus} uploading={uploading} meta={preparingDocument} prepareStatus={prepareStatus} />
    </article>
  );
}

function PaperCard({ document, onOpen, onArchiveDocument }) {
  const title = document.title || document.file_name || "Untitled PDF";
  const highlightCount = Number(document.highlight_count || 0);
  const threadCount = Number(document.thread_count || 0);

  return (
    <article className="pdf-chat-paper-card">
      <button type="button" className="pdf-chat-paper-open" onClick={onOpen}>
        <span className="pdf-chat-file-icon" aria-hidden="true">PDF</span>
        <span className="pdf-chat-paper-title">{title}</span>
        <span className="pdf-chat-paper-meta">{document.file_name || "original.pdf"}</span>
        <span className="pdf-chat-paper-stats">
          {pageCountLabel(document.page_count)} · {preparationStatusLabel(document.prepare_status)}
        </span>
        <span className="pdf-chat-paper-stats">
          {countLabel(highlightCount, "highlight")} · {countLabel(threadCount, "chat thread")}
        </span>
        <span className="pdf-chat-paper-stats">
          Last opened {formatTime(document.last_opened_at) || "not yet"}
        </span>
      </button>
      <details className="pdf-chat-paper-details">
        <summary>Technical details</summary>
        <dl className="pdf-chat-small-grid">
          <dt>retrieval method</dt>
          <dd>{document.retrieval_method || "unknown"}</dd>
          <dt>embedding status</dt>
          <dd>{document.embedding_status || "unknown"}</dd>
          <dt>parsed blocks</dt>
          <dd>{String(document.parsed_blocks_count ?? "-")}</dd>
        </dl>
      </details>
      <div className="pdf-chat-paper-actions">
        <button type="button" className="pdf-chat-danger-action compact" onClick={() => onArchiveDocument?.(document)}>
          Remove
        </button>
      </div>
    </article>
  );
}

function PreparationStatus({ status, uploading, meta, prepareStatus }) {
  const progress_percent = Number(prepareStatus?.progress_percent ?? (status === "Ready" ? 100 : 0));
  const elapsed_seconds = Number(prepareStatus?.elapsed_seconds ?? 0);
  const estimated_remaining_seconds = prepareStatus?.estimated_remaining_seconds;
  const steps = prepareStatus?.steps || defaultPrepareSteps(status);
  const stageLabel = prepareStatus?.stage_label || status || meta?.prepare_status || "Waiting for upload";
  const complete = isPreparationComplete(status, meta, prepareStatus);
  const retrievalLabel = retrievalSummary(prepareStatus?.retrieval_method || meta?.retrieval_method);
  const pageLabel = pageCountLabel(meta?.page_count || prepareStatus?.page_count);

  if (complete) {
    return (
      <div className="pdf-chat-preparation compact">
        <div className="pdf-chat-ready-summary">
          Ready · {pageLabel} · {retrievalLabel} · prepared in {formatDuration(elapsed_seconds)}
        </div>
        <details className="pdf-chat-preparation-details">
          <summary>Preparation details</summary>
          <PreparationStepList steps={steps} />
          <PreparationTechnicalDetails meta={meta} prepareStatus={prepareStatus} />
        </details>
      </div>
    );
  }

  return (
    <div className="pdf-chat-preparation">
      <div className="pdf-chat-progress-summary">
        <span>{stageLabel}</span>
        <strong>{Math.round(progress_percent)}%</strong>
      </div>
      <div className="pdf-chat-progress-track" aria-label="Preparation progress">
        <div className="pdf-chat-progress-bar" style={{ width: `${Math.max(0, Math.min(progress_percent, 100))}%` }} />
      </div>
      <div className="pdf-chat-progress-time">
        <span>Elapsed {formatDuration(elapsed_seconds)}</span>
        <span>{estimated_remaining_seconds == null ? "Estimating..." : `${formatDuration(estimated_remaining_seconds)} remaining`}</span>
      </div>
      <PreparationStepList steps={steps} />
      <PreparationTechnicalDetails meta={meta} prepareStatus={prepareStatus} />
    </div>
  );
}

function PreparationStepList({ steps }) {
  return (
    <ol className="pdf-chat-progress-steps">
      {steps.map((step) => (
        <li key={step.id || step.label} className={`pdf-chat-stage ${step.status || "pending"}`}>
          <span aria-hidden="true">{step.status === "completed" ? "✓" : step.status === "active" ? "•" : "○"}</span>
          {step.label}
        </li>
      ))}
    </ol>
  );
}

function PreparationTechnicalDetails({ meta, prepareStatus }) {
  if (!prepareStatus && !meta) return null;
  return (
    <dl className="pdf-chat-small-grid">
      <dt>parsed blocks</dt>
      <dd>{String(prepareStatus?.block_count ?? meta?.parsed_blocks_count ?? "-")}</dd>
      <dt>embedding</dt>
      <dd>{prepareStatus?.embedding_index_status || meta?.embedding_status || "-"}</dd>
    </dl>
  );
}

function PdfChatWorkspace({ documentSummary, onBack, onDocumentChanged }) {
  const documentId = documentSummary.document_id;
  const documentReady = Boolean(documentId);
  const [documentDetail, setDocumentDetail] = useState(null);
  const [highlights, setHighlights] = useState([]);
  const [pendingSelection, setPendingSelection] = useState(null);
  const [activeSelection, setActiveSelection] = useState(null);
  const [areaMode, setAreaMode] = useState(false);
  const [thread, setThread] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [explainLoading, setExplainLoading] = useState(false);
  const [followUpLoading, setFollowUpLoading] = useState(false);
  const [cleanupLoading, setCleanupLoading] = useState(false);
  const [followUpText, setFollowUpText] = useState("");
  const [followUpInputSource, setFollowUpInputSource] = useState("text");
  const [readingSessionId, setReadingSessionId] = useState("");
  const [learningState, setLearningState] = useState(null);
  const [strategyCandidates, setStrategyCandidates] = useState([]);
  const [strategyPlannerMode, setStrategyPlannerMode] = useState("");
  const [strategyTriggerContext, setStrategyTriggerContext] = useState(null);
  const [selectedStrategy, setSelectedStrategy] = useState(null);
  const [strategyLoading, setStrategyLoading] = useState(false);
  const [strategySourceKey, setStrategySourceKey] = useState("");
  const [modelStatus, setModelStatus] = useState(null);
  const [learningSignalSource, setLearningSignalSource] = useState("simulated");
  const [liveSignalActive, setLiveSignalActive] = useState(false);
  const [liveSignalError, setLiveSignalError] = useState("");
  const [showSelfView, setShowSelfView] = useState(false);
  const [cameraPausedByUser, setCameraPausedByUser] = useState(false);
  const [cameraStartStatus, setCameraStartStatus] = useState("standby");
  const [reactionWindowActive, setReactionWindowActive] = useState(false);
  const [reactionWindowSummary, setReactionWindowSummary] = useState(null);
  const [reactionWindowScrollTargetId, setReactionWindowScrollTargetId] = useState("");
  const highlighterUtilsRef = useRef(null);
  const pendingSelectionRef = useRef(null);
  const selectionDebugRef = useRef(emptySelectionDebug());
  const strategyCooldownRef = useRef({});
  const strategyRequestRef = useRef(false);
  const dismissedStrategyKeysRef = useRef(new Set());
  const reactionWindowRunRef = useRef(0);
  const activeSelectionRef = useRef(null);
  const learningStateRef = useRef(null);
  const threadRef = useRef(null);
  const highlightsRef = useRef([]);
  const latestThreadRequestRef = useRef(0);
  const cameraVideoRef = useRef(null);
  const cameraCanvasRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const cameraTimerRef = useRef(null);
  const cameraPausedByUserRef = useRef(false);
  const cameraSelfViewPreferenceRef = useRef("visible");
  const completedReactionTurnIdsRef = useRef(new Set());
  const reactionWindowTurnIdRef = useRef("");
  const reactionWindowStateRef = useRef(null);
  const reactionWindowSampleTimerRef = useRef(null);
  const reactionWindowMinTimerRef = useRef(null);
  const reactionWindowMaxTimerRef = useRef(null);
  const reactionWindowSentinelTimerRef = useRef(null);
  const reactionWindowProgrammaticScrollTimerRef = useRef(null);
  const reactionWindowLayoutTimerRef = useRef(null);
  const reactionWindowObserverRef = useRef(null);
  const reactionSentinelRefs = useRef(new Map());
  const pdfUrl = documentReady ? `/api/documents/${documentId}/file` : "";

  const currentMeta = documentDetail?.meta || documentSummary;
  const prepareStatus = documentDetail?.prepare_status || {};
  const viewerHighlights = useMemo(() => highlights.map(toViewerHighlight), [highlights]);
  const strategySourceTurnId = strategyTriggerContext?.source_turn_id || reactionWindowSummary?.source_turn_id || "";

  useEffect(() => {
    async function loadWorkspace() {
      if (!documentReady) {
        setError("Document is still loading; wait for a stable document id before opening the workspace.");
        setLoading(false);
        return;
      }
      setLoading(true);
      setError("");
      try {
        const [detail, savedHighlights, model] = await Promise.all([
          fetchJson(`/api/documents/${documentId}`),
          fetchJson(`/api/documents/${documentId}/highlights`),
          fetchJson("/api/emotion/model/status").catch(() => null),
          postJson(`/api/documents/${documentId}/open`, {}),
        ]);
        setDocumentDetail(detail);
        applyLoadedHighlights(savedHighlights.highlights || []);
        setModelStatus(model);
        onDocumentChanged?.(detail.meta || documentSummary);
        const session = await postJson(`/api/documents/${documentId}/reading-session/start`, {});
        setReadingSessionId(session.session_id || "");
        setLearningState(session.learning_state || null);
      } catch (err) {
        setError(err?.message || String(err));
      } finally {
        setLoading(false);
      }
    }
    loadWorkspace();
  }, [documentId, documentReady]);

  useEffect(() => {
    if (!documentDetail || prepareStatus.status === "completed" || prepareStatus.status === "failed") return undefined;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const detail = await fetchJson(`/api/documents/${documentId}`);
        if (cancelled) return;
        setDocumentDetail(detail);
        onDocumentChanged?.(detail.meta || documentSummary);
      } catch {
        // Polling is best-effort; explicit user actions still surface errors.
      }
    }, 1000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [documentId, documentDetail, prepareStatus.status]);

  useEffect(() => {
    if (!readingSessionId) return undefined;
    if (learningSignalSource === "webcam") return undefined;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const payload = await fetchJson(`/api/reading-sessions/${readingSessionId}/learning-state/current`);
        if (!cancelled) setLearningState(payload.learning_state || null);
      } catch {
        // Learning-state polling is passive; core reading actions should continue.
      }
    }, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [readingSessionId, learningSignalSource]);

  useEffect(() => () => {
    clearReactionWindowTimers();
    stopLiveSignal("unmount");
  }, []);

  useEffect(() => {
    activeSelectionRef.current = activeSelection;
  }, [activeSelection]);

  useEffect(() => {
    learningStateRef.current = learningState;
  }, [learningState]);

  useEffect(() => {
    threadRef.current = thread;
  }, [thread]);

  useEffect(() => {
    highlightsRef.current = highlights;
  }, [highlights]);

  function applyLoadedHighlights(rawHighlights) {
    const normalizedHighlights = (rawHighlights || [])
      .map((item) => normalizePersistedHighlight(item, documentId, "restored"))
      .filter(Boolean);
    logPdfChatRestoreDebug("highlights_loaded", {
      event: "highlights_loaded",
      document_id: documentId,
      highlight_count: normalizedHighlights.length,
      highlight_ids: normalizedHighlights.map((item) => item.highlight_id),
    });
    if (shouldIgnoreEmptyHighlightsLoad(normalizedHighlights, highlightsRef.current, documentId)) return;
    setHighlights(normalizedHighlights);
    highlightsRef.current = normalizedHighlights;
  }

  const onPdfDocumentLoaded = useCallback((pdfDocument) => {
    const pageCount = Number(pdfDocument?.numPages || 0);
    if (!Number.isFinite(pageCount) || pageCount <= 0) return;
    setDocumentDetail((current) => {
      if (!current?.meta || Number(current.meta.page_count || 0) === pageCount) return current;
      return { ...current, meta: { ...current.meta, page_count: pageCount } };
    });
  }, []);

  function resetStrategyState() {
    setStrategyCandidates([]);
    setStrategyPlannerMode("");
    setStrategyTriggerContext(null);
    setSelectedStrategy(null);
    setStrategySourceKey("");
    setReactionWindowSummary(null);
    setReactionWindowScrollTargetId("");
    reactionWindowRunRef.current += 1;
    reactionWindowTurnIdRef.current = "";
    reactionWindowStateRef.current = null;
    completedReactionTurnIdsRef.current = new Set();
    setReactionWindowActive(false);
  }

  async function postSessionEvent(eventType, extra = {}) {
    if (!readingSessionId) return null;
    return postJson(`/api/reading-sessions/${readingSessionId}/events`, {
      event_type: eventType,
      document_id: documentId,
      highlight_id: activeSelection?.llmInputPreview?.highlight_id || extra.highlight_id || "",
      ...extra,
    }).catch(() => null);
  }

  function applyThreadStrategyState(payload) {
    if (!payload) return;
    const metadata = latestTurnMetadata(payload);
    completedReactionTurnIdsRef.current = completedReactionTurnIdsFromThread(payload);
    const threadStrategy = payload.selected_strategy && Object.keys(payload.selected_strategy).length
      ? payload.selected_strategy
      : null;
    setSelectedStrategy(threadStrategy);
    setStrategyCandidates(Array.isArray(metadata.strategy_candidates) ? metadata.strategy_candidates : Array.isArray(payload.strategy_candidates) ? payload.strategy_candidates : []);
    setStrategyPlannerMode(metadata.planner_mode || payload.planner_mode || "");
    setStrategyTriggerContext(metadata.trigger_context && Object.keys(metadata.trigger_context).length ? metadata.trigger_context : payload.trigger_context && Object.keys(payload.trigger_context).length ? payload.trigger_context : null);
    setReactionWindowSummary(metadata.reaction_window_summary && Object.keys(metadata.reaction_window_summary).length ? metadata.reaction_window_summary : reactionSummaryFromThread(payload));
  }

  function handleSelection(selection) {
    const browserText = window.getSelection()?.toString() || "";
    const libraryText = selection?.content?.text || "";
    const selectionCropImage = selection?.content?.image || "";
    selectionDebugRef.current = {
      browserText,
      normalizedBrowserText: normalizePdfText(browserText),
      libraryText,
      normalizedLibraryText: normalizePdfText(libraryText),
      selectionCropImage,
      selectionKeys: Object.keys(selection || {}),
      selectionType: selection?.type || "",
    };

    if (!selection || !["text", "area"].includes(selection.type)) {
      pendingSelectionRef.current = null;
      setPendingSelection(null);
      return;
    }
    pendingSelectionRef.current = selection;
    setPendingSelection(selection);
  }

  async function handleCreateHighlight() {
    const currentSelection = pendingSelectionRef.current;
    if (!currentSelection) return;
    if (!documentReady) {
      setError("Document is still loading; wait for a stable document id before saving highlights.");
      return;
    }
    setError("");
    resetStrategyState();
    const ghost = currentSelection.makeGhostHighlight();
    const id = makeId();
    const highlight = { ...ghost, id, highlight_id: id, document_id: documentId };
    const debugSource = selectionDebugRef.current;
    const rawText = ghost?.content?.text || debugSource.browserText || "";
    const cropImage = ghost?.content?.image || debugSource.selectionCropImage || "";
    const highlightDebug = {
      id,
      type: ghost?.type || "",
      rawText,
      normalizedText: normalizePdfText(rawText),
      textLength: rawText.length,
      cropImage,
      empty: rawText.length === 0,
      suspicious: ghost?.type === "area" ? false : isSuspiciousText(rawText),
      pageNumber: pageNumberFromPosition(ghost?.position),
      position: ghost?.position || null,
      boundingRect: ghost?.position?.boundingRect || null,
      rects: ghost?.position?.rects || [],
    };
    const baseMatch = emptyMatchDebug(documentId, highlightDebug.id);
    const initialSelection = selectionStateFrom(highlight, highlightDebug, baseMatch);
    const initialThread = emptyThreadForSelection(documentId, highlightDebug.id, initialSelection);
    setHighlights((current) => [...current, highlight]);
    setActiveSelection(initialSelection);
    setThread(initialThread);
    threadRef.current = initialThread;
    applyThreadStrategyState(initialThread);
    pendingSelectionRef.current = null;
    setPendingSelection(null);
    highlighterUtilsRef.current?.setTip?.(null);
    window.setTimeout(() => highlighterUtilsRef.current?.removeGhostHighlight?.(), 0);

    let nextSelection = selectionStateFrom(highlight, highlightDebug, baseMatch);
    try {
      const matchDebug = await matchHighlightToBlocks(documentId, highlight, highlightDebug);
      nextSelection = selectionStateFrom(highlight, highlightDebug, matchDebug);
      setActiveSelection(nextSelection);
    } catch (err) {
      const failedMatch = { ...baseMatch, error: err?.message || String(err) };
      nextSelection = selectionStateFrom(highlight, highlightDebug, failedMatch);
      setActiveSelection(nextSelection);
    }

    const nextHighlights = [...highlights, {
      ...highlight,
      ...highlightMetadataForSave(nextSelection),
    }];
    setHighlights(nextHighlights);
    try {
      await saveHighlights(nextHighlights);
    } catch (err) {
      setError(`Highlight could not be saved: ${err?.message || err}`);
      setHighlights((current) => current.filter((item) => (item.highlight_id || item.id) !== highlightDebug.id));
      setActiveSelection(null);
      setThread(null);
      threadRef.current = null;
      return;
    }
    if (highlightDebug.cropImage) {
      const crop = await postJson(`/api/documents/${documentId}/highlights/${highlightDebug.id}/crop`, {
        crop_image_data_url: highlightDebug.cropImage,
      }).catch(() => null);
      if (crop?.crop_path || crop?.crop_url) {
        const cropMetadata = {
          crop_path: crop.crop_path || "",
          crop_image_path: crop.crop_image_path || crop.crop_path || "",
          crop_url: crop.crop_url || cropUrlForHighlight({ document_id: documentId, highlight_id: highlightDebug.id }),
        };
        const nextHighlightsWithCrop = nextHighlights.map((item) => (
          (item.highlight_id || item.id) === highlightDebug.id ? { ...item, ...cropMetadata } : item
        ));
        setHighlights(nextHighlightsWithCrop);
        setActiveSelection((current) => current ? {
          ...current,
          highlight: { ...current.highlight, ...cropMetadata },
          highlightDebug: { ...current.highlightDebug, cropImagePath: cropMetadata.crop_image_path, cropImageUrl: cropMetadata.crop_url },
          llmInputPreview: { ...current.llmInputPreview, ...cropMetadata },
        } : current);
      }
    }
    await postSessionEvent(ghost?.type === "area" ? "area_selected" : "highlight_created", {
      highlight_id: highlightDebug.id,
      selection_type: ghost?.type || "",
    });
  }

  async function handleHighlightClick(highlight) {
    return activatePersistedHighlight(highlight);
  }

  async function activatePersistedHighlight(highlight) {
    resetStrategyState();
    setError("");
    const normalized = normalizePersistedHighlight(highlight, documentId, "restored");
    if (!normalized?.highlight_id) {
      setError("Cannot restore this highlight because its id is missing.");
      return null;
    }
    const highlightId = normalized.highlight_id;
    const saved = highlightsRef.current.find((item) => (item.highlight_id || item.id) === normalized.highlight_id) || normalized;
    const restored = normalizePersistedHighlight(saved, documentId, "restored");
    const highlightDebug = highlightDebugFromSaved(restored);
    const matchDebug = matchDebugFromSaved(documentId, restored);
    const nextSelection = selectionStateFrom(restored, highlightDebug, matchDebug);
    setActiveSelection(nextSelection);
    activeSelectionRef.current = nextSelection;
    let threadResult = null;
    try {
      threadResult = await loadThreadForHighlight(normalized.highlight_id);
    } catch (err) {
      threadResult = { status: err?.status ?? null, message_count: err?.message_count ?? null, stale: false };
      setError(err?.message || String(err));
    }
    logPdfChatRestoreDebug("highlight_click", {
      event: "highlight_click",
      document_id: documentId,
      highlight_id: normalized.highlight_id,
      id: normalized.id,
      type: normalized.type,
      source: normalized.source || "restored",
      page_number: normalized.page_number || pageNumberFromPosition(normalized.position),
      has_thread_loaded: threadHasMessages(threadRef.current),
      local_thread_message_count: threadMessageCount(threadRef.current),
      thread_fetch_url: `/api/documents/${documentId}/threads/${highlightId}`,
      thread_fetch_status: threadResult?.status ?? null,
      thread_fetch_message_count: threadResult?.message_count ?? null,
    });
    return threadResult;
  }

  async function loadThreadForHighlight(highlightId) {
    const requestId = latestThreadRequestRef.current + 1;
    latestThreadRequestRef.current = requestId;
    const threadFetchUrl = `/api/documents/${documentId}/threads/${highlightId}`;
    const response = await fetch(threadFetchUrl);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const error = new Error(payload.error || `HTTP ${response.status}`);
      error.status = response.status;
      error.message_count = threadMessageCount(payload);
      throw error;
    }
    if (latestThreadRequestRef.current !== requestId) return;
    if (activeSelectionRef.current?.llmInputPreview?.highlight_id !== highlightId) return;
    if (shouldIgnoreEmptyThreadLoad(payload, threadRef.current, highlightId)) return;
    setThread(payload);
    threadRef.current = payload;
    applyThreadStrategyState(payload);
    logPdfChatRestoreDebug("thread_loaded", {
      event: "thread_loaded",
      document_id: documentId,
      highlight_id: highlightId,
      message_count: threadMessageCount(payload),
      source: "backend",
    });
    return { status: response.status, message_count: threadMessageCount(payload), stale: false };
  }

  async function persistThreadForHighlight(thread) {
    const highlightId = thread?.highlight_id || activeSelectionRef.current?.llmInputPreview?.highlight_id || "";
    if (!highlightId || !thread) return thread;
    if (shouldIgnoreEmptyThreadPersist(thread, threadRef.current, highlightId)) return threadRef.current;
    return persistThreadAfterAssistantMessage(documentId, highlightId, thread);
  }

  async function handleDeleteHighlight() {
    const highlightId = activeSelection?.llmInputPreview?.highlight_id;
    if (!highlightId) return;
    if (!window.confirm("Delete this highlight and its conversation?")) return;
    setCleanupLoading(true);
    setError("");
    try {
      const payload = await deleteJson(`/api/documents/${documentId}/highlights/${highlightId}`);
      setHighlights((payload.highlights || []).map(toViewerHighlight));
      setActiveSelection(null);
      setThread(null);
      threadRef.current = null;
      resetStrategyState();
      onDocumentChanged?.(payload.meta || currentMeta);
      await postSessionEvent("highlight_deleted", { highlight_id: highlightId });
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setCleanupLoading(false);
    }
  }

  async function handleClearConversation() {
    const highlightId = activeSelection?.llmInputPreview?.highlight_id;
    if (!highlightId) return;
    if (!window.confirm("Clear the conversation for this highlight?")) return;
    setCleanupLoading(true);
    setError("");
    try {
      const payload = await postJson(`/api/documents/${documentId}/threads/${highlightId}/clear`, {});
      setThread(payload.thread || null);
      threadRef.current = payload.thread || null;
      resetStrategyState();
      applyThreadStrategyState(payload.thread || null);
      onDocumentChanged?.(payload.meta || currentMeta);
      await postSessionEvent("thread_cleared", { highlight_id: highlightId });
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setCleanupLoading(false);
    }
  }

  async function handleDeleteCurrentTurn(turnId) {
    const highlightId = activeSelection?.llmInputPreview?.highlight_id;
    if (!highlightId || !turnId) return;
    if (!window.confirm("Delete this conversation turn?")) return;
    setCleanupLoading(true);
    setError("");
    try {
      const payload = await deleteJson(`/api/documents/${documentId}/threads/${highlightId}/turns/${turnId}`);
      setThread(payload.thread || null);
      threadRef.current = payload.thread || null;
      applyThreadStrategyState(payload.thread || null);
      onDocumentChanged?.(payload.meta || currentMeta);
      await postSessionEvent("thread_turn_deleted", { highlight_id: highlightId, turn_id: turnId });
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setCleanupLoading(false);
    }
  }

  async function requestStrategyCandidates(
    triggeredBy = "reaction_window",
    summaryOverride = reactionWindowSummary,
    baselineExplanation = "",
    sourceTurnId = "",
    selectionOverride = activeSelection,
  ) {
    const selection = selectionOverride || activeSelectionRef.current;
    const summary = summaryOverride || reactionWindowSummary;
    const stateSnapshot = learningStateRef.current || learningState || {};
    const resolvedSourceTurnId = sourceTurnId || summary?.source_turn_id || "";
    const resolvedBaselineExplanation = normalizePdfText(
      baselineExplanation || summary?.baseline_explanation || baselineExplanationForTurn(threadRef.current, resolvedSourceTurnId),
    );
    const resolvedReactionWindowSummary = summary && Object.keys(summary).length ? summary : null;
    if (!selection?.llmInputPreview?.highlight_id || !resolvedSourceTurnId || !resolvedBaselineExplanation || !resolvedReactionWindowSummary) {
      return;
    }
    if (hasTurnStrategyCandidates(threadRef.current, resolvedSourceTurnId) && triggeredBy !== "manual_refresh") return;
    if (strategyRequestRef.current) return;
    const key = reactionStrategyKey(selection, summary);
    if (triggeredBy !== "manual_refresh" && dismissedStrategyKeysRef.current.has(key)) return;
    const lastTriggeredAt = strategyCooldownRef.current[key] || 0;
    if (triggeredBy !== "manual_refresh" && Date.now() - lastTriggeredAt < 30000) return;
    strategyRequestRef.current = true;
    resolvedReactionWindowSummary.strategy_request_sent = true;
    resolvedReactionWindowSummary.strategy_response_received = false;
    resolvedReactionWindowSummary.reaction_window_status = resolvedReactionWindowSummary.reaction_window_status || "completed";
    setStrategyLoading(true);
    setError("");
    const triggerContext = {
      triggered_by: "reaction_window",
      trigger_reason: summary.trigger_reason || "The recent learning signal suggests trying an adaptive explanation.",
      duration_sec: Number(summary.duration_sec || 0),
      trend: summary.trend || "",
      intensity: Number(summary.avg_confidence ?? summary.max_confidence ?? stateSnapshot.intensity ?? stateSnapshot.confidence ?? 0),
      refresh_requested: triggeredBy === "manual_refresh",
      support_cue: summary.support_cue || "",
      source_turn_id: resolvedSourceTurnId,
      source_turn_type: summary.source_turn_type || "",
    };
    try {
      const preview = selection.llmInputPreview || {};
      const paperContext = strategyPaperContextFromSelection(selection);
      const plannerInputSummary = plannerInputSummaryForStrategyRequest({
        preview,
        paperContext,
        baselineExplanation: resolvedBaselineExplanation,
        reactionWindowSummary: resolvedReactionWindowSummary,
        recentConversation: threadRef.current?.messages || [],
        previousStrategy: selectedStrategy,
      });
      const payload = await postJson(`/api/documents/${documentId}/strategy-candidates`, {
        session_id: readingSessionId,
        document_id: documentId,
        highlight_id: preview.highlight_id,
        source_turn_id: resolvedSourceTurnId,
        source_turn_type: summary.source_turn_type || "",
        input_source: "proactive_recommendation",
        response_mode: "proactive_support",
        selection_type: preview.highlight_type,
        page_number: preview.page_number,
        selected_text: preview.selected_text,
        caption: preview.caption,
        crop_available: Boolean(preview.crop_image_available || preview.crop_url || preview.crop_image_path),
        baseline_explanation: resolvedBaselineExplanation,
        reaction_window_summary: resolvedReactionWindowSummary,
        support_cue: summary.support_cue || "",
        learning_state: stateSnapshot,
        learning_signal_package: stateSnapshot.learning_signal_package || {},
        paper_context: paperContext,
        planner_input_summary: plannerInputSummary,
        recent_conversation: (threadRef.current?.messages || []).slice(-6),
        trigger_context: triggerContext,
      });
      const candidates = payload.candidates || [];
      resolvedReactionWindowSummary.strategy_response_received = true;
      setStrategyCandidates(candidates);
      setStrategyPlannerMode(payload.planner_mode || "");
      setStrategyTriggerContext(triggerContext);
      setStrategySourceKey(key);
      strategyCooldownRef.current[key] = Date.now();
      completedReactionTurnIdsRef.current.add(resolvedSourceTurnId);
      const updatedThread = mergeTurnMetadata(threadRef.current, resolvedSourceTurnId, {
        reaction_window_summary: resolvedReactionWindowSummary,
        strategy_candidates: candidates,
        strategy_status: "suggested",
        support_cue: payload.support_cue || summary.support_cue || "",
        support_cue_label: payload.support_cue_label || summary.support_cue_label || "",
        learning_signal_package: payload.learning_signal_package || stateSnapshot.learning_signal_package || {},
        planner_mode: payload.planner_mode || "",
        planner_prompt_version: payload.planner_prompt_version || "reaction_strategy_planner_v2",
        planner_input_summary: plannerInputSummary,
        trigger_context: triggerContext,
      });
      const persistedThread = await persistThreadForHighlight(updatedThread);
      setThread(persistedThread);
      threadRef.current = persistedThread;
      await postSessionEvent("strategy_candidates_generated", {
        highlight_id: preview.highlight_id,
        strategy_candidates: candidates,
        planner_mode: payload.planner_mode || "",
        reaction_window_summary: resolvedReactionWindowSummary,
        support_cue: summary.support_cue || "",
        support_cue_label: summary.support_cue_label || "",
        learning_signal_package: payload.learning_signal_package || stateSnapshot.learning_signal_package || {},
        academic_state: summary.dominant_state || stateSnapshot.academic_state,
        confidence: summary.avg_confidence ?? stateSnapshot.confidence,
        distribution: summary.avg_distribution || stateSnapshot.distribution,
        trend: summary.trend || stateSnapshot.trend,
        duration_sec: summary.duration_sec,
        end_reason: summary.end_reason,
        read_progress_estimate: summary.read_progress_estimate,
        state_transition: summary.state_transition,
        transition_label: summary.transition_label,
        reaction_window_status: summary.reaction_window_status,
        sample_count: summary.sample_count,
        min_duration_met: summary.min_duration_met,
        min_read_time_met: summary.min_read_time_met,
        min_samples_met: summary.min_samples_met,
        top_seen: summary.top_seen,
        bottom_seen: summary.bottom_seen,
        user_scroll_progressed: summary.user_scroll_progressed,
        suppressed_auto_scroll: summary.suppressed_auto_scroll,
        word_count: summary.word_count,
        message_height: summary.message_height,
        viewport_height: summary.viewport_height,
        is_short_answer: summary.is_short_answer,
        estimated_min_read_sec: summary.estimated_min_read_sec,
        elapsed_sec: summary.elapsed_sec,
        end_blocked_reason: summary.end_blocked_reason,
        strategy_request_sent: summary.strategy_request_sent,
        strategy_response_received: summary.strategy_response_received,
        intensity: triggerContext.intensity,
        trigger_reason: triggerContext.trigger_reason,
        passage_type: paperContext.passage_type,
        difficulty_hint: paperContext.difficulty_hint,
      });
    } catch (err) {
      if (isReactionWindowValidationError(err)) return;
      setError(err?.message || String(err));
    } finally {
      strategyRequestRef.current = false;
      setStrategyLoading(false);
    }
  }

  async function saveHighlights(nextHighlights) {
    const payload = await putJson(`/api/documents/${documentId}/highlights`, {
      highlights: nextHighlights.map((highlight) => ({
        ...highlight,
        document_id: documentId,
        highlight_id: highlight.highlight_id || highlight.id,
      })),
    });
    onDocumentChanged?.(payload.meta || currentMeta);
  }

  async function readLearningStateSample() {
    if (!readingSessionId) return learningStateRef.current || null;
    try {
      const payload = await fetchJson(`/api/reading-sessions/${readingSessionId}/learning-state/current`);
      if (payload.learning_state) {
        setLearningState(payload.learning_state);
        return payload.learning_state;
      }
    } catch {
      // Reaction monitoring should not interrupt the reading flow.
    }
    return learningStateRef.current || null;
  }

  function clearReactionWindowTimers() {
    if (reactionWindowSampleTimerRef.current) window.clearInterval(reactionWindowSampleTimerRef.current);
    if (reactionWindowMinTimerRef.current) window.clearTimeout(reactionWindowMinTimerRef.current);
    if (reactionWindowMaxTimerRef.current) window.clearTimeout(reactionWindowMaxTimerRef.current);
    if (reactionWindowSentinelTimerRef.current) window.clearTimeout(reactionWindowSentinelTimerRef.current);
    if (reactionWindowProgrammaticScrollTimerRef.current) window.clearTimeout(reactionWindowProgrammaticScrollTimerRef.current);
    if (reactionWindowLayoutTimerRef.current) window.clearTimeout(reactionWindowLayoutTimerRef.current);
    reactionWindowSampleTimerRef.current = null;
    reactionWindowMinTimerRef.current = null;
    reactionWindowMaxTimerRef.current = null;
    reactionWindowSentinelTimerRef.current = null;
    reactionWindowProgrammaticScrollTimerRef.current = null;
    reactionWindowLayoutTimerRef.current = null;
    if (reactionWindowObserverRef.current) {
      reactionWindowObserverRef.current.disconnect();
      reactionWindowObserverRef.current = null;
    }
  }

  function registerReactionSentinel(turnId, sentinelKind, node) {
    if (!turnId) return;
    const kind = sentinelKind === "top" ? "top" : "bottom";
    const current = reactionSentinelRefs.current.get(turnId) || {};
    if (!node) {
      delete current[kind];
      if (current.top || current.bottom || current.message) {
        reactionSentinelRefs.current.set(turnId, current);
      } else {
        reactionSentinelRefs.current.delete(turnId);
      }
      return;
    }
    current[kind] = node;
    current.message = node.closest(".pdf-chat-message") || current.message || null;
    reactionSentinelRefs.current.set(turnId, current);
    if (reactionWindowTurnIdRef.current === turnId) {
      scrollReactionMessageTopIntoView(turnId);
      observeReactionSentinels(turnId);
      scheduleReactionWindowLayoutSettle(turnId);
    }
  }

  function observeReactionSentinels(turnId) {
    const sentinels = reactionSentinelRefs.current.get(turnId) || {};
    const nodes = [sentinels.top, sentinels.bottom].filter(Boolean);
    if (!nodes.length || reactionWindowTurnIdRef.current !== turnId) return;
    if (reactionWindowObserverRef.current) {
      reactionWindowObserverRef.current.disconnect();
      reactionWindowObserverRef.current = null;
    }
    if (typeof IntersectionObserver === "undefined") return;
    const root = nodes[0].closest(".pdf-chat-conversation-scroll") || nodes[0].closest(".pdf-chat-side-scroll") || null;
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting && entry.intersectionRatio <= 0) continue;
        const active = reactionWindowStateRef.current;
        if (!active || active.done || active.sourceTurnId !== turnId) continue;
        const kind = entry.target?.dataset?.reactionSentinel || "bottom";
        if (kind === "top") {
          active.topSeen = true;
          continue;
        }
        if (!active.layoutSettled) {
          active.bottomSeenBeforeLayoutSettled = true;
          active.endBlockedReason = "waiting_for_layout_settle";
          continue;
        }
        if (active.programmaticScrollActive || Date.now() < Number(active.programmaticScrollUntil || 0)) {
          active.suppressedAutoScroll = true;
          active.bottomSeenFromProgrammaticScroll = true;
          active.endBlockedReason = "programmatic_scroll_ignored";
          continue;
        }
        markReactionWindowBottomSeen(active);
      }
    }, {
      root,
      threshold: 0.1,
      rootMargin: "0px 0px -15% 0px",
    });
    nodes.forEach((node) => observer.observe(node));
    reactionWindowObserverRef.current = observer;
  }

  function scheduleReactionWindowLayoutSettle(sourceTurnId) {
    if (reactionWindowLayoutTimerRef.current) window.clearTimeout(reactionWindowLayoutTimerRef.current);
    reactionWindowLayoutTimerRef.current = window.setTimeout(() => {
      reactionWindowLayoutTimerRef.current = null;
      updateReactionWindowLayoutState(sourceTurnId);
    }, REACTION_WINDOW_LAYOUT_SETTLE_MS);
  }

  function updateReactionWindowLayoutState(turnId) {
    const active = reactionWindowStateRef.current;
    const sentinels = reactionSentinelRefs.current.get(turnId) || {};
    const root = sentinels.top?.closest(".pdf-chat-conversation-scroll") || sentinels.bottom?.closest(".pdf-chat-conversation-scroll") || null;
    if (!active || active.sourceTurnId !== turnId || !root || !sentinels.top || !sentinels.bottom) return;
    const topRect = sentinels.top.getBoundingClientRect();
    const bottomRect = sentinels.bottom.getBoundingClientRect();
    const rootRect = root.getBoundingClientRect();
    const messageHeight = Math.max(0, bottomRect.bottom - topRect.top);
    const viewportHeight = Number(root.clientHeight || rootRect.height || 0);
    const topVisible = topRect.top >= rootRect.top && topRect.bottom <= rootRect.bottom;
    const bottomVisible = bottomRect.top >= rootRect.top && bottomRect.bottom <= rootRect.bottom;
    active.layoutSettled = true;
    active.messageHeight = roundNumber(messageHeight, 1);
    active.viewportHeight = roundNumber(viewportHeight, 1);
    active.isShortAnswer = Boolean(
      (messageHeight > 0 && viewportHeight > 0 && messageHeight <= viewportHeight * 0.75)
      || (topVisible && bottomVisible)
    );
    if (topVisible) active.topSeen = true;
    if (active.isShortAnswer && bottomVisible) {
      active.bottomSeen = true;
      active.readProgressEstimate = Math.max(Number(active.readProgressEstimate || 0), 1);
    }
    maybeEndReactionWindowForReading(active);
  }

  function scrollReactionMessageTopIntoView(turnId) {
    const active = reactionWindowStateRef.current;
    const sentinels = reactionSentinelRefs.current.get(turnId) || {};
    const target = sentinels.top || sentinels.message;
    if (!active || active.sourceTurnId !== turnId || !target) return;
    if (active.didInitialTopScroll) return;
    active.didInitialTopScroll = true;
    active.programmaticScrollActive = true;
    active.suppressedAutoScroll = true;
    active.topSeen = true;
    active.programmaticScrollUntil = Date.now() + REACTION_PROGRAMMATIC_SCROLL_MS;
    target.scrollIntoView({ behavior: "auto", block: "start" });
    if (reactionWindowProgrammaticScrollTimerRef.current) window.clearTimeout(reactionWindowProgrammaticScrollTimerRef.current);
    reactionWindowProgrammaticScrollTimerRef.current = window.setTimeout(() => {
      const latest = reactionWindowStateRef.current;
      if (latest && latest.sourceTurnId === turnId) latest.programmaticScrollActive = false;
      reactionWindowProgrammaticScrollTimerRef.current = null;
    }, REACTION_PROGRAMMATIC_SCROLL_MS);
  }

  function handleReactionWindowScroll(event) {
    const active = reactionWindowStateRef.current;
    if (!active || active.done) return;
    const target = event?.currentTarget;
    if (!target) return;
    const scrollTop = Number(target.scrollTop || 0);
    const previousScrollTop = Number(active.lastScrollTop ?? scrollTop);
    active.lastScrollTop = scrollTop;
    if (active.programmaticScrollActive || Date.now() < Number(active.programmaticScrollUntil || 0)) {
      active.suppressedAutoScroll = true;
      return; // programmatic scrolls must not count as reading progress
    }
    active.userHasScrolledDuringWindow = true;
    if (scrollTop > previousScrollTop + 4) {
      active.userScrollProgressed = true;
      const progress = (scrollTop + Number(target.clientHeight || 0)) / Math.max(1, Number(target.scrollHeight || 1));
      active.readProgressEstimate = Math.max(Number(active.readProgressEstimate || 0), Math.min(1, progress));
      maybeEndReactionWindowForReading(active);
    }
  }

  function reactionWindowMinSamplesMet(active) {
    return Boolean(active && (active.samples || []).length >= REACTION_WINDOW_MIN_SAMPLES);
  }

  function canEndReactionWindowForReading(active) {
    if (!active || active.done) return false;
    if (!active.layoutSettled) {
      active.endBlockedReason = "waiting_for_layout_settle";
      return false;
    }
    if (!active.topSeen) {
      active.endBlockedReason = "waiting_for_top_seen";
      return false;
    }
    if (!active.bottomSeen) {
      active.endBlockedReason = "waiting_for_bottom_seen";
      return false;
    }
    if (!active.minReadTimeMet) {
      active.endBlockedReason = "waiting_for_min_read_time";
      return false;
    }
    if (!reactionWindowMinSamplesMet(active)) {
      active.endBlockedReason = "waiting_for_min_samples";
      return false;
    }
    if (active.isShortAnswer) {
      active.endBlockedReason = "";
      return true;
    }
    if (!active.userScrollProgressed) {
      active.endBlockedReason = "waiting_for_user_scroll_progress";
      return false;
    }
    active.endBlockedReason = "";
    return true;
  }

  function markReactionWindowBottomSeen(active) {
    if (!active.layoutSettled) {
      active.bottomSeenBeforeLayoutSettled = true;
      active.endBlockedReason = "waiting_for_layout_settle";
      return;
    }
    active.bottomSeen = true;
    active.readProgressEstimate = Math.max(Number(active.readProgressEstimate || 0), 0.9);
    active.pendingEndReason = "near_bottom_reached";
    if (!canEndReactionWindowForReading(active)) return;
    scheduleReactionWindowEnd("near_bottom_reached", { readProgressEstimate: active.readProgressEstimate });
  }

  function maybeEndReactionWindowForReading(active = reactionWindowStateRef.current) {
    if (!active || active.done || active.ending) return;
    if (active.isShortAnswer && canEndReactionWindowForReading(active)) {
      void endReactionWindow("short_answer_min_elapsed", { readProgressEstimate: 1 });
      return;
    }
    if (canEndReactionWindowForReading(active)) {
      scheduleReactionWindowEnd("near_bottom_reached", { readProgressEstimate: active.readProgressEstimate || 0.9 });
    }
  }

  function reactionWindowDebugFields(active, endReason) {
    const elapsedSec = active.windowStartEpoch
      ? Math.max(0, (Date.now() - active.windowStartEpoch) / 1000)
      : 0;
    return {
      reaction_window_status: "completed",
      start_turn_id: active.sourceTurnId || "",
      source_turn_id: active.sourceTurnId || "",
      end_reason: endReason || "",
      sample_count: (active.samples || []).length,
      word_count: Number(active.wordCount || 0),
      message_height: Number(active.messageHeight || 0),
      viewport_height: Number(active.viewportHeight || 0),
      is_short_answer: Boolean(active.isShortAnswer),
      estimated_min_read_sec: Number(active.estimatedMinReadSec || 0),
      elapsed_sec: roundNumber(elapsedSec, 1),
      min_duration_met: Boolean(active.minReadTimeMet),
      min_read_time_met: Boolean(active.minReadTimeMet),
      min_samples_met: reactionWindowMinSamplesMet(active),
      top_seen: Boolean(active.topSeen),
      bottom_seen: Boolean(active.bottomSeen),
      user_scroll_progressed: Boolean(active.userScrollProgressed),
      suppressed_auto_scroll: Boolean(active.suppressedAutoScroll),
      end_blocked_reason: active.endBlockedReason || "",
      strategy_request_sent: false,
      strategy_response_received: false,
    };
  }

  function scheduleReactionWindowEnd(endReason, options = {}) {
    if (reactionWindowSentinelTimerRef.current) window.clearTimeout(reactionWindowSentinelTimerRef.current);
    reactionWindowSentinelTimerRef.current = window.setTimeout(() => {
      if (endReason === "near_bottom_reached") {
        void endReactionWindow("near_bottom_reached", options);
        return;
      }
      void endReactionWindow(endReason, options);
    }, REACTION_WINDOW_SENTINEL_DELAY_MS);
  }

  async function sampleReactionWindow(runId, options = {}) {
    const active = reactionWindowStateRef.current;
    if (!active || (!options.force && active.done) || active.runId !== runId) return;
    const sample = await readLearningStateSample();
    const latest = reactionWindowStateRef.current;
    if (!sample || !latest || (!options.force && latest.done) || latest.runId !== runId) return;
    latest.samples.push({ ...sample, sample_index: latest.samples.length, sample_timestamp: new Date().toISOString() });
    maybeEndReactionWindowForReading(latest);
  }

  async function endReactionWindow(endReason, options = {}) {
    const active = reactionWindowStateRef.current;
    if (!active || active.done) return null;
    active.ending = true;
    clearReactionWindowTimers();
    await sampleReactionWindow(active.runId, { force: true });
    active.done = true;
    const windowEnd = new Date().toISOString();
    const readProgressEstimate = Number(options.readProgressEstimate ?? active.readProgressEstimate ?? (endReason === "max_timeout" ? 0.5 : 0.9));
    const summary = summarizeReactionWindowSamples(
      active.samples,
      active.sourceTurnId,
      active.preview.highlight_id,
      active.windowStart,
      windowEnd,
      active.sourceTurnType,
      endReason,
      readProgressEstimate,
    );
    summary.baseline_explanation = active.baselineExplanation || "";
    Object.assign(summary, reactionWindowDebugFields(active, endReason));
    summary.reliability_notes = Array.isArray(summary.reliability_notes) ? summary.reliability_notes : [];
    if (["followup_submitted", "strategy_clicked"].includes(endReason) && !active.minReadTimeMet) {
      summary.reliability_notes.push("reading window ended early; use conservative strategy inference");
    }
    setReactionWindowSummary(summary);
    const triggerContext = {
      triggered_by: "reaction_window",
      trigger_reason: summary.trigger_reason,
      duration_sec: summary.duration_sec,
      trend: summary.trend,
      intensity: summary.avg_confidence,
      support_cue: summary.support_cue,
      source_turn_id: active.sourceTurnId,
      source_turn_type: active.sourceTurnType,
      end_reason: summary.end_reason,
      read_progress_estimate: summary.read_progress_estimate,
      transition_label: summary.transition_label,
      reaction_window_status: summary.reaction_window_status,
      sample_count: summary.sample_count,
      min_duration_met: summary.min_duration_met,
      min_read_time_met: summary.min_read_time_met,
      min_samples_met: summary.min_samples_met,
      top_seen: summary.top_seen,
      bottom_seen: summary.bottom_seen,
      user_scroll_progressed: summary.user_scroll_progressed,
      suppressed_auto_scroll: summary.suppressed_auto_scroll,
      word_count: summary.word_count,
      message_height: summary.message_height,
      viewport_height: summary.viewport_height,
      is_short_answer: summary.is_short_answer,
      estimated_min_read_sec: summary.estimated_min_read_sec,
      elapsed_sec: summary.elapsed_sec,
      end_blocked_reason: summary.end_blocked_reason,
    };
    setStrategyTriggerContext(triggerContext);
    completedReactionTurnIdsRef.current.add(active.sourceTurnId);
    const threadWithSummary = mergeTurnMetadata(threadRef.current, active.sourceTurnId, {
      reaction_window_summary: summary,
      support_cue: summary.support_cue,
      support_cue_label: summary.support_cue_label,
      trigger_context: triggerContext,
    });
    try {
      const persistedThread = await persistThreadForHighlight(threadWithSummary);
      setThread(persistedThread);
      threadRef.current = persistedThread;
    } catch (err) {
      setThread(threadWithSummary);
      threadRef.current = threadWithSummary;
    }
    const shouldRequestStrategy = options.requestCandidates !== false && !["followup_submitted", "strategy_clicked"].includes(endReason);
    summary.strategy_request_sent = shouldRequestStrategy;
    await postSessionEvent("reaction_window_completed", {
      highlight_id: active.preview.highlight_id,
      source_turn_id: active.sourceTurnId,
      source_turn_type: active.sourceTurnType,
      end_reason: summary.end_reason,
      read_progress_estimate: summary.read_progress_estimate,
      reaction_window_status: summary.reaction_window_status,
      sample_count: summary.sample_count,
      min_duration_met: summary.min_duration_met,
      min_read_time_met: summary.min_read_time_met,
      min_samples_met: summary.min_samples_met,
      top_seen: summary.top_seen,
      bottom_seen: summary.bottom_seen,
      user_scroll_progressed: summary.user_scroll_progressed,
      suppressed_auto_scroll: summary.suppressed_auto_scroll,
      word_count: summary.word_count,
      message_height: summary.message_height,
      viewport_height: summary.viewport_height,
      is_short_answer: summary.is_short_answer,
      estimated_min_read_sec: summary.estimated_min_read_sec,
      elapsed_sec: summary.elapsed_sec,
      end_blocked_reason: summary.end_blocked_reason,
      strategy_request_sent: summary.strategy_request_sent,
      strategy_response_received: summary.strategy_response_received,
      reaction_window_summary: summary,
      support_cue: summary.support_cue,
      support_cue_label: summary.support_cue_label,
    });
    if (shouldRequestStrategy) {
      await requestStrategyCandidates("reaction_window", summary, active.baselineExplanation, active.sourceTurnId, active.selection);
    }
    if (reactionWindowStateRef.current?.runId === active.runId) {
      reactionWindowStateRef.current = null;
      setReactionWindowActive(false);
      setReactionWindowScrollTargetId("");
      reactionWindowTurnIdRef.current = "";
    }
    return summary;
  }

  async function startReactionWindow(sourceTurnId, baselineExplanation, selectionOverride = activeSelectionRef.current, sourceTurnType = "baseline_explanation") {
    const selection = selectionOverride || activeSelectionRef.current;
    const preview = selection?.llmInputPreview || {};
    if (!readingSessionId || !preview.highlight_id || !sourceTurnId || !normalizePdfText(baselineExplanation)) return;
    if (completedReactionTurnIdsRef.current.has(sourceTurnId) || hasTurnReactionMetadata(threadRef.current, sourceTurnId)) {
      completedReactionTurnIdsRef.current.add(sourceTurnId);
      return;
    }

    clearReactionWindowTimers();
    const runId = Date.now();
    const wordCount = wordCountForReadingEstimate(baselineExplanation);
    const estimatedMinReadSec = estimateReactionWindowMinReadSec(baselineExplanation);
    reactionWindowRunRef.current = runId;
    reactionWindowTurnIdRef.current = sourceTurnId;
    setReactionWindowScrollTargetId(sourceTurnId);
    setReactionWindowActive(true);
    setReactionWindowSummary(null);
    setStrategyCandidates([]);
    setStrategyPlannerMode("");
    setStrategyTriggerContext(null);

    const windowStart = new Date().toISOString();
    reactionWindowStateRef.current = {
      runId,
      sourceTurnId,
      sourceTurnType,
      baselineExplanation,
      selection,
      preview,
      windowStart,
      windowStartEpoch: Date.now(),
      samples: [],
      readProgressEstimate: 0,
      wordCount,
      estimatedMinReadSec,
      minReadTimeMet: false,
      topSeen: false,
      bottomSeen: false,
      bottomSeenFromProgrammaticScroll: false,
      bottomSeenBeforeLayoutSettled: false,
      userHasScrolledDuringWindow: false,
      userScrollProgressed: false,
      programmaticScrollActive: false,
      programmaticScrollUntil: 0,
      suppressedAutoScroll: true,
      didInitialTopScroll: false,
      lastScrollTop: null,
      pendingEndReason: "",
      layoutSettled: false,
      messageHeight: 0,
      viewportHeight: 0,
      isShortAnswer: false,
      endBlockedReason: "",
      done: false,
    };
    void postSessionEvent("reaction_window_started", {
      highlight_id: preview.highlight_id,
      source_turn_id: sourceTurnId,
      source_turn_type: sourceTurnType,
      source: liveSignalActive ? "webcam_model" : "simulated_fallback",
      word_count: wordCount,
      estimated_min_read_sec: estimatedMinReadSec,
      maximum_duration_ms: REACTION_WINDOW_MAX_MS,
    });

    void sampleReactionWindow(runId);
    reactionWindowSampleTimerRef.current = window.setInterval(() => {
      void sampleReactionWindow(runId);
    }, REACTION_WINDOW_SAMPLE_MS);
    reactionWindowMinTimerRef.current = window.setTimeout(() => {
      const active = reactionWindowStateRef.current;
      if (!active || active.done || active.runId !== runId) return;
      active.minReadTimeMet = true;
      maybeEndReactionWindowForReading(active);
    }, estimatedMinReadSec * 1000);
    reactionWindowMaxTimerRef.current = window.setTimeout(() => {
      void endReactionWindow("max_timeout", { readProgressEstimate: reactionWindowStateRef.current?.readProgressEstimate || 0.5 });
    }, REACTION_WINDOW_MAX_MS);
    window.setTimeout(() => {
      scrollReactionMessageTopIntoView(sourceTurnId);
      observeReactionSentinels(sourceTurnId);
      scheduleReactionWindowLayoutSettle(sourceTurnId);
    }, 0);
  }

  function monitorableAssistantMessage(message) {
    if (!message || message.role !== "assistant" || !message.turn_id || !normalizePdfText(message.content || "")) return false;
    const turn_type = message.turn_type || "baseline_explanation";
    return (
      turn_type === "baseline_explanation"
      || turn_type === "strategy_reexplanation"
      || turn_type === "follow_up"
    );
  }

  function startReactionWindowForAssistantMessage(message, selectionOverride = activeSelectionRef.current) {
    if (!monitorableAssistantMessage(message)) return;
    startReactionWindow(message.turn_id, message.content || "", selectionOverride, message.turn_type || "previous_explanation");
  }

  async function explainSelection(strategyOverride = null, defaultTask = "baseline_explain_current_selection", strategyContextOverride = null) {
    if (!documentReady) {
      setError("Document is still loading; wait for a stable document id before explaining.");
      return;
    }
    if (!activeSelection?.llmInputPreview?.highlight_id) return;
    if (isDomOrReactEvent(strategyOverride)) {
      strategyOverride = null;
    }
    const effectiveStrategy = isPlainStrategyObject(strategyOverride) ? strategyOverride : null;
    const isStrategyExplain = Boolean(effectiveStrategy);
    const selectionSnapshot = activeSelection;
    const strategyContext = strategyContextOverride || {};
    const currentReactionSummary = strategyContext.reactionWindowSummary || reactionWindowSummary || null;
    const currentStrategyCandidates = strategyContext.strategyCandidates || strategyCandidates;
    const currentTriggerContext = strategyContext.triggerContext || strategyTriggerContext;
    const currentPlannerMode = strategyContext.plannerMode || strategyPlannerMode;
    const currentSourceTurnId = strategyContext.sourceTurnId || currentReactionSummary?.source_turn_id || currentTriggerContext?.source_turn_id || "";
    const strategyPlannerInputSummary = isStrategyExplain
      ? plannerInputSummaryForStrategyRequest({
        preview: activeSelection.llmInputPreview,
        paperContext: strategyPaperContextFromSelection(activeSelection),
        baselineExplanation: baselineExplanationForTurn(threadRef.current, currentSourceTurnId),
        reactionWindowSummary: currentReactionSummary || {},
        recentConversation: threadRef.current?.messages || [],
        previousStrategy: selectedStrategy,
      })
      : {};
    setExplainLoading(true);
    setError("");
    try {
      void ensureCameraSignalStarted("explain_clicked");
      const payload = sanitizeSerializablePayload({
        ...activeSelection.llmInputPreview,
        document_id: documentId,
        selection_type: activeSelection.llmInputPreview.highlight_type || activeSelection.llmInputPreview.type || "",
        response_style: "chat_conversational",
        session_id: readingSessionId,
        learning_state: learningState,
        learning_signal_package: learningState?.learning_signal_package || {},
        input_source: isStrategyExplain ? "strategy_click" : "text",
        response_mode: isStrategyExplain ? "strategy_response" : "normal_followup",
        strategy_candidates: isStrategyExplain ? currentStrategyCandidates : [],
        selected_strategy_id: effectiveStrategy?.strategy_id || "",
        selected_strategy: effectiveStrategy || null,
        trigger_context: isStrategyExplain ? currentTriggerContext : null,
        source_turn_id: isStrategyExplain ? currentSourceTurnId : "",
        reaction_window_summary: isStrategyExplain ? currentReactionSummary : null,
        support_cue: isStrategyExplain ? currentReactionSummary?.support_cue || "" : "",
        planner_mode: isStrategyExplain ? currentPlannerMode : "",
        planner_input_summary: isStrategyExplain ? strategyPlannerInputSummary : {},
        user_question: null,
        default_task: defaultTask || "baseline_explain_current_selection",
        turn_type: isStrategyExplain ? "strategy_reexplanation" : "baseline_explanation",
      });
      assertJsonSerializablePayload(payload);
      const endpoint = `/api/documents/${documentId}/explain-selection`;
      const result = await postJson(endpoint, payload);
      const normalizedResult = normalizeExplainSelectionResponse(result);
      if (!normalizedResult.answerText) {
        logExplainSelectionDebug(endpoint, payload, result, normalizedResult, null);
        if (normalizedResult.errorMessage) throw new Error(normalizedResult.errorMessage);
        throw new Error("The explanation response did not include an answer. Please try again.");
      }
      const responseAssistant = latestAssistantMessage(normalizedResult.thread);
      const responseAssistantText = normalizePdfText(responseAssistant?.content || "");
      const normalizedAnswerText = normalizePdfText(normalizedResult.answerText);
      const nextThread = responseAssistantText && responseAssistantText === normalizedAnswerText
        ? normalizedResult.thread
        : appendAssistantMessage(normalizedResult.thread || threadRef.current, normalizedResult);
      let persistedThread;
      try {
        persistedThread = await persistThreadForHighlight(nextThread);
      } catch (persistError) {
        setThread(nextThread);
        threadRef.current = nextThread;
        setError("Answer generated, but conversation could not be saved.");
        logExplainSelectionDebug(endpoint, payload, result, normalizedResult, nextThread);
        return;
      }
      const generatedAssistantMessage = latestAssistantMessage(persistedThread);
      if (monitorableAssistantMessage(generatedAssistantMessage)) {
        setReactionWindowScrollTargetId(generatedAssistantMessage.turn_id || generatedAssistantMessage.conversation_turn_id || "");
      }
      setThread(persistedThread);
      threadRef.current = persistedThread;
      setActiveSelection((current) => current ? { ...current, explainResult: result } : current);
      applyThreadStrategyState(persistedThread);
      logExplainSelectionDebug(endpoint, payload, result, normalizedResult, persistedThread);
      await postSessionEvent("answer_generated", {
        highlight_id: activeSelection.llmInputPreview.highlight_id,
        selected_strategy_id: effectiveStrategy?.strategy_id || "",
        selected_strategy_title: effectiveStrategy?.title || "",
      });
      if (effectiveStrategy) {
        await postSessionEvent("strategy_answer_generated", {
          highlight_id: activeSelection.llmInputPreview.highlight_id,
          source_turn_id: currentSourceTurnId,
          selected_strategy_id: effectiveStrategy.strategy_id || "",
          selected_strategy_title: effectiveStrategy.title || "",
          reaction_window_summary: currentReactionSummary || {},
        });
      }
      startReactionWindowForAssistantMessage(generatedAssistantMessage, selectionSnapshot);
    } catch (err) {
      setError(userFacingErrorMessage(err));
    } finally {
      setExplainLoading(false);
    }
  }

  async function sendFollowUp(event) {
    event.preventDefault();
    const question = followUpText.trim();
    const highlightId = activeSelection?.llmInputPreview?.highlight_id;
    if (!documentReady) {
      setError("Document is still loading; wait for a stable document id before explaining.");
      return;
    }
    if (!question || !highlightId) return;
    setFollowUpLoading(true);
    setError("");
    try {
      await endReactionWindow("followup_submitted", { requestCandidates: false, readProgressEstimate: 0.75 });
      const result = await postJson(
        `/api/documents/${documentId}/threads/${highlightId}/follow-up`,
        {
          question,
          selection_snapshot: activeSelection.llmInputPreview,
          session_id: readingSessionId,
          learning_state: learningState,
          input_source: followUpInputSource === "speech" ? "speech" : "text",
          response_mode: "normal_followup",
          strategy_candidates: [],
          selected_strategy_id: null,
          selected_strategy: null,
          trigger_context: null,
          reaction_window_summary: null,
          learning_signal_package: {},
          source_turn_id: "",
        },
      );
      const nextThread = result.thread || null;
      const generatedAssistantMessage = latestAssistantMessage(nextThread);
      if (monitorableAssistantMessage(generatedAssistantMessage)) {
        setReactionWindowScrollTargetId(generatedAssistantMessage.turn_id || generatedAssistantMessage.conversation_turn_id || "");
      }
      setThread(nextThread);
      threadRef.current = nextThread;
      setActiveSelection((current) => current ? { ...current, explainResult: result } : current);
      applyThreadStrategyState(nextThread);
      await postSessionEvent("follow_up_sent", {
        highlight_id: highlightId,
        input_source: followUpInputSource === "speech" ? "speech" : "text",
        response_mode: "normal_followup",
        selected_strategy_id: "",
        selected_strategy_title: "",
      });
      startReactionWindowForAssistantMessage(generatedAssistantMessage, activeSelectionRef.current);
      setFollowUpText("");
      setFollowUpInputSource("text");
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setFollowUpLoading(false);
    }
  }

  async function handleSelectStrategy(candidate, sourceContext = {}) {
    const sourceReactionSummary = sourceContext.reactionWindowSummary || reactionWindowSummary || {};
    const sourceCandidates = sourceContext.strategyCandidates || strategyCandidates;
    const sourceTriggerContext = sourceContext.triggerContext || strategyTriggerContext;
    await endReactionWindow("strategy_clicked", { requestCandidates: false, readProgressEstimate: sourceReactionSummary?.read_progress_estimate || 0.9 });
    setSelectedStrategy(candidate);
    if (sourceContext.reactionWindowSummary) setReactionWindowSummary(sourceContext.reactionWindowSummary);
    if (sourceContext.strategyCandidates) setStrategyCandidates(sourceContext.strategyCandidates);
    if (sourceContext.plannerMode) setStrategyPlannerMode(sourceContext.plannerMode);
    if (sourceContext.triggerContext) setStrategyTriggerContext(sourceContext.triggerContext);
    await postSessionEvent("strategy_selected", {
      selected_strategy_id: candidate?.strategy_id || "",
      selected_strategy_title: candidate?.title || "",
      reaction_window_summary: sourceReactionSummary,
      support_cue: sourceReactionSummary?.support_cue || "",
      support_cue_label: sourceReactionSummary?.support_cue_label || "",
      academic_state: learningState?.academic_state || "",
      confidence: learningState?.confidence,
    });
    await explainSelection(candidate, "explain_current_selection_with_selected_strategy", {
      sourceTurnId: sourceContext.sourceTurnId || sourceReactionSummary?.source_turn_id || "",
      reactionWindowSummary: sourceReactionSummary,
      strategyCandidates: sourceCandidates,
      triggerContext: sourceTriggerContext,
      plannerMode: sourceContext.plannerMode || strategyPlannerMode,
    });
  }

  async function ensureCameraSignalStarted(reason = "answer_generated") {
    setLiveSignalError("");
    if (cameraPausedByUserRef.current && reason !== "manual_start") return "paused_by_user";
    if (liveSignalActive || cameraStreamRef.current) {
      setCameraStartStatus("started");
      return "already_active";
    }
    if (!readingSessionId) {
      setCameraStartStatus("fallback_simulated");
      return "fallback_simulated";
    }
    setCameraStartStatus("starting");
    const latestStatus = modelStatus || await fetchJson("/api/emotion/model/status").catch(() => null);
    if (latestStatus) setModelStatus(latestStatus);
    if (latestStatus && latestStatus.model_loaded === false) {
      setLiveSignalError("Live model unavailable. Using simulated learning signal.");
      setLearningSignalSource("simulated");
      setCameraStartStatus("fallback_simulated");
      return "unavailable";
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setLiveSignalError("Live signal is unavailable in this browser. Using simulated learning signal.");
      setLearningSignalSource("simulated");
      setCameraStartStatus("fallback_simulated");
      return "unavailable";
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      cameraStreamRef.current = stream;
      setLearningSignalSource("webcam");
      setLiveSignalActive(true);
      cameraPausedByUserRef.current = false;
      setCameraPausedByUser(false);
      setCameraStartStatus("started");
      setShowSelfView(cameraSelfViewPreferenceRef.current !== "hidden");
      window.setTimeout(() => {
        if (cameraVideoRef.current) {
          cameraVideoRef.current.srcObject = stream;
          cameraVideoRef.current.play?.().catch(() => null);
        }
        sendWebcamFrame();
      }, 250);
      if (cameraTimerRef.current) window.clearInterval(cameraTimerRef.current);
      cameraTimerRef.current = window.setInterval(sendWebcamFrame, 1000);
      return "started";
    } catch (err) {
      setLiveSignalError(`Live signal could not start: ${err?.message || err}. Using simulated learning signal.`);
      stopLiveSignal("fallback_simulated");
      return err?.name === "NotAllowedError" ? "denied" : "unavailable";
    }
  }

  async function startLiveSignal() {
    cameraPausedByUserRef.current = false;
    setCameraPausedByUser(false);
    return ensureCameraSignalStarted("manual_start");
  }

  function pauseLiveSignal() {
    cameraPausedByUserRef.current = true;
    setCameraPausedByUser(true);
    stopLiveSignal("paused_by_user");
  }

  function stopLiveSignal(reason = "fallback_simulated") {
    if (cameraTimerRef.current) {
      window.clearInterval(cameraTimerRef.current);
      cameraTimerRef.current = null;
    }
    if (cameraStreamRef.current) {
      cameraStreamRef.current.getTracks?.().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }
    if (cameraVideoRef.current) cameraVideoRef.current.srcObject = null;
    setLiveSignalActive(false);
    setLearningSignalSource("simulated");
    if (reason === "paused_by_user") {
      setCameraStartStatus("paused_by_user");
    } else if (reason !== "unmount") {
      setCameraStartStatus("fallback_simulated");
    }
  }

  async function sendWebcamFrame() {
    if (!readingSessionId || !cameraVideoRef.current || !cameraCanvasRef.current) return;
    const video = cameraVideoRef.current;
    if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;
    const canvas = cameraCanvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const image = canvas.toDataURL("image/jpeg", 0.72);
    try {
      const payload = await postJson(`/api/reading-sessions/${readingSessionId}/emotion/frame`, {
        document_id: documentId,
        image,
        timestamp: new Date().toISOString(),
      });
      if (payload.model_status) setModelStatus(payload.model_status);
      if (payload.learning_state) setLearningState(payload.learning_state);
      if (payload.learning_state?.source !== "webcam_model") {
        setLiveSignalError("Live model unavailable. Using simulated learning signal.");
        stopLiveSignal("fallback_simulated");
      }
    } catch (err) {
      setLiveSignalError(`Live signal update failed: ${err?.message || err}. Using simulated learning signal.`);
      stopLiveSignal("fallback_simulated");
    }
  }

  function handleDismissStrategies() {
    const key = activeSelection && reactionWindowSummary ? reactionStrategyKey(activeSelection, reactionWindowSummary) : "";
    if (key) dismissedStrategyKeysRef.current.add(key);
    setStrategyCandidates([]);
    setStrategySourceKey("");
  }

  function handleFollowUpTextChange(value, inputSource = "text") {
    setFollowUpText(value);
    setFollowUpInputSource(inputSource === "speech" ? "speech" : "text");
  }

  return (
    <main className="pdf-chat-workspace">
      <section className="pdf-chat-reader-pane">
        <div className="pdf-chat-reader-toolbar">
          <button type="button" onClick={onBack}>Back to Library</button>
          <span>{currentMeta.title || currentMeta.file_name || "Untitled PDF"}</span>
          <button
            type="button"
            className={areaMode ? "active" : ""}
            onClick={() => setAreaMode((value) => !value)}
          >
            Area Select
          </button>
          {pendingSelection ? (
            <button type="button" className="pdf-chat-confirm-highlight" onClick={handleCreateHighlight} disabled={!documentReady}>
              {pendingSelection.type === "area" ? "Save Area" : "Save Highlight"}
            </button>
          ) : null}
        </div>
        <div className="pdf-chat-reader-surface">
          {loading ? <p className="pdf-chat-loading">Loading workspace...</p> : null}
          <PdfLoader
            key={pdfUrl}
            document={pdfUrl}
            beforeLoad={(progress) => (
              <div className="pdf-chat-loading">Loading PDF... {progress?.loaded || 0} bytes</div>
            )}
            errorMessage={(loaderError) => (
              <div className="pdf-chat-error">Failed to load PDF: {String(loaderError?.message || loaderError)}</div>
            )}
            onError={(loaderError) => {
              setError(String(loaderError?.message || loaderError));
            }}
          >
            {(pdfDocument) => (
              <PdfDocumentView
                pdfDocument={pdfDocument}
                highlights={viewerHighlights}
                areaMode={areaMode}
                onPdfDocumentLoaded={onPdfDocumentLoaded}
                onSelection={handleSelection}
                onHighlightClick={handleHighlightClick}
                highlighterUtilsRef={highlighterUtilsRef}
              />
            )}
          </PdfLoader>
        </div>
      </section>

      <ChatSidePanel
        meta={currentMeta}
        prepareStatus={prepareStatus}
        activeSelection={activeSelection}
        thread={thread}
        error={error}
        learningState={learningState}
        modelStatus={modelStatus}
        learningSignalSource={learningSignalSource}
        liveSignalActive={liveSignalActive}
        liveSignalError={liveSignalError}
        showSelfView={showSelfView}
        cameraPausedByUser={cameraPausedByUser}
        cameraStartStatus={cameraStartStatus}
        reactionWindowActive={reactionWindowActive}
        reactionWindowScrollTargetId={reactionWindowScrollTargetId}
        strategyCandidates={strategyCandidates}
        strategyPlannerMode={strategyPlannerMode}
        strategySourceTurnId={strategySourceTurnId}
        selectedStrategy={selectedStrategy}
        strategyLoading={strategyLoading}
        explainLoading={explainLoading}
        followUpText={followUpText}
        followUpLoading={followUpLoading}
        cleanupLoading={cleanupLoading}
        documentReady={documentReady}
        onExplainSelection={explainSelection}
        onSelectStrategy={handleSelectStrategy}
        onDismissStrategies={handleDismissStrategies}
        onRefreshStrategies={() => requestStrategyCandidates("manual_refresh")}
        onClearConversation={handleClearConversation}
        onDeleteHighlight={handleDeleteHighlight}
        onDeleteTurn={handleDeleteCurrentTurn}
        registerReactionSentinel={registerReactionSentinel}
        onConversationScroll={handleReactionWindowScroll}
        onStartLiveSignal={startLiveSignal}
        onUseSimulatedSignal={pauseLiveSignal}
        onToggleSelfView={() => setShowSelfView((value) => {
          cameraSelfViewPreferenceRef.current = value ? "hidden" : "visible";
          return !value;
        })}
        onFollowUpTextChange={handleFollowUpTextChange}
        onSendFollowUp={sendFollowUp}
        cameraVideoRef={cameraVideoRef}
      />
      <canvas ref={cameraCanvasRef} className="pdf-chat-camera-canvas" aria-hidden="true" />
    </main>
  );
}

function ChatSidePanel({
  meta,
  prepareStatus,
  activeSelection,
  thread,
  error,
  learningState,
  modelStatus,
  learningSignalSource,
  liveSignalActive,
  liveSignalError,
  showSelfView,
  cameraPausedByUser,
  cameraStartStatus,
  reactionWindowActive,
  reactionWindowScrollTargetId,
  strategyCandidates,
  strategyPlannerMode,
  strategySourceTurnId,
  selectedStrategy,
  strategyLoading,
  explainLoading,
  followUpText,
  followUpLoading,
  cleanupLoading,
  documentReady,
  onExplainSelection,
  onSelectStrategy,
  onDismissStrategies,
  onRefreshStrategies,
  onClearConversation,
  onDeleteHighlight,
  onDeleteTurn,
  registerReactionSentinel,
  onConversationScroll,
  onStartLiveSignal,
  onUseSimulatedSignal,
  onToggleSelfView,
  onFollowUpTextChange,
  onSendFollowUp,
  cameraVideoRef,
}) {
  const threadEndRef = useRef(null);
  const lastAutoScrollKeyRef = useRef("");
  const [speechState, setSpeechState] = useState("idle");
  const [speechMessage, setSpeechMessage] = useState("");
  const mediaRecorderRef = useRef(null);
  const speechStreamRef = useRef(null);
  const speechChunksRef = useRef([]);
  const speechStopTimerRef = useRef(null);
  const isConversationLoading = explainLoading || followUpLoading;
  const messageCount = thread?.messages?.length || 0;
  const suppressAutoScrollForReactionWindow = Boolean(reactionWindowActive || reactionWindowScrollTargetId);
  const isSpeechRecording = speechState === "recording";
  const isSpeechTranscribing = speechState === "transcribing";
  const speechButtonLabel = isSpeechRecording ? "Stop speech recording" : "Record speech question";
  const speechButtonDisabled = isSpeechTranscribing || (!isSpeechRecording && (!documentReady || !activeSelection || followUpLoading));
  const speechStatusLabel = speechState === "recording"
    ? "Recording..."
    : speechState === "transcribing"
      ? "Transcribing..."
      : speechState === "error"
        ? speechMessage || "Speech unavailable"
        : "";

  useEffect(() => {
    if (!messageCount && !isConversationLoading) return;
    const autoScrollKey = `${messageCount}:${isConversationLoading}`;
    if (lastAutoScrollKeyRef.current === autoScrollKey) return;
    lastAutoScrollKeyRef.current = autoScrollKey;
    if (suppressAutoScrollForReactionWindow) return;
    threadEndRef.current?.scrollIntoView({
      behavior: isConversationLoading ? "auto" : "smooth",
      block: "end",
    });
  }, [messageCount, isConversationLoading, suppressAutoScrollForReactionWindow]);

  useEffect(() => () => {
    cleanupSpeechRecording();
  }, []);

  function clearSpeechStopTimer() {
    if (!speechStopTimerRef.current) return;
    window.clearTimeout(speechStopTimerRef.current);
    speechStopTimerRef.current = null;
  }

  function stopSpeechStream() {
    speechStreamRef.current?.getTracks?.().forEach((track) => track.stop());
    speechStreamRef.current = null;
  }

  function cleanupSpeechRecording() {
    clearSpeechStopTimer();
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.onstop = null;
      recorder.stop();
    }
    mediaRecorderRef.current = null;
    speechChunksRef.current = [];
    stopSpeechStream();
  }

  async function startSpeechRecording() {
    if (!browserSupportsSpeechRecording()) {
      setSpeechState("error");
      setSpeechMessage("Speech unavailable");
      return;
    }
    clearSpeechStopTimer();
    speechChunksRef.current = [];
    setSpeechMessage("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = preferredSpeechMimeType();
      const recorder = mimeType
        ? new MediaRecorder(stream, { mimeType })
        : new MediaRecorder(stream);
      speechStreamRef.current = stream;
      mediaRecorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data?.size) speechChunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        setSpeechState("error");
        setSpeechMessage("Speech unavailable");
        cleanupSpeechRecording();
      };
      recorder.onstop = () => {
        clearSpeechStopTimer();
        stopSpeechStream();
        void transcribeRecordedSpeech();
      };
      setSpeechState("recording");
      recorder.start();
      speechStopTimerRef.current = window.setTimeout(stopSpeechRecording, SPEECH_RECORDING_MAX_MS);
    } catch (err) {
      cleanupSpeechRecording();
      setSpeechState("error");
      setSpeechMessage("Speech unavailable");
    }
  }

  function stopSpeechRecording() {
    clearSpeechStopTimer();
    const recorder = mediaRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.stop();
      return;
    }
    stopSpeechStream();
    if (speechState === "recording") setSpeechState("idle");
  }

  async function transcribeRecordedSpeech() {
    const chunks = speechChunksRef.current;
    speechChunksRef.current = [];
    if (!chunks.length) {
      setSpeechState("error");
      setSpeechMessage("Speech unavailable");
      return;
    }
    setSpeechState("transcribing");
    setSpeechMessage("");
    try {
      const mimeType = chunks.find((chunk) => chunk.type)?.type || preferredSpeechMimeType() || "audio/webm";
      const extension = mimeType.includes("wav") ? "wav" : "webm";
      const blob = new Blob(chunks, { type: mimeType });
      const formData = new FormData();
      formData.append("audio", blob, `speech.${extension}`);
      const response = await fetch("/api/speech/transcribe", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      if (!payload.available) throw new Error(payload.error || payload.message || "Speech transcription unavailable");
      const transcript = normalizePdfText(payload.text || "");
      if (!transcript) throw new Error("Speech transcription unavailable");
      onFollowUpTextChange(transcript, "speech");
      setSpeechState("idle");
      setSpeechMessage("");
    } catch (err) {
      setSpeechState("error");
      setSpeechMessage("Speech unavailable");
    }
  }

  function handleSpeechButtonClick() {
    if (isSpeechRecording) {
      stopSpeechRecording();
      return;
    }
    void startSpeechRecording();
  }

  return (
    <aside className="pdf-chat-side-panel">
      <div className="pdf-chat-side-top">
        <LearningSignalPanel
          learningState={learningState}
          activeSelection={activeSelection}
          meta={meta}
          prepareStatus={prepareStatus}
          modelStatus={modelStatus}
          learningSignalSource={learningSignalSource}
          liveSignalActive={liveSignalActive}
          liveSignalError={liveSignalError}
          showSelfView={showSelfView}
          cameraPausedByUser={cameraPausedByUser}
          cameraStartStatus={cameraStartStatus}
          reactionWindowActive={reactionWindowActive}
          onStartLiveSignal={onStartLiveSignal}
          onUseSimulatedSignal={onUseSimulatedSignal}
          onToggleSelfView={onToggleSelfView}
          cameraVideoRef={cameraVideoRef}
        />
      </div>

      <CurrentSelectionCard
        activeSelection={activeSelection}
        prepareStatus={prepareStatus}
        explainLoading={explainLoading}
        cleanupLoading={cleanupLoading}
        documentReady={documentReady}
        hasThreadMessages={Boolean(thread?.messages?.length)}
        onExplainSelection={onExplainSelection}
        onClearConversation={onClearConversation}
        onDeleteHighlight={onDeleteHighlight}
      />

      <div className="pdf-chat-conversation-region">
      <div className="pdf-chat-side-scroll">
        <div className="pdf-chat-conversation-scroll" onScroll={onConversationScroll}>
        {error ? <p className="pdf-chat-error">{error}</p> : null}

          <HighlightThreadView
          thread={thread}
          isLoading={isConversationLoading}
          endRef={threadEndRef}
          strategyCandidates={strategyCandidates}
          turnMetadata={thread?.turn_metadata || {}}
          strategyPlannerMode={strategyPlannerMode}
          strategySourceTurnId={strategySourceTurnId}
          selectedStrategy={selectedStrategy}
          strategyLoading={strategyLoading}
          answerLoading={explainLoading}
          cleanupLoading={cleanupLoading}
          onSelectStrategy={onSelectStrategy}
          onDismissStrategies={onDismissStrategies}
          onRefreshStrategies={onRefreshStrategies}
          onDeleteTurn={onDeleteTurn}
          registerReactionSentinel={registerReactionSentinel}
        />
        </div>
      </div>
      </div>

      <form className="pdf-chat-follow-up compact" onSubmit={onSendFollowUp}>
        <FollowUpContextLine activeSelection={activeSelection} />
        {speechStatusLabel ? (
          <span className={`pdf-chat-speech-status ${speechState}`} title={speechMessage || speechStatusLabel}>
            {speechStatusLabel}
          </span>
        ) : null}
        <input
          value={followUpText}
          onChange={(event) => onFollowUpTextChange(event.target.value, "text")}
          placeholder="Ask a follow-up about this selection..."
          disabled={!documentReady || !activeSelection || followUpLoading}
        />
        <button
          type="button"
          className={`pdf-chat-speech-button ${isSpeechRecording ? "recording" : ""}`}
          onClick={handleSpeechButtonClick}
          disabled={speechButtonDisabled}
          aria-label={speechButtonLabel}
          title={speechButtonLabel}
        >
          {isSpeechRecording ? "Stop" : "Mic"}
        </button>
        <button type="submit" disabled={!documentReady || !activeSelection || !followUpText.trim() || followUpLoading}>
          Send
        </button>
      </form>
    </aside>
  );
}

function DocumentDetailsToggle({ meta, prepareStatus }) {
  return (
    <details className="pdf-chat-document-details-menu">
      <summary>Document details</summary>
      <div className="pdf-chat-document-details-popover">
        <dl className="pdf-chat-small-grid">
          <dt>file</dt>
          <dd>{meta.file_name || meta.title || "Untitled PDF"}</dd>
          <dt>pages</dt>
          <dd>{pageCountLabel(meta.page_count)}</dd>
          <dt>status</dt>
          <dd>{preparationStatusLabel(meta.prepare_status || prepareStatus.status)}</dd>
          <dt>retrieval method</dt>
          <dd>{prepareStatus.retrieval_method || meta.retrieval_method || "unknown"}</dd>
          <dt>parsed blocks</dt>
          <dd>{String(prepareStatus.block_count ?? meta.parsed_blocks_count ?? "-")}</dd>
          <dt>embedding status</dt>
          <dd>{prepareStatus.embedding_index_status || meta.embedding_status || "unknown"}</dd>
        </dl>
        <PreparationStatus meta={meta} prepareStatus={prepareStatus} />
      </div>
    </details>
  );
}

function CurrentSelectionCard({
  activeSelection,
  prepareStatus,
  explainLoading,
  cleanupLoading,
  documentReady,
  hasThreadMessages,
  onExplainSelection,
  onClearConversation,
  onDeleteHighlight,
}) {
  if (!activeSelection) {
    return (
      <section className="pdf-chat-selection-toolbar empty">
        <h3>Current Selection</h3>
        <p>Select text or an area to start an explanation.</p>
      </section>
    );
  }

  const preview = activeSelection.llmInputPreview;
  const isArea = preview.highlight_type === "area";
  const selectedText = preview.selected_text || "";
  const selectionTitle = `${selectionKindLabel(preview)} · Page ${preview.page_number || "-"}`;
  const showFullSelectionToggle = !isArea && normalizePdfText(selectedText).length > 180;
  const cropThumbnail = preview.crop_image_data_url || preview.crop_url;
  const previewText = isArea ? truncateText(preview.caption || "Selected area", 120) : truncateText(selectedText, 160);

  function handleBaselineExplain() {
    onExplainSelection(null, "baseline_explain_current_selection");
  }

  return (
    <section className="pdf-chat-selection-toolbar">
      <div className={`pdf-chat-selection-main ${isArea && cropThumbnail ? "has-thumb" : ""}`}>
        {isArea && cropThumbnail ? (
          <img className="pdf-chat-crop-thumb compact" src={cropThumbnail} alt="Selected PDF area crop" />
        ) : null}
        <div>
          <h3>{selectionTitle}</h3>
          {previewText ? <p className="pdf-chat-selection-preview-line pdf-chat-selection-text-preview">{previewText}</p> : null}
        </div>
      </div>
      <div className="pdf-chat-selection-actions">
        <button
          type="button"
          className="pdf-chat-explain-button"
          onClick={() => handleBaselineExplain()}
          disabled={!documentReady || explainLoading || cleanupLoading}
        >
          {explainLoading ? "Explaining..." : "Explain"}
        </button>
        <details className="pdf-chat-selection-more">
          <summary>More</summary>
          <div className="pdf-chat-selection-more-menu">
            {showFullSelectionToggle ? (
              <details className="pdf-chat-selection-expand">
                <summary>Show full selection</summary>
                <blockquote>{selectedText}</blockquote>
              </details>
            ) : null}
            {isArea && preview.caption ? (
              <p className="pdf-chat-caption"><span>Matched caption</span>{preview.caption}</p>
            ) : null}
            <button type="button" className="pdf-chat-secondary-action" onClick={onClearConversation} disabled={!hasThreadMessages || cleanupLoading || explainLoading}>
              Clear conversation
            </button>
            <button type="button" className="pdf-chat-danger-action" onClick={onDeleteHighlight} disabled={cleanupLoading || explainLoading}>
              Delete highlight
            </button>
            <ContextInspector activeSelection={activeSelection} prepareStatus={prepareStatus} />
          </div>
        </details>
      </div>
    </section>
  );
}

function LearningSignalPanel({
  learningState,
  activeSelection,
  meta,
  prepareStatus,
  modelStatus,
  learningSignalSource,
  liveSignalActive,
  liveSignalError,
  showSelfView,
  cameraPausedByUser,
  cameraStartStatus,
  reactionWindowActive,
  onStartLiveSignal,
  onUseSimulatedSignal,
  onToggleSelfView,
  cameraVideoRef,
}) {
  const distribution = learningState?.distribution || {};
  const states = ["boredom", "confusion", "engagement", "frustration"];
  const liveUnavailable = modelStatus && modelStatus.model_loaded === false;
  const faceDetection = learningState?.face_detection || modelStatus?.face_detector || {};
  const faceDetectionLabel = faceDetectionLabelFor(faceDetection);
  const modelOutputType = learningState?.model_output_type || modelStatus?.emotion_pipeline_status?.model_output_type || modelStatus?.model_output_type || "academic_state";
  const modelOutputLabel = modelModeLabelForLearningSignal(modelOutputType);
  const rawSignalLabel = rawSignalLabelForLearningSignal(learningState, modelStatus);
  const statusLabel = reactionWindowActive
    ? "Observing response..."
    : cameraStartStatus === "starting"
      ? "Starting live signal"
      : cameraPausedByUser
        ? "Camera paused"
        : liveSignalActive
          ? "Live signal active"
          : liveSignalError
            ? "Simulated fallback"
            : "Camera standby";
  const cueLabel = learningSupportCueLabel(learningState);
  const trendLabel = learningState?.trend || "stable";
  return (
    <section className="pdf-chat-learning-strip" aria-label="Live learning signal">
      <div className="pdf-chat-signal-summary">
        <div className={`pdf-chat-camera-preview-wrap compact ${liveSignalActive && showSelfView ? "visible" : ""}`}>
          <video ref={cameraVideoRef} className="pdf-chat-camera-preview" playsInline muted aria-hidden={!showSelfView} />
          {!liveSignalActive ? <p>standby</p> : !showSelfView ? <p>hidden</p> : null}
        </div>
        <div className="pdf-chat-signal-main">
          <div className="pdf-chat-learning-signal-header">
            <h3>Learning signal</h3>
          </div>
          <p className="pdf-chat-signal-line">
            {liveSignalActive || learningState ? <>Learning cue: {cueLabel} · {formatPercent(learningState?.confidence)} · {trendLabel}</> : statusLabel}
          </p>
          {reactionWindowActive ? <p className="pdf-chat-learning-note compact">Observing response...</p> : null}
        </div>
        <div className="pdf-chat-learning-source-controls">
          {liveSignalActive ? (
            <button type="button" onClick={onUseSimulatedSignal} aria-label="Pause camera signal" title="Pause camera signal">Pause</button>
          ) : (
            <button type="button" onClick={onStartLiveSignal} aria-label="Start camera signal" title="Start camera signal">Start</button>
          )}
          {liveSignalActive ? (
            <button
              type="button"
              onClick={onToggleSelfView}
              aria-label={showSelfView ? "Hide self-view" : "Show self-view"}
              title={showSelfView ? "Hide self-view" : "Show self-view"}
            >
              {showSelfView ? "Hide" : "Show"}
            </button>
          ) : null}
          <details className="pdf-chat-learning-details">
            <summary aria-label="Signal details" title="Signal details">Details</summary>
            <div className="pdf-chat-signal-details-popover">
              <section className="pdf-chat-signal-details-section">
                <h4>Signal details</h4>
                {!liveSignalActive ? <p className="pdf-chat-learning-note">Camera signal standby. Signal monitoring starts after an explanation is shown.</p> : null}
                {liveUnavailable ? <p className="pdf-chat-learning-note">Live model unavailable. Using simulated learning signal.</p> : null}
                {liveSignalError ? <p className="pdf-chat-learning-note">{liveSignalError}</p> : null}
                {!activeSelection ? (
                  <p className="pdf-chat-learning-note">
                    Select a passage or figure to see support strategies when the learning signal becomes relevant.
                  </p>
                ) : null}
                <dl className="pdf-chat-learning-grid">
                  <dt>Status</dt>
                  <dd>{statusLabel}</dd>
                  <dt>Source</dt>
                  <dd>{sourceLabelForLearningSignal(learningSignalSource, learningState)}</dd>
                  <dt>Face detection</dt>
                  <dd>Face detection: {faceDetectionLabel}</dd>
                  <dt>Model</dt>
                  <dd>Model output type: {modelOutputLabel}</dd>
                  <dt>Learning support</dt>
                  <dd>{rawSignalLabel}</dd>
                  <dt>Tools</dt>
                  <dd>
                    <a href="/settings" target="_blank" rel="noreferrer">Settings</a>
                    {" · "}
                    <a href="/llm-compare" target="_blank" rel="noreferrer">LLM compare</a>
                    {" · "}
                    <a href="/camera-debug" target="_blank" rel="noreferrer">Camera debug</a>
                    {" · "}
                    <a href="/pdf-test" target="_blank" rel="noreferrer" title="Open PDF/RAG debug workspace">PDF debug</a>
                  </dd>
                  <dt>Duration</dt>
                  <dd>{formatDuration(learningState?.duration_sec || 0)}</dd>
                </dl>
                <div className="pdf-chat-distribution-list" aria-label="Learning-state distribution">
                  {states.map((state) => (
                    <div className="pdf-chat-distribution-row" key={state}>
                      <span>{learningStateDisplayLabel(state)}</span>
                      <div className="pdf-chat-distribution-track">
                        <div className="pdf-chat-distribution-bar" style={{ width: `${Math.round(Number(distribution[state] || 0) * 100)}%` }} />
                      </div>
                      <span>{formatPercent(distribution[state])}</span>
                    </div>
                  ))}
                </div>
              </section>
              <section className="pdf-chat-signal-details-section">
                <h4>Document details</h4>
                <PreparationStatus meta={meta} prepareStatus={prepareStatus} />
              </section>
            </div>
          </details>
        </div>
      </div>
    </section>
  );
}

function StrategyCandidatePanel({
  candidates,
  plannerMode,
  sourceTurnType,
  selectedStrategy,
  loading,
  answerLoading,
  onSelectStrategy,
  onDismiss,
  onRefresh,
}) {
  if (loading && !candidates.length) {
    return (
      <section className="pdf-chat-strategy-panel">
        <h3>{strategyPanelHeading(sourceTurnType)}</h3>
        <p className="pdf-chat-muted">Preparing context-specific strategies...</p>
      </section>
    );
  }
  if (!candidates.length) return null;
  const orderedCandidates = normalizeRecommendedCandidatesForDisplay(candidates);
  const recommendedCandidate = orderedCandidates.find((candidate) => candidate.recommended) || orderedCandidates[0];
  const alternativeCandidates = orderedCandidates.filter((candidate) => candidate.strategy_id !== recommendedCandidate?.strategy_id);
  return (
    <section className="pdf-chat-strategy-panel">
      <div className="pdf-chat-strategy-header">
        <div>
          <h3>{strategyPanelHeading(sourceTurnType)}</h3>
          <p>Based on the recent learning signal while this explanation was being read, here is one recommended way to continue.</p>
        </div>
        <span>{plannerMode || "heuristic"}</span>
      </div>
      {recommendedCandidate ? (
        <div className="pdf-chat-recommended-strategy">
          <StrategyCard
            candidate={recommendedCandidate}
            selected={selectedStrategy?.strategy_id === recommendedCandidate.strategy_id}
            loading={loading}
            answerLoading={answerLoading}
            badgeText="Recommended"
            onSelectStrategy={onSelectStrategy}
          />
        </div>
      ) : null}
      {alternativeCandidates.length ? (
        <details className="pdf-chat-alternative-strategies">
          <summary>Other ways to explain this</summary>
          <div className="pdf-chat-strategy-list">
            {alternativeCandidates.map((candidate) => (
              <StrategyCard
                key={candidate.strategy_id}
                candidate={candidate}
                selected={selectedStrategy?.strategy_id === candidate.strategy_id}
                loading={loading}
                answerLoading={answerLoading}
                badgeText="Alternative"
                onSelectStrategy={onSelectStrategy}
              />
            ))}
          </div>
        </details>
      ) : null}
      <div className="pdf-chat-strategy-actions">
        <button type="button" onClick={onRefresh} disabled={loading}>Refresh</button>
        <button type="button" onClick={onDismiss}>Dismiss</button>
      </div>
    </section>
  );
}

function StrategyCard({ candidate, selected, loading, answerLoading, badgeText, onSelectStrategy }) {
  const move = candidate.pedagogical_move || strategyPedagogicalMove(candidate);
  const focus = candidate.context_focus || strategyContextFocus(candidate);
  const evidenceText = strategyAcademicStateEvidenceText(candidate);
  const supportCue = strategySupportCueLabel(candidate);
  const reason = strategyReasonText(candidate);
  return (
    <article className={`pdf-chat-strategy-card ${selected ? "selected" : ""}`}>
      <div className="pdf-chat-strategy-card-title">
        <h4>{move}</h4>
        {badgeText ? <span className="pdf-chat-strategy-badge">{badgeText}</span> : null}
      </div>
      {focus ? (
        <p className="pdf-chat-strategy-focus">
          <span>Focus</span>
          {focus}
        </p>
      ) : null}
      <p>{candidate.short_description}</p>
      {evidenceText ? (
        <p className="pdf-chat-strategy-evidence"><span>Learning-state evidence: </span>{evidenceText}</p>
      ) : null}
      {supportCue ? (
        <p className="pdf-chat-strategy-evidence"><span>Support cue: </span>{supportCue}</p>
      ) : null}
      <p className="pdf-chat-strategy-why"><span>Why this appeared:</span>{reason}</p>
      <button type="button" onClick={() => onSelectStrategy(candidate)} disabled={loading || answerLoading}>
        {selected && answerLoading ? "Explaining this way..." : "Explain with this strategy"}
      </button>
    </article>
  );
}

function normalizeRecommendedCandidatesForDisplay(candidates) {
  const normalized = (candidates || []).map((candidate, index) => ({
    ...candidate,
    recommended_score: finiteScore(candidate.recommended_score, Math.max(0.5, 0.72 - index * 0.04)),
  }));
  if (!normalized.length) return [];
  const bestIndex = normalized.reduce((best, candidate, index) => (
    candidate.recommended_score > normalized[best].recommended_score ? index : best
  ), 0);
  return normalized
    .map((candidate, index) => ({ ...candidate, recommended: index === bestIndex }))
    .sort((a, b) => Number(b.recommended) - Number(a.recommended) || b.recommended_score - a.recommended_score);
}

function fallbackStrategyFamily(strategy) {
  return strategy?.strategy_family || strategy?.strategy_id || "custom_strategy";
}

function strategyPedagogicalMove(strategy) {
  return strategy?.pedagogical_move || strategy?.strategy_title || strategy?.title || strategy?.strategy_id || "Selected strategy";
}

function strategyContextFocus(strategy) {
  return strategy?.context_focus || "";
}

function strategyReasonText(strategy) {
  return strategy?.reason_text || strategy?.why_recommended || "";
}

function strategyAcademicStateEvidenceText(strategy) {
  return strategy?.academic_state_evidence_text || "";
}

function strategySupportCueLabel(strategy) {
  return strategy?.support_cue_label || strategy?.support_cue_display_label || (strategy?.support_cue ? supportCueReasonLabel(canonicalSupportCueForDisplay(strategy.support_cue)) : "");
}

function strategyPanelHeading(sourceTurnType) {
  return sourceTurnType === "strategy_reexplanation" ? "Suggested next move" : "Suggested ways to improve this explanation";
}

function FollowUpContextLine({ activeSelection, strategy }) {
  const selectionLabel = activeSelection?.llmInputPreview ? selectionKindLabel(activeSelection.llmInputPreview) : "No active selection";
  const pageLabel = activeSelection?.llmInputPreview?.page_number ? ` · Page ${activeSelection.llmInputPreview.page_number}` : "";
  const strategyLabel = strategy ? ` · Strategy: ${strategyPedagogicalMove(strategy)}` : "";
  return (
    <p className="pdf-chat-follow-up-context-line">{selectionLabel}{pageLabel}{strategyLabel}</p>
  );
}

function HighlightThreadView({
  thread,
  isLoading,
  endRef,
  strategyCandidates,
  turnMetadata,
  strategyPlannerMode,
  strategySourceTurnId,
  selectedStrategy,
  strategyLoading,
  answerLoading,
  cleanupLoading,
  onSelectStrategy,
  onDismissStrategies,
  onRefreshStrategies,
  onDeleteTurn,
  registerReactionSentinel,
}) {
  const messages = thread?.messages || [];
  const turns = useMemo(() => deriveConversationTurns(messages), [messages]);
  const [currentTurnIndex, setCurrentTurnIndex] = useState(0);
  const currentTurn = turns[currentTurnIndex] || null;

  useEffect(() => {
    setCurrentTurnIndex(Math.max(0, turns.length - 1));
  }, [turns.length, isLoading]);

  function goToPreviousTurn() {
    setCurrentTurnIndex((index) => Math.max(0, index - 1));
  }

  function goToNextTurn() {
    setCurrentTurnIndex((index) => Math.min(Math.max(0, turns.length - 1), index + 1));
  }

  function goToLatestTurn() {
    setCurrentTurnIndex(Math.max(0, turns.length - 1));
  }

  return (
    <section className="pdf-chat-thread">
      <div className="pdf-chat-thread-toolbar">
        <h3>Conversation</h3>
        {turns.length ? (
          <div className="pdf-chat-turn-controls">
            <span className="pdf-chat-turn-indicator">Turn {currentTurnIndex + 1} / {turns.length}</span>
            <button type="button" onClick={goToPreviousTurn} disabled={currentTurnIndex <= 0}>Previous</button>
            <button type="button" onClick={goToNextTurn} disabled={currentTurnIndex >= turns.length - 1}>Next</button>
            <button type="button" onClick={goToLatestTurn} disabled={currentTurnIndex >= turns.length - 1}>Latest</button>
            <button type="button" className="pdf-chat-turn-delete" onClick={() => onDeleteTurn(currentTurn?.turn_id)} disabled={!currentTurn?.turn_id || cleanupLoading}>
              Delete turn
            </button>
          </div>
        ) : null}
      </div>
      {!turns.length && !isLoading ? <p className="pdf-chat-muted">Ask for an explanation to start this selection's conversation.</p> : null}
      {currentTurn ? currentTurn.messages.map((message, index) => {
        const roleClass = message.role === "user" ? "user" : "assistant";
        const continuingStrategy = currentTurn.messages.some((item) => item.role === "user");
        return (
          <article className={`pdf-chat-message ${roleClass}`} key={`${message.created_at || index}-${index}`}>
            <div className="pdf-chat-bubble">
              {message.role === "assistant" ? (
                <ConversationStrategyBadge message={message} continuing={continuingStrategy} />
              ) : null}
              <div className="pdf-chat-message-content">
                {message.role === "assistant" ? (
                  <span
                    className="pdf-chat-reaction-sentinel top"
                    data-reaction-turn-id={message.turn_id || message.conversation_turn_id || ""}
                    data-reaction-sentinel="top"
                    ref={(node) => registerReactionSentinel?.(message.turn_id || message.conversation_turn_id || "", "top", node)}
                    aria-hidden="true"
                  />
                ) : null}
                <MarkdownText content={message.content || ""} />
                {message.role === "assistant" ? (
                  <span
                    className="pdf-chat-reaction-sentinel bottom"
                    data-reaction-turn-id={message.turn_id || message.conversation_turn_id || ""}
                    data-reaction-sentinel="bottom"
                    ref={(node) => registerReactionSentinel?.(message.turn_id || message.conversation_turn_id || "", "bottom", node)}
                    aria-hidden="true"
                  />
                ) : null}
              </div>
            </div>
          </article>
        );
      }) : null}
      <TurnStrategySuggestions
        turn={currentTurn}
        turnMetadata={turnMetadata}
        strategySourceTurnId={strategySourceTurnId}
        candidates={strategyCandidates}
        plannerMode={strategyPlannerMode}
        selectedStrategy={selectedStrategy}
        loading={strategyLoading}
        answerLoading={answerLoading}
        onSelectStrategy={onSelectStrategy}
        onDismiss={onDismissStrategies}
        onRefresh={onRefreshStrategies}
      />
      {isLoading ? (
        <article className="pdf-chat-message assistant">
          <div className="pdf-chat-bubble pdf-chat-loading-bubble">
            <span className="pdf-chat-dot" />
            <span className="pdf-chat-dot" />
            <span className="pdf-chat-dot" />
          </div>
        </article>
      ) : null}
      <div className="pdf-chat-thread-end" ref={endRef} />
    </section>
  );
}

function TurnStrategySuggestions({
  turn,
  turnMetadata,
  strategySourceTurnId,
  candidates,
  plannerMode,
  selectedStrategy,
  loading,
  answerLoading,
  onSelectStrategy,
  onDismiss,
  onRefresh,
}) {
  const matchesSourceTurn = Boolean(turn && turn.turn_id === strategySourceTurnId);
  const turnSpecificCandidates = strategyCandidatesForTurn(turn, turnMetadata, candidates, strategySourceTurnId);
  const metadata = turnMetadataForThread({ turn_metadata: turnMetadata || {} }, turn?.turn_id);
  if (metadata.strategy_status === "applied") {
    return <AppliedStrategyRecord metadata={metadata} />;
  }
  if (!matchesSourceTurn && !turnSpecificCandidates.length) return null;
  return (
    <StrategyCandidatePanel
      candidates={turnSpecificCandidates}
      plannerMode={metadata?.planner_mode || plannerMode}
      sourceTurnType={metadata?.reaction_window_summary?.source_turn_type || metadata?.source_turn_type || ""}
      selectedStrategy={selectedStrategy}
      loading={matchesSourceTurn && loading}
      answerLoading={answerLoading}
      onSelectStrategy={(candidate) => onSelectStrategy(candidate, {
        sourceTurnId: turn?.turn_id || "",
        reactionWindowSummary: metadata?.reaction_window_summary || {},
        strategyCandidates: turnSpecificCandidates,
        triggerContext: metadata?.trigger_context || {},
        plannerMode: metadata?.planner_mode || plannerMode,
      })}
      onDismiss={onDismiss}
      onRefresh={onRefresh}
    />
  );
}

function AppliedStrategyRecord({ metadata }) {
  const reactionSummary = metadata?.reaction_window_summary || {};
  const observedWindow = observedWindowLabel(reactionSummary);
  const applied = {
    strategy_id: metadata?.applied_strategy_id || "",
    strategy_family: metadata?.applied_strategy_family || "",
    pedagogical_move: metadata?.applied_pedagogical_move || "",
    context_focus: metadata?.applied_context_focus || "",
    title: metadata?.applied_strategy_title || "",
    strategy_status: "applied",
  };
  const move = strategyPedagogicalMove(applied);
  const reason = strategyReasonText(metadata) || metadata?.strategy_reason || reactionSummary.reason_text || reactionSummary.trigger_reason || metadata?.trigger_context?.trigger_reason || "";
  const evidenceText = strategyAcademicStateEvidenceText(metadata) || reactionSummary.academic_state_evidence_text || "";
  const supportCue = strategySupportCueLabel(metadata) || reactionSummary.support_cue_display_label || reactionSummary.support_cue_label || "";
  if (!metadata?.applied_strategy_id && !metadata?.applied_strategy_family && !move) return null;
  return (
    <section className="pdf-chat-strategy-panel applied">
      <div className="pdf-chat-strategy-header">
        <div>
          <h3>Strategy used</h3>
          <p>Applied strategy: {move}</p>
        </div>
        <span>Applied</span>
      </div>
      {evidenceText ? (
        <p className="pdf-chat-strategy-evidence"><span>Learning-state evidence: </span>{evidenceText}</p>
      ) : null}
      {supportCue ? (
        <p className="pdf-chat-strategy-evidence"><span>Support cue: </span>{supportCue}</p>
      ) : null}
      {reason ? (
        <p className="pdf-chat-strategy-why"><span>Why this strategy was used:</span>{reason}</p>
      ) : null}
      {observedWindow ? <p className="pdf-chat-muted">Observed window: {observedWindow}</p> : null}
    </section>
  );
}

function ConversationStrategyBadge({ message, continuing }) {
  if (!message.strategy_title && !message.pedagogical_move) return null;
  const reactionSummary = message.reaction_window_summary || {};
  const observedWindow = observedWindowLabel(reactionSummary);
  const strategyReason = strategyReasonText(message) || message.strategy_reason || reactionSummary.reason_text || reactionSummary.trigger_reason || message.trigger_context?.trigger_reason || "";
  const evidenceText = strategyAcademicStateEvidenceText(message) || reactionSummary.academic_state_evidence_text || "";
  const supportCue = strategySupportCueLabel(message) || reactionSummary.support_cue_display_label || reactionSummary.support_cue_label || "";
  const move = strategyPedagogicalMove(message);
  const focus = strategyContextFocus(message);
  const tracePayload = {
    source_turn_id: message.source_turn_id || reactionSummary.source_turn_id || "",
    reaction_window_summary: reactionSummary,
    planner_mode: message.planner_mode || null,
    planner_prompt_version: message.planner_prompt_version || "",
    selected_strategy_id: message.applied_strategy_id || message.strategy_id || "",
    selected_strategy: {
      strategy_id: message.applied_strategy_id || message.strategy_id || "",
      strategy_family: message.applied_strategy_family || message.strategy_family || fallbackStrategyFamily(message),
      pedagogical_move: message.applied_pedagogical_move || message.pedagogical_move || "",
      context_focus: message.applied_context_focus || message.context_focus || "",
      title: message.strategy_title || "",
      short_description: message.strategy_short_description || "",
    },
    legacy_selected_strategy: {
      pedagogical_move: message.pedagogical_move || "",
      context_focus: message.context_focus || "",
    },
    applied_strategy_id: message.applied_strategy_id || "",
    applied_strategy_family: message.applied_strategy_family || "",
    source_reaction_turn_id: message.source_reaction_turn_id || "",
    reason_text: message.reason_text || "",
    academic_state_evidence_text: message.academic_state_evidence_text || reactionSummary.academic_state_evidence_text || "",
    confidence_handling: message.confidence_handling || reactionSummary.confidence_handling || "",
    source_turn_type: message.source_turn_type || reactionSummary.source_turn_type || "",
    support_cue_label: message.support_cue_label || reactionSummary.support_cue_display_label || reactionSummary.support_cue_label || "",
    strategy_status: message.strategy_status || "",
    trigger_reason: strategyReason,
    support_cue: message.support_cue || reactionSummary.support_cue || "",
    average_distribution: reactionSummary.avg_distribution || {},
    face_detector: reactionSummary.face_detection_summary?.mode || message.face_detection_summary?.mode || "",
    planner_input_summary: message.planner_input_summary || {},
  };
  return (
    <div className="pdf-chat-strategy-message-badge">
      <span>{continuing ? "Continuing with strategy:" : "Using strategy:"}</span>
      <strong>{move}</strong>
      {focus ? (
        <div className="pdf-chat-strategy-focus compact">
          <span>Focus:</span>
          <p>{focus}</p>
        </div>
      ) : null}
      {message.strategy_short_description ? <small>{message.strategy_short_description}</small> : null}
      {evidenceText ? (
        <div className="pdf-chat-strategy-focus compact">
          <span>Learning-state evidence: </span>
          <p>{evidenceText}</p>
        </div>
      ) : null}
      {supportCue ? (
        <div className="pdf-chat-strategy-focus compact">
          <span>Support cue: </span>
          <p>{supportCue}</p>
        </div>
      ) : null}
      {strategyReason ? (
        <div className="pdf-chat-strategy-reason">
          <span>Why this strategy was used</span>
          <p>{strategyReason}</p>
        </div>
      ) : null}
      {observedWindow ? <small>Observed window: {observedWindow}</small> : null}
      <details className="pdf-chat-strategy-trace">
        <summary>Strategy trace</summary>
        <pre>{formatJson(tracePayload)}</pre>
      </details>
    </div>
  );
}

function ContextInspector({ activeSelection, prepareStatus }) {
  if (!activeSelection) return null;
  const preview = activeSelection.llmInputPreview;
  const explain = activeSelection.explainResult || {};
  const raw_payload = {
    llm_input_preview: preview,
    match_debug: activeSelection.matchDebug,
    prepare_status: prepareStatus,
  };
  return (
    <details className="pdf-chat-inspector" open={false}>
      <summary>Context used</summary>
      <dl className="pdf-chat-small-grid">
        <dt>selection</dt>
        <dd>{preview.highlight_type || "-"}</dd>
        <dt>page</dt>
        <dd>{preview.page_number || "-"}</dd>
        <dt>LLM mode</dt>
        <dd>{preview.recommended_llm_mode || "-"}</dd>
        <dt>provider</dt>
        <dd>{explain.provider || "-"}</dd>
        <dt>model</dt>
        <dd>{explain.model || "-"}</dd>
        <dt>retrieval</dt>
        <dd>{explain.retrieval_method || prepareStatus.retrieval_method || "-"}</dd>
      </dl>
      <h4>Matched block</h4>
      <pre>{formatJson(preview.matched_block)}</pre>
      <h4>Nearby context</h4>
      <pre>{formatJson(preview.nearby_useful_context)}</pre>
      <h4>Related blocks</h4>
      <pre>{formatJson(explain.retrieved_blocks || explain.global_rag_context || [])}</pre>
      {shouldShowCaptionFields(preview) ? (
        <>
          <h4>Caption candidates</h4>
          <pre>{formatJson(preview.candidate_captions || [])}</pre>
        </>
      ) : null}
      <h4>prompt_preview</h4>
      <pre>{explain.prompt_preview || "(not generated yet)"}</pre>
      <h4>raw_payload</h4>
      <pre>{formatJson(raw_payload)}</pre>
    </details>
  );
}

function MarkdownText({ content }) {
  const blocks = parseMarkdownBlocks(content);
  return (
    <div className="pdf-chat-markdown">
      {blocks.map((block, index) => {
        if (block.type === "ul") {
          return (
            <ul key={index}>
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}><MarkdownInline text={item} /></li>
              ))}
            </ul>
          );
        }
        if (block.type === "ol") {
          return (
            <ol key={index}>
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}><MarkdownInline text={item} /></li>
              ))}
            </ol>
          );
        }
        return <p key={index}><MarkdownInline text={block.text} /></p>;
      })}
    </div>
  );
}

function MarkdownInline({ text }) {
  return parseInlineMarkdown(text).map((part, index) => {
    if (part.type === "strong") return <strong key={index}>{part.text}</strong>;
    if (part.type === "em") return <em key={index}>{part.text}</em>;
    if (part.type === "code") return <code key={index}>{part.text}</code>;
    return <React.Fragment key={index}>{part.text}</React.Fragment>;
  });
}

function parseMarkdownBlocks(content) {
  const lines = String(content || "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let paragraphLines = [];
  let currentList = null;

  function flushParagraph() {
    if (!paragraphLines.length) return;
    blocks.push({ type: "paragraph", text: normalizeText(paragraphLines.join(" ")) });
    paragraphLines = [];
  }

  function flushList() {
    if (!currentList) return;
    blocks.push(currentList);
    currentList = null;
  }

  for (const line of lines) {
    const trimmed = line.trim();
    const unordered = line.match(/^\s*[-*+]\s+(.+)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.+)$/);

    if (!trimmed) {
      flushParagraph();
      flushList();
      continue;
    }

    if (unordered || ordered) {
      const type = unordered ? "ul" : "ol";
      const itemText = (unordered || ordered)[1].trim();
      flushParagraph();
      if (!currentList || currentList.type !== type) {
        flushList();
        currentList = { type, items: [] };
      }
      currentList.items.push(itemText);
      continue;
    }

    if (currentList && /^\s{2,}\S/.test(line) && currentList.items.length) {
      currentList.items[currentList.items.length - 1] = `${currentList.items[currentList.items.length - 1]} ${trimmed}`;
      continue;
    }

    flushList();
    paragraphLines.push(trimmed);
  }

  flushParagraph();
  flushList();
  return blocks.length ? blocks : [{ type: "paragraph", text: "" }];
}

function parseInlineMarkdown(text) {
  const source = String(text || "");
  const tokenPattern = /(`[^`\n]+`|\*\*[^*\n]+?\*\*|__[^_\n]+?__|\*[^*\s][^*\n]*?\*|_[^_\s][^_\n]*?_)/g;
  const parts = [];
  let cursor = 0;
  let match;

  while ((match = tokenPattern.exec(source)) !== null) {
    if (match.index > cursor) {
      parts.push({ type: "text", text: source.slice(cursor, match.index) });
    }
    const token = match[0];
    if (token.startsWith("`")) {
      parts.push({ type: "code", text: token.slice(1, -1) });
    } else if (token.startsWith("**") || token.startsWith("__")) {
      parts.push({ type: "strong", text: token.slice(2, -2) });
    } else {
      parts.push({ type: "em", text: token.slice(1, -1) });
    }
    cursor = match.index + token.length;
  }

  if (cursor < source.length) {
    parts.push({ type: "text", text: source.slice(cursor) });
  }
  return parts.length ? parts : [{ type: "text", text: "" }];
}

async function matchHighlightToBlocks(documentId, highlight, highlightDebug) {
  const viewportRects = viewportRectsFromPosition(highlight.position);
  const normalizedRects = normalizedRectsFromPosition(highlight.position);
  const parserRects1000 = parserRects1000FromPosition(highlight.position);
  const matchResult = await postJson("/api/document/match-blocks", {
    document_id: documentId,
    highlight_id: highlightDebug.id,
    page_number: highlightDebug.pageNumber,
    selected_text: highlightDebug.normalizedText,
    viewport_rects: viewportRects,
    normalized_rects: normalizedRects,
    parser_rects_1000: parserRects1000,
    position: highlight.position,
  });
  return {
    documentId,
    highlightId: highlightDebug.id,
    pageNumber: highlightDebug.pageNumber,
    selectedText: highlightDebug.normalizedText,
    viewportRects: matchResult.viewport_rects || viewportRects,
    normalizedRects: matchResult.normalized_rects || normalizedRects,
    parserRects1000: matchResult.parser_rects_1000 || parserRects1000,
    matchedBlocks: matchResult.matched_blocks || [],
    previousBlocks: matchResult.previous_blocks || [],
    nextBlocks: matchResult.next_blocks || [],
    selectedCaption: matchResult.selected_caption || null,
    captionConfidence: matchResult.caption_confidence || "none",
    candidateCaptions: matchResult.candidate_captions || [],
    rawResponse: matchResult,
    error: "",
  };
}

function selectionStateFrom(highlight, highlightDebug, matchDebug) {
  const llmInputPreview = buildLlmInputPreview(highlightDebug, matchDebug);
  return {
    highlight,
    highlightDebug,
    matchDebug,
    llmInputPreview,
    cleanedPromptPreview: buildCleanedPromptPreview(llmInputPreview),
    explainResult: null,
  };
}

function highlightMetadataForSave(selection) {
  if (!selection?.llmInputPreview) return {};
  const preview = selection.llmInputPreview;
  return {
    highlight_id: preview.highlight_id,
    document_id: preview.document_id,
    type: preview.highlight_type,
    highlight_type: preview.highlight_type,
    page_number: preview.page_number,
    selected_text: preview.selected_text,
    text_preview: preview.selected_text?.slice(0, 240) || "",
    caption: preview.caption || "",
    caption_confidence: preview.caption_confidence || "",
    crop_path: preview.crop_path || preview.crop_image_path || "",
    crop_image_path: preview.crop_image_path || preview.crop_path || "",
    crop_url: preview.crop_url || "",
    viewport_rects: preview.viewport_rects,
    normalized_rects: preview.normalized_rects,
    parser_rects_1000: preview.parser_rects_1000,
    matched_block: preview.matched_block,
    nearby_context: preview.nearby_useful_context,
  };
}

function emptySelectionDebug() {
  return {
    browserText: "",
    normalizedBrowserText: "",
    libraryText: "",
    normalizedLibraryText: "",
    selectionCropImage: "",
    selectionKeys: [],
    selectionType: "",
  };
}

function emptyMatchDebug(documentId, highlightId = "") {
  return {
    documentId,
    highlightId,
    pageNumber: "",
    selectedText: "",
    viewportRects: [],
    normalizedRects: [],
    parserRects1000: [],
    matchedBlocks: [],
    previousBlocks: [],
    nextBlocks: [],
    selectedCaption: null,
    captionConfidence: "none",
    candidateCaptions: [],
    rawResponse: null,
    error: "",
  };
}

function buildLlmInputPreview(lastHighlightDebug, matchDebug) {
  const matchedBlock = matchDebug.matchedBlocks?.[0] || null;
  const previousBlock = matchDebug.previousBlocks?.[0] || null;
  const nextBlock = matchDebug.nextBlocks?.[0] || null;
  const normalizedSelectedText = normalizePdfText(lastHighlightDebug.rawText || matchDebug.selectedText || "");
  const hasText = Boolean(normalizedSelectedText);
  const contextBlocks = usefulContextBlocks(matchDebug);
  const selectedCaption = selectCaptionForHighlight(lastHighlightDebug, matchDebug, matchedBlock);
  const showCaptionDebug = lastHighlightDebug.type === "area" || isCaptionBlock(matchedBlock);

  return {
    document_id: matchDebug.documentId,
    highlight_id: lastHighlightDebug.id || "",
    highlight_type: lastHighlightDebug.type,
    page_number: lastHighlightDebug.pageNumber,
    selected_text: normalizedSelectedText,
    text_available: hasText,
    reason: textUnavailableReason(lastHighlightDebug, hasText),
    viewport_rects: matchDebug.viewportRects,
    normalized_rects: matchDebug.normalizedRects,
    parser_rects_1000: matchDebug.parserRects1000,
    crop_image_available: Boolean(lastHighlightDebug.cropImage || lastHighlightDebug.cropImagePath || lastHighlightDebug.cropImageUrl),
    crop_image_path: lastHighlightDebug.cropImagePath || "",
    crop_path: lastHighlightDebug.cropImagePath || "",
    crop_url: lastHighlightDebug.cropImageUrl || "",
    crop_image_data_url: isDataImageUrl(lastHighlightDebug.cropImage) ? lastHighlightDebug.cropImage : "",
    crop_image_data_url_length: lastHighlightDebug.cropImage?.length || 0,
    matched_block_id: matchedBlock?.block_id || "",
    matched_block_type: matchedBlock?.block_type || "",
    coordinate_overlap: coordinateOverlap(matchedBlock),
    text_bonus: textBonus(matchedBlock),
    match_score: blockMatchScore(matchedBlock),
    caption: selectedCaption?.markdown_content || "",
    selected_caption: showCaptionDebug ? selectedCaption || null : null,
    caption_confidence: showCaptionDebug ? matchDebug.captionConfidence || (selectedCaption ? "medium" : "none") : "",
    candidate_captions: showCaptionDebug ? matchDebug.candidateCaptions || [] : [],
    matched_block: summarizeBlock(matchedBlock),
    previous_block: summarizeBlock(previousBlock),
    next_block: summarizeBlock(nextBlock),
    nearby_useful_context: contextBlocks.map(summarizeBlock),
    recommended_llm_mode: recommendLlmMode(lastHighlightDebug, matchDebug, matchedBlock),
    response_style: "chat_conversational",
  };
}

function buildCleanedPromptPreview(llmInputPreview) {
  const showCaptionFields = shouldShowCaptionFields(llmInputPreview);
  return {
    mode: llmInputPreview.recommended_llm_mode,
    page_number: llmInputPreview.page_number,
    crop_image_available: llmInputPreview.crop_image_available,
    caption: showCaptionFields ? llmInputPreview.caption || "" : "",
    selected_caption: showCaptionFields ? llmInputPreview.selected_caption?.markdown_content || "" : "",
    caption_confidence: showCaptionFields ? llmInputPreview.caption_confidence || "" : "",
    candidate_captions: showCaptionFields ? (llmInputPreview.candidate_captions || [])
      .map((caption) => caption?.markdown_content || "")
      .filter(Boolean) : [],
    nearby_useful_context: (llmInputPreview.nearby_useful_context || [])
      .filter((block) => !isLowValueContextBlock(block))
      .map((block) => block?.markdown_content || "")
      .filter(Boolean),
    selected_text: llmInputPreview.text_available ? normalizePdfText(llmInputPreview.selected_text) : "",
  };
}

function recommendLlmMode(lastHighlightDebug, matchDebug, matchedBlock) {
  const blockType = matchedBlock?.block_type || "";
  const matchScore = blockMatchScore(matchedBlock);
  const hasCrop = Boolean(lastHighlightDebug.cropImage);
  const hasText = Boolean(normalizePdfText(lastHighlightDebug.rawText || matchDebug.selectedText || ""));
  const contextBlocks = usefulContextBlocks(matchDebug);
  const captionBlock = selectCaptionForHighlight(lastHighlightDebug, matchDebug, matchedBlock)
    || (lastHighlightDebug.type === "area" ? findCaptionBlock(contextBlocks) : null);
  const hasContext = contextBlocks.length > 0;
  const weakMatch = !matchedBlock || matchScore < 0.08;

  if (hasCrop && captionBlock) return "image_plus_context";
  if (lastHighlightDebug.type === "text" && hasText && !weakMatch) return "text_context";
  if (blockType === "table" && !weakMatch) return "table_context";
  if (blockType === "formula" && !weakMatch) return "formula_context";
  if (["image", "caption"].includes(blockType) && hasCrop && !weakMatch) {
    return hasContext ? "image_plus_context" : "image_multimodal";
  }
  if (lastHighlightDebug.type === "area" && hasCrop && hasContext && !weakMatch) return "image_plus_context";
  if (hasCrop && weakMatch && hasContext) return "image_multimodal";
  if (hasCrop && weakMatch) return "fallback_image_only";
  if (hasCrop) return "image_multimodal";
  if (hasText) return "text_context";
  return "fallback_image_only";
}

function selectCaptionForHighlight(lastHighlightDebug, matchDebug, matchedBlock) {
  if (lastHighlightDebug.type === "area" && matchDebug.selectedCaption?.markdown_content) {
    return matchDebug.selectedCaption;
  }
  if (lastHighlightDebug.type !== "area" && isCaptionBlock(matchedBlock)) {
    return matchedBlock;
  }
  return null;
}

function shouldShowCaptionFields(llmInputPreview) {
  return llmInputPreview.highlight_type === "area" || llmInputPreview.matched_block_type === "caption";
}

function textUnavailableReason(lastHighlightDebug, hasText) {
  if (hasText) return "";
  if (lastHighlightDebug.type === "area") return "area_selection";
  return "empty_selection_text";
}

function usefulContextBlocks(matchDebug) {
  return uniqueBlocks([
    ...(matchDebug.matchedBlocks || []),
    ...(matchDebug.previousBlocks || []),
    ...(matchDebug.nextBlocks || []),
  ]).filter((block) => !isLowValueContextBlock(block));
}

function uniqueBlocks(blocks) {
  const seen = new Set();
  const unique = [];
  for (const block of blocks) {
    const key = block?.block_id || `${block?.page_number || ""}:${block?.markdown_content || ""}`;
    if (!key || seen.has(key)) continue;
    seen.add(key);
    unique.push(block);
  }
  return unique;
}

function findCaptionBlock(blocks) {
  return (blocks || []).find((block) => isCaptionBlock(block)) || null;
}

function isCaptionBlock(block) {
  const blockType = String(block?.block_type || "").toLowerCase();
  const text = normalizePdfText(block?.markdown_content || "");
  return blockType === "caption" || /^(fig\.?|figure|table)\s*\d+/i.test(text);
}

function isLowValueContextBlock(block) {
  const blockType = String(block?.block_type || "").toLowerCase();
  const text = normalizePdfText(block?.markdown_content || "");
  if (!text) return true;
  if (["caption", "formula", "table"].includes(blockType)) return false;
  if (/^(fig\.?|figure|table|equation|eq\.)\s*\d+/i.test(text)) return false;
  if (/^(-\s*)?\d+(\s*\/\s*\d+|\s+of\s+\d+)?(\s*-)?$/i.test(text)) return true;
  if (/^[A-Z][A-Za-z-]+,?\s+et al\.?$/i.test(text)) return true;
  if (/^[A-Z][A-Za-z0-9& .-]*\s*['\u2019]\d{2},\s+[A-Z][a-z]+\s+\d{1,2}[-\u2013]\d{1,2},\s+\d{4},/.test(text)) return true;
  if (/^(research article|original article|article)$/i.test(text)) return true;
  if (/^[A-Za-z ]+\s+\d+\(\d+\)$/i.test(text)) return true;
  if (/^(doi:|https?:|www\.)/i.test(text)) return true;
  if (text.length < 28 && text.split(/\s+/).length <= 3) return true;
  if (text.length < 120 && /,\s+et al\.|running head/i.test(text)) return true;
  return false;
}

function summarizeBlock(block) {
  if (!block) return null;
  return {
    block_id: block.block_id || "",
    page_number: block.page_number || "",
    block_type: block.block_type || "",
    coordinate_overlap: coordinateOverlap(block),
    text_bonus: textBonus(block),
    match_score: blockMatchScore(block),
    markdown_content: block.markdown_content || "",
  };
}

function blockMatchScore(block) {
  return clampedScore(block?.match_score ?? block?.overlap_score ?? block?.coordinate_overlap_score);
}

function coordinateOverlap(block) {
  return clampedScore(block?.coordinate_overlap ?? block?.coordinate_overlap_score);
}

function textBonus(block) {
  return clampedScore(block?.text_bonus ?? block?.selected_text_similarity);
}

function clampedScore(value) {
  const score = Number(value ?? 0);
  if (!Number.isFinite(score)) return 0;
  return Math.max(0, Math.min(score, 1));
}

function viewportRectsFromPosition(position) {
  const rects = Array.isArray(position?.rects) && position.rects.length > 0
    ? position.rects
    : position?.boundingRect
      ? [position.boundingRect]
      : [];
  return rects.map((rect) => ({
    x1: Number(rect.x1 ?? rect.left ?? 0),
    y1: Number(rect.y1 ?? rect.top ?? 0),
    x2: Number(rect.x2 ?? ((rect.left ?? 0) + (rect.width ?? 0))),
    y2: Number(rect.y2 ?? ((rect.top ?? 0) + (rect.height ?? 0))),
    width: Number(rect.width ?? 0),
    height: Number(rect.height ?? 0),
    pageNumber: Number(rect.pageNumber || position?.boundingRect?.pageNumber || 1),
  }));
}

function normalizedRectsFromPosition(position) {
  return viewportRectsFromPosition(position)
    .map(normalizedRectFromViewportRect)
    .filter(Boolean);
}

function parserRects1000FromPosition(position) {
  return normalizedRectsFromPosition(position).map((rect) => ({
    x1: roundNumber(rect.x1 * 1000, 3),
    y1: roundNumber(rect.y1 * 1000, 3),
    x2: roundNumber(rect.x2 * 1000, 3),
    y2: roundNumber(rect.y2 * 1000, 3),
    pageNumber: rect.pageNumber,
  }));
}

function normalizedRectFromViewportRect(rect) {
  const x1 = Number(rect.x1 || 0);
  const y1 = Number(rect.y1 || 0);
  const x2 = Number(rect.x2 || 0);
  const y2 = Number(rect.y2 || 0);
  const width = Number(rect.width || 0);
  const height = Number(rect.height || 0);
  const pageNumber = Number(rect.pageNumber || 1);

  if (Math.max(x2, y2) <= 1) return clampedNormalizedRect(x1, y1, x2, y2, pageNumber);
  if (width > 1 && height > 1 && x2 <= width * 1.1 && y2 <= height * 1.1) {
    return clampedNormalizedRect(x1 / width, y1 / height, x2 / width, y2 / height, pageNumber);
  }
  if (Math.max(x2, y2) <= 1000) return clampedNormalizedRect(x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000, pageNumber);
  return null;
}

function clampedNormalizedRect(x1, y1, x2, y2, pageNumber) {
  const left = Math.max(0, Math.min(Number(x1), Number(x2)));
  const top = Math.max(0, Math.min(Number(y1), Number(y2)));
  const right = Math.min(1, Math.max(Number(x1), Number(x2)));
  const bottom = Math.min(1, Math.max(Number(y1), Number(y2)));
  if (right <= left || bottom <= top) return null;
  return {
    x1: roundNumber(left, 6),
    y1: roundNumber(top, 6),
    x2: roundNumber(right, 6),
    y2: roundNumber(bottom, 6),
    pageNumber,
  };
}

function normalizeText(text) {
  return (text || "").replace(/\s+/g, " ").trim();
}

function normalizePdfText(text) {
  return normalizeText(text)
    .replace(/\uFB00/g, "ff")
    .replace(/\uFB01/g, "fi")
    .replace(/\uFB02/g, "fl")
    .replace(/\uFB03/g, "ffi")
    .replace(/\uFB04/g, "ffl")
    .replace(/([A-Za-z0-9])-\s+([A-Za-z0-9])/g, "$1$2")
    .replace(/\s+/g, " ")
    .trim();
}

function pageNumberFromPosition(position) {
  return Number(position?.boundingRect?.pageNumber || position?.rects?.[0]?.pageNumber || 1);
}

function isSuspiciousText(text) {
  const normalizedText = normalizeText(text);
  if (!normalizedText) return true;
  if (normalizedText.includes("\uFFFD")) return true;
  const visibleCharacters = [...normalizedText].filter((char) => char.trim()).length;
  const alphanumericCharacters = [...normalizedText].filter((char) => /[A-Za-z0-9]/.test(char)).length;
  return visibleCharacters > 0 && alphanumericCharacters / visibleCharacters < 0.25;
}

function isDataImageUrl(value) {
  return /^data:image\/(png|jpe?g);base64,/i.test(String(value || ""));
}

function cropUrlForHighlight(highlight) {
  const documentId = highlight?.document_id || "";
  const highlightId = highlight?.highlight_id || highlight?.id || "";
  if (!documentId || !highlightId || !highlight?.crop_image_path && !highlight?.crop_path) return "";
  return `/api/documents/${documentId}/highlights/${highlightId}/crop`;
}

function normalizePersistedHighlight(highlight, documentId, source = "restored") {
  const highlightId = highlight?.highlight_id || highlight?.id || "";
  if (!highlightId) return null;
  const content = highlight.content || {};
  const type = highlight.type || highlight.highlight_type || "text";
  const pageNumber = highlight.page_number || pageNumberFromPosition(highlight.position);
  const normalized = {
    ...highlight,
    id: highlightId,
    highlight_id: highlightId,
    document_id: highlight.document_id || documentId || "",
    type,
    highlight_type: type,
    page_number: pageNumber,
    source,
  };
  const cropUrl = cropUrlForHighlight(normalized);
  return {
    ...normalized,
    crop_url: normalized.crop_url || cropUrl,
    content: {
      ...content,
      text: content.text || highlight.selected_text || "",
      image: content.image || highlight.crop_image_data_url || highlight.crop_url || cropUrl || "",
    },
  };
}

function toViewerHighlight(highlight) {
  return normalizePersistedHighlight(highlight, highlight?.document_id || "", highlight?.source || "restored");
}

function highlightDebugFromSaved(highlight) {
  const cropImage = highlight.crop_image_data_url || highlight.content?.image || "";
  return {
    id: highlight.highlight_id || highlight.id || "",
    type: highlight.type || highlight.highlight_type || "text",
    rawText: highlight.selected_text || highlight.content?.text || "",
    normalizedText: normalizePdfText(highlight.selected_text || highlight.content?.text || ""),
    textLength: String(highlight.selected_text || highlight.content?.text || "").length,
    cropImage,
    cropImagePath: highlight.crop_image_path || highlight.crop_path || "",
    cropImageUrl: highlight.crop_url || cropUrlForHighlight(highlight),
    empty: !highlight.selected_text && !highlight.content?.text,
    suspicious: false,
    pageNumber: highlight.page_number || pageNumberFromPosition(highlight.position),
    position: highlight.position || null,
    boundingRect: highlight.position?.boundingRect || null,
    rects: highlight.position?.rects || [],
  };
}

function matchDebugFromSaved(documentId, highlight) {
  return {
    documentId,
    highlightId: highlight.highlight_id || highlight.id || "",
    pageNumber: highlight.page_number || pageNumberFromPosition(highlight.position),
    selectedText: highlight.selected_text || "",
    viewportRects: highlight.viewport_rects || viewportRectsFromPosition(highlight.position),
    normalizedRects: highlight.normalized_rects || normalizedRectsFromPosition(highlight.position),
    parserRects1000: highlight.parser_rects_1000 || parserRects1000FromPosition(highlight.position),
    matchedBlocks: highlight.matched_block ? [highlight.matched_block] : [],
    previousBlocks: [],
    nextBlocks: Array.isArray(highlight.nearby_context) ? highlight.nearby_context : [],
    selectedCaption: highlight.caption ? { markdown_content: highlight.caption } : null,
    captionConfidence: highlight.caption_confidence || "none",
    candidateCaptions: highlight.candidate_captions || [],
    rawResponse: null,
    error: "",
  };
}

function emptyThreadForSelection(documentId, highlightId, selection) {
  return {
    document_id: documentId || "",
    highlight_id: highlightId || "",
    selection_snapshot: selection?.llmInputPreview || {},
    messages: [],
    strategy_candidates: [],
    selected_strategy_id: "",
    selected_strategy: null,
    trigger_context: null,
    reaction_window_summary: null,
    turn_metadata: {},
  };
}

function normalizeExplainSelectionResponse(response) {
  const raw = response && typeof response === "object" ? response : {};
  const thread = raw.thread && typeof raw.thread === "object" ? raw.thread : null;
  const assistantMessage = latestAssistantMessage(thread);
  const answerText = firstNonEmptyText([
    raw.answer,
    typeof raw.assistant_message === "string" ? raw.assistant_message : raw.assistant_message?.content,
    typeof raw.message === "string" ? raw.message : raw.message?.content,
    assistantMessage?.content,
    raw.result?.answer,
    raw.explanation,
    raw.content,
    raw.result?.explanation,
  ]);
  const errorMessage = raw.ok === false || raw.error ? String(raw.error || "Explanation failed.") : "";
  return {
    ...raw,
    thread,
    assistantMessage,
    answerText,
    answer: answerText,
    errorMessage,
  };
}

function firstNonEmptyText(values) {
  for (const value of values) {
    const text = typeof value === "string" ? value : value == null ? "" : String(value);
    if (normalizePdfText(text)) return text;
  }
  return "";
}

function logExplainSelectionDebug(endpoint, payload, response, normalizedResult, nextThread) {
  if (typeof console === "undefined" || typeof console.debug !== "function") return;
  const latestAssistant = latestAssistantMessage(nextThread);
  console.debug("[pdf-chat] explain-selection", {
    endpoint_called: endpoint,
    payload_keys: Object.keys(payload || {}),
    response_keys: response && typeof response === "object" ? Object.keys(response) : [],
    extracted_answer_length: normalizePdfText(normalizedResult?.answerText || "").length,
    created_turn_id: latestAssistant?.turn_id || "",
    message_appended: Boolean(normalizePdfText(latestAssistant?.content || "")),
  });
}

function appendAssistantMessage(thread, result) {
  const now = Date.now() / 1000;
  const assistantMessage = result.assistantMessage || {};
  return {
    ...(thread || {}),
    messages: [
      ...((thread && thread.messages) || []),
      {
        role: "assistant",
        content: result.answerText || result.answer || "",
        created_at: now,
        provider: result.provider,
        model: result.model,
        turn_id: assistantMessage.turn_id || result.turn_id || `turn_${crypto?.randomUUID?.()?.replaceAll("-", "") || Date.now()}`,
        turn_type: result.selected_strategy ? "strategy_reexplanation" : "baseline_explanation",
        prompt_snapshot_id: assistantMessage.prompt_snapshot_id || result.prompt_snapshot_id || "",
        snapshot_stage: result.stage || result.snapshot_stage || "",
        strategy_id: result.selected_strategy_id || result.strategy_id || "",
        strategy_title: result.selected_strategy?.title || result.strategy_title || "",
      },
    ],
  };
}

function roundNumber(value, precision) {
  const factor = 10 ** precision;
  return Math.round(Number(value) * factor) / factor;
}

function makeId() {
  return crypto?.randomUUID?.() || `highlight-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function pageCountLabel(value) {
  const count = Number(value || 0);
  if (!Number.isFinite(count) || count <= 0) return "Unknown pages";
  return `${count} ${count === 1 ? "page" : "pages"}`;
}

function countLabel(value, singular) {
  const count = Number(value || 0);
  return `${count} ${count === 1 ? singular : `${singular}s`}`;
}

function preparationStatusLabel(status) {
  const normalized = String(status || "").toLowerCase();
  if (normalized === "completed" || normalized === "ready") return "Ready";
  if (normalized === "preparing") return "Preparing";
  if (normalized === "failed") return "Needs attention";
  return "Unknown status";
}

function isPreparationComplete(status, meta, prepareStatus) {
  const normalized = String(prepareStatus?.status || meta?.prepare_status || status || "").toLowerCase();
  const progress = Number(prepareStatus?.progress_percent ?? 0);
  return normalized === "completed" || normalized === "ready" || progress >= 100;
}

function retrievalSummary(method) {
  const normalized = String(method || "").trim().toLowerCase();
  if (!normalized || normalized === "unknown") return "retrieval ready";
  return `${normalized.replace(/[_-]+/g, " ")} retrieval`;
}

function selectionKindLabel(preview) {
  return preview?.highlight_type === "area" ? "Area selection" : "Text selection";
}

function truncateText(text, limit) {
  const normalized = normalizePdfText(text);
  if (normalized.length <= limit) return normalized;
  return `${normalized.slice(0, Math.max(0, limit - 1)).trim()}…`;
}

function formatTime(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) return "";
  return new Date(number * 1000).toLocaleString();
}

function formatPercent(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) return "0%";
  return `${Math.round(number * 100)}%`;
}

function formatDuration(value) {
  const seconds = Math.max(0, Math.round(Number(value || 0)));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function sourceLabelForLearningSignal(source, learningState) {
  if (source === "webcam" || learningState?.source === "webcam_model") return "Source: live webcam model";
  return "Source: simulated camera stream";
}

function isDomOrReactEvent(value) {
  if (!value || typeof value !== "object") return false;
  if (typeof Event !== "undefined" && value instanceof Event) return true;
  if (typeof HTMLElement !== "undefined" && value instanceof HTMLElement) return true;
  if (typeof value.preventDefault === "function" && typeof value.stopPropagation === "function") return true;
  if ("nativeEvent" in value && ("target" in value || "currentTarget" in value)) return true;
  const target = value.target || value.currentTarget;
  return Boolean(typeof HTMLElement !== "undefined" && target instanceof HTMLElement);
}

function isPlainStrategyObject(value) {
  if (!value || typeof value !== "object" || Array.isArray(value) || isDomOrReactEvent(value)) return false;
  return Boolean(value.strategy_id || value.title || value.prompt_instruction);
}

function sanitizeSerializablePayload(value, seen = new WeakSet()) {
  if (value === null || typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return value;
  }
  if (value === undefined || typeof value === "function" || typeof value === "symbol" || isDomOrReactEvent(value)) {
    return undefined;
  }
  if (value instanceof Date) return value.toISOString();
  if (typeof File !== "undefined" && value instanceof File) return undefined;
  if (typeof Blob !== "undefined" && value instanceof Blob) return undefined;
  if (typeof MediaStream !== "undefined" && value instanceof MediaStream) return undefined;
  if (Array.isArray(value)) {
    return value
      .map((item) => sanitizeSerializablePayload(item, seen))
      .filter((item) => item !== undefined);
  }
  if (typeof value === "object") {
    if (seen.has(value)) return undefined;
    seen.add(value);
    const prototype = Object.getPrototypeOf(value);
    if (prototype && prototype !== Object.prototype) return undefined;
    const clean = {};
    for (const [key, item] of Object.entries(value)) {
      const sanitized = sanitizeSerializablePayload(item, seen);
      if (sanitized !== undefined) clean[key] = sanitized;
    }
    return clean;
  }
  return undefined;
}

function assertJsonSerializablePayload(payload) {
  try {
    JSON.stringify(payload);
  } catch (err) {
    throw new Error("Could not prepare the explanation request. Please try again.");
  }
}

function userFacingErrorMessage(err) {
  const message = err?.message || String(err);
  if (/circular structure|reactfiber|html/i.test(message)) {
    return "Could not prepare the explanation request. Please try again.";
  }
  return message;
}

function isReactionWindowValidationError(err) {
  return /baseline_explanation,\s*source_turn_id,\s*and\s*reaction_window_summary/i.test(err?.message || String(err));
}

function faceDetectionLabelFor(faceDetection) {
  const mode = String(faceDetection?.actual_detector || faceDetection?.detector || faceDetection?.mode || faceDetection?.fallback || "").trim();
  if (!mode) return "center crop fallback";
  if (mode === "openface") return "openface";
  if (mode === "opencv_haar") return "OpenCV Haar";
  if (mode === "center_crop") return "center crop fallback";
  if (mode === "yolo") return "YOLO";
  return mode.replace(/_/g, " ");
}

function modelModeLabelForLearningSignal(modelOutputType) {
  const mode = String(modelOutputType || "").trim();
  if (mode === "raw_emotion") return "raw emotion model";
  if (mode === "academic_state" || mode === "academic_state_model") return "academic-state model";
  return mode ? mode.replace(/_/g, " ") : "unknown";
}

function rawSignalLabelForLearningSignal(learningState, modelStatus) {
  const statusAvailable = Boolean(modelStatus?.emotion_pipeline_status?.raw_detection_available || modelStatus?.raw_emotion_available);
  const rawAvailable = Boolean(learningState?.raw_facial_emotion_available || statusAvailable);
  if (rawAvailable) return "Learning-support signal available";
  return "Learning-support signal unavailable for this checkpoint";
}

function learningSupportCueLabel(learningState) {
  const packageCue = learningState?.learning_signal_package?.support_cue || learningState?.learning_signal_package?.reaction_window_summary?.support_cue;
  if (packageCue) return supportCueDisplayLabel(packageCue);
  const state = String(learningState?.academic_state || "").toLowerCase();
  const labels = {
    boredom: "possible re-engagement cue",
    confusion: "clarification cue",
    engagement: "deepening cue",
    frustration: "continued difficulty cue",
  };
  return labels[state] || "warming up";
}

function learningStateDisplayLabel(state) {
  const academicState = String(state || "").toLowerCase();
  return ["boredom", "confusion", "engagement", "frustration"].includes(academicState) ? academicState : state;
}

function supportCueDisplayLabel(supportCue) {
  const labels = {
    clarification: "clarification cue",
    difficulty_support: "difficulty-support cue",
    sustained_clarification: "clarification cue",
    reduce_load: "reduce-load cue",
    re_engagement: "re-engagement cue",
    deepening: "deepening cue",
    clarify_and_reengage: "clarify and re-engage cue",
    gentle_clarification: "gentle clarification cue",
    uncertain: "mixed signal",
    neutral_or_uncertain: "mixed signal",
    neutral: "neutral cue",
  };
  return labels[supportCue] || "learning-support cue";
}

function reactionStrategyKey(activeSelection, reactionSummary) {
  const highlightId = activeSelection?.llmInputPreview?.highlight_id || "";
  const sourceTurnId = reactionSummary?.source_turn_id || "turn";
  const cue = reactionSummary?.support_cue || "neutral";
  return `${highlightId}:${sourceTurnId}:${cue}`;
}

function summarizeReactionWindowSamples(
  samples,
  sourceTurnId,
  highlightId,
  windowStart,
  windowEnd,
  sourceTurnType = "baseline_explanation",
  endReason = "near_bottom_reached",
  readProgressEstimate = 0.9,
) {
  const usableSamples = (samples || []).filter(Boolean);
  const states = ["boredom", "confusion", "engagement", "frustration"];
  const totals = Object.fromEntries(states.map((state) => [state, 0]));
  let confidenceTotal = 0;
  let maxConfidence = 0;
  const trends = [];
  const detectorModes = [];
  const cropStrategies = [];
  const modelOutputTypes = [];
  let rawDetectionAvailable = false;
  for (const sample of usableSamples) {
    const distribution = sample.distribution || sample.state_distribution || {};
    for (const state of states) {
      totals[state] += Number(distribution[state] || (sample.academic_state === state ? sample.confidence || 0 : 0));
    }
    const confidence = Number(sample.confidence || 0);
    confidenceTotal += confidence;
    maxConfidence = Math.max(maxConfidence, confidence);
    if (sample.trend) trends.push(String(sample.trend));
    if (sample.model_output_type) modelOutputTypes.push(String(sample.model_output_type));
    rawDetectionAvailable = rawDetectionAvailable || Boolean(sample.raw_facial_emotion_available);
    const faceDetection = sample.face_detection || {};
    if (faceDetection.actual_detector || faceDetection.detector || faceDetection.mode) detectorModes.push(faceDetection.actual_detector || faceDetection.detector || faceDetection.mode);
    if (faceDetection.crop_strategy) cropStrategies.push(faceDetection.crop_strategy);
  }
  const divisor = Math.max(1, usableSamples.length);
  const avgDistribution = Object.fromEntries(states.map((state) => [state, roundNumber(totals[state] / divisor, 3)]));
  const orderedStates = [...states].sort((a, b) => avgDistribution[b] - avgDistribution[a]);
  const dominantState = orderedStates[0] || "engagement";
  const secondaryState = orderedStates[1] || "";
  const avgConfidence = roundNumber(confidenceTotal / divisor, 3);
  const phases = splitReactionPhaseSamples(usableSamples);
  const earlyDistribution = aggregateReactionDistribution(phases.early, states);
  const middleDistribution = aggregateReactionDistribution(phases.middle, states);
  const lateDistribution = aggregateReactionDistribution(phases.late, states);
  const earlyDominantState = dominantStateFromDistribution(earlyDistribution, states);
  const middleDominantState = dominantStateFromDistribution(middleDistribution, states);
  const lateDominantState = dominantStateFromDistribution(lateDistribution, states);
  let transition = reactionTransitionSummary(earlyDominantState, middleDominantState, lateDominantState, earlyDistribution, lateDistribution, usableSamples.length);
  const trend = transition.trend || (trends.includes("rising") ? "rising" : trends.includes("stable") ? "stable" : trends[0] || "stable");
  if (isResolvingDifficultyPattern(avgDistribution, dominantState, secondaryState, trend)) {
    transition = {
      ...transition,
      state_transition: "resolving_difficulty",
      transition_label: "resolving_difficulty",
      stability: "transitioning",
    };
  }
  const stability = transition.stability || (Math.abs((avgDistribution[dominantState] || 0) - (avgDistribution[secondaryState] || 0)) < 0.12 ? "mixed" : "stable");
  const supportCue = supportCueForReadingTransition(
    transition.state_transition,
    dominantState,
    secondaryState,
    avgDistribution,
    avgConfidence,
    endReason,
    usableSamples.length,
  );
  const modeCounts = detectorModes.reduce((counts, mode) => ({ ...counts, [mode]: (counts[mode] || 0) + 1 }), {});
  const faceMode = Object.entries(modeCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "center_crop";
  const outputCounts = modelOutputTypes.reduce((counts, mode) => ({ ...counts, [mode]: (counts[mode] || 0) + 1 }), {});
  const modelOutputType = Object.entries(outputCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "academic_state";
  const cropStrategyCounts = cropStrategies.reduce((counts, strategy) => ({ ...counts, [strategy]: (counts[strategy] || 0) + 1 }), {});
  const cropStrategy = Object.entries(cropStrategyCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
  const windowDuration = Math.max(0, (Date.parse(windowEnd) - Date.parse(windowStart)) / 1000);
  const reason = buildStrategyReasonText({
    source_turn_type: sourceTurnType,
    academic_state_scores: avgDistribution,
    dominant_academic_state: dominantState,
    secondary_academic_state: secondaryState,
    support_cue: supportCue.support_cue,
    trend,
    avg_confidence: avgConfidence,
    recommended_strategy: recommendedStrategyForSupportCue(supportCue.support_cue),
    pedagogical_move: supportCue.support_cue === "deepening" ? "deeper explanation strategy" : "step-by-step strategy",
  });
  const phaseEvidence = transition.transition_label && transition.state_transition !== "mixed_or_uncertain"
    ? `${transition.transition_label} · ${reason.academic_state_evidence_text}`
    : reason.academic_state_evidence_text;
  return {
    source_turn_id: sourceTurnId || "",
    source_turn_type: sourceTurnType || "previous_explanation",
    highlight_id: highlightId || "",
    start_time: windowStart,
    end_time: windowEnd,
    window_start: windowStart,
    window_end: windowEnd,
    end_reason: endReason || "near_bottom_reached",
    read_progress_estimate: roundNumber(finiteScore(readProgressEstimate, 0), 2),
    duration_sec: roundNumber(windowDuration || (usableSamples.length * REACTION_WINDOW_SAMPLE_MS / 1000), 1),
    sample_count: usableSamples.length,
    evidence_count: usableSamples.length,
    early_distribution: earlyDistribution,
    middle_distribution: middleDistribution,
    late_distribution: lateDistribution,
    early_dominant_state: earlyDominantState,
    middle_dominant_state: middleDominantState,
    late_dominant_state: lateDominantState,
    state_transition: transition.state_transition,
    transition_label: transition.transition_label,
    dominant_state: dominantState,
    secondary_state: secondaryState,
    avg_confidence: avgConfidence,
    max_confidence: roundNumber(maxConfidence, 3),
    avg_distribution: avgDistribution,
    trend,
    stability,
    support_cue: supportCue.support_cue,
    support_cue_label: supportCue.support_cue_label,
    support_cue_display_label: reason.support_cue_label,
    academic_state_evidence_text: phaseEvidence,
    confidence_handling: reason.confidence_handling,
    reason_text: reason.reason_text,
    trigger_reason: reason.reason_text,
    reliability_notes: [
      "reaction window summary is used as learning-support evidence, not diagnosis",
      ...(usableSamples.length < REACTION_WINDOW_MIN_SAMPLES ? ["low sample count; avoid strong process inference"] : []),
    ],
    model_output_type: modelOutputType,
    raw_detection_available: rawDetectionAvailable,
    face_detection_summary: {
      mode: faceMode,
      fallback_used: faceMode === "center_crop",
      crop_strategy: cropStrategy,
    },
  };
}

function splitReactionPhaseSamples(samples) {
  const usable = (samples || []).filter(Boolean);
  if (!usable.length) return { early: [], middle: [], late: [] };
  if (usable.length === 1) return { early: usable, middle: usable, late: usable };
  if (usable.length === 2) return { early: [usable[0]], middle: usable, late: [usable[1]] };
  const third = Math.max(1, Math.ceil(usable.length / 3));
  const early = usable.slice(0, third);
  const middle = usable.slice(third, Math.max(third + 1, usable.length - third));
  const late = usable.slice(Math.max(third, usable.length - third));
  return {
    early: early.length ? early : usable,
    middle: middle.length ? middle : usable,
    late: late.length ? late : usable,
  };
}

function aggregateReactionDistribution(samples, states) {
  const usable = (samples || []).filter(Boolean);
  const totals = Object.fromEntries(states.map((state) => [state, 0]));
  if (!usable.length) return Object.fromEntries(states.map((state) => [state, 0]));
  for (const sample of usable) {
    const distribution = sample.distribution || sample.state_distribution || {};
    for (const state of states) {
      totals[state] += Number(distribution[state] || (sample.academic_state === state ? sample.confidence || 0 : 0));
    }
  }
  return Object.fromEntries(states.map((state) => [state, roundNumber(totals[state] / usable.length, 3)]));
}

function dominantStateFromDistribution(distribution, states) {
  const ordered = [...states].sort((a, b) => Number(distribution?.[b] || 0) - Number(distribution?.[a] || 0));
  return ordered[0] || "uncertain";
}

function reactionTransitionSummary(earlyState, middleState, lateState, earlyDistribution, lateDistribution, sampleCount) {
  if (!sampleCount) {
    return {
      state_transition: "mixed_or_uncertain",
      transition_label: "mixed or uncertain",
      trend: "uncertain",
      stability: "low_evidence",
    };
  }
  const earlyDifficulty = ["confusion", "frustration"].includes(earlyState);
  const lateDifficulty = ["confusion", "frustration"].includes(lateState);
  const earlyScore = Number(earlyDistribution?.[earlyState] || 0);
  const lateScore = Number(lateDistribution?.[lateState] || 0);
  const scoreDelta = lateScore - earlyScore;
  let stateTransition = "mixed_or_uncertain";
  if (earlyDifficulty && lateState === "engagement") {
    stateTransition = `${earlyState}_to_engagement`;
  } else if (earlyState === "engagement" && lateDifficulty) {
    stateTransition = `engagement_to_${lateState}`;
  } else if (lateState === "confusion" && (earlyState === "confusion" || middleState === "confusion")) {
    stateTransition = "persistent_confusion";
  } else if (lateState === "frustration" && (earlyState === "frustration" || middleState === "frustration")) {
    stateTransition = "persistent_frustration";
  } else if (earlyState === "engagement" && middleState === "engagement" && lateState === "engagement") {
    stateTransition = "stable_engagement";
  } else if (earlyState === "boredom" && middleState === "boredom" && lateState === "boredom") {
    stateTransition = "persistent_boredom";
  }
  const labels = {
    confusion_to_engagement: "early confusion -> late engagement",
    frustration_to_engagement: "early frustration -> late engagement",
    engagement_to_confusion: "early engagement -> late confusion",
    engagement_to_frustration: "early engagement -> late frustration",
    persistent_confusion: "persistent confusion",
    persistent_frustration: "persistent frustration",
    persistent_boredom: "persistent low engagement",
    stable_engagement: "stable engagement",
    mixed_or_uncertain: "mixed or uncertain",
  };
  return {
    state_transition: stateTransition,
    transition_label: labels[stateTransition] || "mixed or uncertain",
    trend: Math.abs(scoreDelta) < 0.08 ? "stable" : scoreDelta > 0 ? "rising" : "falling",
    stability: stateTransition.startsWith("persistent") || stateTransition === "stable_engagement" ? "stable" : "transitioning",
  };
}

function supportCueForReadingTransition(stateTransition, dominantState, secondaryState, distribution, avgConfidence, endReason, sampleCount) {
  if (endReason === "max_timeout" || sampleCount < 2) {
    return { support_cue: "neutral_or_uncertain", support_cue_label: "Possible ways to continue" };
  }
  const baseCue = supportCueForReaction(dominantState, secondaryState, distribution, avgConfidence);
  if (stateTransition === "resolving_difficulty") {
    return { support_cue: "deepening", support_cue_label: "Cautious deepening cue" };
  }
  if (["confusion_to_engagement", "frustration_to_engagement", "stable_engagement"].includes(stateTransition)) {
    return { support_cue: "deepening", support_cue_label: "Deepening cue" };
  }
  if (["clarify_and_reengage", "gentle_clarification", "neutral_or_uncertain"].includes(baseCue.support_cue)) {
    return baseCue;
  }
  if (stateTransition === "persistent_confusion" || stateTransition === "engagement_to_confusion") {
    return { support_cue: "sustained_clarification", support_cue_label: "Sustained clarification cue" };
  }
  if (stateTransition === "persistent_frustration" || stateTransition === "engagement_to_frustration") {
    return { support_cue: "difficulty_support", support_cue_label: "Reduce cognitive load cue" };
  }
  if (stateTransition === "persistent_boredom") {
    return { support_cue: "re_engagement", support_cue_label: "Re-engagement cue" };
  }
  return baseCue;
}

function supportCueForReaction(dominantState, secondaryState, distribution, avgConfidence) {
  const states = ["boredom", "confusion", "engagement", "frustration"];
  const values = Object.fromEntries(states.map((state) => [state, Number(distribution?.[state] || 0)]));
  const ordered = [...states].sort((a, b) => values[b] - values[a]);
  const top = values[ordered[0]] || 0;
  const second = values[ordered[1]] || 0;
  const spread = Math.max(...states.map((state) => values[state])) - Math.min(...states.map((state) => values[state]));
  if (values.confusion >= 0.35 && values.boredom >= 0.25) {
    return { support_cue: "clarify_and_reengage", support_cue_label: "Clarify and re-engage cue" };
  }
  if (values.confusion >= 0.35 && values.frustration >= 0.25) {
    return { support_cue: "gentle_clarification", support_cue_label: "Gentle clarification cue" };
  }
  if (top >= 0.55) {
    if (ordered[0] === "engagement") return { support_cue: "deepening", support_cue_label: "Deepening cue" };
    if (ordered[0] === "boredom") return { support_cue: "re_engagement", support_cue_label: "Re-engagement cue" };
    if (ordered[0] === "frustration") return { support_cue: "difficulty_support", support_cue_label: "Reduce cognitive load cue" };
    if (ordered[0] === "confusion") return { support_cue: "sustained_clarification", support_cue_label: "Sustained clarification cue" };
  }
  if (top < 0.45 || top - second < 0.05 || spread < 0.18) {
    return { support_cue: "neutral_or_uncertain", support_cue_label: "Possible ways to continue" };
  }
  if (values.engagement >= 0.65 || dominantState === "engagement") return { support_cue: "deepening", support_cue_label: "Deepening cue" };
  if (values.boredom >= 0.50 || dominantState === "boredom") return { support_cue: "re_engagement", support_cue_label: "Re-engagement cue" };
  if (values.frustration >= 0.45 || dominantState === "frustration") return { support_cue: "difficulty_support", support_cue_label: "Reduce cognitive load cue" };
  if (values.confusion >= 0.45 || dominantState === "confusion") return { support_cue: "sustained_clarification", support_cue_label: "Sustained clarification cue" };
  return { support_cue: "neutral_or_uncertain", support_cue_label: "Possible ways to continue" };
}

function isResolvingDifficultyPattern(scores, dominantState, secondaryState, trend) {
  if (dominantState !== "frustration" || secondaryState !== "engagement" || trend !== "falling") return false;
  const frustration = finiteScore(scores?.frustration, 0);
  const engagement = finiteScore(scores?.engagement, 0);
  return frustration > 0 && engagement > 0 && frustration - engagement >= 0 && frustration - engagement <= 0.15;
}

function buildStrategyReasonText({
  source_turn_type,
  academic_state_scores,
  dominant_academic_state,
  secondary_academic_state,
  support_cue,
  trend,
  avg_confidence,
  recommended_strategy,
  pedagogical_move,
}) {
  const states = ["boredom", "confusion", "engagement", "frustration"];
  const scores = Object.fromEntries(states.map((state) => [state, finiteScore(academic_state_scores?.[state], 0)]));
  const ordered = [...states].sort((a, b) => scores[b] - scores[a]);
  const dominant = states.includes(dominant_academic_state) ? dominant_academic_state : ordered[0] || "uncertain";
  const secondary = states.includes(secondary_academic_state) && secondary_academic_state !== dominant
    ? secondary_academic_state
    : ordered.find((state) => state !== dominant && scores[state] > 0) || "";
  const confidenceHandling = Number(avg_confidence) < 0.5 ? "low_confidence" : "standard_confidence";
  const sourceLabel = sourceExplanationLabel(source_turn_type);
  const cue = canonicalSupportCueForDisplay(support_cue, dominant);
  const evidenceText = academicStateEvidenceText(scores, dominant, secondary, trend, confidenceHandling);
  const stateDetail = [
    `${dominant} ${formatPercent(scores[dominant] || 0)}`,
    secondary ? `secondary ${secondary} ${formatPercent(scores[secondary] || 0)}` : "",
  ].filter(Boolean).join(", ");
  const trendClause = trend ? ` with a ${trend} trend` : "";
  const confidenceClause = confidenceHandling === "low_confidence" ? ", but the average confidence was low" : "";
  const resolvingDifficulty = isResolvingDifficultyPattern(scores, dominant, secondary, trend);
  let strategySentence = strategyReasonSentence(cue, recommended_strategy, pedagogical_move, confidenceHandling);
  if (resolvingDifficulty && cue === "deepening") {
    strategySentence = "The difficulty signal was falling and engagement was close behind, suggesting the earlier difficulty may be resolving. The system therefore selected a cautious deepening strategy.";
  } else if (cue === "deepening" && dominant !== "engagement") {
    strategySentence = "The signal did not show persistent difficulty or clarification needs across the full window, so the system selected a cautious continuation strategy.";
  }
  const reasonText = `While ${sourceLabel} was being read, the reaction window was ${dominant}-dominant (${stateDetail})${trendClause}${confidenceClause}. ${strategySentence}`;
  return {
    reason_text: reasonText,
    academic_state_evidence_text: evidenceText,
    confidence_handling: confidenceHandling,
    source_turn_type: source_turn_type || "previous_explanation",
    support_cue_label: supportCueReasonLabel(cue),
  };
}

function sourceExplanationLabel(sourceTurnType) {
  if (sourceTurnType === "baseline_explanation" || sourceTurnType === "baseline") return "the baseline explanation";
  if (sourceTurnType === "strategy_reexplanation" || sourceTurnType === "adaptive") return "the previous adaptive explanation";
  return "the previous explanation";
}

function canonicalSupportCueForDisplay(supportCue, dominantState) {
  const aliases = {
    sustained_clarification: "clarification",
    gentle_clarification: "clarification",
    clarify_and_reengage: "clarification",
    reduce_load: "difficulty_support",
    difficulty_support: "difficulty_support",
    deepening: "deepening",
    re_engagement: "re_engagement",
    neutral_or_uncertain: "uncertain",
    uncertain: "uncertain",
    neutral: "neutral",
  };
  if (aliases[supportCue]) return aliases[supportCue];
  const byState = {
    boredom: "re_engagement",
    confusion: "clarification",
    engagement: "deepening",
    frustration: "difficulty_support",
  };
  return byState[dominantState] || "uncertain";
}

function academicStateEvidenceText(scores, dominant, secondary, trend, confidenceHandling) {
  const pieces = [
    [dominant, scores[dominant]],
    secondary ? [secondary, scores[secondary]] : null,
  ].filter((item) => item && Number(item[1]) > 0).map(([state, score]) => `${state} ${formatPercent(score)}`);
  const parts = [pieces.join(", ") || "mixed learning-state evidence"];
  if (confidenceHandling === "low_confidence") parts.push("low-confidence signal");
  if (trend) parts.push(`trend ${trend}`);
  return parts.join(" · ");
}

function supportCueReasonLabel(supportCue) {
  const labels = {
    re_engagement: "re-engagement",
    clarification: "clarification",
    deepening: "deepening",
    difficulty_support: "reduce cognitive load",
    neutral: "neutral",
    uncertain: "mixed / low-confidence signal",
  };
  return labels[supportCue] || "mixed / low-confidence signal";
}

function recommendedStrategyForSupportCue(supportCue) {
  const cue = canonicalSupportCueForDisplay(supportCue);
  const strategies = {
    clarification: "simplify_and_define_terms",
    difficulty_support: "step_by_step_decomposition",
    re_engagement: "relevance_hook",
    deepening: "deepen_or_extend",
  };
  return strategies[cue] || "baseline_explanation";
}

function strategyReasonSentence(supportCue, recommendedStrategy, pedagogicalMove, confidenceHandling) {
  const strategy = String(recommendedStrategy || "").toLowerCase();
  const move = String(pedagogicalMove || "").toLowerCase();
  const stepLike = strategy.includes("step") || strategy.includes("decomposition") || move.includes("step");
  if (supportCue === "deepening") return "This suggested the answer could be extended, so the system selected a deeper explanation strategy.";
  if (supportCue === "difficulty_support") {
    if (confidenceHandling === "low_confidence") {
      return `The system therefore chose a conservative ${stepLike ? "step-by-step strategy" : "support strategy"} to reduce cognitive load.`;
    }
    return `This suggested a strategy to reduce cognitive load, so the system selected a ${stepLike ? "step-by-step support strategy" : "support strategy"}.`;
  }
  if (supportCue === "clarification") return "This suggested clarification would help, so the system selected a clarification strategy.";
  if (supportCue === "re_engagement") return "This suggested a light re-engagement move, so the system selected a relevance-focused strategy.";
  return "Because the signal was mixed or uncertain, the system selected a conservative continuation strategy.";
}

function observedWindowLabel(summary) {
  if (!summary || !Object.keys(summary).length) return "";
  const endReason = endReasonObservedLabel(summary.end_reason);
  const duration = formatDuration(summary.duration_sec || 0);
  const confidence = formatPercent(summary.avg_confidence ?? summary.max_confidence ?? 0);
  const trend = summary.trend || "stable";
  return ["Observed while the explanation was being read", endReason, duration, `avg confidence ${confidence}`, `trend ${trend}`].filter(Boolean).join(" · ");
}

function endReasonObservedLabel(endReason) {
  const labels = {
    near_bottom_reached: "reached near the end of the explanation",
    short_answer_min_elapsed: "observed for the minimum reading window",
    max_timeout: "observed until the reading window timed out",
    followup_submitted: "observed until a follow-up was submitted",
    strategy_clicked: "observed until a strategy was selected",
  };
  return labels[endReason] || "";
}

function reactionSummaryFromThread(thread) {
  if (!thread) return null;
  if (thread.reaction_window_summary && Object.keys(thread.reaction_window_summary).length) {
    return thread.reaction_window_summary;
  }
  const metadata = latestTurnMetadata(thread);
  if (metadata.reaction_window_summary && Object.keys(metadata.reaction_window_summary).length) {
    return metadata.reaction_window_summary;
  }
  const message = [...(thread.messages || [])].reverse().find((item) => item.reaction_window_summary && Object.keys(item.reaction_window_summary).length);
  return message?.reaction_window_summary || null;
}

function turnMetadataForThread(thread, turnId) {
  if (!thread?.turn_metadata || !turnId) return {};
  return thread.turn_metadata[turnId] || {};
}

function latestTurnMetadata(thread) {
  if (!thread?.turn_metadata) return {};
  const latestAssistant = latestAssistantMessage(thread);
  const latestTurnId = latestAssistant?.turn_id || latestAssistant?.conversation_turn_id || "";
  if (latestTurnId && thread.turn_metadata[latestTurnId]) return thread.turn_metadata[latestTurnId];
  const entries = Object.entries(thread.turn_metadata);
  return entries.length ? entries[entries.length - 1][1] || {} : {};
}

function mergeTurnMetadata(thread, turnId, metadata) {
  if (!thread || !turnId) return thread;
  return {
    ...thread,
    turn_metadata: {
      ...(thread.turn_metadata || {}),
      [turnId]: {
        ...turnMetadataForThread(thread, turnId),
        ...metadata,
      },
    },
  };
}

function hasTurnReactionMetadata(thread, turnId) {
  const metadata = turnMetadataForThread(thread, turnId);
  return Boolean(metadata.reaction_window_summary && Object.keys(metadata.reaction_window_summary).length);
}

function hasTurnStrategyCandidates(thread, turnId) {
  const metadata = turnMetadataForThread(thread, turnId);
  return Boolean(Array.isArray(metadata.strategy_candidates) && metadata.strategy_candidates.length);
}

function completedReactionTurnIdsFromThread(thread) {
  const completed = new Set();
  const metadata = thread?.turn_metadata || {};
  for (const [turnId, value] of Object.entries(metadata)) {
    if (value?.reaction_window_summary || (Array.isArray(value?.strategy_candidates) && value.strategy_candidates.length)) {
      completed.add(turnId);
    }
  }
  return completed;
}

function strategyCandidatesForTurn(turn, turnMetadata, candidates, strategySourceTurnId) {
  if (!turn?.turn_id) return [];
  const metadata = turnMetadataForThread({ turn_metadata: turnMetadata || {} }, turn.turn_id);
  if (metadata.strategy_status === "applied") return [];
  if (Array.isArray(metadata.strategy_candidates) && metadata.strategy_candidates.length) return metadata.strategy_candidates;
  if (turn.turn_id === strategySourceTurnId) return candidates || [];
  return [];
}

function latestAssistantMessage(thread) {
  return [...((thread && thread.messages) || [])].reverse().find((message) => message.role === "assistant") || null;
}

function threadHasMessages(thread) {
  return Boolean((thread?.messages || []).some((message) => normalizePdfText(message?.content || "")));
}

function threadMessageCount(thread) {
  return (thread?.messages || []).filter((message) => normalizePdfText(message?.content || "")).length;
}

function shouldIgnoreEmptyHighlightsLoad(incomingHighlights, currentHighlights, documentId) {
  const incoming = Array.isArray(incomingHighlights) ? incomingHighlights : [];
  const current = Array.isArray(currentHighlights) ? currentHighlights : [];
  return (
    incoming.length === 0
    && current.some((highlight) => (highlight.document_id || documentId) === documentId)
  );
}

function shouldIgnoreEmptyThreadLoad(incomingThread, currentThread, highlightId) {
  const incomingHighlightId = incomingThread?.highlight_id || "";
  const currentHighlightId = currentThread?.highlight_id || "";
  return (
    !threadHasMessages(incomingThread)
    && threadHasMessages(currentThread)
    && currentHighlightId === highlightId
    && (!incomingHighlightId || incomingHighlightId === highlightId)
  );
}

function shouldIgnoreEmptyThreadPersist(nextThread, currentThread, highlightId) {
  const nextHighlightId = nextThread?.highlight_id || "";
  const currentHighlightId = currentThread?.highlight_id || "";
  return (
    !threadHasMessages(nextThread)
    && threadHasMessages(currentThread)
    && currentHighlightId === highlightId
    && (!nextHighlightId || nextHighlightId === highlightId)
  );
}

async function persistThreadAfterAssistantMessage(documentId, highlightId, thread) {
  const saved = await putJson(`/api/documents/${documentId}/threads/${highlightId}`, thread);
  if (!saved || !Array.isArray(saved.messages)) {
    throw new Error("Answer generated, but conversation could not be saved.");
  }
  return saved;
}

function logPdfChatRestoreDebug(event, payload) {
  if (!pdfChatRestoreDebugEnabled()) return;
  if (typeof console === "undefined" || typeof console.debug !== "function") return;
  console.debug("[pdf-chat] highlight restore", { event, ...(payload || {}) });
}

function pdfChatRestoreDebugEnabled() {
  if (typeof import.meta !== "undefined" && import.meta.env?.DEV) return true;
  if (typeof window === "undefined") return false;
  return new URLSearchParams(window.location.search).has("pdfChatDebug");
}

function baselineExplanationForTurn(thread, sourceTurnId) {
  if (!sourceTurnId) return "";
  const message = (thread?.messages || []).find((item) => (
    item.role === "assistant"
    && (item.turn_id === sourceTurnId || item.conversation_turn_id === sourceTurnId)
  ));
  return message?.content || "";
}

function strategyPaperContextFromSelection(activeSelection) {
  const preview = activeSelection?.llmInputPreview || {};
  const passageType = inferPassageType(preview);
  const difficultyHint = inferDifficultyHint(preview, passageType);
  return {
    matched_block: preview.matched_block || null,
    nearby_context: preview.nearby_useful_context || [],
    retrieved_chunks: [],
    paper_profile: {},
    passage_type: passageType,
    difficulty_hint: difficultyHint,
  };
}

function plannerInputSummaryForStrategyRequest({
  preview,
  paperContext,
  baselineExplanation,
  reactionWindowSummary,
  recentConversation,
  previousStrategy,
}) {
  return {
    selection_type: preview?.highlight_type || "",
    selected_text_length: normalizePdfText(preview?.selected_text || "").length,
    caption_length: normalizePdfText(preview?.caption || "").length,
    baseline_explanation_length: normalizePdfText(baselineExplanation || "").length,
    rag_chunk_count: Array.isArray(paperContext?.retrieved_chunks) ? paperContext.retrieved_chunks.length : 0,
    nearby_context_count: Array.isArray(paperContext?.nearby_context) ? paperContext.nearby_context.length : 0,
    reaction_window_duration_sec: Number(reactionWindowSummary?.duration_sec || 0),
    reaction_window_end_reason: reactionWindowSummary?.end_reason || "",
    read_progress_estimate: Number(reactionWindowSummary?.read_progress_estimate || 0),
    word_count: Number(reactionWindowSummary?.word_count || 0),
    estimated_min_read_sec: Number(reactionWindowSummary?.estimated_min_read_sec || 0),
    elapsed_sec: Number(reactionWindowSummary?.elapsed_sec || reactionWindowSummary?.duration_sec || 0),
    min_read_time_met: Boolean(reactionWindowSummary?.min_read_time_met),
    min_samples_met: Boolean(reactionWindowSummary?.min_samples_met),
    top_seen: Boolean(reactionWindowSummary?.top_seen),
    bottom_seen: Boolean(reactionWindowSummary?.bottom_seen),
    user_scroll_progressed: Boolean(reactionWindowSummary?.user_scroll_progressed),
    is_short_answer: Boolean(reactionWindowSummary?.is_short_answer),
    end_blocked_reason: reactionWindowSummary?.end_blocked_reason || "",
    state_transition: reactionWindowSummary?.state_transition || "",
    transition_label: reactionWindowSummary?.transition_label || "",
    early_distribution: reactionWindowSummary?.early_distribution || {},
    middle_distribution: reactionWindowSummary?.middle_distribution || {},
    late_distribution: reactionWindowSummary?.late_distribution || {},
    support_cue: reactionWindowSummary?.support_cue || "",
    allowed_strategy_families: allowedStrategyFamiliesForSupportCue(reactionWindowSummary?.support_cue || "neutral_or_uncertain"),
    recent_conversation_count: Array.isArray(recentConversation) ? recentConversation.length : 0,
    previous_strategy_id: previousStrategy?.strategy_id || "",
    passage_type: paperContext?.passage_type || "unknown",
    difficulty_hint: paperContext?.difficulty_hint || "unknown",
  };
}

function allowedStrategyFamiliesForSupportCue(supportCue) {
  const families = {
    sustained_clarification: ["step_by_step_breakdown", "define_key_terms", "concrete_example", "input_process_output_map", "mechanism_walkthrough", "formula_intuition"],
    difficulty_support: ["concrete_example", "example_based_explanation", "analogy_or_reframe", "connect_to_paper_argument", "simplest_version_first", "one_small_next_step", "reduce_information_density", "key_takeaway_first"],
    reduce_load: ["concrete_example", "example_based_explanation", "analogy_or_reframe", "connect_to_paper_argument", "simplest_version_first", "one_small_next_step", "reduce_information_density", "key_takeaway_first"],
    re_engagement: ["why_it_matters", "one_sentence_takeaway", "make_it_relevant", "compare_with_familiar_method", "quick_quiz"],
    deepening: ["deep_technical_explanation", "critique_assumptions", "connect_to_related_work", "limitations_and_implications", "compare_methods"],
    clarify_and_reengage: ["concise_explanation", "concrete_example", "why_it_matters", "step_by_step_breakdown", "compare_with_familiar_method"],
    gentle_clarification: ["simplest_version_first", "one_small_next_step", "define_key_terms", "analogy_or_reframe", "concrete_example"],
    neutral_or_uncertain: ["concise_explanation", "structured_breakdown", "example_based_explanation", "connect_to_paper_argument"],
  };
  return families[supportCue] || families.neutral_or_uncertain;
}

function inferPassageType(preview) {
  if (preview.highlight_type === "area") {
    if (preview.caption || preview.matched_block_type === "image") return "figure";
    if (preview.matched_block_type === "table") return "table";
  }
  const blockType = String(preview.matched_block_type || "").toLowerCase();
  if (blockType === "formula") return "formula";
  if (blockType === "table") return "result";
  const text = normalizePdfText(`${preview.selected_text || ""} ${preview.matched_block?.markdown_content || ""}`).toLowerCase();
  if (/\b(method|algorithm|procedure|pipeline|we compute|we train|we extract)\b/.test(text)) return "method";
  if (/\b(result|accuracy|performance|baseline|evaluation|significant)\b/.test(text)) return "result";
  if (/\bdefine|definition|is called|refers to\b/.test(text)) return "definition";
  if (/\bdiscussion|limitation|future work|implication\b/.test(text)) return "discussion";
  return "unknown";
}

function inferDifficultyHint(preview, passageType) {
  const text = normalizePdfText(`${preview.selected_text || ""} ${preview.matched_block?.markdown_content || ""}`);
  if (passageType === "formula" || /[=∑∫√≤≥]/.test(text)) return "formula";
  if (passageType === "method" && /\b(step|stage|pipeline|algorithm|first|then|finally)\b/i.test(text)) return "multi_step_process";
  if (/\b(theorem|proof|assumption|latent|objective|optimization)\b/i.test(text)) return "dense_theory";
  if (/\b[A-Z]{2,}\b/.test(text) || text.split(/\s+/).some((word) => word.length > 16)) return "technical_terms";
  if (preview.highlight_type === "area") return "unclear_reference";
  return "unknown";
}

function deriveConversationTurns(messages) {
  const turns = [];
  const explicitTurns = new Map();
  let pendingLegacyUser = null;

  function addTurn(turn) {
    turns.push({
      turn_id: turn.turn_id || `derived-${turns.length}`,
      messages: turn.messages || [],
    });
  }

  for (const message of messages || []) {
    const explicitTurnId = message.turn_id || message.conversation_turn_id;
    if (explicitTurnId) {
      if (pendingLegacyUser) {
        addTurn({ messages: [pendingLegacyUser] });
        pendingLegacyUser = null;
      }
      if (!explicitTurns.has(explicitTurnId)) {
        const turn = { turn_id: explicitTurnId, messages: [] };
        explicitTurns.set(explicitTurnId, turn);
        turns.push(turn);
      }
      explicitTurns.get(explicitTurnId).messages.push(message);
      continue;
    }

    if (message.role === "user") {
      if (pendingLegacyUser) addTurn({ messages: [pendingLegacyUser] });
      pendingLegacyUser = message;
      continue;
    }

    if (message.role === "assistant") {
      if (pendingLegacyUser) {
        addTurn({ messages: [pendingLegacyUser, message] });
        pendingLegacyUser = null;
      } else {
        addTurn({ messages: [message] });
      }
      continue;
    }

    addTurn({ messages: [message] });
  }

  if (pendingLegacyUser) addTurn({ messages: [pendingLegacyUser] });
  return turns.filter((turn) => turn.messages.length);
}

function wordCountForReadingEstimate(text) {
  const normalized = normalizePdfText(text || "");
  if (!normalized) return 0;
  return normalized.split(/\s+/).filter(Boolean).length;
}

function estimateReactionWindowMinReadSec(text) {
  const wordCount = wordCountForReadingEstimate(text);
  const estimated = wordCount / REACTION_WINDOW_READING_WORDS_PER_SEC;
  return Math.max(
    REACTION_WINDOW_MIN_READ_SEC_FLOOR,
    Math.min(REACTION_WINDOW_MIN_READ_SEC_CEIL, Math.ceil(estimated)),
  );
}

function finiteScore(value, fallback) {
  const score = Number(value);
  if (!Number.isFinite(score)) return fallback;
  return Math.max(0, Math.min(score, 1));
}

function defaultPrepareSteps(status) {
  const labels = [
    "Uploading PDF",
    "Extracting text and layout",
    "Building paper profile",
    "Building keyword index",
    "Building embedding index",
    "Ready",
  ];
  return labels.map((label, index) => ({
    id: label.toLowerCase().replace(/[^a-z0-9]+/g, "-"),
    label,
    status: status === "Ready" || (!status && index === 0) ? "completed" : index === 0 ? "active" : "pending",
  }));
}

function formatJson(value) {
  return JSON.stringify(value ?? null, null, 2);
}

function browserSupportsSpeechRecording() {
  return Boolean(
    typeof navigator !== "undefined"
    && navigator.mediaDevices?.getUserMedia
    && typeof MediaRecorder !== "undefined",
  );
}

function preferredSpeechMimeType() {
  if (typeof MediaRecorder === "undefined" || typeof MediaRecorder.isTypeSupported !== "function") return "";
  for (const mimeType of ["audio/webm;codecs=opus", "audio/webm", "audio/wav"]) {
    if (MediaRecorder.isTypeSupported(mimeType)) return mimeType;
  }
  return "";
}

async function pollPreparationStatus(documentId, onStatus) {
  const started = Date.now();
  let latest = null;
  while (Date.now() - started < 120000) {
    latest = await fetchJson(`/api/documents/${documentId}`);
    onStatus?.(latest.prepare_status || null);
    const status = latest.prepare_status?.status;
    if (status === "completed" || status === "failed") {
      return latest;
    }
    await delay(750);
  }
  return latest || { document_id: documentId };
}

function delay(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function putJson(url, body) {
  const response = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

async function deleteJson(url, body = {}) {
  const response = await fetch(url, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

createRoot(document.getElementById("pdf-chat-root")).render(<PdfChatApp />);
