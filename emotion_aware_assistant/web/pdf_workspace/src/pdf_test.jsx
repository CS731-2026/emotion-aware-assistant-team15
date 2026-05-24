import React, { useCallback, useEffect, useRef, useState } from "react";
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
import "./pdf_test.css";

const DEBUG_DOCUMENT_ID = "debug-pdf";

function HighlightContainer() {
  const { highlight, isScrolledTo } = useHighlightContainerContext();

  if (highlight.type === "area") {
    return <AreaHighlight highlight={highlight} isScrolledTo={isScrolledTo} />;
  }

  return <TextHighlight highlight={highlight} isScrolledTo={isScrolledTo} />;
}

function PdfDocumentView({
  pdfDocument,
  highlights,
  areaMode,
  onPdfDocumentLoaded,
  onSelection,
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
      <HighlightContainer />
    </PdfHighlighter>
  );
}

function PdfDebugPanel({
  selectionDebug,
  lastHighlightDebug,
  matchDebug,
  matchStatus,
  copyText,
  copyStatus,
  explainStatus,
  explainResult,
  explainLoading,
  currentDocument,
  prepareStatus,
  uploadStatus,
  onCopyCapturedText,
  onExplainSelection,
}) {
  const llmInputPreview = buildLlmInputPreview(lastHighlightDebug, matchDebug);
  const cleanedPromptPreview = buildCleanedPromptPreview(llmInputPreview);
  const canExplain = Boolean(llmInputPreview.highlight_id) && !explainLoading;
  const showCaptionFields = shouldShowCaptionFields(llmInputPreview);

  return (
    <aside className="pdf-test-debug-panel" aria-label="Captured selection debug">
      <div className="pdf-test-debug-heading">
        <h2>Captured Selection</h2>
        <button
          type="button"
          className="pdf-test-copy-button"
          onClick={onCopyCapturedText}
          disabled={!copyText}
        >
          Copy Captured Text
        </button>
      </div>
      {copyStatus ? <p className="pdf-test-copy-status">{copyStatus}</p> : null}
      {uploadStatus ? <p className="pdf-test-copy-status">{uploadStatus}</p> : null}

      <h3>Document preparation</h3>
      <dl className="pdf-test-debug-grid">
        <dt>document_id</dt>
        <dd>{currentDocument.document_id || "-"}</dd>
        <dt>file name</dt>
        <dd>{currentDocument.file_name || "-"}</dd>
        <dt>page count</dt>
        <dd>{String(currentDocument.page_count || "-")}</dd>
        <dt>parsed blocks count</dt>
        <dd>{String(prepareStatus?.block_count ?? "-")}</dd>
        <dt>paper profile status</dt>
        <dd>{prepareStatus?.paper_profile_path ? "ready" : "-"}</dd>
        <dt>keyword index status</dt>
        <dd>{prepareStatus?.keyword_index_path ? "ready" : "-"}</dd>
        <dt>embedding/File Search index status</dt>
        <dd>{prepareStatus?.embedding_index_status || prepareStatus?.file_search_status || "-"}</dd>
        <dt>errors</dt>
        <dd>{prepareStatus?.error || prepareStatus?.embedding_message || "-"}</dd>
      </dl>

      <h3>Browser selection text</h3>
      <p>Raw</p>
      <pre>{selectionDebug.browserText || "(empty)"}</pre>
      <p>Normalized</p>
      <pre>{selectionDebug.normalizedBrowserText || "(empty)"}</pre>

      <h3>Library selection text</h3>
      <p>Raw</p>
      <pre>{selectionDebug.libraryText || "(empty)"}</pre>
      <p>Normalized</p>
      <pre>{selectionDebug.normalizedLibraryText || "(empty)"}</pre>

      <h3>Final highlight text</h3>
      <p>Raw</p>
      <pre>{lastHighlightDebug.rawText || "(empty)"}</pre>
      <p>Normalized</p>
      <pre>{lastHighlightDebug.normalizedText || "(empty)"}</pre>

      <h3>Crop preview</h3>
      {lastHighlightDebug.cropImage ? (
        <img
          className="pdf-test-crop-preview"
          src={lastHighlightDebug.cropImage}
          alt="Selected PDF area crop"
        />
      ) : (
        <p>No crop captured.</p>
      )}

      <h3>Highlight metadata</h3>
      <dl className="pdf-test-debug-grid">
        <dt>Highlight id</dt>
        <dd>{lastHighlightDebug.id || "-"}</dd>
        <dt>Highlight type</dt>
        <dd>{lastHighlightDebug.type || "-"}</dd>
        <dt>Page number</dt>
        <dd>{lastHighlightDebug.pageNumber || "-"}</dd>
        <dt>Text length</dt>
        <dd>{String(lastHighlightDebug.textLength || 0)}</dd>
        <dt>Empty</dt>
        <dd>{lastHighlightDebug.empty ? "yes" : "no"}</dd>
        <dt>Suspicious</dt>
        <dd>{lastHighlightDebug.suspicious ? "yes" : "no"}</dd>
        <dt>Selection keys</dt>
        <dd>{selectionDebug.selectionKeys.join(", ") || "-"}</dd>
      </dl>

      <h3>Position</h3>
      <pre>{formatJson(lastHighlightDebug.position)}</pre>
      <h3>Bounding rect</h3>
      <pre>{formatJson(lastHighlightDebug.boundingRect)}</pre>
      <h3>Rects</h3>
      <pre>{formatJson(lastHighlightDebug.rects)}</pre>

      <h3>Local coordinate context</h3>
      <h3>Matched Markdown Blocks</h3>
      {matchStatus ? <p>{matchStatus}</p> : null}
      <dl className="pdf-test-debug-grid">
        <dt>highlight_id</dt>
        <dd>{matchDebug.highlightId || "-"}</dd>
        <dt>page_number</dt>
        <dd>{matchDebug.pageNumber || "-"}</dd>
        <dt>selected text</dt>
        <dd>{matchDebug.selectedText || "-"}</dd>
      </dl>
      <h3>Viewport rects</h3>
      <pre>{formatJson(matchDebug.viewportRects)}</pre>
      <h3>Normalized rects</h3>
      <pre>{formatJson(matchDebug.normalizedRects)}</pre>
      <h3>Parser rects 1000</h3>
      <pre>{formatJson(matchDebug.parserRects1000)}</pre>
      <MatchBlockList title="Matched block" blocks={matchDebug.matchedBlocks} />
      <MatchBlockList title="Previous block" blocks={matchDebug.previousBlocks} />
      <MatchBlockList title="Next block" blocks={matchDebug.nextBlocks} />
      {matchDebug.error ? (
        <>
          <h3>Matcher error</h3>
          <pre>{matchDebug.error}</pre>
        </>
      ) : null}

      <h3>LLM Input Preview</h3>
      <dl className="pdf-test-debug-grid">
        <dt>Recommended LLM mode</dt>
        <dd>{llmInputPreview.recommended_llm_mode}</dd>
        <dt>highlight_type</dt>
        <dd>{llmInputPreview.highlight_type || "-"}</dd>
        <dt>page_number</dt>
        <dd>{llmInputPreview.page_number || "-"}</dd>
        <dt>text_available</dt>
        <dd>{llmInputPreview.text_available ? "true" : "false"}</dd>
        <dt>reason</dt>
        <dd>{llmInputPreview.reason || "-"}</dd>
        <dt>crop_image_available</dt>
        <dd>{llmInputPreview.crop_image_available ? "true" : "false"}</dd>
        <dt>crop_image_data_url_length</dt>
        <dd>{String(llmInputPreview.crop_image_data_url_length)}</dd>
        {showCaptionFields ? (
          <>
            <dt>caption</dt>
            <dd>{llmInputPreview.caption || "-"}</dd>
            <dt>selected_caption</dt>
            <dd>{llmInputPreview.selected_caption?.markdown_content || "-"}</dd>
            <dt>caption_confidence</dt>
            <dd>{llmInputPreview.caption_confidence || "-"}</dd>
          </>
        ) : null}
        <dt>matched_block_id</dt>
        <dd>{llmInputPreview.matched_block_id || "-"}</dd>
        <dt>matched_block_type</dt>
        <dd>{llmInputPreview.matched_block_type || "-"}</dd>
        <dt>coordinate_overlap</dt>
        <dd>{String(llmInputPreview.coordinate_overlap ?? "-")}</dd>
        <dt>text_bonus</dt>
        <dd>{String(llmInputPreview.text_bonus ?? "-")}</dd>
        <dt>match_score</dt>
        <dd>{String(llmInputPreview.match_score ?? "-")}</dd>
      </dl>
      {showCaptionFields ? (
        <details className="pdf-test-raw-debug">
          <summary>Candidate captions</summary>
          <pre>{formatJson(llmInputPreview.candidate_captions || [])}</pre>
        </details>
      ) : null}
      <h3>Cleaned Prompt Preview</h3>
      <pre>{formatJson(cleanedPromptPreview)}</pre>

      <h3>Explain Selection</h3>
      <button
        type="button"
        className="pdf-test-explain-button"
        onClick={() => onExplainSelection(llmInputPreview)}
        disabled={!canExplain}
      >
        {explainLoading ? "Explaining..." : "Explain Selection"}
      </button>
      {explainStatus ? <p className="pdf-test-explain-status">{explainStatus}</p> : null}
      {explainResult ? (
        <div className="pdf-test-explain-result">
          <dl className="pdf-test-debug-grid">
            <dt>Provider</dt>
            <dd>{explainResult.provider || "-"}</dd>
            <dt>Model</dt>
            <dd>{explainResult.model || "-"}</dd>
            <dt>Mode</dt>
            <dd>{explainResult.mode || explainResult.recommended_llm_mode || "-"}</dd>
            <dt>Used image</dt>
            <dd>{explainResult.used_image ? "true" : "false"}</dd>
            <dt>Paper profile used</dt>
            <dd>{explainResult.paper_profile_used ? "true" : "false"}</dd>
            <dt>Retrieved block count</dt>
            <dd>{String(explainResult.retrieved_block_count ?? 0)}</dd>
          </dl>
          {explainResult.error ? (
            <p className="pdf-test-explain-error">{explainResult.error}</p>
          ) : null}
          <p>Paper profile summary</p>
          <pre>{explainResult.paper_profile_summary || "(empty)"}</pre>
          <p>Retrieved related blocks</p>
          <ExplainBlockList blocks={explainResult.retrieved_blocks || []} />
          <p>Global RAG context</p>
          <ExplainBlockList blocks={explainResult.global_rag_context || []} />
          <dl className="pdf-test-debug-grid">
            <dt>retrieval_method</dt>
            <dd>{explainResult.retrieval_method || "-"}</dd>
          </dl>
          <p>Answer</p>
          <pre>{explainResult.answer || "(empty)"}</pre>
          <details className="pdf-test-prompt-debug">
            <summary>Prompt preview</summary>
            <pre>{explainResult.prompt_preview || "(empty)"}</pre>
          </details>
        </div>
      ) : null}
      <details className="pdf-test-raw-debug">
        <summary>Raw LLM debug payload</summary>
        <pre>{formatJson(llmInputPreview)}</pre>
      </details>
    </aside>
  );
}

function MatchBlockList({ title, blocks }) {
  if (!blocks?.length) {
    return (
      <>
        <h3>{title}</h3>
        <p>No block returned.</p>
      </>
    );
  }

  return (
    <>
      {blocks.map((block, index) => (
        <article className="pdf-test-match-block" key={`${title}-${block.block_id || index}`}>
          <h3>{index === 0 ? title : `${title} ${index + 1}`}</h3>
          <dl className="pdf-test-debug-grid">
            <dt>Matched block id</dt>
            <dd>{block.block_id || "-"}</dd>
            <dt>Page number</dt>
            <dd>{block.page_number || "-"}</dd>
            <dt>Block type</dt>
            <dd>{block.block_type || "-"}</dd>
            <dt>Coordinate overlap</dt>
            <dd>{String(coordinateOverlap(block) ?? "-")}</dd>
            <dt>Text bonus</dt>
            <dd>{String(textBonus(block) ?? "-")}</dd>
            <dt>Match score</dt>
            <dd>{String(blockMatchScore(block) ?? "-")}</dd>
          </dl>
          <p>Markdown content</p>
          <pre>{block.markdown_content || "(empty)"}</pre>
        </article>
      ))}
    </>
  );
}

function ExplainBlockList({ blocks }) {
  if (!blocks?.length) {
    return <p>No related blocks returned.</p>;
  }
  return (
    <div className="pdf-test-explain-blocks">
      {blocks.map((block, index) => (
        <article className="pdf-test-explain-block" key={block.block_id || index}>
          <dl className="pdf-test-debug-grid">
            <dt>block_id</dt>
            <dd>{block.block_id || "-"}</dd>
            <dt>page_number</dt>
            <dd>{block.page_number || "-"}</dd>
            <dt>block_type</dt>
            <dd>{block.block_type || "-"}</dd>
            <dt>score</dt>
            <dd>{String(block.score ?? block.keyword_score ?? "-")}</dd>
          </dl>
          <pre>{block.markdown_content || "(empty)"}</pre>
        </article>
      ))}
    </div>
  );
}

function PdfTestApp() {
  const [highlights, setHighlights] = useState([]);
  const [pendingSelection, setPendingSelection] = useState(null);
  const [areaMode, setAreaMode] = useState(false);
  const [pdfUrl, setPdfUrl] = useState("/api/debug/pdf");
  const [currentDocument, setCurrentDocument] = useState(() => ({
    document_id: DEBUG_DOCUMENT_ID,
    file_name: "debug PDF",
    page_count: "",
    pdf_url: "/api/debug/pdf",
  }));
  const [prepareStatus, setPrepareStatus] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [selectionDebug, setSelectionDebug] = useState(() => emptySelectionDebug());
  const [lastHighlightDebug, setLastHighlightDebug] = useState(() => emptyHighlightDebug());
  const [matchDebug, setMatchDebug] = useState(() => emptyMatchDebug());
  const [matchStatus, setMatchStatus] = useState("");
  const [copyStatus, setCopyStatus] = useState("");
  const [explainStatus, setExplainStatus] = useState("");
  const [explainResult, setExplainResult] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const pendingSelectionRef = useRef(null);
  const selectionDebugRef = useRef(emptySelectionDebug());
  const highlighterUtilsRef = useRef(null);
  const parsePromiseRef = useRef(null);
  const currentDocumentId = currentDocument.document_id || DEBUG_DOCUMENT_ID;

  const onPdfDocumentLoaded = useCallback((pdfDocument) => {
    const pageCount = Number(pdfDocument?.numPages || 0);
    if (!Number.isFinite(pageCount) || pageCount <= 0) return;
    setCurrentDocument((current) => {
      if (Number(current.page_count || 0) === pageCount) return current;
      return { ...current, page_count: pageCount };
    });
    setPrepareStatus((current) => {
      if (!current || Number(current.page_count || 0) === pageCount) return current;
      return { ...current, page_count: pageCount };
    });
  }, []);

  function handleSelection(selection) {
    const browserText = window.getSelection()?.toString() || "";
    const libraryText = selection?.content?.text || "";
    const selectionCropImage = selection?.content?.image || "";
    const nextSelectionDebug = {
      browserText,
      normalizedBrowserText: normalizePdfText(browserText),
      libraryText,
      normalizedLibraryText: normalizePdfText(libraryText),
      selectionCropImage,
      selectionKeys: Object.keys(selection || {}),
      selectionType: selection?.type || "",
    };

    console.log("[pdf-test] selection", { selection, ...nextSelectionDebug });
    selectionDebugRef.current = nextSelectionDebug;
    setSelectionDebug(nextSelectionDebug);
    setCopyStatus("");

    if (!["text", "area"].includes(selection.type)) {
      pendingSelectionRef.current = null;
      setPendingSelection(null);
      return;
    }

    pendingSelectionRef.current = selection;
    setPendingSelection(selection);
  }

  function handleHighlightSelection() {
    const currentSelection = pendingSelectionRef.current;
    if (!currentSelection) return;

    const ghost = currentSelection.makeGhostHighlight();
    const id = crypto.randomUUID();
    const highlight = { ...ghost, id };
    const debugSource = selectionDebugRef.current;
    const rawText = ghost?.content?.text || debugSource.browserText || "";
    const cropImage = ghost?.content?.image || debugSource.selectionCropImage || "";
    const normalizedText = normalizePdfText(rawText);
    const position = ghost?.position || null;
    const nextHighlightDebug = {
      id,
      type: ghost?.type || "",
      rawText,
      normalizedText,
      textLength: rawText.length,
      cropImage,
      empty: rawText.length === 0,
      suspicious: ghost?.type === "area" ? false : isSuspiciousText(rawText),
      pageNumber: pageNumberFromPosition(position),
      position,
      boundingRect: position?.boundingRect || null,
      rects: position?.rects || [],
    };

    console.log("[pdf-test] ghost highlight", highlight);
    setHighlights((currentHighlights) => [
      ...currentHighlights,
      highlight,
    ]);
    setLastHighlightDebug(nextHighlightDebug);
    setExplainStatus("");
    setExplainResult(null);
    matchHighlightToBlocks(highlight, nextHighlightDebug);
    pendingSelectionRef.current = null;
    setPendingSelection(null);
    highlighterUtilsRef.current?.setTip?.(null);
    window.setTimeout(() => {
      highlighterUtilsRef.current?.removeGhostHighlight?.();
    }, 0);
  }

  async function handleCopyCapturedText() {
    const text = capturedTextForCopy(selectionDebug, lastHighlightDebug);
    if (!text) return;

    try {
      await navigator.clipboard.writeText(text);
      setCopyStatus("Copied captured text.");
    } catch (error) {
      console.warn("[pdf-test] copy failed", error);
      setCopyStatus("Copy unavailable.");
    }
  }

  async function handleUploadPdf(event) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setUploadStatus("Preparing PDF...");
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("/api/document/upload", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }
      parsePromiseRef.current = null;
      const nextDocument = {
        document_id: payload.document_id || "",
        file_name: payload.file_name || file.name,
        page_count: payload.page_count || "",
        pdf_url: payload.pdf_url || (payload.document_id ? `/api/document/file/${payload.document_id}` : ""),
      };
      setCurrentDocument(nextDocument);
      setPdfUrl(nextDocument.pdf_url || "/api/debug/pdf");
      setPrepareStatus(payload.parse_status || null);
      setHighlights([]);
      setPendingSelection(null);
      setSelectionDebug(emptySelectionDebug());
      setLastHighlightDebug(emptyHighlightDebug());
      setMatchDebug(emptyMatchDebug());
      setExplainResult(null);
      setExplainStatus("");
      setUploadStatus("PDF prepared.");
    } catch (error) {
      console.error("[pdf-test] upload failed", error);
      setUploadStatus(`Upload failed: ${error?.message || String(error)}`);
    }
  }

  async function explainSelection(llmInputPreview) {
    if (!llmInputPreview?.highlight_id) return;

    setExplainLoading(true);
    setExplainStatus("Explaining selection...");
    setExplainResult(null);
    try {
      const result = await postJson("/api/debug/explain-selection", llmInputPreview);
      setExplainResult(result);
      setExplainStatus("Explanation ready.");
    } catch (error) {
      setExplainResult({
        provider: "",
        model: "",
        mode: llmInputPreview.recommended_llm_mode || "",
        used_image: false,
        paper_profile_used: false,
        paper_profile_summary: "",
        retrieved_block_count: 0,
        retrieved_blocks: [],
        global_rag_context: [],
        retrieval_method: "",
        prompt_preview: "",
        answer: "",
        error: error?.message || String(error),
      });
      setExplainStatus("Explanation failed.");
    } finally {
      setExplainLoading(false);
    }
  }

  async function ensureDebugParse() {
    if (currentDocumentId !== DEBUG_DOCUMENT_ID) {
      if (prepareStatus) return prepareStatus;
      const status = await fetchJson(`/api/document/parse-status/${currentDocumentId}`);
      setPrepareStatus(status.parse_status || null);
      return status.parse_status;
    }
    if (!parsePromiseRef.current) {
      parsePromiseRef.current = postJson("/api/debug/parse", {});
    }
    const parsed = await parsePromiseRef.current;
    setPrepareStatus(parsed.parsed || null);
    return parsed;
  }

  async function matchHighlightToBlocks(highlight, highlightDebug) {
    const viewportRects = viewportRectsFromPosition(highlight.position);
    const normalizedRects = normalizedRectsFromPosition(highlight.position);
    const parserRects1000 = parserRects1000FromPosition(highlight.position);
    setMatchStatus("Parsing debug PDF and matching coordinates...");
    setMatchDebug((current) => ({
      ...current,
      documentId: currentDocumentId,
      highlightId: highlightDebug.id,
      pageNumber: highlightDebug.pageNumber,
      selectedText: highlightDebug.normalizedText,
      viewportRects,
      normalizedRects,
      parserRects1000,
      selectedCaption: null,
      captionConfidence: "none",
      candidateCaptions: [],
      error: "",
    }));

    try {
      await ensureDebugParse();
      const matchResult = await postJson("/api/document/match-blocks", {
        document_id: currentDocumentId,
        highlight_id: highlightDebug.id,
        page_number: highlightDebug.pageNumber,
        selected_text: highlightDebug.normalizedText,
        viewport_rects: viewportRectsFromPosition(highlight.position),
        normalized_rects: normalizedRectsFromPosition(highlight.position),
        parser_rects_1000: parserRects1000FromPosition(highlight.position),
        position: highlight.position,
      });
      console.log("[pdf-test] matched markdown blocks", matchResult);
      setMatchDebug({
        documentId: currentDocumentId,
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
      });
      setMatchStatus("Coordinate match complete.");
    } catch (error) {
      console.error("[pdf-test] coordinate match failed", error);
      setMatchDebug((current) => ({
        ...current,
        error: error?.message || String(error),
      }));
      setMatchStatus("Coordinate match failed.");
    }
  }

  const copyText = capturedTextForCopy(selectionDebug, lastHighlightDebug);

  return (
    <div className="pdf-test-app">
      <div className="pdf-test-toolbar">
        <span>PDF test viewer loaded. Select text, then highlight it.</span>
        <label className="pdf-test-upload-button">
          Upload PDF
          <input type="file" accept="application/pdf" onChange={handleUploadPdf} hidden />
        </label>
        <button
          type="button"
          className={`pdf-test-area-button${areaMode ? " active" : ""}`}
          onClick={() => setAreaMode((value) => !value)}
        >
          Area Select
        </button>
        {pendingSelection ? (
          <button
            type="button"
            className="pdf-test-highlight-button"
            onClick={handleHighlightSelection}
          >
            {pendingSelection.type === "area" ? "Highlight Area" : "Highlight"}
          </button>
        ) : null}
      </div>

      <div className="pdf-test-content">
        <div className="pdf-test-viewer">
          <PdfLoader
            key={pdfUrl}
            document={pdfUrl}
            beforeLoad={(progress) => (
              <div style={{ padding: 16 }}>
                Loading PDF... {progress?.loaded || 0} bytes
              </div>
            )}
            errorMessage={(error) => (
              <div style={{ padding: 16, color: "red" }}>
                Failed to load PDF: {String(error?.message || error)}
              </div>
            )}
            onError={(error) => {
              console.error("PdfLoader error", error);
            }}
          >
            {(pdfDocument) => (
              <PdfDocumentView
                pdfDocument={pdfDocument}
                highlights={highlights}
                areaMode={areaMode}
                onPdfDocumentLoaded={onPdfDocumentLoaded}
                onSelection={handleSelection}
                highlighterUtilsRef={highlighterUtilsRef}
              />
            )}
          </PdfLoader>
        </div>
        <PdfDebugPanel
          selectionDebug={selectionDebug}
          lastHighlightDebug={lastHighlightDebug}
          matchDebug={matchDebug}
          matchStatus={matchStatus}
          copyText={copyText}
          copyStatus={copyStatus}
          explainStatus={explainStatus}
          explainResult={explainResult}
          explainLoading={explainLoading}
          currentDocument={currentDocument}
          prepareStatus={prepareStatus}
          uploadStatus={uploadStatus}
          onCopyCapturedText={handleCopyCapturedText}
          onExplainSelection={explainSelection}
        />
      </div>
    </div>
  );
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

function emptyHighlightDebug() {
  return {
    id: "",
    type: "",
    rawText: "",
    normalizedText: "",
    textLength: 0,
    cropImage: "",
    empty: true,
    suspicious: false,
    pageNumber: "",
    position: null,
    boundingRect: null,
    rects: [],
  };
}

function emptyMatchDebug() {
  return {
    documentId: DEBUG_DOCUMENT_ID,
    highlightId: "",
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
  const matchScore = blockMatchScore(matchedBlock);
  const normalizedSelectedText = normalizePdfText(lastHighlightDebug.rawText || matchDebug.selectedText || "");
  const hasText = Boolean(normalizedSelectedText);
  const contextBlocks = usefulContextBlocks(matchDebug);
  const selectedCaption = selectCaptionForHighlight(lastHighlightDebug, matchDebug, matchedBlock);
  const showCaptionDebug = lastHighlightDebug.type === "area" || isCaptionBlock(matchedBlock);

  return {
    document_id: matchDebug.documentId || DEBUG_DOCUMENT_ID,
    highlight_id: lastHighlightDebug.id || "",
    highlight_type: lastHighlightDebug.type,
    page_number: lastHighlightDebug.pageNumber,
    selected_text: normalizedSelectedText,
    text_available: hasText,
    reason: textUnavailableReason(lastHighlightDebug, hasText),
    viewport_rects: matchDebug.viewportRects,
    normalized_rects: matchDebug.normalizedRects,
    parser_rects_1000: matchDebug.parserRects1000,
    crop_image_available: Boolean(lastHighlightDebug.cropImage),
    crop_image_path: "",
    crop_image_data_url: lastHighlightDebug.cropImage || "",
    crop_image_data_url_length: lastHighlightDebug.cropImage?.length || 0,
    matched_block_id: matchedBlock?.block_id || "",
    matched_block_type: matchedBlock?.block_type || "",
    coordinate_overlap: coordinateOverlap(matchedBlock),
    text_bonus: textBonus(matchedBlock),
    match_score: matchScore,
    caption: selectedCaption?.markdown_content || "",
    selected_caption: showCaptionDebug ? selectedCaption || null : null,
    caption_confidence: showCaptionDebug ? matchDebug.captionConfidence || (selectedCaption ? "medium" : "none") : "",
    candidate_captions: showCaptionDebug ? matchDebug.candidateCaptions || [] : [],
    matched_block: summarizeBlock(matchedBlock),
    previous_block: summarizeBlock(previousBlock),
    next_block: summarizeBlock(nextBlock),
    nearby_useful_context: contextBlocks.map(summarizeBlock),
    recommended_llm_mode: recommendLlmMode(lastHighlightDebug, matchDebug, matchedBlock),
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

  if (hasCrop && captionBlock) {
    return "image_plus_context";
  }
  if (lastHighlightDebug.type === "text" && hasText && !weakMatch) {
    return "text_context";
  }
  if (blockType === "table" && !weakMatch) {
    return "table_context";
  }
  if (blockType === "formula" && !weakMatch) {
    return "formula_context";
  }
  if (["image", "caption"].includes(blockType) && hasCrop && !weakMatch) {
    return hasContext ? "image_plus_context" : "image_multimodal";
  }
  if (lastHighlightDebug.type === "area" && hasCrop && hasContext && !weakMatch) {
    return "image_plus_context";
  }
  if (hasCrop && weakMatch && hasContext) {
    return "image_multimodal";
  }
  if (hasCrop && weakMatch) {
    return "fallback_image_only";
  }
  if (hasCrop) {
    return "image_multimodal";
  }
  if (hasText) {
    return "text_context";
  }
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
  if (text.length < 120 && /,\s+et al\.|running head|identifying hearing difficulty moments/i.test(text)) return true;
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

function capturedTextForCopy(selectionDebug, lastHighlightDebug) {
  return (
    lastHighlightDebug.rawText ||
    selectionDebug.libraryText ||
    selectionDebug.browserText ||
    ""
  );
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

  if (Math.max(x2, y2) <= 1) {
    return clampedNormalizedRect(x1, y1, x2, y2, pageNumber);
  }
  if (width > 1 && height > 1 && x2 <= width * 1.1 && y2 <= height * 1.1) {
    return clampedNormalizedRect(x1 / width, y1 / height, x2 / width, y2 / height, pageNumber);
  }
  if (Math.max(x2, y2) <= 1000) {
    return clampedNormalizedRect(x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000, pageNumber);
  }
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

function roundNumber(value, precision) {
  const factor = 10 ** precision;
  return Math.round(Number(value) * factor) / factor;
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
  return Number(
    position?.boundingRect?.pageNumber || position?.rects?.[0]?.pageNumber || 1,
  );
}

function isSuspiciousText(text) {
  const normalizedText = normalizeText(text);
  if (!normalizedText) return true;
  if (normalizedText.includes("\uFFFD")) return true;

  const visibleCharacters = [...normalizedText].filter((char) => char.trim()).length;
  const alphanumericCharacters = [...normalizedText].filter((char) =>
    /[A-Za-z0-9]/.test(char),
  ).length;
  return visibleCharacters > 0 && alphanumericCharacters / visibleCharacters < 0.25;
}

function formatJson(value) {
  return JSON.stringify(value ?? null, null, 2);
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

createRoot(document.getElementById("pdf-test-root")).render(<PdfTestApp />);
