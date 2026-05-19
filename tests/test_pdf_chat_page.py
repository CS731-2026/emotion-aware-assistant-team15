import unittest
from pathlib import Path


class PdfChatPageTests(unittest.TestCase):
    def test_pdf_chat_static_page_is_separate_from_pdf_test(self):
        html_path = Path("emotion_aware_assistant/web/static/pdf_chat.html")
        source_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx")
        style_path = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.css")

        self.assertTrue(html_path.exists(), "/pdf-chat should have its own static HTML page.")
        self.assertTrue(source_path.exists(), "/pdf-chat should use a scoped React entry.")
        self.assertTrue(style_path.exists(), "/pdf-chat should keep its layout CSS isolated.")

        html = html_path.read_text(encoding="utf-8")
        source = source_path.read_text(encoding="utf-8")

        self.assertIn('id="pdf-chat-root"', html)
        self.assertIn("/pdf-workspace/pdf-chat.js", html)
        self.assertIn("Paper Reading Assistant", source)
        self.assertNotIn("pdf-test-root", html)
        self.assertNotIn("PDF test viewer loaded", source)

    def test_pdf_chat_has_paper_library_upload_and_cards(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "PaperLibrary",
            "PaperCard",
            "UploadPaperCard",
            "PreparationStatus",
            'fetchJson("/api/documents?library_only=1")',
            'fetch("/api/documents/upload"',
            "pollPreparationStatus",
            "Uploading PDF",
            "Extracting text and layout",
            "Building paper profile",
            "Building keyword index",
            "Building embedding index",
            "Ready",
            "progress_percent",
            "elapsed_seconds",
            "estimated_remaining_seconds",
            "pdf-chat-progress-bar",
            "highlight_count",
            "thread_count",
            "retrieval_method",
            "pdf-chat-library-tools",
            "Open settings",
            "Users can configure API keys and model roles before uploading.",
            "Remove this paper from the library?",
            "`/api/documents/${document.document_id}/archive`",
            "onArchiveDocument",
            "loadDocuments();",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_reuses_pdf_highlighter_and_selection_context_shape(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "PdfLoader",
            "PdfHighlighter",
            "TextHighlight",
            "AreaHighlight",
            "useHighlightContainerContext",
            "enableAreaSelection",
            "viewport_rects",
            "normalized_rects",
            "parser_rects_1000",
            "crop_image_data_url",
            "buildLlmInputPreview",
            'postJson("/api/document/match-blocks"',
            "normalizePdfText",
            "isLowValueContextBlock",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_restores_clickable_area_highlights_with_crop_urls(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")
        styles = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.css").read_text(encoding="utf-8")

        for required in [
            "onClickCapture={handleClick}",
            "cropUrlForHighlight(highlight)",
            "cropImagePath: highlight.crop_image_path || highlight.crop_path || \"\"",
            "crop_url: normalized.crop_url || cropUrl",
            "cropImageUrl: highlight.crop_url || cropUrlForHighlight(highlight)",
            "preview.crop_image_data_url || preview.crop_url",
            "loadThreadForHighlight(normalized.highlight_id)",
        ]:
            self.assertIn(required, source)
        self.assertIn("pointer-events: auto;", styles)

    def test_pdf_chat_workspace_has_clean_chat_panel_and_collapsed_inspector(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "PdfChatWorkspace",
            "ChatSidePanel",
            "CurrentSelectionCard",
            "HighlightThreadView",
            "ContextInspector",
            "Current Selection",
            "selectionKindLabel",
            "Text selection",
            "Area selection",
            "Explain",
            "Conversation",
            "pdf-chat-loading-bubble",
            "Ask a follow-up about this selection",
            "Context used",
            "provider",
            "model",
            "prompt_preview",
            "raw_payload",
            "open={false}",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_chat_panel_uses_safe_markdown_and_anchored_messages(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")
        styles = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.css").read_text(encoding="utf-8")

        for required in [
            "const threadEndRef = useRef(null);",
            "threadEndRef.current?.scrollIntoView",
            "className=\"pdf-chat-side-scroll\"",
            "className=\"pdf-chat-bubble\"",
            "className=\"pdf-chat-message-content\"",
            "function parseMarkdownBlocks",
            "function MarkdownInline",
            "function parseInlineMarkdown",
            "<ul key={index}>",
            "<ol key={index}>",
            "<code key={index}>",
            "<strong key={index}>",
            "<em key={index}>",
        ]:
            self.assertIn(required, source)

        for required_style in [
            ".pdf-chat-side-panel",
            ".pdf-chat-side-scroll",
            ".pdf-chat-message.assistant .pdf-chat-bubble",
            ".pdf-chat-message.user .pdf-chat-bubble",
            ".pdf-chat-markdown ul",
            ".pdf-chat-markdown ol",
            ".pdf-chat-follow-up",
        ]:
            self.assertIn(required_style, styles)

        self.assertNotIn("dangerouslySetInnerHTML", source)

    def test_pdf_chat_selection_preparation_and_paper_cards_are_user_facing(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")
        styles = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.css").read_text(encoding="utf-8")

        for required in [
            "const selectionTitle = `${selectionKindLabel(preview)} · Page ${preview.page_number || \"-\"}`;",
            "pdf-chat-selection-text-preview",
            "Show full selection",
            "Preparation details",
            "pdf-chat-ready-summary",
            "isPreparationComplete",
            "prepared in",
            "<article className=\"pdf-chat-paper-card\">",
            "Technical details",
            "retrieval method",
            "embedding status",
            "parsed blocks",
        ]:
            self.assertIn(required, source)

        for required_style in [
            ".pdf-chat-selection-text-preview",
            "-webkit-line-clamp: 2;",
            ".pdf-chat-preparation.compact",
            ".pdf-chat-ready-summary",
            ".pdf-chat-paper-details",
            ".pdf-chat-paper-open",
        ]:
            self.assertIn(required_style, styles)

    def test_pdf_chat_calls_plural_document_explain_and_thread_routes(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "`/api/documents/${documentId}/file`",
            "`/api/documents/${documentId}/open`",
            "`/api/documents/${documentId}/highlights`",
            "`/api/documents/${documentId}/threads/${highlightId}`",
            "`/api/documents/${documentId}/explain-selection`",
            "`/api/documents/${documentId}/threads/${highlightId}/follow-up`",
            'response_style: "chat_conversational"',
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_guards_persistence_until_document_id_is_stable(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "const documentReady = Boolean(documentId);",
            "Document is still loading; wait for a stable document id before saving highlights.",
            "Document is still loading; wait for a stable document id before explaining.",
            "disabled={!documentReady}",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_learning_signal_panel_is_passive_and_polling_based(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")
        styles = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.css").read_text(encoding="utf-8")

        for required in [
            "LearningSignalPanel",
            "Live learning signal",
            "Document details",
            "Source: simulated camera stream",
            "Model output type:",
            "academic-state model",
            "Raw emotion unavailable for this checkpoint",
            "Raw emotion available",
            "Settings",
            "LLM compare",
            "Camera debug",
            "PDF debug",
            "href=\"/settings\"",
            "href=\"/llm-compare\"",
            "href=\"/camera-debug\"",
            "href=\"/pdf-test\"",
            "Open PDF/RAG debug workspace",
            "Cue:",
            "Face detection:",
            "Camera signal standby",
            "Signal monitoring starts after an explanation is shown.",
            "distribution",
            "`/api/documents/${documentId}/reading-session/start`",
            "`/api/reading-sessions/${readingSessionId}/learning-state/current`",
            "`/api/reading-sessions/${readingSessionId}/events`",
            "window.setInterval",
        ]:
            self.assertIn(required, source)

        for forbidden in [
            "manual academic-state selector",
            "Detected emotion",
            "You are confused",
            "Camera detected",
            "I can see you",
            "Last analyzed frame",
            "OpenFace landmarks",
            "crop_preview_data_url",
            "model_input_preview_data_url",
            "selected_text=${",
        ]:
            self.assertNotIn(forbidden, source)

        self.assertIn(".pdf-chat-learning-signal", styles)
        self.assertIn(".pdf-chat-document-details", styles)
        self.assertIn(".pdf-chat-distribution-bar", styles)

    def test_pdf_chat_can_switch_to_live_webcam_model_signal_without_manual_state_selector(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "modelStatus",
            "learningSignalSource",
            "cameraVideoRef",
            "showSelfView",
            "Show self-view",
            "Hide self-view",
            "Start camera signal",
            "Pause camera signal",
            "Live model unavailable. Using simulated learning signal.",
            "navigator.mediaDevices.getUserMedia",
            "window.setInterval(sendWebcamFrame, 1000)",
            "`/api/reading-sessions/${readingSessionId}/emotion/frame`",
            "setLearningSignalSource(\"webcam\")",
            "sourceLabelForLearningSignal",
            "Source: live webcam model",
            "modelModeLabelForLearningSignal",
            "rawSignalLabelForLearningSignal",
        ]:
            self.assertIn(required, source)

        for forbidden in [
            "manual academic-state selector",
            "choose confusion",
            "choose frustration",
            "Detected emotion",
            "Camera detected",
            "your face shows",
        ]:
            self.assertNotIn(forbidden, source)

    def test_pdf_chat_generates_strategy_cards_after_reaction_window(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")
        styles = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.css").read_text(encoding="utf-8")

        for required in [
            "REACTION_WINDOW_DURATION_MS",
            "startReactionWindow",
            "summarizeReactionWindowSamples",
            "reactionWindowSummary",
            "triggered_by: \"reaction_window\"",
            "baseline_explanation",
            "source_turn_id",
            "strategyCooldownRef",
            "30000",
            "`/api/documents/${documentId}/strategy-candidates`",
            "StrategyCandidatePanel",
            "Suggested ways to improve this explanation",
            "Explain with this strategy",
            "onSelectStrategy",
            "onDismissStrategies",
            "postSessionEvent(\"strategy_selected\"",
            "postSessionEvent(\"reaction_window_started\"",
            "postSessionEvent(\"reaction_window_completed\"",
            "postSessionEvent(\"strategy_candidates_generated\"",
        ]:
            self.assertIn(required, source)

        for required_style in [
            ".pdf-chat-strategy-panel",
            ".pdf-chat-strategy-card",
            ".pdf-chat-strategy-badge",
        ]:
            self.assertIn(required_style, styles)

    def test_pdf_chat_strategy_panel_prioritizes_one_recommended_card(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "normalizeRecommendedCandidatesForDisplay(candidates)",
            "recommendedCandidate",
            "alternativeCandidates",
            "Other ways to explain this",
            "Alternative",
            "pdf-chat-recommended-strategy",
            "pdf-chat-alternative-strategies",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_right_panel_uses_compact_signal_and_sticky_selection_actions(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")
        styles = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.css").read_text(encoding="utf-8")

        for required in [
            "pdf-chat-side-top",
            "pdf-chat-conversation-region",
            "pdf-chat-learning-strip",
            "pdf-chat-signal-summary",
            "pdf-chat-learning-details",
            "DocumentDetailsToggle",
            "Document details",
            "pdf-chat-selection-toolbar",
            "pdf-chat-selection-preview-line",
            "pdf-chat-selection-more",
            "More",
            "Clear conversation",
            "Delete highlight",
            "onClearConversation",
            "onDeleteHighlight",
            "Select text or an area to start an explanation.",
        ]:
            self.assertIn(required, source)

        self.assertIn("<h3>Learning signal</h3>", source)
        self.assertIn('aria-label="Live learning signal"', source)
        self.assertNotIn("<h3>Live learning signal</h3>", source)
        self.assertIn("aria-label=\"Start camera signal\"", source)
        self.assertIn(">Start</button>", source)
        self.assertIn("aria-label=\"Pause camera signal\"", source)
        self.assertIn(">Pause</button>", source)
        self.assertIn("aria-label=\"Signal details\"", source)
        self.assertIn(">Details</summary>", source)
        self.assertNotIn("</LearningSignalPanel>\n        <DocumentDetailsToggle", source)
        self.assertNotIn("pdf-chat-signal-chips", source)
        self.assertNotIn("Current support cue:", source)
        self.assertIn("pdf-chat-signal-line", source)
        self.assertIn("Cue: {cueLabel}", source)
        self.assertIn("<section className=\"pdf-chat-signal-details-section\">", source)
        self.assertIn("<h4>Signal details</h4>", source)
        self.assertIn("<h4>Document details</h4>", source)

        for required_style in [
            "display: flex;",
            "flex-direction: column;",
            "container-type: inline-size;",
            ".pdf-chat-conversation-region",
            "flex: 1 1 auto;",
            ".pdf-chat-side-top",
            ".pdf-chat-selection-toolbar",
            ".pdf-chat-learning-strip",
            ".pdf-chat-signal-main",
            ".pdf-chat-signal-line",
            ".pdf-chat-follow-up.compact",
            "@container (max-width: 380px)",
        ]:
            self.assertIn(required_style, styles)

        self.assertNotIn("className=\"pdf-chat-selection-card pdf-chat-selection-action-bar\"", source)
        self.assertNotIn("<CurrentStrategyLabel", source)

    def test_pdf_chat_right_panel_responsive_css_prevents_overflow_and_overlap(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")
        styles = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.css").read_text(encoding="utf-8")

        for required in [
            'className={`pdf-chat-selection-main ${isArea && cropThumbnail ? "has-thumb" : ""}`}',
            "pdf-chat-document-details-popover",
            "pdf-chat-signal-details-popover",
        ]:
            self.assertIn(required, source)

        for required_style in [
            ".pdf-chat-selection-main {",
            "grid-template-columns: minmax(0, 1fr);",
            ".pdf-chat-selection-main.has-thumb",
            ".pdf-chat-selection-main > div",
            ".pdf-chat-selection-preview-line",
            "overflow: hidden;",
            "text-overflow: ellipsis;",
            "white-space: nowrap;",
            ".pdf-chat-bubble",
            "max-width: 100%;",
            ".pdf-chat-message.assistant .pdf-chat-bubble",
            "width: 100%;",
            ".pdf-chat-reader-pane .PdfHighlighter__tip-container",
            "z-index: 4;",
        ]:
            self.assertIn(required_style, styles)

    def test_pdf_chat_strategy_suggestions_attach_to_source_conversation_turn(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "strategySourceTurnId",
            "TurnStrategySuggestions",
            "turn.turn_id === strategySourceTurnId",
            "StrategyCandidatePanel",
            "Suggested ways to improve this explanation",
            "Other ways to explain this",
            "planner_input_summary",
            "selected_text_length",
            "baseline_explanation_length",
            "reaction_window_duration_sec",
        ]:
            self.assertIn(required, source)

        self.assertNotIn("<StrategyCandidatePanel\n          candidates={strategyCandidates}", source)

    def test_pdf_chat_cleanup_controls_call_document_cleanup_endpoints(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "async function handleDeleteHighlight()",
            "async function handleClearConversation()",
            "async function handleDeleteCurrentTurn(turnId)",
            "Delete this highlight and its conversation?",
            "Clear the conversation for this highlight?",
            "Delete this conversation turn?",
            "`/api/documents/${documentId}/highlights/${highlightId}`",
            "`/api/documents/${documentId}/threads/${highlightId}/clear`",
            "`/api/documents/${documentId}/threads/${highlightId}/turns/${turnId}`",
            "onDeleteTurn",
            "Delete turn",
        ]:
            self.assertIn(required, source)

        highlight_delete = source.index("const payload = await deleteJson(`/api/documents/${documentId}/highlights/${highlightId}`);")
        highlight_update = source.index("setHighlights((payload.highlights || []).map(toViewerHighlight));", highlight_delete)
        turn_delete = source.index("const payload = await deleteJson(`/api/documents/${documentId}/threads/${highlightId}/turns/${turnId}`);")
        turn_update = source.index("setThread(payload.thread || null);", turn_delete)
        self.assertLess(highlight_delete, highlight_update)
        self.assertLess(turn_delete, turn_update)
        self.assertIn("setError(err?.message || String(err));", source)

    def test_pdf_chat_strategy_trace_keeps_answer_outside_collapsed_details(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        strategy_badge_call = source.index("<ConversationStrategyBadge")
        answer_index = source.index("className=\"pdf-chat-message-content\"", strategy_badge_call)
        self.assertLess(strategy_badge_call, answer_index)
        self.assertIn("planner_input_summary: message.planner_input_summary || {}", source)
        self.assertIn("<MarkdownText content={message.content || \"\"} />", source)

    def test_pdf_chat_sends_selected_strategy_to_explain_and_follow_up(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "selectedStrategy",
            "selected_strategy_id: selectedStrategy?.strategy_id || \"\"",
            "selected_strategy: selectedStrategy || null",
            "strategy_candidates: strategyCandidates",
            "learning_state: learningState",
            "reaction_window_summary: reactionWindowSummary",
            "trigger_context: strategyTriggerContext",
            "Strategy:",
            "FollowUpContextLine",
            "postSessionEvent(\"answer_generated\"",
            "postSessionEvent(\"follow_up_sent\"",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_strategy_card_click_generates_answer_without_follow_up_text(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "async function handleSelectStrategy(candidate, sourceContext = {})",
            "await explainSelection(candidate, \"explain_current_selection_with_selected_strategy\", {",
            "default_task: defaultTask || \"baseline_explain_current_selection\"",
            "user_question: null",
            "answerLoading={explainLoading}",
            "disabled={loading || answerLoading}",
            "Explain with this strategy",
        ]:
            self.assertIn(required, source)

        self.assertNotIn("Use this strategy", source)

    def test_pdf_chat_normal_explain_button_does_not_pass_click_event(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "function handleBaselineExplain()",
            "onClick={() => handleBaselineExplain()}",
            "function isDomOrReactEvent(value)",
            "if (isDomOrReactEvent(strategyOverride))",
            "const effectiveStrategy = isPlainStrategyObject(strategyOverride) ? strategyOverride : null;",
            "const isStrategyExplain = Boolean(effectiveStrategy);",
            "strategy_candidates: isStrategyExplain ? currentStrategyCandidates : [],",
            "trigger_context: isStrategyExplain ? currentTriggerContext : null,",
            "source_turn_id: isStrategyExplain ? currentSourceTurnId : \"\",",
            "reaction_window_summary: isStrategyExplain ? currentReactionSummary : null,",
            "selected_strategy_id: effectiveStrategy?.strategy_id || \"\"",
            "selected_strategy: effectiveStrategy || null",
            "default_task: defaultTask || \"baseline_explain_current_selection\"",
            "turn_type: isStrategyExplain ? \"strategy_reexplanation\" : \"baseline_explanation\"",
            "document_id: documentId",
            "selection_type: activeSelection.llmInputPreview.highlight_type || activeSelection.llmInputPreview.type || \"\"",
            "sanitizeSerializablePayload(",
            "assertJsonSerializablePayload(payload);",
        ]:
            self.assertIn(required, source)

        self.assertNotIn("onClick={onExplainSelection}", source)

    def test_pdf_chat_baseline_explain_normalizes_response_and_renders_next_thread(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "function normalizeExplainSelectionResponse(response)",
            "const normalizedResult = normalizeExplainSelectionResponse(result);",
            "if (!normalizedResult.answerText) {",
            "const responseAssistant = latestAssistantMessage(normalizedResult.thread);",
            "const responseAssistantText = normalizePdfText(responseAssistant?.content || \"\");",
            "const normalizedAnswerText = normalizePdfText(normalizedResult.answerText);",
            "const nextThread = responseAssistantText && responseAssistantText === normalizedAnswerText",
            ": appendAssistantMessage(normalizedResult.thread || threadRef.current, normalizedResult);",
            "persistedThread = await persistThreadForHighlight(nextThread);",
            "setThread(persistedThread);",
            "threadRef.current = persistedThread;",
            "startReactionWindowForAssistantMessage(latestAssistantMessage(persistedThread), selectionSnapshot);",
            "logExplainSelectionDebug(",
            "extracted_answer_length",
            "message_appended",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_persists_explain_thread_and_ignores_stale_empty_loads(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "async function persistThreadForHighlight(thread)",
            "async function persistThreadAfterAssistantMessage(documentId, highlightId, thread)",
            "`/api/documents/${documentId}/threads/${highlightId}`",
            "throw new Error(\"Answer generated, but conversation could not be saved.\");",
            "function threadHasMessages(thread)",
            "function shouldIgnoreEmptyThreadLoad(incomingThread, currentThread, highlightId)",
            "if (shouldIgnoreEmptyThreadLoad(payload, threadRef.current, highlightId)) return;",
            "if (shouldIgnoreEmptyThreadPersist(thread, threadRef.current, highlightId)) return threadRef.current;",
            "threadRef.current = payload;",
            "persistedThread = await persistThreadForHighlight(nextThread);",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_restored_highlight_click_uses_stable_highlight_id_and_request_guard(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "function normalizePersistedHighlight(highlight, documentId, source = \"restored\")",
            "id: highlightId,",
            "highlight_id: highlightId,",
            "async function activatePersistedHighlight(highlight)",
            "const normalized = normalizePersistedHighlight(highlight, documentId, \"restored\");",
            "await loadThreadForHighlight(normalized.highlight_id);",
            "return activatePersistedHighlight(highlight);",
            "const latestThreadRequestRef = useRef(0);",
            "const requestId = latestThreadRequestRef.current + 1;",
            "if (latestThreadRequestRef.current !== requestId) return;",
            "if (activeSelectionRef.current?.llmInputPreview?.highlight_id !== highlightId) return;",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_highlight_restore_diagnostics_and_stale_highlight_guard_exist(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "function logPdfChatRestoreDebug(event, payload)",
            "event: \"highlights_loaded\"",
            "event: \"highlight_click\"",
            "event: \"thread_loaded\"",
            "thread_fetch_url: `/api/documents/${documentId}/threads/${highlightId}`",
            "function shouldIgnoreEmptyHighlightsLoad(incomingHighlights, currentHighlights, documentId)",
            "if (shouldIgnoreEmptyHighlightsLoad(normalizedHighlights, highlightsRef.current, documentId)) return;",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_reaction_window_starts_only_after_persisted_thread(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        persist_index = source.index("persistedThread = await persistThreadForHighlight(nextThread);")
        set_thread_index = source.index("setThread(persistedThread);", persist_index)
        reaction_index = source.index("startReactionWindowForAssistantMessage(latestAssistantMessage(persistedThread), selectionSnapshot);", set_thread_index)
        self.assertLess(persist_index, set_thread_index)
        self.assertLess(set_thread_index, reaction_index)
        self.assertIn("Answer generated, but conversation could not be saved.", source)

    def test_pdf_chat_normalizer_prefers_stable_answer_contract_and_provider_error(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "raw.answer,",
            "typeof raw.assistant_message === \"string\" ? raw.assistant_message : raw.assistant_message?.content,",
            "typeof raw.message === \"string\" ? raw.message : raw.message?.content,",
            "assistantMessage?.content,",
            "raw.result?.answer,",
            "raw.explanation,",
            "raw.content,",
            "const errorMessage = raw.ok === false || raw.error ? String(raw.error || \"Explanation failed.\") : \"\";",
            "if (normalizedResult.errorMessage) throw new Error(normalizedResult.errorMessage);",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_new_highlight_initializes_empty_thread_without_stale_reload(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "function emptyThreadForSelection(documentId, highlightId, selection)",
            "const initialThread = emptyThreadForSelection(documentId, highlightDebug.id, initialSelection);",
            "setThread(initialThread);",
            "threadRef.current = initialThread;",
            "applyThreadStrategyState(initialThread);",
        ]:
            self.assertIn(required, source)

        self.assertEqual(source.count("await loadThreadForHighlight(normalized.highlight_id);"), 1)

    def test_pdf_chat_strategy_request_requires_completed_reaction_window_fields(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "const resolvedSourceTurnId = sourceTurnId || summary?.source_turn_id || \"\";",
            "const resolvedBaselineExplanation = normalizePdfText(",
            "const resolvedReactionWindowSummary = summary && Object.keys(summary).length ? summary : null;",
            "if (!selection?.llmInputPreview?.highlight_id || !resolvedSourceTurnId || !resolvedBaselineExplanation || !resolvedReactionWindowSummary) {",
            "return;",
            "baseline_explanation: resolvedBaselineExplanation",
            "reaction_window_summary: resolvedReactionWindowSummary",
            "source_turn_id: resolvedSourceTurnId",
            "isReactionWindowValidationError(err)",
            "if (!readingSessionId || !preview.highlight_id || !sourceTurnId || !normalizePdfText(baselineExplanation)) return;",
            "startReactionWindowForAssistantMessage(latestAssistantMessage(persistedThread), selectionSnapshot);",
            "if (completedReactionTurnIdsRef.current.has(sourceTurnId) || hasTurnReactionMetadata(threadRef.current, sourceTurnId)) {",
            "await requestStrategyCandidates(\"reaction_window\", summary, baselineExplanation, sourceTurnId, selection);",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_auto_starts_camera_after_explain_without_page_load_start(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "async function ensureCameraSignalStarted(reason = \"answer_generated\")",
            "const [cameraPausedByUser, setCameraPausedByUser] = useState(false);",
            "const [cameraStartStatus, setCameraStartStatus] = useState(\"standby\");",
            "cameraPausedByUserRef.current",
            "if (cameraPausedByUserRef.current && reason !== \"manual_start\") return \"paused_by_user\";",
            "ensureCameraSignalStarted(\"explain_clicked\")",
            "setCameraStartStatus(\"starting\")",
            "setCameraStartStatus(\"started\")",
            "setShowSelfView(cameraSelfViewPreferenceRef.current !== \"hidden\");",
            "function pauseLiveSignal()",
            "setCameraPausedByUser(true);",
            "Camera paused",
            "Starting live signal",
            "Live signal active",
            "Simulated fallback",
        ]:
            self.assertIn(required, source)

        load_workspace = source[source.index("async function loadWorkspace()"):source.index("useEffect(() => {", source.index("async function loadWorkspace()") + 1)]
        self.assertNotIn("ensureCameraSignalStarted", load_workspace)

    def test_pdf_chat_reaction_windows_restart_for_each_assistant_turn(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "function monitorableAssistantMessage(message)",
            "function startReactionWindowForAssistantMessage(message, selectionOverride = activeSelectionRef.current)",
            "startReactionWindowForAssistantMessage(latestAssistantMessage(persistedThread), selectionSnapshot);",
            "startReactionWindowForAssistantMessage(latestAssistantMessage(nextThread), activeSelectionRef.current);",
            "completedReactionTurnIdsRef.current.has(sourceTurnId)",
            "if (hasTurnStrategyCandidates(threadRef.current, resolvedSourceTurnId) && triggeredBy !== \"manual_refresh\") return;",
            "reactionWindowTurnIdRef.current = sourceTurnId;",
            "turn_type === \"baseline_explanation\"",
            "turn_type === \"strategy_reexplanation\"",
            "turn_type === \"follow_up\"",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_persists_and_renders_strategy_candidates_per_source_turn(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "turn_metadata",
            "function mergeTurnMetadata(thread, turnId, metadata)",
            "function turnMetadataForThread(thread, turnId)",
            "function latestTurnMetadata(thread)",
            "strategyCandidatesForTurn(turn, turnMetadata, candidates, strategySourceTurnId)",
            "turnMetadata={thread?.turn_metadata || {}}",
            "const turnSpecificCandidates = strategyCandidatesForTurn(turn, turnMetadata, candidates, strategySourceTurnId);",
            "const updatedThread = mergeTurnMetadata(threadRef.current, resolvedSourceTurnId, {",
            "strategy_candidates: candidates,",
            "reaction_window_summary: resolvedReactionWindowSummary,",
            "planner_mode: payload.planner_mode || \"\",",
        ]:
            self.assertIn(required, source)

    def test_pdf_chat_renders_strategy_badge_inside_paginated_conversation_turns(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")
        styles = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.css").read_text(encoding="utf-8")

        for required in [
            "deriveConversationTurns",
            "currentTurn",
            "currentTurn.messages.map",
            "ConversationStrategyBadge",
            "Using strategy:",
            "Continuing with strategy:",
            "Why this strategy appeared",
            "Observed window:",
            "Strategy trace",
            "message.reaction_window_summary",
            "Previous",
            "Next",
            "Latest",
            "Turn",
            "setCurrentTurnIndex(Math.max(0, turns.length - 1));",
            "message.strategy_title",
            "message.strategy_short_description",
            "strategyPedagogicalMove(message)",
            "strategyContextFocus(message)",
            "Focus:",
            "pedagogical_move: message.pedagogical_move",
            "context_focus: message.context_focus",
        ]:
            self.assertIn(required, source)

        for required_style in [
            ".pdf-chat-thread-toolbar",
            ".pdf-chat-turn-indicator",
            ".pdf-chat-strategy-message-badge",
        ]:
            self.assertIn(required_style, styles)

    def test_pdf_chat_strategy_cards_prioritize_pedagogical_move_and_focus(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "strategyPedagogicalMove(candidate)",
            "strategyContextFocus(candidate)",
            "<span>Focus</span>",
            "candidate.pedagogical_move",
            "candidate.context_focus",
            "fallbackStrategyFamily",
        ]:
            self.assertIn(required, source)

        self.assertIn("<h4>{move}</h4>", source)

    def test_pdf_chat_follow_up_input_shows_current_selection_context(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        for required in [
            "FollowUpContextLine",
            "Strategy:",
            "pdf-chat-follow-up-context-line",
            "className=\"pdf-chat-follow-up compact\"",
        ]:
            self.assertIn(required, source)

        self.assertNotIn("Follow-up will use the latest selected strategy.", source)
        self.assertNotIn("function CurrentStrategyLabel", source)

    def test_pdf_chat_hides_debug_language_from_normal_chat_surface(self):
        source = Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx").read_text(encoding="utf-8")

        self.assertNotIn("Explanation ready.", source)
        self.assertNotIn("<dt>type</dt>", source)
        self.assertNotIn("<dt>mode</dt>", source)
        self.assertNotIn("<h3>Thread</h3>", source)

    def test_pdf_chat_does_not_reference_old_chat_camera_or_secrets(self):
        paths = [
            Path("emotion_aware_assistant/web/static/pdf_chat.html"),
            Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.jsx"),
            Path("emotion_aware_assistant/web/pdf_workspace/src/pdf_chat.css"),
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.exists())

        for forbidden in [
            "/api/chat",
            "Load Sample",
            "Selected Passage",
            "X-goog-api-key",
            "GEMINI_API_KEY",
            "AI" + "za",
        ]:
            self.assertNotIn(forbidden, combined)


if __name__ == "__main__":
    unittest.main()
