import json
import unittest


class LearningSignalPackageTests(unittest.TestCase):
    def test_learning_signal_package_serializes_process_aware_context(self):
        from emotion_aware_assistant.emotion.learning_signal import (
            AcademicStateMapping,
            LearningProcessContext,
            RawEmotionEvidence,
            ReactionWindowSummary,
            build_learning_signal_package,
        )

        package = build_learning_signal_package(
            raw_emotion_evidence=RawEmotionEvidence(
                source="raw_8class_checkpoint",
                checkpoint_path="models/emotion_model/raw_8class_best.pt",
                raw_emotion_label="fear",
                raw_emotion_confidence=0.62,
                raw_emotion_probabilities={
                    "anger": 0.02,
                    "contempt": 0.03,
                    "disgust": 0.01,
                    "fear": 0.62,
                    "happy": 0.04,
                    "neutral": 0.10,
                    "sad": 0.05,
                    "surprise": 0.13,
                },
                raw_top_emotions=[{"label": "fear", "probability": 0.62}, {"label": "surprise", "probability": 0.13}],
                frame_timestamp="2026-05-21T12:00:00Z",
                crop_hash="abc123",
            ),
            academic_state_mapping=AcademicStateMapping(
                mapping_rule_version="required_8class_probability_aggregation_v1",
                mapped_academic_state="confusion",
                mapped_academic_scores={"frustration": 0.08, "confusion": 0.75, "boredom": 0.03, "engagement": 0.14},
                mapping_explanation="fear + surprise -> confusion",
                confidence_notes=["mapped from raw 8-class probabilities"],
            ),
            reaction_window_summary=ReactionWindowSummary(
                window_start="2026-05-21T12:00:00Z",
                window_end="2026-05-21T12:00:08Z",
                dominant_raw_emotions=[{"label": "fear", "probability": 0.62}],
                dominant_mapped_states=[{"state": "confusion", "probability": 0.75}],
                trend="rising_confusion_like",
                evidence_count=4,
                reliability_notes=["short reaction window"],
                avg_mapped_scores={"frustration": 0.08, "confusion": 0.75, "boredom": 0.03, "engagement": 0.14},
                support_cue="sustained_clarification",
            ),
            learning_process_context=LearningProcessContext(
                document_id="doc1",
                highlight_id="h1",
                turn_index=3,
                followup_count_for_highlight=2,
                same_highlight_repeated_questions=True,
                last_user_question="Can you simplify this?",
                current_user_question="I still do not understand why this works.",
                question_intent="still_confused",
                previous_strategy="simplify_and_define_terms",
                recent_strategies=["baseline_explanation", "simplify_and_define_terms"],
                last_response_was_baseline_or_adaptive="adaptive",
                time_since_last_assistant_answer=4.5,
            ),
            diagnostic_only_direct_4class={"state": "engagement", "used_for_strategy": True},
        )

        payload = package.to_dict()

        self.assertEqual(payload["active_source"], "raw_8class_process_aware")
        self.assertEqual(payload["raw_emotion_evidence"]["source"], "raw_8class_checkpoint")
        self.assertEqual(payload["academic_state_mapping"]["mapped_academic_state"], "confusion")
        self.assertEqual(set(payload["academic_state_scores"]), {"boredom", "confusion", "engagement", "frustration"})
        self.assertEqual(payload["dominant_academic_state"], "confusion")
        self.assertEqual(payload["secondary_academic_state"], "engagement")
        self.assertEqual(payload["support_cue"], "clarification")
        self.assertEqual(payload["learning_process_context"]["question_intent"], "still_confused")
        self.assertEqual(payload["inferred_process_state"], "continued_difficulty")
        self.assertEqual(payload["recommended_strategy"], "worked_example")
        self.assertFalse(payload["diagnostic_only_direct_4class"]["used_for_strategy"])
        self.assertIn("not as a psychological diagnosis", payload["non_diagnostic_disclaimer"])
        json.dumps(payload)

    def test_engagement_academic_state_maps_to_deepening_support_cue_after_baseline(self):
        from emotion_aware_assistant.emotion.learning_signal import (
            AcademicStateMapping,
            LearningProcessContext,
            ReactionWindowSummary,
            build_learning_signal_package,
        )

        package = build_learning_signal_package(
            raw_emotion_evidence=None,
            academic_state_mapping=AcademicStateMapping(
                mapping_rule_version="required_8class_probability_aggregation_v1",
                mapped_academic_state="engagement",
                mapped_academic_scores={"boredom": 0.04, "confusion": 0.10, "engagement": 0.78, "frustration": 0.08},
                mapping_explanation="happy + neutral -> engagement",
                confidence_notes=[],
            ),
            reaction_window_summary=ReactionWindowSummary(
                window_start="2026-05-21T12:00:00Z",
                window_end="2026-05-21T12:00:08Z",
                dominant_raw_emotions=[],
                dominant_mapped_states=[{"state": "engagement", "probability": 0.78}],
                trend="stable_engagement_like",
                evidence_count=4,
                reliability_notes=[],
                avg_mapped_scores={"boredom": 0.04, "confusion": 0.10, "engagement": 0.78, "frustration": 0.08},
                support_cue="deepening",
            ),
            learning_process_context=LearningProcessContext(
                document_id="doc1",
                highlight_id="h1",
                turn_index=1,
                followup_count_for_highlight=0,
                same_highlight_repeated_questions=False,
                last_user_question=None,
                current_user_question=None,
                question_intent="general_followup",
                previous_strategy=None,
                recent_strategies=[],
                last_response_was_baseline_or_adaptive="baseline",
                time_since_last_assistant_answer=3.0,
            ),
        ).to_dict()

        self.assertEqual(package["dominant_academic_state"], "engagement")
        self.assertEqual(package["support_cue"], "deepening")
        self.assertEqual(package["inferred_process_state"], "engaged_deepening")
        self.assertEqual(package["recommended_strategy"], "deepen_or_extend")
        self.assertNotEqual(package["recommended_strategy"], "baseline_explanation")

    def test_reaction_context_after_baseline_uses_adaptive_fallback_not_baseline_ready(self):
        from emotion_aware_assistant.emotion.learning_signal import (
            AcademicStateMapping,
            LearningProcessContext,
            ReactionWindowSummary,
            build_learning_signal_package,
        )

        package = build_learning_signal_package(
            raw_emotion_evidence=None,
            academic_state_mapping=AcademicStateMapping(
                mapping_rule_version="required_8class_probability_aggregation_v1",
                mapped_academic_state="uncertain",
                mapped_academic_scores={"boredom": 0.24, "confusion": 0.26, "engagement": 0.25, "frustration": 0.25},
                mapping_explanation="mixed academic-state evidence",
                confidence_notes=["flat distribution"],
            ),
            reaction_window_summary=ReactionWindowSummary(
                window_start="2026-05-21T12:00:00Z",
                window_end="2026-05-21T12:00:08Z",
                dominant_raw_emotions=[],
                dominant_mapped_states=[],
                trend="stable",
                evidence_count=4,
                reliability_notes=[],
                avg_mapped_scores={"boredom": 0.24, "confusion": 0.26, "engagement": 0.25, "frustration": 0.25},
                support_cue="neutral_or_uncertain",
            ),
            learning_process_context=LearningProcessContext(
                document_id="doc1",
                highlight_id="h1",
                turn_index=1,
                followup_count_for_highlight=0,
                same_highlight_repeated_questions=False,
                last_user_question=None,
                current_user_question=None,
                question_intent="general_followup",
                previous_strategy=None,
                recent_strategies=["baseline_explanation"],
                last_response_was_baseline_or_adaptive="baseline",
                time_since_last_assistant_answer=5.0,
            ),
        ).to_dict()

        self.assertEqual(package["inferred_process_state"], "adaptive_uncertain_or_mixed")
        self.assertNotEqual(package["inferred_process_state"], "baseline_ready")
        self.assertNotEqual(package["recommended_strategy"], "baseline_explanation")

    def test_reaction_support_cues_drive_process_state_names(self):
        from emotion_aware_assistant.emotion.learning_signal import (
            AcademicStateMapping,
            LearningProcessContext,
            ReactionWindowSummary,
            build_learning_signal_package,
        )

        cases = [
            ("confusion", {"boredom": 0.06, "confusion": 0.70, "engagement": 0.16, "frustration": 0.08}, "sustained_clarification", "clarification_needed"),
            ("frustration", {"boredom": 0.04, "confusion": 0.16, "engagement": 0.12, "frustration": 0.68}, "reduce_load", "continued_difficulty"),
            ("boredom", {"boredom": 0.66, "confusion": 0.12, "engagement": 0.16, "frustration": 0.06}, "re_engagement", "possible_low_engagement"),
        ]

        for dominant_state, scores, support_cue, expected_process_state in cases:
            with self.subTest(support_cue=support_cue):
                package = build_learning_signal_package(
                    raw_emotion_evidence=None,
                    academic_state_mapping=AcademicStateMapping(
                        mapping_rule_version="required_8class_probability_aggregation_v1",
                        mapped_academic_state=dominant_state,
                        mapped_academic_scores=scores,
                        mapping_explanation="reaction-window academic state evidence",
                        confidence_notes=[],
                    ),
                    reaction_window_summary=ReactionWindowSummary(
                        window_start="2026-05-21T12:00:00Z",
                        window_end="2026-05-21T12:00:08Z",
                        dominant_raw_emotions=[],
                        dominant_mapped_states=[{"state": dominant_state, "probability": scores[dominant_state]}],
                        trend="stable",
                        evidence_count=4,
                        reliability_notes=[],
                        avg_mapped_scores=scores,
                        support_cue=support_cue,
                    ),
                    learning_process_context=LearningProcessContext(
                        document_id="doc1",
                        highlight_id="h1",
                        turn_index=2,
                        followup_count_for_highlight=0,
                        same_highlight_repeated_questions=False,
                        last_user_question=None,
                        current_user_question=None,
                        question_intent="general_followup",
                        previous_strategy=None,
                        recent_strategies=["baseline_explanation"],
                        last_response_was_baseline_or_adaptive="baseline",
                        time_since_last_assistant_answer=5.0,
                    ),
                ).to_dict()

                self.assertEqual(package["dominant_academic_state"], dominant_state)
                self.assertEqual(package["inferred_process_state"], expected_process_state)
                self.assertNotEqual(package["recommended_strategy"], "baseline_explanation")

    def test_strategy_reason_text_explains_engagement_evidence_for_deepening(self):
        from emotion_aware_assistant.emotion.learning_signal import build_strategy_reason_text

        reason = build_strategy_reason_text(
            source_turn_type="baseline_explanation",
            academic_state_scores={"boredom": 0.036, "confusion": 0.031, "engagement": 0.869, "frustration": 0.064},
            dominant_academic_state="engagement",
            secondary_academic_state="frustration",
            support_cue="deepening",
            trend="rising",
            avg_confidence=0.869,
            recommended_strategy="deepen_or_extend",
            pedagogical_move="Deepen the technical explanation",
        )

        self.assertIn("baseline explanation", reason["reason_text"])
        self.assertIn("engagement-dominant", reason["reason_text"])
        self.assertIn("engagement 87%", reason["reason_text"])
        self.assertIn("secondary frustration 6%", reason["reason_text"])
        self.assertIn("rising trend", reason["reason_text"])
        self.assertIn("deeper explanation strategy", reason["reason_text"])
        self.assertEqual(reason["academic_state_evidence_text"], "engagement 87%, frustration 6% · trend rising")
        self.assertEqual(reason["support_cue_label"], "deepening")

    def test_strategy_reason_text_treats_falling_frustration_with_close_engagement_as_resolving(self):
        from emotion_aware_assistant.emotion.learning_signal import build_strategy_reason_text

        reason = build_strategy_reason_text(
            source_turn_type="baseline_explanation",
            academic_state_scores={"boredom": 0.0, "confusion": 0.01, "engagement": 0.45, "frustration": 0.54},
            dominant_academic_state="frustration",
            secondary_academic_state="engagement",
            support_cue="deepening",
            trend="falling",
            avg_confidence=0.54,
            recommended_strategy="deepen_or_extend",
            pedagogical_move="Deepen the technical explanation",
        )

        self.assertIn("frustration-dominant", reason["reason_text"])
        self.assertIn("frustration 54%", reason["reason_text"])
        self.assertIn("secondary engagement 45%", reason["reason_text"])
        self.assertIn("difficulty signal was falling", reason["reason_text"])
        self.assertIn("engagement was close behind", reason["reason_text"])
        self.assertIn("earlier difficulty may be resolving", reason["reason_text"])
        self.assertIn("cautious deepening strategy", reason["reason_text"])
        self.assertNotIn("answer could be extended", reason["reason_text"])
        forbidden = reason["reason_text"].lower()
        for phrase in ["you are", "camera detected", "face showed"]:
            self.assertNotIn(phrase, forbidden)

    def test_strategy_reason_text_avoids_simple_extension_claim_for_non_resolving_frustration_dominant_deepening(self):
        from emotion_aware_assistant.emotion.learning_signal import build_strategy_reason_text

        reason = build_strategy_reason_text(
            source_turn_type="baseline_explanation",
            academic_state_scores={"boredom": 0.02, "confusion": 0.04, "engagement": 0.32, "frustration": 0.62},
            dominant_academic_state="frustration",
            secondary_academic_state="engagement",
            support_cue="deepening",
            trend="rising",
            avg_confidence=0.62,
            recommended_strategy="deepen_or_extend",
            pedagogical_move="Deepen the technical explanation",
        )

        self.assertIn("frustration-dominant", reason["reason_text"])
        self.assertNotIn("answer could be extended", reason["reason_text"])
        self.assertIn("cautious continuation", reason["reason_text"])

    def test_strategy_reason_text_keeps_low_confidence_separate_from_difficulty_cue(self):
        from emotion_aware_assistant.emotion.learning_signal import build_strategy_reason_text

        reason = build_strategy_reason_text(
            source_turn_type="strategy_reexplanation",
            academic_state_scores={"boredom": 0.0, "confusion": 0.0, "engagement": 0.206, "frustration": 0.789},
            dominant_academic_state="frustration",
            secondary_academic_state="engagement",
            support_cue="difficulty_support",
            trend="rising",
            avg_confidence=0.406,
            recommended_strategy="step_by_step_decomposition",
            pedagogical_move="Walk through the passage step by step",
        )

        self.assertIn("previous adaptive explanation", reason["reason_text"])
        self.assertIn("frustration-dominant", reason["reason_text"])
        self.assertIn("frustration 79%", reason["reason_text"])
        self.assertIn("secondary engagement 21%", reason["reason_text"])
        self.assertIn("average confidence was low", reason["reason_text"])
        self.assertIn("step-by-step strategy to reduce cognitive load", reason["reason_text"])
        self.assertEqual(reason["confidence_handling"], "low_confidence")
        self.assertEqual(reason["support_cue_label"], "reduce cognitive load")
        self.assertEqual(
            reason["academic_state_evidence_text"],
            "frustration 79%, engagement 21% · low-confidence signal · trend rising",
        )
        forbidden = reason["reason_text"].lower()
        for phrase in ["you are", "camera detected", "face showed", "sad", "anger", "disgust", "fear", "surprise"]:
            self.assertNotIn(phrase, forbidden)

    def test_question_intent_detection_is_conservative(self):
        from emotion_aware_assistant.emotion.learning_signal import detect_question_intent

        self.assertEqual(detect_question_intent("Can you simplify this term?"), "simplify")
        self.assertEqual(detect_question_intent("Give me an example."), "example")
        self.assertEqual(detect_question_intent("How does this compare with BERT?"), "compare")
        self.assertEqual(detect_question_intent("Why does this work?"), "why_how")
        self.assertEqual(detect_question_intent("I still do not understand."), "still_confused")
        self.assertEqual(detect_question_intent("Is this assumption valid?"), "critique")
        self.assertEqual(detect_question_intent("Can you explain this?"), "explain")
        self.assertEqual(detect_question_intent("What about the next sentence?"), "general_followup")

    def test_provider_prompt_includes_internal_learning_signal_safety(self):
        from emotion_aware_assistant.llm.providers import build_prompt

        package = {
            "active_source": "raw_8class_process_aware",
            "inferred_process_state": "frustration_like_difficulty",
            "recommended_strategy": "step_by_step_decomposition",
            "strategy_reason": "recent learning signal suggested continued difficulty after the previous explanation",
            "prompt_guidance": ["Use calm, supportive, short steps."],
            "learning_process_context": {"question_intent": "still_confused", "followup_count_for_highlight": 2},
            "academic_state_scores": {"boredom": 0.0, "confusion": 0.0, "engagement": 0.0, "frustration": 0.72},
            "dominant_academic_state": "frustration",
            "support_cue": "difficulty_support",
            "academic_state_mapping": {"mapped_academic_state": "frustration", "mapped_academic_scores": {"frustration": 0.72}},
            "raw_emotion_evidence": {"raw_top_emotions": [{"label": "sad", "probability": 0.4}]},
            "non_diagnostic_disclaimer": "Use this as a lightweight learning-support signal, not as a psychological diagnosis.",
        }

        prompt = build_prompt(
            {
                "highlight_type": "text",
                "selected_text": "The model retrieves paper evidence before answering.",
                "text_available": True,
                "response_style": "chat_conversational",
                "learning_signal_package": package,
                "selected_strategy": {
                    "strategy_id": "step_by_step_breakdown",
                    "strategy_family": "step_by_step_breakdown",
                    "pedagogical_move": "Walk through the passage step by step",
                    "context_focus": "retrieval before answer generation",
                    "title": "Walk through the passage step by step",
                    "why_recommended": "recent learning signal suggested continued difficulty after the previous explanation",
                    "prompt_instruction": "Use short numbered steps.",
                    "expected_answer_shape": ["Main idea", "Steps"],
                },
                "default_task": "explain_current_selection_with_selected_strategy",
            }
        )

        self.assertIn("Internal learning-support signal", prompt)
        self.assertIn("Use this as a lightweight learning-support signal, not as a psychological diagnosis.", prompt)
        self.assertIn("Do not mention camera, face analysis, raw emotion labels, or detected emotion unless the user explicitly asks.", prompt)
        self.assertIn("Raw expression evidence:", prompt)
        self.assertIn("sad 0.40", prompt)
        self.assertIn("Academic-state evidence:", prompt)
        self.assertIn("frustration 0.72", prompt)
        self.assertIn("Support cue: difficulty_support", prompt)
        self.assertIn("frustration-like difficulty", prompt)
        self.assertNotIn("you are sad", prompt.lower())
        self.assertNotIn("you are frustrated", prompt.lower())
