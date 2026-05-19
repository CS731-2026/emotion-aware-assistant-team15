import contextlib
import io
import json
import os
import stat
import subprocess
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest.mock import patch


class LocalEnvConfigTests(unittest.TestCase):
    def test_env_local_loader_loads_supported_values_without_overriding_existing_env(self):
        from emotion_aware_assistant.core.config import load_env_file

        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env.local"
            env_path.write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=gemini",
                        "GEMINI_MODEL=gemini-flash-latest",
                        "GEMINI_EMBEDDING_MODEL=gemini-embedding-001",
                        "STRATEGY_PLANNER_PROVIDER=gemini",
                        "GEMINI_API_" + "KEY=local-secret-value",
                    ]
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            with patch.dict(os.environ, {"GEMINI_MODEL": "already-set"}, clear=True):
                with contextlib.redirect_stdout(stdout):
                    result = load_env_file(env_path)

                self.assertTrue(result["present"])
                self.assertEqual(os.environ["LLM_PROVIDER"], "gemini")
                self.assertEqual(os.environ["GEMINI_MODEL"], "already-set")
                self.assertEqual(os.environ["GEMINI_API_KEY"], "local-secret-value")
                self.assertIn("GEMINI_API_KEY", result["loaded_keys"])
                self.assertNotIn("local-secret-value", stdout.getvalue())

    def test_configure_api_key_writes_env_local_gitignore_and_safe_permissions(self):
        from scripts.configure_api_key import configure_gemini_key

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env.local").write_text("UNRELATED=value\nGEMINI_API_KEY=old-key\n", encoding="utf-8")
            (root / ".gitignore").write_text("runtime_uploads/\n", encoding="utf-8")
            secret = "new-" + "gemini-" + "key"
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                result = configure_gemini_key(root, secret)

            env_text = (root / ".env.local").read_text(encoding="utf-8")
            gitignore_text = (root / ".gitignore").read_text(encoding="utf-8")
            mode = stat.S_IMODE((root / ".env.local").stat().st_mode)

            self.assertTrue(result["updated_existing_key"])
            self.assertIn("UNRELATED=value", env_text)
            self.assertIn("GEMINI_API_KEY=" + secret, env_text)
            self.assertNotIn("old-key", env_text)
            self.assertIn("LLM_PROVIDER=gemini", env_text)
            self.assertIn("GEMINI_MODEL=gemini-flash-latest", env_text)
            self.assertIn("GEMINI_EMBEDDING_MODEL=gemini-embedding-001", env_text)
            self.assertIn("STRATEGY_PLANNER_PROVIDER=gemini", env_text)
            self.assertIn(".env.local", gitignore_text)
            self.assertEqual(mode, 0o600)
            self.assertNotIn(secret, stdout.getvalue())
            self.assertIn("Updated GEMINI_API_KEY in .env.local", stdout.getvalue())

    def test_diagnose_environment_reports_key_presence_without_printing_key(self):
        from scripts.diagnose_environment import environment_config_status

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            secret = "diagnostic-" + "secret"
            (root / ".env.local").write_text(
                f"LLM_PROVIDER=gemini\nSTRATEGY_PLANNER_PROVIDER=gemini\nGEMINI_API_KEY={secret}\n",
                encoding="utf-8",
            )

            status = environment_config_status(root)
            output = "\n".join(f"{key}: {value}" for key, value in status.items())

            self.assertTrue(status["env_local_present"])
            self.assertTrue(status["gemini_api_key_configured"])
            self.assertEqual(status["llm_provider"], "gemini")
            self.assertEqual(status["strategy_planner_provider"], "gemini")
            self.assertNotIn(secret, output)

    def test_web_status_reports_provider_configuration_without_secret_value(self):
        from emotion_aware_assistant.web.server import create_web_app

        secret = "status-" + "secret"
        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "gemini",
                "STRATEGY_PLANNER_PROVIDER": "gemini",
                "GEMINI_API_KEY": secret,
            },
            clear=True,
        ):
            app = create_web_app(force_dummy_llm=True, load_local_env=False)
            response = app.test_request("GET", "/api/status")

        payload = response["json"]
        serialized = repr(payload)
        self.assertEqual(response["status"], 200)
        self.assertTrue(payload["llm_provider_configured"])
        self.assertTrue(payload["strategy_planner_provider_configured"])
        self.assertNotIn(secret, serialized)

    def test_web_app_startup_attempts_to_load_project_env_local_by_default(self):
        import emotion_aware_assistant.web.server as server

        calls = []

        def fake_loader():
            calls.append(".env.local")
            return {"present": False, "loaded_keys": [], "skipped_existing_keys": []}

        with patch.object(server, "load_project_local_env", fake_loader):
            server.create_web_app(force_dummy_llm=True, load_local_env=True)

        self.assertEqual(calls, [".env.local"])

    def test_run_web_server_loads_env_local_before_serving(self):
        import emotion_aware_assistant.web.server as server

        calls = []

        class FakeServer:
            server_address = ("127.0.0.1", 8000)

            def serve_forever(self):
                raise KeyboardInterrupt()

            def server_close(self):
                calls.append("closed")

        def fake_loader():
            calls.append("loaded")
            return {"present": True, "loaded_keys": ["LLM_PROVIDER"], "skipped_existing_keys": []}

        with (
            patch.object(server, "load_project_local_env", fake_loader),
            patch.object(server, "create_web_app", lambda **kwargs: object()),
            patch.object(server, "_bind_server", lambda *args, **kwargs: FakeServer()),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(KeyboardInterrupt),
        ):
            server.run_web_server()

        self.assertEqual(calls, ["loaded", "closed"])

    def test_local_config_status_masks_key_and_returns_safe_provider_status(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            secret = "AI" + "za" + "localwebsecret1234"
            (root / ".env.local").write_text(
                f"LLM_PROVIDER=gemini\nSTRATEGY_PLANNER_PROVIDER=gemini\nGEMINI_API_KEY={secret}\n",
                encoding="utf-8",
            )
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                response = app.test_request("GET", "/api/local-config/status")

            payload = response["json"]
            serialized = json.dumps(payload)
            self.assertEqual(response["status"], 200)
            self.assertTrue(payload["env_local_present"])
            self.assertTrue(payload["gemini_api_key_configured"])
            self.assertEqual(payload["llm_provider"], "gemini")
            self.assertEqual(payload["strategy_planner_provider"], "gemini")
            self.assertIn("masked_key", payload)
            self.assertNotEqual(payload["masked_key"], secret)
            self.assertNotIn(secret, serialized)

    def test_local_config_status_includes_safe_face_crop_settings(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env.local").write_text(
                "\n".join(
                    [
                        "FACE_CROP_SCALE=2.1",
                        "FACE_CROP_Y_BIAS=0.24",
                        "FACE_CROP_BOTTOM_EXTRA=0.34",
                        "FACE_CROP_MAKE_SQUARE=false",
                        "GEMINI_API_KEY=secret-value",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                response = app.test_request("GET", "/api/local-config/status")

        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertEqual(response["status"], 200)
        self.assertEqual(payload["crop_scale"], 2.1)
        self.assertEqual(payload["crop_y_bias"], 0.24)
        self.assertEqual(payload["crop_bottom_extra"], 0.34)
        self.assertFalse(payload["crop_make_square"])
        self.assertNotIn("secret-value", serialized)

    def test_local_config_save_writes_env_local_updates_process_env_without_returning_key(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env.local").write_text("UNRELATED=value\nGEMINI_API_KEY=old-key\n", encoding="utf-8")
            (root / ".gitignore").write_text("runtime_uploads/\n", encoding="utf-8")
            secret = "AI" + "za" + "postedlocal5678"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                response = app.test_request(
                    "POST",
                    "/api/local-config/gemini",
                    {
                        "gemini_api_key": secret,
                        "gemini_model": "gemini-flash-latest",
                        "gemini_embedding_model": "gemini-embedding-001",
                        "strategy_planner_provider": "gemini",
                    },
                )
                process_key = os.environ.get("GEMINI_API_KEY")

            payload = response["json"]
            serialized = json.dumps(payload)
            env_text = (root / ".env.local").read_text(encoding="utf-8")
            gitignore_text = (root / ".gitignore").read_text(encoding="utf-8")
            mode = stat.S_IMODE((root / ".env.local").stat().st_mode)

            self.assertEqual(response["status"], 200)
            self.assertTrue(payload["saved"])
            self.assertFalse(payload["restart_required"])
            self.assertTrue(payload["gemini_api_key_configured"])
            self.assertEqual(process_key, secret)
            self.assertIn("UNRELATED=value", env_text)
            self.assertIn("GEMINI_API_KEY=" + secret, env_text)
            self.assertNotIn("old-key", env_text)
            self.assertIn("LLM_PROVIDER=gemini", env_text)
            self.assertIn("GEMINI_MODEL=gemini-flash-latest", env_text)
            self.assertIn("GEMINI_EMBEDDING_MODEL=gemini-embedding-001", env_text)
            self.assertIn("STRATEGY_PLANNER_PROVIDER=gemini", env_text)
            self.assertIn(".env.local", gitignore_text)
            self.assertEqual(mode, 0o600)
            self.assertNotIn(secret, serialized)

    def test_llm_status_returns_provider_roles_and_masks_all_keys(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            gemini_secret = "AI" + "za" + "settings-secret-1234"
            openrouter_secret = "or-secret-5678"
            openai_secret = "sk-openai-compatible"
            (root / ".env.local").write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openrouter",
                        "LLM_MODEL=openai/gpt-4o-mini",
                        "STRATEGY_PLANNER_PROVIDER=gemini",
                        "STRATEGY_PLANNER_MODEL=gemini-flash-latest",
                        "EMBEDDING_PROVIDER=gemini",
                        "EMBEDDING_MODEL=gemini-embedding-001",
                        "GEMINI_API_KEY=" + gemini_secret,
                        "OPENROUTER_API_KEY=" + openrouter_secret,
                        "OPENAI_API_KEY=" + openai_secret,
                        "OPENAI_BASE_URL=http://localhost:11434/v1",
                        "OPENAI_MODEL=local/model",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                response = app.test_request("GET", "/api/local-config/llm/status")

        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertEqual(response["status"], 200, response)
        self.assertTrue(payload["providers"]["gemini"]["configured"])
        self.assertTrue(payload["providers"]["openrouter"]["configured"])
        self.assertTrue(payload["providers"]["openai_compatible"]["configured"])
        self.assertEqual(payload["roles"]["answer_model"]["provider"], "openrouter")
        self.assertEqual(payload["roles"]["answer_model"]["model"], "openai/gpt-4o-mini")
        self.assertEqual(payload["roles"]["strategy_planner_model"]["provider"], "gemini")
        self.assertEqual(payload["roles"]["embedding_model"]["model"], "gemini-embedding-001")
        self.assertIn("comparison_models", payload)
        for secret in (gemini_secret, openrouter_secret, openai_secret):
            self.assertNotIn(secret, serialized)

    def test_llm_provider_endpoint_writes_env_local_and_updates_process_env_safely(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env.local").write_text("UNRELATED=value\nOPENROUTER_API_KEY=old\n", encoding="utf-8")
            (root / ".gitignore").write_text("runtime_uploads/\n", encoding="utf-8")
            secret = "or-new-secret"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                response = app.test_request(
                    "POST",
                    "/api/local-config/llm/provider",
                    {
                        "provider": "openrouter",
                        "api_key": secret,
                        "default_model": "openai/gpt-4o-mini",
                        "site_url": "http://localhost:8000",
                        "site_name": "CS731 Local Assistant",
                    },
                )
                process_values = {
                    "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
                    "OPENROUTER_MODEL": os.environ.get("OPENROUTER_MODEL"),
                    "OPENROUTER_SITE_URL": os.environ.get("OPENROUTER_SITE_URL"),
                    "OPENROUTER_SITE_NAME": os.environ.get("OPENROUTER_SITE_NAME"),
                }

            payload = response["json"]
            serialized = json.dumps(payload)
            env_text = (root / ".env.local").read_text(encoding="utf-8")
            mode = stat.S_IMODE((root / ".env.local").stat().st_mode)

        self.assertEqual(response["status"], 200, response)
        self.assertTrue(payload["saved"])
        self.assertEqual(process_values["OPENROUTER_API_KEY"], secret)
        self.assertEqual(process_values["OPENROUTER_MODEL"], "openai/gpt-4o-mini")
        self.assertEqual(process_values["OPENROUTER_SITE_URL"], "http://localhost:8000")
        self.assertEqual(process_values["OPENROUTER_SITE_NAME"], "CS731 Local Assistant")
        self.assertIn("UNRELATED=value", env_text)
        self.assertIn("OPENROUTER_API_KEY=" + secret, env_text)
        self.assertIn("OPENROUTER_MODEL=openai/gpt-4o-mini", env_text)
        self.assertIn("OPENROUTER_SITE_URL=http://localhost:8000", env_text)
        self.assertIn("OPENROUTER_SITE_NAME=CS731 Local Assistant", env_text)
        self.assertNotIn("old", env_text)
        self.assertNotIn(secret, serialized)
        self.assertEqual(mode, 0o600)

    def test_llm_roles_endpoint_updates_role_env_and_warns_on_unsupported_embedding_provider(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env.local").write_text("UNRELATED=value\n", encoding="utf-8")
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                response = app.test_request(
                    "POST",
                    "/api/local-config/llm/roles",
                    {
                        "answer_model": {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
                        "strategy_planner_model": {"provider": "gemini", "model": "gemini-flash-latest"},
                        "embedding_model": {"provider": "openrouter", "model": "not-an-embedding-model"},
                    },
                )
                process_values = {
                    "LLM_PROVIDER": os.environ.get("LLM_PROVIDER"),
                    "LLM_MODEL": os.environ.get("LLM_MODEL"),
                    "STRATEGY_PLANNER_PROVIDER": os.environ.get("STRATEGY_PLANNER_PROVIDER"),
                    "STRATEGY_PLANNER_MODEL": os.environ.get("STRATEGY_PLANNER_MODEL"),
                    "EMBEDDING_PROVIDER": os.environ.get("EMBEDDING_PROVIDER"),
                    "EMBEDDING_MODEL": os.environ.get("EMBEDDING_MODEL"),
                }

            payload = response["json"]
            env_text = (root / ".env.local").read_text(encoding="utf-8")

        self.assertEqual(response["status"], 200, response)
        self.assertTrue(payload["saved"])
        self.assertEqual(process_values["LLM_PROVIDER"], "openrouter")
        self.assertEqual(process_values["LLM_MODEL"], "openai/gpt-4o-mini")
        self.assertEqual(process_values["STRATEGY_PLANNER_PROVIDER"], "gemini")
        self.assertEqual(process_values["STRATEGY_PLANNER_MODEL"], "gemini-flash-latest")
        self.assertEqual(process_values["EMBEDDING_PROVIDER"], "openrouter")
        self.assertEqual(process_values["EMBEDDING_MODEL"], "not-an-embedding-model")
        self.assertTrue(any("embedding" in warning.lower() and "openrouter" in warning.lower() for warning in payload["warnings"]))
        self.assertIn("LLM_PROVIDER=openrouter", env_text)
        self.assertIn("STRATEGY_PLANNER_MODEL=gemini-flash-latest", env_text)
        self.assertIn("EMBEDDING_PROVIDER=openrouter", env_text)

    def test_llm_comparison_models_are_saved_without_keys(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                response = app.test_request(
                    "PUT",
                    "/api/local-config/llm/comparison-models",
                    {
                        "comparison_models": [
                            {
                                "id": "openrouter_model_a",
                                "label": "OpenRouter Model A",
                                "provider": "openrouter",
                                "model": "openai/gpt-4o-mini",
                                "enabled": True,
                                "role": "comparison",
                                "api_key": "must-not-persist",
                            }
                        ]
                    },
                )
                loaded = app.test_request("GET", "/api/local-config/llm/comparison-models")

            profile_path = runtime_dir / "config" / "llm_profiles.json"
            profile_text = profile_path.read_text(encoding="utf-8")

        self.assertEqual(response["status"], 200, response)
        self.assertTrue(response["json"]["saved"])
        self.assertEqual(loaded["status"], 200)
        self.assertEqual(loaded["json"]["comparison_models"][0]["provider"], "openrouter")
        self.assertNotIn("api_key", profile_text)
        self.assertNotIn("must-not-persist", profile_text)

    def test_llm_test_endpoint_configured_only_never_returns_full_key(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            secret = "AI" + "za" + "configuredonly"
            (root / ".env.local").write_text(
                f"GEMINI_API_KEY={secret}\nGEMINI_MODEL=gemini-flash-latest\n",
                encoding="utf-8",
            )
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {"GEMINI_API_KEY": secret, "GEMINI_MODEL": "gemini-flash-latest"}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                response = app.test_request(
                    "POST",
                    "/api/local-config/llm/test",
                    {"provider": "gemini", "model": "gemini-flash-latest", "role": "answer_model", "test_type": "configured_only"},
                )

        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertEqual(response["status"], 200, response)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["configured"])
        self.assertEqual(payload["tested"], "configured_only")
        self.assertNotIn(secret, serialized)

    def test_explain_selection_uses_answer_model_role_config(self):
        from emotion_aware_assistant.llm.providers import explain_selection

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": "Role model answer."}]}}]}).encode("utf-8")

        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "gemini",
                "LLM_MODEL": "gemini-answer-role",
                "GEMINI_MODEL": "legacy-gemini-model",
                "GEMINI_API_KEY": "AI" + "za" + "answerrole",
            },
            clear=True,
        ), patch("urllib.request.urlopen", fake_urlopen):
            result = explain_selection({"selected_text": "A selected passage", "page_number": 1})

        self.assertEqual(result["provider"], "gemini")
        self.assertEqual(result["model"], "gemini-answer-role")
        self.assertIn("/models/gemini-answer-role:generateContent", requests[0].full_url)
        self.assertEqual(result["answer"], "Role model answer.")

    def test_strategy_planner_uses_strategy_model_role_config(self):
        from emotion_aware_assistant.web.server import create_web_app

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "text": json.dumps(
                                                {
                                                    "state_interpretation": {
                                                        "support_need": "clarification",
                                                        "confidence_handling": "use as cue",
                                                        "context_reasoning": "selected passage",
                                                        "safety_note": "Affective signal used only as a support cue, not as diagnosis.",
                                                    },
                                                    "candidates": [
                                                        {
                                                            "strategy_id": "step_by_step_breakdown",
                                                            "strategy_family": "step_by_step_breakdown",
                                                            "title": "Break it into steps",
                                                            "short_description": "Step through the passage.",
                                                            "why_recommended": "Clarification cue.",
                                                            "prompt_instruction": "Use steps.",
                                                            "expected_answer_shape": ["Main point", "Steps"],
                                                            "recommended": True,
                                                            "recommended_score": 0.9,
                                                        },
                                                        {
                                                            "strategy_id": "define_key_terms",
                                                            "strategy_family": "define_key_terms",
                                                            "title": "Define terms",
                                                            "short_description": "Define terms first.",
                                                            "why_recommended": "Clarification cue.",
                                                            "prompt_instruction": "Define terms.",
                                                            "expected_answer_shape": ["Terms", "Explanation"],
                                                            "recommended": False,
                                                            "recommended_score": 0.7,
                                                        },
                                                        {
                                                            "strategy_id": "concrete_example",
                                                            "strategy_family": "concrete_example",
                                                            "title": "Use an example",
                                                            "short_description": "Add an example.",
                                                            "why_recommended": "Clarification cue.",
                                                            "prompt_instruction": "Use an example.",
                                                            "expected_answer_shape": ["Example", "Back to paper"],
                                                            "recommended": False,
                                                            "recommended_score": 0.65,
                                                        },
                                                    ],
                                                    "warnings": [],
                                                }
                                            )
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.dict(
            os.environ,
            {
                "STRATEGY_PLANNER_PROVIDER": "gemini",
                "STRATEGY_PLANNER_MODEL": "gemini-strategy-role",
                "GEMINI_MODEL": "legacy-gemini-model",
                "GEMINI_API_KEY": "AI" + "za" + "strategyrole",
            },
            clear=True,
        ), patch("urllib.request.urlopen", fake_urlopen):
            app = create_web_app(force_dummy_llm=True, load_local_env=False)
            payload = app.state._call_strategy_planner_llm(
                {
                    "allowed_strategy_families": ["step_by_step_breakdown", "define_key_terms", "concrete_example"],
                    "selected_text": "Dense method passage",
                    "baseline_explanation": "Baseline answer.",
                    "reaction_window_summary": {"support_cue": "sustained_clarification"},
                }
            )

        self.assertIsInstance(payload, dict)
        self.assertIn("/models/gemini-strategy-role:generateContent", requests[0].full_url)

    def test_strategy_planner_supports_openrouter_role_config(self):
        from emotion_aware_assistant.web.server import create_web_app

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(
                                        {
                                            "state_interpretation": {
                                                "support_need": "clarification",
                                                "confidence_handling": "use as cue",
                                                "context_reasoning": "selected passage",
                                                "safety_note": "Affective signal used only as a support cue, not as diagnosis.",
                                            },
                                            "candidates": [
                                                {
                                                    "strategy_id": "step_by_step_breakdown",
                                                    "strategy_family": "step_by_step_breakdown",
                                                    "title": "Break it into steps",
                                                    "short_description": "Step through the passage.",
                                                    "why_recommended": "Clarification cue.",
                                                    "prompt_instruction": "Use steps.",
                                                    "expected_answer_shape": ["Main point", "Steps"],
                                                    "recommended": True,
                                                    "recommended_score": 0.9,
                                                }
                                            ],
                                            "warnings": [],
                                        }
                                    )
                                }
                            }
                        ]
                    }
                ).encode("utf-8")

        requests = []

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with patch.dict(
            os.environ,
            {
                "STRATEGY_PLANNER_PROVIDER": "openrouter",
                "STRATEGY_PLANNER_MODEL": "openai/gpt-4o-mini",
                "OPENROUTER_API_KEY": "router-key",
            },
            clear=True,
        ), patch("urllib.request.urlopen", fake_urlopen):
            app = create_web_app(force_dummy_llm=True, load_local_env=False)
            payload = app.state._call_strategy_planner_llm(
                {
                    "allowed_strategy_families": ["step_by_step_breakdown"],
                    "selected_text": "Dense method passage",
                    "baseline_explanation": "Baseline answer.",
                    "reaction_window_summary": {"support_cue": "sustained_clarification"},
                }
            )

        body = json.loads(requests[0].data.decode("utf-8"))
        self.assertIsInstance(payload, dict)
        self.assertEqual(body["model"], "openai/gpt-4o-mini")
        self.assertIn("/chat/completions", requests[0].full_url)

    def test_embedding_index_uses_embedding_model_role_config(self):
        from emotion_aware_assistant.paper import paper_rag

        seen = []

        def fake_embedding(text, api_key, model, task_type):
            seen.append((model, task_type))
            return [0.1, 0.2, 0.3]

        blocks = [{"block_id": "b1", "markdown_content": "Useful paper content about retrieval and learning.", "page_number": 1}]
        with tempfile.TemporaryDirectory() as temp_dir, patch.dict(
            os.environ,
            {
                "LLM_PROVIDER": "openrouter",
                "EMBEDDING_PROVIDER": "gemini",
                "EMBEDDING_MODEL": "gemini-role-embedding",
                "GEMINI_API_KEY": "AI" + "za" + "embeddingrole",
            },
            clear=True,
        ), patch.object(paper_rag, "_gemini_embedding", fake_embedding):
            status = paper_rag.build_embedding_index("doc1", Path(temp_dir), blocks)

        self.assertEqual(status["embedding_provider"], "gemini")
        self.assertEqual(status["embedding_model"], "gemini-role-embedding")
        self.assertEqual(status["embedding_index_status"], "completed")
        self.assertEqual(seen[0], ("gemini-role-embedding", "RETRIEVAL_DOCUMENT"))

    def test_local_config_face_crop_endpoint_writes_env_local_safely(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / ".env.local").write_text("UNRELATED=value\nFACE_CROP_SCALE=1.2\n", encoding="utf-8")
            (root / ".gitignore").write_text("runtime_uploads/\n", encoding="utf-8")
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                response = app.test_request(
                    "POST",
                    "/api/local-config/face-crop",
                    {
                        "FACE_CROP_SCALE": "2.0",
                        "FACE_CROP_Y_BIAS": "0.22",
                        "FACE_CROP_BOTTOM_EXTRA": "0.30",
                        "FACE_CROP_MAKE_SQUARE": "true",
                    },
                )
                process_values = {
                    "FACE_CROP_SCALE": os.environ.get("FACE_CROP_SCALE"),
                    "FACE_CROP_Y_BIAS": os.environ.get("FACE_CROP_Y_BIAS"),
                    "FACE_CROP_BOTTOM_EXTRA": os.environ.get("FACE_CROP_BOTTOM_EXTRA"),
                    "FACE_CROP_MAKE_SQUARE": os.environ.get("FACE_CROP_MAKE_SQUARE"),
                }

            payload = response["json"]
            env_text = (root / ".env.local").read_text(encoding="utf-8")
            gitignore_text = (root / ".gitignore").read_text(encoding="utf-8")
            mode = stat.S_IMODE((root / ".env.local").stat().st_mode)

            self.assertEqual(response["status"], 200, response)
            self.assertTrue(payload["saved"])
            self.assertFalse(payload["restart_required"])
            self.assertEqual(payload["crop_scale"], 2.0)
            self.assertEqual(payload["crop_y_bias"], 0.22)
            self.assertEqual(payload["crop_bottom_extra"], 0.3)
            self.assertTrue(payload["crop_make_square"])
            self.assertEqual(process_values["FACE_CROP_SCALE"], "2.0")
            self.assertEqual(process_values["FACE_CROP_Y_BIAS"], "0.22")
            self.assertEqual(process_values["FACE_CROP_BOTTOM_EXTRA"], "0.3")
            self.assertEqual(process_values["FACE_CROP_MAKE_SQUARE"], "true")
            self.assertIn("UNRELATED=value", env_text)
            self.assertIn("FACE_CROP_SCALE=2.0", env_text)
            self.assertIn("FACE_CROP_Y_BIAS=0.22", env_text)
            self.assertIn("FACE_CROP_BOTTOM_EXTRA=0.3", env_text)
            self.assertIn("FACE_CROP_MAKE_SQUARE=true", env_text)
            self.assertIn(".env.local", gitignore_text)
            self.assertEqual(mode, 0o600)

    def test_local_config_openface_endpoint_writes_env_local_safely(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary = root / "OpenFace" / "build" / "bin" / "FeatureExtraction"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            os.chmod(binary, 0o755)
            (root / ".env.local").write_text("UNRELATED=value\nFACE_DETECTOR=yolo\n", encoding="utf-8")
            (root / ".gitignore").write_text("runtime_uploads/\n", encoding="utf-8")
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state._face_detector = object()
                response = app.test_request(
                    "POST",
                    "/api/local-config/openface",
                    {
                        "FACE_DETECTOR": "openface",
                        "OPENFACE_FEATURE_EXTRACTION_BIN": str(binary),
                    },
                )
                detector_env = os.environ.get("FACE_DETECTOR")
                openface_env = os.environ.get("OPENFACE_FEATURE_EXTRACTION_BIN")
                detector_cache = app.state._face_detector

            payload = response["json"]
            env_text = (root / ".env.local").read_text(encoding="utf-8")
            gitignore_text = (root / ".gitignore").read_text(encoding="utf-8")
            mode = stat.S_IMODE((root / ".env.local").stat().st_mode)

            self.assertEqual(response["status"], 200, response)
            self.assertTrue(payload["saved"])
            self.assertFalse(payload["restart_required"])
            self.assertEqual(payload["face_detector_status"]["requested_detector"], "openface")
            self.assertTrue(payload["face_detector_status"]["openface"]["available"])
            self.assertEqual(payload["face_detector_status"]["openface"]["binary_path"], str(binary))
            self.assertEqual(detector_env, "openface")
            self.assertEqual(openface_env, str(binary))
            self.assertIsNone(detector_cache)
            self.assertIn("UNRELATED=value", env_text)
            self.assertIn("FACE_DETECTOR=openface", env_text)
            self.assertIn(f"OPENFACE_FEATURE_EXTRACTION_BIN={binary}", env_text)
            self.assertIn(".env.local", gitignore_text)
            self.assertEqual(mode, 0o600)

    def test_configure_openface_script_writes_env_local_and_preserves_unrelated_entries(self):
        from scripts.configure_openface import configure_openface

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            binary = root / "FeatureExtraction"
            binary.write_text("#!/bin/sh\nprintf 'FeatureExtraction help\\n'\n", encoding="utf-8")
            os.chmod(binary, 0o755)
            (root / ".env.local").write_text("UNRELATED=value\n", encoding="utf-8")
            (root / ".gitignore").write_text("runtime_uploads/\n", encoding="utf-8")

            result = configure_openface(root, binary)

            env_text = (root / ".env.local").read_text(encoding="utf-8")
            gitignore_text = (root / ".gitignore").read_text(encoding="utf-8")
            mode = stat.S_IMODE((root / ".env.local").stat().st_mode)

        self.assertTrue(result["saved"])
        self.assertEqual(result["binary_path"], str(binary))
        self.assertIn("UNRELATED=value", env_text)
        self.assertIn("FACE_DETECTOR=openface", env_text)
        self.assertIn(f"OPENFACE_FEATURE_EXTRACTION_BIN={binary}", env_text)
        self.assertIn(".env.local", gitignore_text)
        self.assertEqual(mode, 0o600)

    def test_openface_gitignore_entries_are_present(self):
        gitignore = Path(".gitignore").read_text(encoding="utf-8")

        for entry in [
            "external/",
            "runtime_uploads/openface_build_logs/",
            "FeatureExtraction",
            "FeatureExtraction.exe",
            "*.exe",
            "external/OpenFace/",
        ]:
            self.assertIn(entry, gitignore)

    def test_local_config_save_rejects_empty_key_safely(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                response = app.test_request("POST", "/api/local-config/gemini", {"gemini_api_key": "   "})

        self.assertEqual(response["status"], 400)
        self.assertIn("Gemini API key is required", response["json"]["error"])

    def test_local_settings_page_source_uses_password_field_and_no_browser_storage(self):
        page = Path("emotion_aware_assistant/web/static/local_settings.html")
        source = page.read_text(encoding="utf-8")

        self.assertIn("Local Model & API Settings", source)
        self.assertIn("Provider credentials", source)
        self.assertIn("Role-based model settings", source)
        self.assertIn("Comparison models", source)
        self.assertIn("Provider test/status", source)
        self.assertIn("Advanced local config", source)
        self.assertIn("Gemini", source)
        self.assertIn("OpenRouter", source)
        self.assertIn("OpenAI-compatible", source)
        self.assertIn("openrouter-site-url", source)
        self.assertIn("openrouter-site-name", source)
        self.assertIn("/api/local-config/llm/status", source)
        self.assertIn("/api/local-config/llm/provider", source)
        self.assertIn("/api/local-config/llm/roles", source)
        self.assertIn("/api/local-config/llm/comparison-models", source)
        self.assertIn("/api/local-config/llm/test", source)
        self.assertIn('type="password"', source)
        self.assertIn("This page writes local configuration only. API keys are stored in `.env.local` and are not stored in the browser.", source)
        self.assertNotIn("localStorage", source)
        self.assertNotIn("sessionStorage", source)

    def test_settings_route_serves_local_static_page(self):
        import emotion_aware_assistant.web.server as server

        app = server.create_web_app(force_dummy_llm=True, load_local_env=False)

        class Handler(server.WebRequestHandler):
            web_app = app

        httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = httpd.server_address
            with urllib.request.urlopen(f"http://{host}:{port}/settings", timeout=5) as response:
                body = response.read().decode("utf-8")
                status = response.status
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=5)

        self.assertEqual(status, 200)
        self.assertIn("Local Model & API Settings", body)


if __name__ == "__main__":
    unittest.main()
