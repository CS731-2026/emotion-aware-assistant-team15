import contextlib
import io
import json
import os
import stat
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
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
                        "GEMINI_MODEL=gemini-2.5-flash",
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
            self.assertIn("GEMINI_MODEL=gemini-2.5-flash", env_text)
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
                        "gemini_model": "gemini-2.5-flash",
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
            self.assertIn("GEMINI_MODEL=gemini-2.5-flash", env_text)
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
                        "STRATEGY_PLANNER_MODEL=gemini-2.5-flash",
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
                        "strategy_planner_model": {"provider": "gemini", "model": "gemini-2.5-flash"},
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
        self.assertEqual(process_values["STRATEGY_PLANNER_MODEL"], "gemini-2.5-flash")
        self.assertEqual(process_values["EMBEDDING_PROVIDER"], "openrouter")
        self.assertEqual(process_values["EMBEDDING_MODEL"], "not-an-embedding-model")
        self.assertTrue(any("embedding" in warning.lower() and "openrouter" in warning.lower() for warning in payload["warnings"]))
        self.assertIn("LLM_PROVIDER=openrouter", env_text)
        self.assertIn("STRATEGY_PLANNER_MODEL=gemini-2.5-flash", env_text)
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
                f"GEMINI_API_KEY={secret}\nGEMINI_MODEL=gemini-2.5-flash\n",
                encoding="utf-8",
            )
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {"GEMINI_API_KEY": secret, "GEMINI_MODEL": "gemini-2.5-flash"}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                response = app.test_request(
                    "POST",
                    "/api/local-config/llm/test",
                    {"provider": "gemini", "model": "gemini-2.5-flash", "role": "answer_model", "test_type": "configured_only"},
                )

        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertEqual(response["status"], 200, response)
        self.assertTrue(payload["ok"])
        self.assertTrue(payload["configured"])
        self.assertEqual(payload["tested"], "configured_only")
        self.assertNotIn(secret, serialized)

    def test_settings_llm_save_masks_key_and_preserves_blank_key(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            secret = "or-settings-secret-1234"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                save = app.test_request(
                    "POST",
                    "/api/settings/llm",
                    {
                        "providers": [
                            {
                                "provider": "openrouter",
                                "api_key": secret,
                                "model": "openai/gpt-4o-mini",
                                "display_name": "OpenRouter Demo",
                            }
                        ],
                        "default_model": {
                            "provider": "openrouter",
                            "model": "openai/gpt-4o-mini",
                            "label": "OpenRouter Demo",
                        },
                    },
                )
                preserve = app.test_request(
                    "POST",
                    "/api/settings/llm",
                    {
                        "providers": [
                            {
                                "provider": "openrouter",
                                "api_key": "",
                                "model": "openai/gpt-4o-mini-updated",
                                "display_name": "OpenRouter Updated",
                            }
                        ]
                    },
                )
                status = app.test_request("GET", "/api/settings/llm")

            settings_path = runtime_dir / "local_llm_settings.json"
            stored = json.loads(settings_path.read_text(encoding="utf-8"))
            serialized = json.dumps({"save": save["json"], "preserve": preserve["json"], "status": status["json"]})

        self.assertEqual(save["status"], 200, save)
        self.assertEqual(preserve["status"], 200, preserve)
        self.assertEqual(status["status"], 200, status)
        self.assertEqual(stored["openrouter"]["api_key"], secret)
        self.assertTrue(
            any(profile["model_id"] == "openai/gpt-4o-mini-updated" for profile in stored["model_profiles"])
        )
        self.assertTrue(status["json"]["providers"]["openrouter"]["key_configured"])
        self.assertIn("configured (...1234)", status["json"]["providers"]["openrouter"]["masked_key_display"])
        self.assertNotIn(secret, serialized)

    def test_settings_llm_clear_key_requires_explicit_clear_flag(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            secret = "or-clear-secret-5678"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request(
                    "POST",
                    "/api/settings/llm",
                    {"providers": [{"provider": "openrouter", "api_key": secret, "model": "openai/gpt-4o-mini"}]},
                )
                cleared = app.test_request(
                    "POST",
                    "/api/settings/llm",
                    {"providers": [{"provider": "openrouter", "clear_key": True, "model": "openai/gpt-4o-mini"}]},
                )
                status = app.test_request("GET", "/api/settings/llm")

            settings_path = runtime_dir / "local_llm_settings.json"
            stored = json.loads(settings_path.read_text(encoding="utf-8"))
            serialized = json.dumps(cleared["json"])

        self.assertEqual(cleared["status"], 200, cleared)
        self.assertEqual(stored["openrouter"].get("api_key", ""), "")
        self.assertFalse(status["json"]["providers"]["openrouter"]["key_configured"])
        self.assertNotIn(secret, serialized)

    def test_settings_llm_test_reports_missing_key_and_missing_model(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request(
                    "POST",
                    "/api/settings/llm",
                    {"providers": [{"provider": "openrouter", "model": "openai/gpt-4o-mini"}]},
                )
                missing_key = app.test_request(
                    "POST",
                    "/api/settings/llm/test",
                    {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
                )
                app.test_request(
                    "POST",
                    "/api/settings/llm",
                    {"providers": [{"provider": "openrouter", "api_key": "or-test-secret", "model": ""}]},
                )
                missing_model = app.test_request(
                    "POST",
                    "/api/settings/llm/test",
                    {"provider": "openrouter", "model": ""},
                )

        self.assertEqual(missing_key["status"], 200, missing_key)
        self.assertFalse(missing_key["json"]["ok"])
        self.assertEqual(missing_key["json"]["status"], "skipped")
        self.assertEqual(missing_key["json"]["error_type"], "missing_key")
        self.assertEqual(missing_model["status"], 200, missing_model)
        self.assertFalse(missing_model["json"]["ok"])
        self.assertEqual(missing_model["json"]["status"], "skipped")
        self.assertEqual(missing_model["json"]["error_type"], "missing_model")

    def test_settings_llm_test_normalizes_failed_provider_response_safely(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module
        import urllib.error

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            secret = "or-provider-secret-9012"

            def fake_urlopen(request, timeout=0):
                raise urllib.error.HTTPError(
                    request.full_url,
                    401,
                    "Unauthorized: " + secret,
                    hdrs={},
                    fp=io.BytesIO(("bad " + secret).encode("utf-8")),
                )

            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request(
                    "POST",
                    "/api/settings/llm",
                    {"providers": [{"provider": "openrouter", "api_key": secret, "model": "openai/gpt-4o-mini"}]},
                )
                response = app.test_request(
                    "POST",
                    "/api/settings/llm/test",
                    {"provider": "openrouter", "model": "openai/gpt-4o-mini"},
                )

        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertEqual(response["status"], 200, response)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["error_type"], "unauthorized")
        self.assertNotIn(secret, serialized)

    def test_openrouter_settings_save_preserve_clear_and_mask_key(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            secret = "or-openrouter-secret-1234"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                saved = app.test_request(
                    "POST",
                    "/api/settings/openrouter",
                    {
                        "api_key": secret,
                        "base_url": "https://openrouter.ai/api/v1",
                        "site_url": "http://localhost:8000",
                        "site_name": "CS731 Local Assistant",
                    },
                )
                preserved = app.test_request(
                    "POST",
                    "/api/settings/openrouter",
                    {"api_key": "", "site_name": "Updated Site Name"},
                )
                status = app.test_request("GET", "/api/settings/llm")
                cleared = app.test_request("POST", "/api/settings/openrouter", {"clear_key": True})

            settings = json.loads((runtime_dir / "local_llm_settings.json").read_text(encoding="utf-8"))
            serialized = json.dumps({"saved": saved["json"], "preserved": preserved["json"], "status": status["json"], "cleared": cleared["json"]})

        self.assertEqual(saved["status"], 200, saved)
        self.assertEqual(preserved["status"], 200, preserved)
        self.assertEqual(cleared["status"], 200, cleared)
        self.assertEqual(settings["openrouter"]["api_key"], "")
        self.assertEqual(status["json"]["openrouter"]["site_name"], "Updated Site Name")
        self.assertTrue(status["json"]["openrouter"]["key_configured"])
        self.assertIn("configured (...1234)", status["json"]["openrouter"]["masked_key_display"])
        self.assertFalse(cleared["json"]["openrouter"]["key_configured"])
        self.assertNotIn(secret, serialized)

    def test_gemini_settings_save_preserve_clear_and_mask_key(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            secret = "AI" + "za" + "gemini-direct-secret-1234"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                saved = app.test_request(
                    "POST",
                    "/api/settings/gemini",
                    {
                        "api_key": secret,
                        "display_name": "Gemini Direct",
                        "model": "gemini-2.5-flash",
                        "embedding_model": "gemini-embedding-001",
                    },
                )
                preserved = app.test_request(
                    "POST",
                    "/api/settings/gemini",
                    {"api_key": "", "display_name": "Gemini Updated", "model": "gemini-2.5-flash"},
                )
                status = app.test_request("GET", "/api/settings/llm")
                cleared = app.test_request("POST", "/api/settings/gemini", {"clear_key": True})

            settings = json.loads((runtime_dir / "local_llm_settings.json").read_text(encoding="utf-8"))
            serialized = json.dumps({"saved": saved["json"], "preserved": preserved["json"], "status": status["json"], "cleared": cleared["json"]})

        self.assertEqual(saved["status"], 200, saved)
        self.assertEqual(preserved["status"], 200, preserved)
        self.assertEqual(cleared["status"], 200, cleared)
        self.assertEqual(settings["gemini"]["api_key"], "")
        self.assertEqual(status["json"]["gemini"]["display_name"], "Gemini Updated")
        self.assertEqual(status["json"]["gemini"]["model"], "gemini-2.5-flash")
        self.assertEqual(status["json"]["embedding"]["model"], "gemini-embedding-001")
        self.assertTrue(status["json"]["gemini"]["key_configured"])
        self.assertIn("configured (...1234)", status["json"]["gemini"]["masked_key_display"])
        self.assertFalse(cleared["json"]["gemini"]["key_configured"])
        self.assertNotIn(secret, serialized)

    def test_legacy_gemini_provider_settings_are_migrated_and_preserved(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            runtime_dir.mkdir(parents=True)
            secret = "AI" + "za" + "legacy-gemini-secret-1234"
            (runtime_dir / "local_llm_settings.json").write_text(
                json.dumps(
                    {
                        "providers": {
                            "gemini": {
                                "api_key": secret,
                                "display_name": "Legacy Gemini",
                                "model": "gemini-legacy-chat",
                                "embedding_model": "gemini-legacy-embedding",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                status = app.test_request("GET", "/api/settings/llm")
                saved_openrouter = app.test_request("POST", "/api/settings/openrouter", {"site_name": "Router Only"})

            stored = json.loads((runtime_dir / "local_llm_settings.json").read_text(encoding="utf-8"))
            serialized = json.dumps({"status": status["json"], "saved_openrouter": saved_openrouter["json"]})

        self.assertEqual(status["status"], 200, status)
        self.assertEqual(saved_openrouter["status"], 200, saved_openrouter)
        self.assertTrue(status["json"]["gemini"]["key_configured"])
        self.assertEqual(status["json"]["gemini"]["display_name"], "Legacy Gemini")
        self.assertEqual(status["json"]["gemini"]["model"], "gemini-legacy-chat")
        self.assertEqual(status["json"]["gemini"]["embedding_model"], "gemini-legacy-embedding")
        self.assertEqual(stored["gemini"]["api_key"], secret)
        self.assertEqual(stored["gemini"]["model"], "gemini-legacy-chat")
        self.assertNotIn(secret, serialized)

    def test_openrouter_model_profiles_can_be_added_updated_tested_disabled_and_deleted(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"choices": [{"message": {"content": "OK"}}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            secret = "or-profile-secret-5678"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request("POST", "/api/settings/openrouter", {"api_key": secret})
                created = app.test_request(
                    "POST",
                    "/api/settings/models",
                    {
                        "display_name": "Claude Opus 4.7 Fast",
                        "model_id": "anthropic/claude-opus-4.7-fast",
                        "family": "Claude",
                        "enabled": True,
                        "notes": "fast preset",
                    },
                )
                profile_id = created["json"]["model_profile"]["id"]
                updated = app.test_request(
                    "PUT",
                    f"/api/settings/models/{profile_id}",
                    {"display_name": "Claude Fast", "enabled": False, "notes": "disabled for now"},
                )
                tested = app.test_request("POST", f"/api/settings/models/{profile_id}/test")
                deleted = app.test_request("DELETE", f"/api/settings/models/{profile_id}")
                status = app.test_request("GET", "/api/settings/llm")

        serialized = json.dumps({"created": created["json"], "updated": updated["json"], "tested": tested["json"], "deleted": deleted["json"], "status": status["json"]})
        body = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(created["status"], 200, created)
        self.assertEqual(updated["status"], 200, updated)
        self.assertEqual(tested["status"], 200, tested)
        self.assertEqual(deleted["status"], 200, deleted)
        self.assertEqual(body["model"], "anthropic/claude-opus-4.7-fast")
        self.assertEqual(tested["json"]["provider"], "openrouter")
        self.assertEqual(tested["json"]["display_name"], "Claude Fast")
        self.assertEqual(tested["json"]["model_id"], "anthropic/claude-opus-4.7-fast")
        self.assertTrue(tested["json"]["ok"])
        self.assertEqual(status["json"]["model_profiles"], [])
        self.assertNotIn(secret, serialized)

    def test_gemini_model_profile_can_be_added_tested_and_deleted(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            secret = "AI" + "za" + "profile-secret-5678"
            stale_env_secret = "AI" + "za" + "profile-stale-env"
            (root / ".env.local").write_text(
                f"GEMINI_API_KEY={stale_env_secret}\nGEMINI_MODEL=gemini-flash-latest\n",
                encoding="utf-8",
            )
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request("POST", "/api/settings/gemini", {"api_key": secret})
                created = app.test_request(
                    "POST",
                    "/api/settings/models",
                    {
                        "provider": "gemini",
                        "display_name": "Gemini 2.5 Flash",
                        "model_id": "gemini-2.5-flash",
                        "enabled": True,
                    },
                )
                profile_id = created["json"]["model_profile"]["id"]
                tested = app.test_request("POST", f"/api/settings/models/{profile_id}/test")
                deleted = app.test_request("DELETE", f"/api/settings/models/{profile_id}")
                status = app.test_request("GET", "/api/settings/llm")

        serialized = json.dumps({"created": created["json"], "tested": tested["json"], "deleted": deleted["json"], "status": status["json"]})
        self.assertEqual(created["status"], 200, created)
        self.assertEqual(created["json"]["model_profile"]["provider"], "gemini")
        self.assertEqual(tested["status"], 200, tested)
        self.assertEqual(tested["json"]["provider"], "gemini")
        self.assertEqual(tested["json"]["display_name"], "Gemini 2.5 Flash")
        self.assertEqual(tested["json"]["model_id"], "gemini-2.5-flash")
        self.assertEqual(tested["json"]["test_type"], "chat")
        self.assertTrue(tested["json"]["ok"])
        self.assertIn("/models/gemini-2.5-flash:generateContent", requests[0].full_url)
        headers = {key.lower(): value for key, value in requests[0].header_items()}
        body = json.loads(requests[0].data.decode("utf-8"))
        self.assertEqual(headers["x-goog-api-key"], secret)
        self.assertNotEqual(headers["x-goog-api-key"], stale_env_secret)
        self.assertNotIn("authorization", headers)
        self.assertEqual(body["contents"][0]["parts"][0]["text"], "Reply with exactly: OK")
        self.assertNotIn("messages", body)
        self.assertNotIn("generationConfig", body)
        self.assertEqual(deleted["status"], 200, deleted)
        self.assertEqual(status["json"]["model_profiles"], [])
        self.assertNotIn(secret, serialized)

    def test_gemini_chat_settings_test_uses_minimal_direct_generate_content_request(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            secret = "AI" + "za" + "chat-test-secret-1234"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request("POST", "/api/settings/gemini", {"api_key": secret, "model": "gemini-2.5-flash"})
                tested = app.test_request("POST", "/api/settings/gemini/test-chat", {"model_id": "gemini-2.5-flash"})

        headers = {key.lower(): value for key, value in requests[0].header_items()}
        body = json.loads(requests[0].data.decode("utf-8"))
        serialized = json.dumps(tested["json"])
        self.assertEqual(tested["status"], 200, tested)
        self.assertTrue(tested["json"]["ok"])
        self.assertEqual(tested["json"]["test_type"], "chat")
        self.assertEqual(tested["json"]["model_id"], "gemini-2.5-flash")
        self.assertEqual(tested["json"]["payload_shape"], "contents.parts")
        self.assertIn("/models/gemini-2.5-flash:generateContent", tested["json"]["endpoint_url"])
        self.assertEqual(tested["json"]["key_source"], "local_settings")
        self.assertEqual(tested["json"]["masked_key_suffix"], "1234")
        self.assertIn("/models/gemini-2.5-flash:generateContent", requests[0].full_url)
        self.assertEqual(headers["x-goog-api-key"], secret)
        self.assertEqual(headers["content-type"], "application/json")
        self.assertNotIn("authorization", headers)
        self.assertEqual(body, {"contents": [{"parts": [{"text": "Reply with exactly: OK"}]}]})
        self.assertNotIn("messages", body)
        self.assertNotIn("generationConfig", body)
        self.assertNotIn("temperature", body)
        self.assertNotIn("top_p", body)
        self.assertNotIn("topP", body)
        self.assertNotIn("top_k", body)
        self.assertNotIn("topK", body)
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("maxOutputTokens", body)
        self.assertNotIn(secret, serialized)

    def test_gemini_chat_settings_test_prefers_saved_key_over_stale_env_file_key(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            local_secret = "AI" + "za" + "saved-settings-secret-1234"
            stale_env_secret = "AI" + "za" + "stale-env-secret-0000"
            (root / ".env.local").write_text(
                f"GEMINI_API_KEY={stale_env_secret}\nGEMINI_MODEL=gemini-flash-latest\n",
                encoding="utf-8",
            )
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request("POST", "/api/settings/gemini", {"api_key": local_secret, "model": "gemini-2.5-flash"})
                tested = app.test_request("POST", "/api/settings/gemini/test-chat", {"model_id": "gemini-2.5-flash"})

        headers = {key.lower(): value for key, value in requests[0].header_items()}
        serialized = json.dumps(tested["json"])
        self.assertEqual(tested["status"], 200, tested)
        self.assertTrue(tested["json"]["ok"])
        self.assertEqual(tested["json"]["key_source"], "local_settings")
        self.assertEqual(tested["json"]["masked_key_suffix"], "1234")
        self.assertEqual(tested["json"]["payload_shape"], "contents.parts")
        self.assertEqual(headers["x-goog-api-key"], local_secret)
        self.assertNotEqual(headers["x-goog-api-key"], stale_env_secret)
        self.assertNotIn(local_secret, serialized)
        self.assertNotIn(stale_env_secret, serialized)

    def test_gemini_embedding_settings_test_is_separate_from_chat(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        requests = []

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps({"embedding": {"values": [0.1, 0.2]}}).encode("utf-8")

        def fake_urlopen(request, timeout=0):
            requests.append(request)
            return FakeResponse()

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            secret = "AI" + "za" + "embedding-test-secret-9999"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request("POST", "/api/settings/gemini", {"api_key": secret, "model": "gemini-2.5-flash", "embedding_model": "gemini-embedding-001"})
                tested = app.test_request("POST", "/api/settings/gemini/test-embedding", {"model_id": "gemini-embedding-001"})
                status = app.test_request("GET", "/api/settings/llm")

        headers = {key.lower(): value for key, value in requests[0].header_items()}
        body = json.loads(requests[0].data.decode("utf-8"))
        serialized = json.dumps({"tested": tested["json"], "status": status["json"]})
        self.assertEqual(tested["status"], 200, tested)
        self.assertTrue(tested["json"]["ok"])
        self.assertEqual(tested["json"]["test_type"], "embedding")
        self.assertEqual(tested["json"]["model_id"], "gemini-embedding-001")
        self.assertEqual(tested["json"]["payload_shape"], "embedding.content.parts")
        self.assertIn("/models/gemini-embedding-001:embedContent", requests[0].full_url)
        self.assertEqual(headers["x-goog-api-key"], secret)
        self.assertNotIn("authorization", headers)
        self.assertEqual(body["model"], "models/gemini-embedding-001")
        self.assertEqual(body["content"]["parts"][0]["text"], "Reply with exactly: OK")
        self.assertEqual(body["taskType"], "RETRIEVAL_DOCUMENT")
        self.assertIsNone(status["json"]["gemini"].get("last_test"))
        self.assertEqual(status["json"]["embedding"]["last_test"]["test_type"], "embedding")
        self.assertEqual(status["json"]["embedding"]["last_test"]["model_id"], "gemini-embedding-001")
        self.assertNotIn(secret, serialized)

    def test_gemini_direct_failure_returns_safe_provider_message(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        def fake_urlopen(request, timeout=0):
            body = io.BytesIO(json.dumps({"error": {"status": "INVALID_ARGUMENT", "message": "Model does not support generateContent for this request."}}).encode("utf-8"))
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, body)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            secret = "AI" + "za" + "safe-error-secret-4321"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request("POST", "/api/settings/gemini", {"api_key": secret, "model": "gemini-2.5-flash"})
                tested = app.test_request("POST", "/api/settings/gemini/test-chat", {"model_id": "gemini-2.5-flash"})

        serialized = json.dumps(tested["json"])
        self.assertEqual(tested["status"], 200, tested)
        self.assertFalse(tested["json"]["ok"])
        self.assertEqual(tested["json"]["test_type"], "chat")
        self.assertEqual(tested["json"]["status_code"], 400)
        self.assertEqual(tested["json"]["error_type"], "provider_error")
        self.assertEqual(tested["json"]["google_error_status"], "INVALID_ARGUMENT")
        self.assertIn("Model does not support generateContent", tested["json"]["message"])
        self.assertEqual(tested["json"]["masked_key_suffix"], "4321")
        self.assertNotIn(secret, serialized)

    def test_stale_gemini_flash_latest_profile_warns_and_is_not_auto_selected_for_roles(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            runtime_dir.mkdir(parents=True)
            (runtime_dir / "local_llm_settings.json").write_text(
                json.dumps(
                    {
                        "gemini": {"api_key": "AI" + "za" + "stale-secret", "model": "gemini-flash-latest"},
                        "model_profiles": [
                            {"id": "old-gemini", "provider": "gemini", "display_name": "Old Gemini", "model_id": "gemini-flash-latest", "enabled": True}
                        ],
                        "roles": {
                            "default_answer_model_profile_id": "old-gemini",
                            "strategy_model_profile_id": "old-gemini",
                            "compare_model_profile_ids": ["old-gemini"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                status = app.test_request("GET", "/api/settings/llm")

        self.assertEqual(status["status"], 200, status)
        self.assertIsNone(status["json"]["roles"]["default_answer_model_profile_id"])
        self.assertIsNone(status["json"]["roles"]["strategy_model_profile_id"])
        self.assertEqual(status["json"]["roles"]["compare_model_profile_ids"], [])
        warnings = "\n".join(status["json"]["warnings"])
        self.assertIn("gemini-flash-latest", warnings)
        self.assertIn("gemini-2.5-flash", warnings)

    def test_deleting_model_profile_invalidates_role_assignments_safely(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request("POST", "/api/settings/openrouter", {"api_key": "or-role-secret"})
                created = app.test_request(
                    "POST",
                    "/api/settings/models",
                    {"display_name": "OpenRouter Free", "model_id": "openrouter/free", "enabled": True},
                )
                profile_id = created["json"]["model_profile"]["id"]
                roles = app.test_request(
                    "POST",
                    "/api/settings/roles",
                    {
                        "default_answer_model_profile_id": profile_id,
                        "strategy_model_profile_id": profile_id,
                        "compare_model_profile_ids": [profile_id],
                    },
                )
                deleted = app.test_request("DELETE", f"/api/settings/models/{profile_id}")
                status = app.test_request("GET", "/api/settings/llm")

        self.assertEqual(roles["status"], 200, roles)
        self.assertEqual(deleted["status"], 200, deleted)
        self.assertIsNone(status["json"]["roles"]["default_answer_model_profile_id"])
        self.assertIsNone(status["json"]["roles"]["strategy_model_profile_id"])
        self.assertEqual(status["json"]["roles"]["compare_model_profile_ids"], [])
        self.assertTrue(any("deleted" in warning.lower() or "missing" in warning.lower() for warning in status["json"]["warnings"]))

    def test_explain_selection_uses_answer_model_role_config(self):
        from emotion_aware_assistant.llm.providers import explain_selection
        import emotion_aware_assistant.core.llm_config as llm_config

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

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(
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
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

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

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(
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

    def test_strategy_planner_gemini_uses_saved_settings_key_before_environment_key(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

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
                                                        }
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

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            saved_secret = "AI" + "za" + "strategy-saved-secret"
            stale_secret = "AI" + "za" + "strategy-stale-secret"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {"GEMINI_API_KEY": stale_secret}, clear=True), patch("urllib.request.urlopen", fake_urlopen):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request("POST", "/api/settings/gemini", {"api_key": saved_secret, "model": "gemini-2.5-flash"})
                profile = app.test_request(
                    "POST",
                    "/api/settings/models",
                    {"provider": "gemini", "display_name": "Gemini Strategy", "model_id": "gemini-2.5-flash", "enabled": True},
                )["json"]["model_profile"]["id"]
                app.test_request("POST", "/api/settings/roles", {"strategy_model_profile_id": profile})
                payload = app.state._call_strategy_planner_llm(
                    {
                        "allowed_strategy_families": ["step_by_step_breakdown"],
                        "selected_text": "Dense method passage",
                        "baseline_explanation": "Baseline answer.",
                        "reaction_window_summary": {"support_cue": "sustained_clarification"},
                    }
                )

        headers = {key.lower(): value for key, value in requests[0].header_items()}
        serialized = json.dumps(payload)
        self.assertIsInstance(payload, dict)
        self.assertEqual(headers["x-goog-api-key"], saved_secret)
        self.assertNotEqual(headers["x-goog-api-key"], stale_secret)
        self.assertNotIn(saved_secret, serialized)
        self.assertNotIn(stale_secret, serialized)

    def test_strategy_planner_supports_openrouter_role_config(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

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

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(
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

    def test_strategy_planner_uses_changed_strategy_role_without_restart(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

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
                                            "candidates": [],
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

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            new_secret = "or-new-strategy-key"
            stale_secret = "or-stale-strategy-env"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": stale_secret,
                    "STRATEGY_PLANNER_PROVIDER": "openrouter",
                    "STRATEGY_PLANNER_MODEL": "openai/stale-strategy-model",
                },
                clear=True,
            ), patch("urllib.request.urlopen", fake_urlopen):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request("POST", "/api/settings/openrouter", {"api_key": new_secret})
                old_profile = app.test_request(
                    "POST",
                    "/api/settings/models",
                    {"provider": "openrouter", "display_name": "Old Strategy", "model_id": "openai/old-strategy", "enabled": True},
                )["json"]["model_profile"]["id"]
                app.test_request("POST", "/api/settings/roles", {"strategy_model_profile_id": old_profile})
                new_profile = app.test_request(
                    "POST",
                    "/api/settings/models",
                    {"provider": "openrouter", "display_name": "New Strategy", "model_id": "openai/new-strategy", "enabled": True},
                )["json"]["model_profile"]["id"]
                app.test_request("POST", "/api/settings/roles", {"strategy_model_profile_id": new_profile})
                app.state._call_strategy_planner_llm(
                    {
                        "allowed_strategy_families": ["step_by_step_breakdown"],
                        "selected_text": "Dense method passage",
                        "baseline_explanation": "Baseline answer.",
                        "reaction_window_summary": {"support_cue": "sustained_clarification"},
                    }
                )

        body = json.loads(requests[0].data.decode("utf-8"))
        headers = {key.lower(): value for key, value in requests[0].header_items()}
        self.assertEqual(body["model"], "openai/new-strategy")
        self.assertEqual(headers["authorization"], f"Bearer {new_secret}")
        self.assertNotEqual(headers["authorization"], f"Bearer {stale_secret}")

    def test_embedding_index_uses_embedding_model_role_config(self):
        from emotion_aware_assistant.core import llm_config
        from emotion_aware_assistant.paper import paper_rag

        seen = []

        def fake_embedding(text, api_key, model, task_type):
            seen.append((model, task_type))
            return [0.1, 0.2, 0.3]

        blocks = [{"block_id": "b1", "markdown_content": "Useful paper content about retrieval and learning.", "page_number": 1}]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(
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

    def test_gemini_embedding_uses_saved_settings_key_before_environment_key(self):
        from emotion_aware_assistant.core import llm_config
        from emotion_aware_assistant.paper import paper_rag

        seen = []

        def fake_embedding(text, api_key, model, task_type):
            seen.append((api_key, model, task_type))
            return [0.1, 0.2, 0.3]

        blocks = [{"block_id": "b1", "markdown_content": "Useful paper content about retrieval and learning.", "page_number": 1}]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            saved_secret = "AI" + "za" + "embedding-saved-secret"
            stale_secret = "AI" + "za" + "embedding-stale-secret"
            llm_config.save_gemini_settings(root, runtime_dir, {"api_key": saved_secret, "embedding_model": "gemini-embedding-001"})
            with patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {"GEMINI_API_KEY": stale_secret}, clear=True), patch.object(paper_rag, "_gemini_embedding", fake_embedding):
                status = paper_rag.build_embedding_index("doc1", Path(temp_dir), blocks)

        serialized = json.dumps(status)
        self.assertEqual(status["embedding_provider"], "gemini")
        self.assertEqual(status["embedding_index_status"], "completed")
        self.assertEqual(seen[0][0], saved_secret)
        self.assertNotEqual(seen[0][0], stale_secret)
        self.assertNotIn(saved_secret, serialized)
        self.assertNotIn(stale_secret, serialized)

    def test_read_llm_values_prefers_saved_openrouter_and_roles_over_environment(self):
        from emotion_aware_assistant.core import llm_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            saved_secret = "or-saved-runtime-key"
            stale_secret = "or-stale-env-key"
            llm_config.save_openrouter_settings(root, runtime_dir, {"api_key": saved_secret})
            profile_id = llm_config.create_model_profile(
                root,
                runtime_dir,
                {"provider": "openrouter", "display_name": "Saved Runtime Model", "model_id": "openai/saved-model", "enabled": True},
            )["model_profile"]["id"]
            llm_config.save_profile_roles(root, runtime_dir, {"default_answer_model_profile_id": profile_id})
            with patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(
                os.environ,
                {
                    "OPENROUTER_API_KEY": stale_secret,
                    "LLM_PROVIDER": "gemini",
                    "LLM_MODEL": "gemini-env-model",
                },
                clear=True,
            ):
                values = llm_config.read_llm_values(root, runtime_dir)
                role = llm_config.role_config("answer_model", values)

        self.assertEqual(values["OPENROUTER_API_KEY"], saved_secret)
        self.assertNotEqual(values["OPENROUTER_API_KEY"], stale_secret)
        self.assertEqual(role["provider"], "openrouter")
        self.assertEqual(role["model"], "openai/saved-model")

    def test_gemini_embedding_settings_change_takes_effect_without_restart(self):
        from emotion_aware_assistant.core import llm_config
        from emotion_aware_assistant.paper import paper_rag

        seen = []

        def fake_embedding(text, api_key, model, task_type):
            seen.append((api_key, model, task_type))
            return [0.1, 0.2, 0.3]

        blocks = [{"block_id": "b1", "markdown_content": "Useful paper content about retrieval and learning.", "page_number": 1}]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            old_secret = "AI" + "za" + "old-embedding-settings"
            new_secret = "AI" + "za" + "new-embedding-settings"
            stale_secret = "AI" + "za" + "stale-embedding-env"
            llm_config.save_gemini_settings(root, runtime_dir, {"api_key": old_secret, "embedding_model": "old-embedding-model"})
            llm_config.save_gemini_settings(root, runtime_dir, {"api_key": new_secret, "embedding_model": "new-embedding-model"})
            with patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(
                os.environ,
                {
                    "GEMINI_API_KEY": stale_secret,
                    "GEMINI_EMBEDDING_MODEL": "stale-env-embedding-model",
                },
                clear=True,
            ), patch.object(paper_rag, "_gemini_embedding", fake_embedding):
                status = paper_rag.build_embedding_index("doc1", Path(temp_dir), blocks)

        serialized = json.dumps(status)
        self.assertEqual(status["embedding_index_status"], "completed")
        self.assertEqual(seen[0][0], new_secret)
        self.assertEqual(seen[0][1], "new-embedding-model")
        self.assertNotEqual(seen[0][0], stale_secret)
        self.assertNotIn(old_secret, serialized)
        self.assertNotIn(new_secret, serialized)
        self.assertNotIn(stale_secret, serialized)

    def test_gemini_key_resolution_falls_back_to_environment_without_saved_settings_key(self):
        from emotion_aware_assistant.core import llm_config

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            env_secret = "AI" + "za" + "environment-fallback"
            with patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {"GOOGLE_API_KEY": env_secret}, clear=True):
                resolved = llm_config.resolve_gemini_api_key(root, root / "runtime_uploads")

        serialized = json.dumps(resolved)
        self.assertEqual(resolved["key"], env_secret)
        self.assertEqual(resolved["key_source"], "environment")
        self.assertEqual(resolved["masked_suffix"], "back")
        self.assertNotIn(env_secret, json.dumps({key: value for key, value in resolved.items() if key != "key"}))

    def test_runtime_status_masks_keys_and_reports_local_settings_sources(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            saved_gemini = "AI" + "za" + "runtime-saved-gemini"
            stale_gemini = "AI" + "za" + "runtime-stale-gemini"
            router_secret = "or-runtime-secret"
            (root / ".env.local").write_text(
                "\n".join(
                    [
                        "GEMINI_API_KEY=" + stale_gemini,
                        "GEMINI_MODEL=gemini-flash-latest",
                        "GEMINI_EMBEDDING_MODEL=old-embedding-model",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.state.server_port = 8004
                app.test_request("POST", "/api/settings/gemini", {"api_key": saved_gemini, "model": "gemini-2.5-flash"})
                app.test_request("POST", "/api/settings/openrouter", {"api_key": router_secret})
                profile = app.test_request(
                    "POST",
                    "/api/settings/models",
                    {"provider": "gemini", "display_name": "Gemini Runtime", "model_id": "gemini-2.5-flash", "enabled": True},
                )["json"]["model_profile"]["id"]
                app.test_request("POST", "/api/settings/roles", {"default_answer_model_profile_id": profile})
                response = app.test_request("GET", "/api/runtime/status")

        payload = response["json"]
        serialized = json.dumps(payload)
        warnings = "\n".join(payload["warnings"])
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(payload["server_port"], 8004)
        self.assertEqual(payload["gemini"]["key_source"], "local_settings")
        self.assertEqual(payload["gemini"]["masked_key_suffix"], "mini")
        self.assertEqual(payload["gemini"]["model_source"], "role_profile")
        self.assertEqual(payload["gemini"]["model_id"], "gemini-2.5-flash")
        self.assertEqual(payload["openrouter"]["key_source"], "local_settings")
        self.assertEqual(payload["openrouter"]["masked_key_suffix"], "cret")
        self.assertEqual(payload["resolved_default_model"]["provider"], "gemini")
        self.assertIn("Gemini environment variables are present", warnings)
        self.assertIn("GEMINI_MODEL=gemini-flash-latest", warnings)
        self.assertIn("GEMINI_API_KEY", payload["env_gemini_variables_present"])
        self.assertNotIn(saved_gemini, serialized)
        self.assertNotIn(stale_gemini, serialized)
        self.assertNotIn(router_secret, serialized)

    def test_runtime_status_uses_environment_key_only_when_settings_key_is_missing(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            env_secret = "AI" + "za" + "runtime-env-fallback"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {"GOOGLE_API_KEY": env_secret}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                response = app.test_request("GET", "/api/runtime/status")

        payload = response["json"]
        serialized = json.dumps(payload)
        self.assertEqual(response["status"], 200, response)
        self.assertEqual(payload["gemini"]["key_source"], "environment")
        self.assertEqual(payload["gemini"]["masked_key_suffix"], "back")
        self.assertIn("GOOGLE_API_KEY", payload["env_gemini_variables_present"])
        self.assertNotIn(env_secret, serialized)

    def test_runtime_status_revision_changes_when_settings_file_changes(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                app.test_request("POST", "/api/settings/gemini", {"api_key": "AI" + "za" + "revision-secret"})
                first = app.test_request("GET", "/api/runtime/status")["json"]
                time.sleep(0.02)
                app.test_request(
                    "POST",
                    "/api/settings/models",
                    {"provider": "gemini", "display_name": "Revision Model", "model_id": "gemini-2.5-flash", "enabled": True},
                )
                second = app.test_request("GET", "/api/runtime/status")["json"]

        self.assertNotEqual(first["settings_revision"], second["settings_revision"])
        self.assertNotEqual(first["settings_file_mtime_epoch"], second["settings_file_mtime_epoch"])
        self.assertIsNotNone(second["settings_file_mtime"])

    def test_settings_save_invalidates_runtime_llm_config_cache(self):
        from emotion_aware_assistant.web.server import create_web_app
        import emotion_aware_assistant.core.llm_config as llm_config
        import emotion_aware_assistant.web.state as state_module

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime_dir = root / "runtime_uploads"
            with patch.object(state_module, "PROJECT_ROOT", root), patch.object(llm_config, "PROJECT_ROOT", root, create=True), patch.dict(os.environ, {}, clear=True):
                app = create_web_app(force_dummy_llm=True, load_local_env=False)
                app.state.upload_dir = runtime_dir
                before = app.state._llm_runtime_config_invalidated_at
                time.sleep(0.02)
                saved = app.test_request("POST", "/api/settings/openrouter", {"api_key": "or-cache-invalidated"})
                runtime = app.test_request("GET", "/api/runtime/status")["json"]

        self.assertEqual(saved["status"], 200, saved)
        self.assertGreater(app.state._llm_runtime_config_invalidated_at, before)
        self.assertGreaterEqual(runtime["settings_revision"], 1)

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
        self.assertIn("OpenRouter API configuration", source)
        self.assertIn("Gemini API configuration", source)
        self.assertIn("Saved model library", source)
        self.assertIn("Role assignments", source)
        self.assertIn("Embedding configuration", source)
        self.assertIn("Runtime status", source)
        self.assertIn("Advanced local config", source)
        self.assertIn("OpenRouter", source)
        self.assertIn("Gemini Direct", source)
        self.assertIn("OpenRouter Free", source)
        self.assertIn("Claude Opus 4.7 Fast", source)
        self.assertIn("Gemini 2.5 Flash", source)
        self.assertIn("Test Gemini Chat", source)
        self.assertIn("Test Gemini Embedding", source)
        self.assertNotIn("OpenAI-compatible", source)
        self.assertNotIn("gemini-flash-latest", source)
        self.assertNotIn("openai-key", source)
        self.assertNotIn("OPENAI_API_KEY", source)
        self.assertIn("gemini-key", source)
        self.assertIn("gemini-clear-key", source)
        self.assertIn("embedding-model", source)
        self.assertIn("model-profiles", source)
        self.assertIn("openrouter-site-url", source)
        self.assertIn("openrouter-site-name", source)
        self.assertIn("openrouter-clear-key", source)
        self.assertIn("/api/settings/llm", source)
        self.assertIn("/api/settings/openrouter", source)
        self.assertIn("/api/settings/gemini", source)
        self.assertIn("/api/settings/gemini/test-chat", source)
        self.assertIn("/api/settings/gemini/test-embedding", source)
        self.assertIn("/api/runtime/status", source)
        self.assertIn("/api/settings/models", source)
        self.assertIn("/api/settings/roles", source)
        self.assertNotIn("/api/local-config/llm/provider", source)
        self.assertIn('type="password"', source)
        self.assertIn("API keys are stored backend-side in the ignored runtime settings file or environment variables", source)
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
