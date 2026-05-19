from __future__ import annotations


def run_gui(config_path: str = "config.yaml") -> int:
    try:
        from PyQt5.QtWidgets import QApplication  # type: ignore
    except Exception as exc:
        print("GUI mode requires PyQt5, but it is not available in this environment.")
        print(f"Import error: {exc}")
        print("Use terminal mode instead: python main.py --mode terminal")
        return 0

    from emotion_aware_assistant.core.config import load_config
    from emotion_aware_assistant.ui.main_window import MainWindow

    app = QApplication([])
    window = MainWindow(load_config(config_path))
    window.show()
    return int(app.exec_())
