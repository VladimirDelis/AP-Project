"""
Main window + app entry point — Person C's responsibility.

Ties the Live view (Person B) and Offline view (Person C) together in a
tabbed main window, forwards Model status messages to a shared status
bar, and auto-switches to the Offline tab when the connection drops so
the user immediately sees their recorded session.

Run with:
    python main.py
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget, QStatusBar

from viewmodels.offline_viewmodel import OfflineViewModel
from views.offline_view import OfflineView

# --- Swap these two imports out once A and B deliver their real files ---
from models.fake_model_stub import FakeSignalModel as SignalModel
from views.live_view_stub import LiveViewStub as LiveView
# --------------------------------------------------------------------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TCP Signal Visualization")
        self.resize(1000, 650)

        # The Model is shared by both the live view and the offline
        # view — created once here and injected into each ViewModel/View.
        self.model = SignalModel()

        self.offline_view_model = OfflineViewModel(self.model)
        self.offline_view = OfflineView(self.offline_view_model)
        self.live_view = LiveView(self.model)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.live_view, "Live")
        self.tabs.addTab(self.offline_view, "Offline Inspection")
        self.setCentralWidget(self.tabs)

        self.setStatusBar(QStatusBar())
        self.model.status_updated.connect(self._on_status_updated)
        self.model.connection_state_changed.connect(self._on_connection_state_changed)

    def _on_status_updated(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def _on_connection_state_changed(self, connected: bool) -> None:
        if not connected:
            # Auto-switch to Offline tab so the user can immediately
            # inspect what was just recorded.
            self.tabs.setCurrentWidget(self.offline_view)

    def closeEvent(self, event) -> None:
        self.model.disconnect()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
