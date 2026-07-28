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

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QStatusBar,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QPushButton,
)

from viewmodels.offline_viewmodel import OfflineViewModel
from views.offline_view import OfflineView

# --- Swap this import out once A delivers the real Model ---
from models.fake_model_stub import FakeSignalModel as SignalModel
# --------------------------------------------------------------------------

from viewmodels.live_viewmodel import LiveViewModel
from views.connection_widget import ConnectionWidget
from views.live_plot_view import LivePlotView
from views.all_channels_plot_view import AllChannelsPlotView
from views.channel_selector_widget import ChannelSelectorWidget
from views.mode_selector_widget import ModeSelectorWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TCP Signal Visualization")
        self.resize(1000, 650)

        # ------------------------------------------------------------
        # TEMPORARY / NOT YET INTEGRATED:
        # This is a stand-in "fake" Model used only for the Offline
        # side for now. The Live side below currently builds its OWN
        # separate LiveViewModel (with its own TCPWorker/thread) rather
        # than sharing this Model instance. That means, right now, the
        # Offline view is NOT actually seeing data from the Live view's
        # real connection. This needs to be resolved once we agree
        # with Person A on a single shared Model object/interface.
        # ------------------------------------------------------------
        self.model = SignalModel()

        self.offline_view_model = OfflineViewModel(self.model)
        self.offline_view = OfflineView(self.offline_view_model)

        # --- Live side (Person B) ---
        # LivePlotView/AllChannelsPlotView are plain Views: they take
        # config values (sample_rate_hz etc.), not a ViewModel, and are
        # wired up via explicit signal connections below.
        self.live_view_model = LiveViewModel()  # builds its own worker/thread

        connection_widget = ConnectionWidget(self.live_view_model)
        self.live_plot_view = LivePlotView(sample_rate_hz=2000)
        self.all_channels_view = AllChannelsPlotView(sample_rate_hz=2000)

        # Both views keep receiving data regardless of which is
        # currently visible, so switching back never shows stale data.
        self.live_view_model.processed_data_ready.connect(self.live_plot_view.append_window)
        self.live_view_model.processed_data_ready.connect(self.all_channels_view.append_window)

        # Channel selection: the ViewModel is the single source of truth
        # (see LiveViewModel.channel_changed's docstring) -- the selector
        # widget only ever calls set_channel(); it never touches
        # live_plot_view directly. AllChannelsPlotView is intentionally
        # NOT connected to this signal at all: it always shows all 32
        # channels regardless of what's selected here.
        channel_selector = ChannelSelectorWidget(self.live_view_model)
        self.live_view_model.channel_changed.connect(self.live_plot_view.set_channel)
        self.live_plot_view.set_channel(self.live_view_model.selected_channel)  # initial sync

        # Mode selection: same single-source-of-truth pattern. Mode
        # affects BOTH views, since both listen to processed_data_ready
        # (which is already mode-aware -- see LiveViewModel._process_window),
        # not to a per-view mode setting of their own.
        mode_selector = ModeSelectorWidget(self.live_view_model)

        # Stack holds both plots; only one is shown at a time.
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.live_plot_view)     # index 0: single channel
        self.plot_stack.addWidget(self.all_channels_view)  # index 1: all channels

        self.toggle_all_channels_btn = QPushButton("Plot All Channels")
        self.toggle_all_channels_btn.setCheckable(True)
        self.toggle_all_channels_btn.toggled.connect(self._on_toggle_all_channels)

        controls_row = QWidget()
        controls_layout = QHBoxLayout(controls_row)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.addWidget(channel_selector)
        controls_layout.addWidget(mode_selector)
        controls_layout.addWidget(self.toggle_all_channels_btn)
        controls_layout.addStretch(1)

        self.live_view = QWidget()
        live_layout = QVBoxLayout(self.live_view)
        live_layout.addWidget(connection_widget)
        live_layout.addWidget(controls_row)
        live_layout.addWidget(self.plot_stack)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.live_view, "Live")
        self.tabs.addTab(self.offline_view, "Offline Inspection")
        self.setCentralWidget(self.tabs)

        self.setStatusBar(QStatusBar())

        # NOTE: status/connection signals are wired to live_view_model,
        # since that's what actually owns the real TCP connection right
        # now. self.model (the offline fake stub) has its own separate
        # signals that aren't reflecting live connection state — this
        # will need to be unified once there's one shared Model.
        self.live_view_model.status_updated.connect(self._on_status_updated)
        self.live_view_model.connection_state_changed.connect(self._on_connection_state_changed)

    def _on_toggle_all_channels(self, checked: bool) -> None:
        if checked:
            self.plot_stack.setCurrentWidget(self.all_channels_view)
            self.toggle_all_channels_btn.setText("Show Single Channel")
        else:
            self.plot_stack.setCurrentWidget(self.live_plot_view)
            self.toggle_all_channels_btn.setText("Plot All Channels")

    def _on_status_updated(self, message: str) -> None:
        self.statusBar().showMessage(message, 5000)

    def _on_connection_state_changed(self, connected: bool) -> None:
        if not connected:
            # Auto-switch to Offline tab so the user can immediately
            # inspect what was just recorded.
            self.tabs.setCurrentWidget(self.offline_view)

    def closeEvent(self, event) -> None:
        self.live_view_model.disconnect()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())