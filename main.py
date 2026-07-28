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
    QScrollArea,
)

from viewmodels.offline_viewmodel import OfflineViewModel
from views.offline_view import OfflineView

from models.tcp_client_model import TcpClientModel
from models.model_adapter import SharedSessionModel

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
        # ONE shared session, seen by both ViewModels.
        #
        # TcpClientModel and signal_processing.py are Person A's real
        # models -- both treated as read-only here (see
        # models/model_adapter.py's own docstring for why). TcpClientModel
        # itself doesn't match either ViewModel's expected interface
        # exactly (different signal names, different connect() signature,
        # method-based vs. attribute-based data access), so
        # SharedSessionModel bridges those mismatches without touching
        # Person A's file. Constructing ONE TcpClientModel + ONE
        # SharedSessionModel here, then handing that single adapter
        # instance to BOTH ViewModels below, is what makes the Offline
        # view actually see the Live connection's real data now --
        # previously each side had its own separate, disconnected model.
        # ------------------------------------------------------------
        # sampling_rate=2000 confirmed by running the actual test server and
        # reading its "Sampling rate: 2000 Hz" startup output. MUST be
        # re-confirmed if the grading server uses a different recording.pkl
        # file with a different sample rate.
        self.model = TcpClientModel(sampling_rate=2000)
        self.shared_model = SharedSessionModel(self.model)

        self.offline_view_model = OfflineViewModel(self.shared_model)
        self.offline_view = OfflineView(self.offline_view_model)

        # --- Live side (Person B) ---
        # LivePlotView/AllChannelsPlotView are plain Views: they take
        # config values (sample_rate_hz etc.), not a ViewModel, and are
        # wired up via explicit signal connections below.
        self.live_view_model = LiveViewModel(self.shared_model)

        connection_widget = ConnectionWidget(self.live_view_model)
        # sample_rate_hz now comes from the shared model's real
        # sampling_rate (TcpClientModel's actual attribute) instead of a
        # hardcoded 2000 duplicated in three places -- resolves the TODO
        # that used to be in LiveViewModel about this exact duplication.
        self.live_plot_view = LivePlotView(sample_rate_hz=self.model.sampling_rate)
        self.all_channels_view = AllChannelsPlotView(sample_rate_hz=self.model.sampling_rate)

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

        # AllChannelsPlotView enforces a minimum height for all 32 channels
        # (N_CHANNELS * MIN_PX_PER_CHANNEL + 60, ~636px) so its labels never
        # overlap -- but that can exceed the available window/screen height.
        # Wrapping it in a QScrollArea lets the widget render at its full,
        # unsqueezed natural size while the scroll area itself stays within
        # whatever space is actually available, showing a vertical scrollbar
        # instead of clipping any channels off the top/bottom.
        self.all_channels_scroll = QScrollArea()
        self.all_channels_scroll.setWidget(self.all_channels_view)
        self.all_channels_scroll.setWidgetResizable(True)
        # Enough viewport height to show roughly half the channels at once
        # without excessive scrolling, while still leaving room to shrink on
        # small screens instead of forcing the full ~636px minimum.
        self.all_channels_scroll.setMinimumHeight(320)

        # Stack holds both plots; only one is shown at a time.
        self.plot_stack = QStackedWidget()
        self.plot_stack.addWidget(self.live_plot_view)          # index 0: single channel
        self.plot_stack.addWidget(self.all_channels_scroll)     # index 1: all channels (scrollable)

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

        # Both ViewModels now observe the SAME shared_model, so this is no
        # longer "only reflecting the Live side" the way it used to when
        # self.model was a separate, disconnected fake stub -- there's
        # only one real connection now, and both tabs see it.
        self.live_view_model.status_updated.connect(self._on_status_updated)
        self.live_view_model.connection_state_changed.connect(self._on_connection_state_changed)

    def _on_toggle_all_channels(self, checked: bool) -> None:
        if checked:
            self.plot_stack.setCurrentWidget(self.all_channels_scroll)
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