"""
Offline ViewModel — Person C's responsibility.

Owns offline-inspection state (selected channel, selected signal mode) and
pulls processed data from the Model's full-session buffer. The Model is
injected (dependency-injected) so this works against either the fake stub
or Person A's real TcpClientModel without any code changes here, as long
as the Model exposes:

    has_full_session_data() -> bool
    get_full_session_channel(channel_index) -> (t, y)
    sampling_rate: float

Contains no GUI/plotting code (MVVM: ViewModel layer).
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal, Slot

from models import signal_processing
from models.config import N_CHANNELS


class OfflineViewModel(QObject):
    channel_changed = Signal(int)
    mode_changed = Signal(str)
    data_changed = Signal()  # tells the View "re-read and redraw"

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        self.selected_channel: int = 0
        self.signal_mode: str = "original"  # 'original' | 'rms' | 'filtered'

        # Redraw the offline plot whenever new data arrives, in case the
        # dialog/tab is left open while still recording live.
        if hasattr(model, "status_updated"):
            model.status_updated.connect(lambda _msg: self.data_changed.emit())

    @Slot(int)
    def set_channel(self, channel_index: int) -> None:
        if not (0 <= channel_index < N_CHANNELS):
            return
        self.selected_channel = channel_index
        self.channel_changed.emit(channel_index)
        self.data_changed.emit()

    @Slot(str)
    def set_mode(self, mode: str) -> None:
        if mode not in ("original", "rms", "filtered"):
            return
        self.signal_mode = mode
        self.mode_changed.emit(mode)
        self.data_changed.emit()

    def has_data(self) -> bool:
        return self.model.has_full_session_data()

    def get_current_channel_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (time_axis, values) for the currently selected channel
        and mode, ready to plot."""
        if not self.has_data():
            return np.array([]), np.array([])
        t, y = self.model.get_full_session_channel(self.selected_channel)
        processed = signal_processing.apply_mode(y, self.signal_mode, self.model.sampling_rate)
        return t, processed
