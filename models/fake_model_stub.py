"""
STUB — stand-in for Person A's real Model layer.

This exists ONLY so the offline view / main window (Person C's
responsibility) can be built and tested end-to-end before A's real TCP
client is ready, per the plan's "B and C can develop against mocked/fake
data" approach.

REPLACE the internals of this file with Person A's real TcpClientModel
once it lands — keep the same signal names/signatures below (that's the
agreed contract) so nothing else needs to change:

Signals:
    status_updated(str)             -- human-readable status/error messages
    connection_state_changed(bool)  -- True when connected, False otherwise

Expected methods (used by ViewModels/Views):
    connect(host, port)
    disconnect()
    has_full_session_data() -> bool
    get_full_session_channel(channel_index) -> (t: np.ndarray, y: np.ndarray)
    has_data() -> bool                      (live / rolling window)
    get_window() -> (t, y)                  (live single channel)
    get_window_all_channels() -> (t, y)     (live all channels, y shape (N_CHANNELS, n))
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from models.config import N_CHANNELS, SAMPLING_RATE_HZ, ROLLING_WINDOW_SECONDS


class FakeSignalModel(QObject):
    status_updated = Signal(str)
    connection_state_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sampling_rate = SAMPLING_RATE_HZ
        self._connected = False
        self._t = 0.0
        window_samples = int(ROLLING_WINDOW_SECONDS * SAMPLING_RATE_HZ)
        self._rolling = np.zeros((N_CHANNELS, 0))
        self._rolling_max = window_samples
        self._full = np.zeros((N_CHANNELS, 0))

        self._timer = QTimer(self)
        self._timer.setInterval(20)
        self._timer.timeout.connect(self._generate_fake_packet)

    def connect(self, host: str, port: int) -> None:
        # Fake connect always "succeeds" so C can test without a real server.
        self._connected = True
        self._timer.start()
        self.status_updated.emit(f"[FAKE MODEL] Connected to {host}:{port}")
        self.connection_state_changed.emit(True)

    def disconnect(self) -> None:
        self._connected = False
        self._timer.stop()
        self.status_updated.emit("[FAKE MODEL] Disconnected")
        self.connection_state_changed.emit(False)

    def _generate_fake_packet(self) -> None:
        n_new = 5
        t = self._t + np.arange(n_new) / self.sampling_rate
        self._t += n_new / self.sampling_rate
        packet = np.vstack(
            [np.sin(2 * np.pi * (1 + ch) * t) + 0.1 * np.random.randn(n_new)
             for ch in range(N_CHANNELS)]
        )
        self._rolling = np.concatenate([self._rolling, packet], axis=1)[:, -self._rolling_max:]
        self._full = np.concatenate([self._full, packet], axis=1)

    def has_data(self) -> bool:
        return self._rolling.shape[1] >= 2

    def get_window(self):
        y = self._rolling[0]
        x = np.arange(y.shape[0]) / self.sampling_rate
        return x, y

    def get_window_all_channels(self):
        y = self._rolling
        x = np.arange(y.shape[1]) / self.sampling_rate
        return x, y

    def has_full_session_data(self) -> bool:
        return self._full.shape[1] >= 2

    def get_full_session_channel(self, channel_index: int):
        y = self._full[channel_index]
        x = np.arange(y.shape[0]) / self.sampling_rate
        return x, y
