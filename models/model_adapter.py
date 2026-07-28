"""
models/model_adapter.py

SharedSessionModel -- adapts Person A's TcpClientModel to the interface
this app's ViewModels actually expect, WITHOUT modifying TcpClientModel or
signal_processing.py (both treated as read-only/owned by Person A, to
avoid merge conflicts).

Two ViewModels need to share ONE live session:
- LiveViewModel (push-based): wants signals -- connect(), disconnect(),
  status_updated, connection_state_changed, data_received.
- OfflineViewModel (pull-based, already built by Person C, also read-only
  here): wants methods -- has_data(), get_window(),
  get_window_all_channels(), has_full_session_data(),
  get_full_session_channel(channel_index).

TcpClientModel provides neither interface exactly. This class WRAPS a
TcpClientModel instance via composition (not inheritance), so it can freely
rename signals and add accessor methods without touching Person A's class
at all. Every bridged mismatch is flagged in a comment below (search for
"MISMATCH") so this whole adapter is easy to delete later if Person A's
model is ever updated to match either ViewModel's original contract
directly -- at which point callers could just use TcpClientModel itself
again in place of this class.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal

from models.tcp_client_model import TcpClientModel

MIN_CHANNEL = 0
MAX_CHANNEL = 31


class SharedSessionModel(QObject):
    """
    Adapter around a single TcpClientModel instance, shared by both
    LiveViewModel and OfflineViewModel so they see the same session data
    through one object instead of two independently-connected clients.

    Signals
    -------
    status_updated : Signal(str)
        Forwarded directly from TcpClientModel -- names already match.
    connection_state_changed : Signal(bool)
        MISMATCH BRIDGED: TcpClientModel emits this as `connection_changed`.
        Re-exposed under the name the rest of the app already expects;
        purely a rename, no logic change.
    data_received : Signal(np.ndarray)
        Forwarded directly from TcpClientModel -- names already match.
        NOTE: chunks are NOT a fixed shape. TcpClientModel polls on a
        QTimer and emits whatever complete packets arrived since the last
        poll (possibly several concatenated together), so shape is
        (channels, new_samples) with new_samples varying call to call.
        Anything downstream must not assume a fixed sample count (see
        LivePlotView/AllChannelsPlotView's append_window()).
    """

    status_updated = Signal(str)
    connection_state_changed = Signal(bool)
    data_received = Signal(np.ndarray)

    def __init__(self, model: TcpClientModel, parent: Optional[QObject] = None) -> None:
        """
        Parameters
        ----------
        model : TcpClientModel
            An existing, not-yet-connected TcpClientModel instance. Owned
            and constructed by the caller (main.py) -- this adapter only
            wraps it, it doesn't manage its lifetime.
        parent : QObject, optional
        """
        super().__init__(parent)
        self._model = model

        # Used only by get_window() below, for OfflineViewModel's
        # single-channel pull from rolling_buffer -- unrelated to
        # LiveViewModel's own, separate channel selection (which operates
        # on pushed/processed data via LiveViewModel.set_channel(), not
        # through this adapter at all).
        self._selected_channel = 0

        # --- MISMATCH: signal rename ---
        # TcpClientModel emits `connection_changed`; everything downstream
        # (built against the original contract) expects
        # `connection_state_changed`. Straight rename, re-emitted as-is.
        self._model.connection_changed.connect(self.connection_state_changed.emit)

        # These two already match by name -- straight passthrough.
        self._model.status_updated.connect(self.status_updated.emit)
        self._model.data_received.connect(self.data_received.emit)

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self, host: str, port: int) -> None:
        """
        MISMATCH BRIDGED: callers expect connect(host, port), but
        TcpClientModel.connect() takes NO arguments -- it reads host/port
        off plain instance attributes (self.host / self.port) set ahead of
        time. This just sets those attributes, then calls the real
        connect() with no arguments.
        """
        self._model.host = host
        self._model.port = port
        self._model.connect()

    def disconnect(self) -> None:
        """Straight passthrough -- name and signature already match."""
        self._model.disconnect()

    # ------------------------------------------------------------------
    # Shared state
    # ------------------------------------------------------------------
    @property
    def sampling_rate(self):
        """Passthrough to TcpClientModel.sampling_rate -- names already match."""
        return self._model.sampling_rate

    def set_channel(self, index: int) -> None:
        """
        MISMATCH BRIDGED: TcpClientModel has no concept of "selected
        channel" -- it just stores every channel in rolling_buffer/
        full_buffer. get_window() below needs to know which single
        channel to slice out, so that selection has to live somewhere;
        it lives here since this adapter is what OfflineViewModel actually
        calls get_window() on.
        """
        if not (MIN_CHANNEL <= index <= MAX_CHANNEL):
            raise ValueError(f"Channel index must be between {MIN_CHANNEL} and {MAX_CHANNEL}, got {index}")
        self._selected_channel = index

    # ------------------------------------------------------------------
    # Pull-based accessors -- for OfflineViewModel
    # ------------------------------------------------------------------
    # MISMATCH BRIDGED (all five methods below): none of these exist on
    # TcpClientModel. It only exposes raw numpy attributes (rolling_buffer,
    # full_buffer) plus sampling_rate. Everything here derives
    # OfflineViewModel's expected method-based interface from those raw
    # attributes, computing a matching time axis from sampling_rate each
    # call. Returned arrays are views into TcpClientModel's current
    # buffers, not copies -- safe for a pull-based read like this because
    # TcpClientModel replaces the whole buffer array on each new chunk
    # (np.concatenate assigned back to self.rolling_buffer/full_buffer)
    # rather than mutating one in place, so a previously-returned view
    # simply becomes a stale-but-still-valid snapshot, never corrupted
    # data.

    def has_data(self) -> bool:
        return self._model.rolling_buffer.shape[1] >= 2

    def get_window(self) -> Tuple[np.ndarray, np.ndarray]:
        """Single-channel (currently selected via set_channel()) view of rolling_buffer."""
        y = self._model.rolling_buffer[self._selected_channel]
        t = np.arange(y.shape[0]) / self._model.sampling_rate
        return t, y

    def get_window_all_channels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Full rolling_buffer (all channels) plus a matching time axis."""
        y = self._model.rolling_buffer
        t = np.arange(y.shape[1]) / self._model.sampling_rate
        return t, y

    def has_full_session_data(self) -> bool:
        return self._model.full_buffer.shape[1] >= 2

    def get_full_session_channel(self, channel_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Single channel row from full_buffer (the whole recorded session so far), plus time axis."""
        y = self._model.full_buffer[channel_index]
        t = np.arange(y.shape[0]) / self._model.sampling_rate
        return t, y