"""
viewmodels/live_viewmodel.py

LiveViewModel -- the "VM" in this app's MVVM structure.

Role in MVVM
------------
The ViewModel sits between the View (ConnectionWidget, LivePlotView,
AllChannelsPlotView, the selector widgets) and the Model layer (a shared
SharedSessionModel, adapting Person A's TcpClientModel):

    View  <---signals/methods--->  ViewModel  <---signals/methods--->  Model

- The View never talks to SharedSessionModel or TcpClientModel directly.
  It only calls methods on the ViewModel and listens to signals re-exposed
  by the ViewModel.
- The ViewModel never touches Qt widgets -- it has no idea a QPushButton
  or QLineEdit exists. It just exposes signals, simple methods, and state.
- SharedSessionModel/TcpClientModel know nothing about the ViewModel or
  View; they just emit signals / return data when asked.

ARCHITECTURE CHANGE from the previous version of this file: this used to
own a QThread + TCPWorker pair to keep a *blocking* EMGTCPClient off the
GUI thread. TcpClientModel (Person A's real model) is different: it uses a
non-blocking socket polled by its own QTimer, so nothing here ever blocks
the GUI thread in the first place -- there's no thread to own anymore.
LiveViewModel is now a thin pass-through: it takes an already-constructed
SharedSessionModel via constructor injection (rather than building its own
client), forwards its signals, and runs incoming data through
`_process_window()` before re-emitting it as `processed_data_ready`.

The shared model itself (one TcpClientModel wrapped in one
SharedSessionModel) is constructed once in main.py and handed to both this
class and Person C's OfflineViewModel, so both ViewModels observe the same
live session instead of each opening their own separate connection.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, Signal

from models.signal_processing import apply_mode

VALID_MODES = ("original", "rms", "filtered")
MIN_CHANNEL = 0
MAX_CHANNEL = 31


class LiveViewModel(QObject):
    """
    ViewModel for the live signal view.

    Wraps a SharedSessionModel (constructor injection) and re-exposes its
    signals so the View only ever depends on this class.

    Signals (re-exposed to the View)
    ---------------------------------
    data_received : Signal(np.ndarray)
        Forwarded from the shared model, unchanged/raw. Shape is
        (channels, n_samples) with n_samples VARYING call to call (see
        SharedSessionModel's docstring) -- not a fixed (32, 18) window.
        Kept around for anything that specifically wants raw data; the
        plot views should NOT connect to this one (see
        `processed_data_ready` below).
    processed_data_ready : Signal(np.ndarray)
        The array views should actually display: `data_received`'s raw
        chunk run through `_process_window()`, i.e. through whatever the
        current mode (original/RMS/filtered) implies. This is what
        LivePlotView and AllChannelsPlotView connect to, so they stay
        completely agnostic to *what* the data represents -- they just
        draw whatever array they're handed. See `_process_window()` for
        the mode-dispatch logic itself.
    status_updated : Signal(str)
        Forwarded from the shared model. Status/error text for display.
    connection_state_changed : Signal(bool)
        Forwarded from the shared model. True = connected, False = not.
    channel_changed : Signal(int)
        Emitted whenever `set_channel()` updates the selection. This is
        what makes the ViewModel the single source of truth for "current
        channel" for the LIVE single-channel plot: a channel-selector
        widget calls `set_channel()` and never touches the plot widget
        directly, and the plot widget reacts to this signal instead.
        (Separate from SharedSessionModel's own set_channel(), which is
        for OfflineViewModel's pull-based rolling_buffer access -- these
        two "selected channel" concepts are intentionally independent,
        since a user could reasonably be watching one channel live while
        inspecting a different one offline.)
    mode_changed : Signal(str)
        Emitted whenever `set_mode()` updates the selection, for the same
        single-source-of-truth reason as `channel_changed`.

    State (read via properties)
    ----------------------------
    selected_channel : int
        Currently selected channel index, 0-31.
    selected_mode : str
        Currently selected signal mode: "original" / "rms" / "filtered".
    is_connected : bool
        Current connection state, mirrored from the shared model.
    """

    data_received = Signal(np.ndarray)
    processed_data_ready = Signal(np.ndarray)
    status_updated = Signal(str)
    connection_state_changed = Signal(bool)
    channel_changed = Signal(int)
    mode_changed = Signal(str)

    def __init__(self, shared_model, parent: Optional[QObject] = None) -> None:
        """
        Parameters
        ----------
        shared_model : SharedSessionModel
            An already-constructed adapter (see models/model_adapter.py),
            shared with OfflineViewModel. This is the ONLY place this
            class needs to change if the underlying model's construction
            ever changes -- everything else here just calls methods on
            whatever it's handed.
        parent : QObject, optional
            Standard Qt parent, for ownership/cleanup.
        """
        super().__init__(parent)

        self._shared_model = shared_model

        # --- state owned by the ViewModel ---
        self._selected_channel: int = 0
        self._selected_mode: str = "original"
        self._is_connected: bool = False

        # Model -> ViewModel. The View never sees these; it only ever
        # connects to this class's own signals. Raw chunks go through
        # _on_shared_model_window(), which forwards the raw data unchanged
        # AND runs it through the current mode's processing before
        # re-emitting it -- see _process_window() for that dispatch.
        self._shared_model.data_received.connect(self._on_shared_model_window)
        self._shared_model.status_updated.connect(self.status_updated.emit)
        self._shared_model.connection_state_changed.connect(self._on_connection_state_changed)

    # ------------------------------------------------------------------
    # Commands -- called by the View in response to user actions
    # ------------------------------------------------------------------
    def connect(self, host: str, port: int) -> None:
        """
        Ask the shared model to connect.

        No thread management here anymore -- the shared model's
        underlying TcpClientModel uses a non-blocking socket polled by its
        own QTimer, so this is just a direct delegating call.

        Parameters
        ----------
        host : str
            Host/IP of the EMG TCP server.
        port : int
            Port of the EMG TCP server.
        """
        self._shared_model.connect(host, port)

    def disconnect(self) -> None:
        """Ask the shared model to disconnect. Direct delegating call."""
        self._shared_model.disconnect()

    def set_channel(self, index: int) -> None:
        """
        Update the currently selected channel for the LIVE single-channel
        view.

        Parameters
        ----------
        index : int
            Channel index, expected to be in range [0, 31]. An out-of-range
            index is rejected gracefully (a status message is emitted and
            the call is a no-op) rather than raising -- matches
            OfflineViewModel.set_channel()'s pattern, since both selector
            widgets already hard-bound their input to this range and an
            exception here would have nowhere safe to be caught.
        """
        if not (MIN_CHANNEL <= index <= MAX_CHANNEL):
            self.status_updated.emit(
                f"Ignored invalid channel index {index}: must be between "
                f"{MIN_CHANNEL} and {MAX_CHANNEL}."
            )
            return
        if index == self._selected_channel:
            return
        self._selected_channel = index
        self.channel_changed.emit(index)

    def set_mode(self, mode: str) -> None:
        """
        Update the currently selected signal mode.

        This is what `_process_window()` reads on every incoming chunk to
        decide how to process it (see that method for the actual
        original/RMS/filtered dispatch). This method itself just validates
        and stores the selection -- it doesn't process anything directly.

        Parameters
        ----------
        mode : str
            One of "original", "rms", "filtered". An unrecognized mode is
            rejected gracefully (a status message is emitted and the call
            is a no-op) rather than raising -- matches
            OfflineViewModel.set_mode()'s pattern, since the selector
            widget already only ever offers these three values.
        """
        if mode not in VALID_MODES:
            self.status_updated.emit(
                f"Ignored invalid mode {mode!r}: must be one of {VALID_MODES}."
            )
            return
        if mode == self._selected_mode:
            return
        self._selected_mode = mode
        self.mode_changed.emit(mode)

    # ------------------------------------------------------------------
    # Read-only state accessors -- for the View to query current state
    # ------------------------------------------------------------------
    @property
    def selected_channel(self) -> int:
        return self._selected_channel

    @property
    def selected_mode(self) -> str:
        return self._selected_mode

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _on_shared_model_window(self, window: np.ndarray) -> None:
        """
        Runs whenever the shared model delivers a new raw chunk.

        Forwards it unchanged via `data_received` (kept for anything that
        still wants raw data), then separately runs it through
        `_process_window()` and forwards *that* result via
        `processed_data_ready` -- which is the signal the plot views
        should actually be connected to.
        """
        self.data_received.emit(window)
        processed = self._process_window(window)
        self.processed_data_ready.emit(processed)

    def _process_window(self, window: np.ndarray) -> np.ndarray:
        """
        Turn one raw (channels, n_samples) chunk into whatever should
        actually be displayed, based on the current mode, using the real
        `models.signal_processing.apply_mode()`. This is the single
        integration point between "current mode selection" and "what the
        views draw" -- LivePlotView and AllChannelsPlotView never know or
        care whether they're looking at original, RMS, or filtered data;
        they just plot whatever `processed_data_ready` hands them.

        Processes the FULL (channels, n_samples) chunk in one call, rather
        than extracting the currently-selected channel's row first:
        apply_mode already loops per-channel internally for 2D input, so
        one call here covers both AllChannelsPlotView (which needs all
        channels anyway) and LivePlotView (which extracts its own single
        row from this same processed array in its existing
        append_window(), same as it already does for raw data). Processing
        per-selected-channel instead would mean calling apply_mode() a
        second time on the same data for no benefit.

        sample_rate_hz now comes from the shared model's real
        `sampling_rate` property (from the actual TcpClientModel), rather
        than a guessed hardcoded constant -- this resolves the earlier
        placeholder/TODO from before the real models existed.

        KNOWN LIMITATION (confirmed against the real module): `apply_mode
        (..., 'filtered', ...)` internally requires at least ~27 samples
        for `scipy.signal.filtfilt` to run (falls back to unchanged input
        otherwise). Individual chunks here can still be shorter than that
        (TcpClientModel's default is 18 samples per packet, sometimes
        several packets concatenated if multiple arrived between polls) --
        so "Filtered" mode may still look identical to "Original" on
        short/typical chunks, same limitation flagged before, now just
        against the real model's real chunk sizes instead of a fixed
        assumption of 18. RMS is unaffected (its window clamps to whatever
        length it's given).
        """
        return apply_mode(window, self._selected_mode, self._shared_model.sampling_rate)

    def _on_connection_state_changed(self, connected: bool) -> None:
        """Track connection state locally, then forward it on to the View."""
        self._is_connected = connected
        self.connection_state_changed.emit(connected)