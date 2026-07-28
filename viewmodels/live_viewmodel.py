"""
viewmodels/live_viewmodel.py

LiveViewModel -- the "VM" in this app's MVVM structure.

Role in MVVM
------------
The ViewModel sits between the View (ConnectionWidget today, and later the
VisPy plot widget) and the Model layer (EMGTCPClient, run on a background
thread via TCPWorker):

    View  <---signals/methods--->  ViewModel  <---signals/methods--->  Model

- The View never talks to TCPWorker or EMGTCPClient directly. It only calls
  methods on the ViewModel and listens to signals re-exposed by the
  ViewModel.
- The ViewModel never touches Qt widgets -- it has no idea a QPushButton or
  QLineEdit exists. It just exposes signals, simple methods, and state.
- TCPWorker/EMGTCPClient know nothing about the ViewModel or View; they just
  emit signals / return data when asked.

This one-directional layering means the View can change (e.g. gain a VisPy
plot alongside the connection widget) independently of the data source, and
vice versa, without this middle layer caring about either change beyond a
small, well-contained area.

Threading ownership
--------------------
EMGTCPClient's connect/receive calls are blocking, so they can't run on the
GUI thread. This ViewModel owns the QThread + TCPWorker pair that makes that
possible:

- `connect(host, port)` creates a fresh EMGTCPClient + TCPWorker, moves the
  worker to a new QThread, wires up signals, and starts the thread. The
  worker's `run()` method (connected to `QThread.started`) then does the
  actual blocking connect + stream loop, entirely off the GUI thread.
- `disconnect()` asks the worker to stop (see TCPWorker.stop() for why that
  call is intentionally made directly, not via a queued signal). The
  worker's `finished` signal is already wired to quit the thread and clean
  up, so disconnecting doesn't block the GUI thread at all.
- `shutdown()` is a separate, synchronous "stop and join" method meant to be
  called once when the whole application is closing, to guarantee the
  worker thread has actually exited before the process does. It uses
  QThread's blocking `wait()`, which `disconnect()` deliberately avoids
  during normal use so the UI never stalls on it.

See models/tcp_worker.py for the full explanation of why `stop()` is called
directly across threads rather than through a queued signal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from utils.Dev_tcp_client_model import EMGTCPClient
from utils.tcp_worker import TCPWorker

VALID_MODES = ("original", "rms", "filtered")
MIN_CHANNEL = 0
MAX_CHANNEL = 31


class LiveViewModel(QObject):
    """
    ViewModel for the live signal view.

    Owns the QThread + TCPWorker pair that drives the (blocking)
    EMGTCPClient, and re-exposes their signals so the View only ever
    depends on this class.

    Signals (re-exposed to the View)
    ---------------------------------
    data_received : Signal(np.ndarray)
        Forwarded from TCPWorker, unchanged/raw. New window, shape
        (32, 18). Kept around for anything that specifically wants raw
        data; the plot views should NOT connect to this one (see
        `processed_data_ready` below).
    processed_data_ready : Signal(np.ndarray)
        The window views should actually display: `data_received`'s raw
        window run through `_process_window()`, i.e. through whatever the
        current mode (original/RMS/filtered) implies. This is what
        LivePlotView and AllChannelsPlotView connect to, so they stay
        completely agnostic to *what* the data represents -- they just
        draw whatever array they're handed. See `_process_window()` for
        the mode-dispatch logic itself.
    status_updated : Signal(str)
        Forwarded from TCPWorker. Status/error text for display.
    connection_state_changed : Signal(bool)
        Forwarded from TCPWorker. True = connected, False = not.
    channel_changed : Signal(int)
        Emitted whenever `set_channel()` updates the selection. This is
        what makes the ViewModel the single source of truth for "current
        channel": a channel-selector widget calls `set_channel()` and
        never touches the plot widget directly, and the plot widget (or
        anything else that cares) reacts to this signal instead of being
        told about a new channel by a sibling View. That keeps two Views
        from ever disagreeing about which channel is selected.
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
        Current connection state, mirrored from the worker.
    """

    data_received = Signal(np.ndarray)
    processed_data_ready = Signal(np.ndarray)
    status_updated = Signal(str)
    connection_state_changed = Signal(bool)
    channel_changed = Signal(int)
    mode_changed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """
        Parameters
        ----------
        parent : QObject, optional
            Standard Qt parent, for ownership/cleanup.
        """
        super().__init__(parent)

        # --- state owned by the ViewModel ---
        self._selected_channel: int = 0
        self._selected_mode: str = "original"
        self._is_connected: bool = False

        # Tracks which modes we've already printed a "processing function
        # not available" warning for, so a missing models.signal_processing
        # module doesn't spam the console once per incoming window (which
        # can arrive many times a second) -- just once per mode selection.
        self._processing_warned_modes: set = set()

        # --- threading ---
        self._thread: Optional[QThread] = None
        self._worker: Optional[TCPWorker] = None

    # ------------------------------------------------------------------
    # Commands -- called by the View in response to user actions
    # ------------------------------------------------------------------
    def connect(self, host: str, port: int) -> None:
        """
        Start connecting to the EMG TCP server on a background thread.

        Creates a fresh EMGTCPClient + TCPWorker, moves the worker onto a
        new QThread, wires up its signals, and starts the thread. The
        actual (blocking) connect + streaming work all happens inside
        TCPWorker.run(), on that thread -- this method itself returns
        immediately.

        Parameters
        ----------
        host : str
            Host/IP of the EMG TCP server.
        port : int
            Port of the EMG TCP server.
        """
        if self._thread is not None:
            # Already connected or connecting; ignore duplicate requests.
            return

        # host/port are set here, at construction time -- EMGTCPClient.connect()
        # takes no arguments, it just uses whatever host/port it was built with.
        client = EMGTCPClient(host=host, port=port)
        worker = TCPWorker(client)
        thread = QThread(self)
        worker.moveToThread(thread)

        # Worker -> ViewModel. The View never sees these; it only ever
        # connects to this class's own signals. Raw windows go through
        # _on_worker_window(), which forwards the raw data unchanged AND
        # runs it through the current mode's processing before re-emitting
        # it -- see _process_window() for that dispatch.
        worker.data_received.connect(self._on_worker_window)
        worker.status_updated.connect(self.status_updated.emit)
        worker.connection_state_changed.connect(self._on_connection_state_changed)

        # Thread lifecycle (the canonical Qt "worker object" pattern):
        #   - run() starts once the thread's event loop is up.
        #   - once run() finishes (clean stop, drop, or failed connect),
        #     the worker asks the thread to quit and schedules its own
        #     deletion.
        #   - once the thread has actually stopped, we clear our references.
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_thread_finished)

        self._worker = worker
        self._thread = thread
        thread.start()

    def disconnect(self) -> None:
        """
        Ask the current connection to stop.

        This does not block the GUI thread. It calls `TCPWorker.stop()`
        directly (an intentional, documented exception to "only touch the
        worker from its own thread" -- see TCPWorker.stop() for why that's
        safe/necessary here), which interrupts the worker's blocked socket
        read. The worker's `finished` signal is already wired (in
        `connect()`) to quit the thread and clean things up, so the rest of
        teardown happens asynchronously via signals.
        """
        if self._worker is None:
            return
        self._worker.stop()

    def set_channel(self, index: int) -> None:
        """
        Update the currently selected channel.

        Parameters
        ----------
        index : int
            Channel index, expected to be in range [0, 31].

        Raises
        ------
        ValueError
            If index is outside the valid channel range.
        """
        if not (MIN_CHANNEL <= index <= MAX_CHANNEL):
            raise ValueError(
                f"Channel index must be between {MIN_CHANNEL} and {MAX_CHANNEL}, got {index}"
            )
        if index == self._selected_channel:
            return
        self._selected_channel = index
        self.channel_changed.emit(index)

    def set_mode(self, mode: str) -> None:
        """
        Update the currently selected signal mode.

        This is what `_process_window()` reads on every incoming window to
        decide how to process it (see that method for the actual
        original/RMS/filtered dispatch). This method itself just validates
        and stores the selection -- it doesn't process anything directly.

        Parameters
        ----------
        mode : str
            One of "original", "rms", "filtered".

        Raises
        ------
        ValueError
            If mode is not one of the supported values.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if mode == self._selected_mode:
            return
        self._selected_mode = mode
        self.mode_changed.emit(mode)

    def shutdown(self) -> None:
        """
        Synchronously stop and join any running background thread.

        Intended to be called once, when the application itself is
        closing (e.g. from the main window's closeEvent), to guarantee the
        worker thread has fully exited before the process does -- otherwise
        Qt can print "QThread: Destroyed while thread is still running"
        warnings, or worse. This deliberately blocks briefly on
        `QThread.wait()`, which is why `disconnect()` avoids doing that
        during normal, interactive use.
        """
        if self._worker is not None:
            self._worker.stop()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)

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
    def _on_worker_window(self, window: np.ndarray) -> None:
        """
        Runs whenever TCPWorker delivers a new raw window.

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
        Turn one raw (32, 18) window into whatever should actually be
        displayed, based on the current mode. This is the single
        integration point between "current mode selection" and "what the
        views draw" -- LivePlotView and AllChannelsPlotView never know or
        care whether they're looking at original, RMS, or filtered data;
        they just plot whatever `processed_data_ready` hands them.

        GUESSED INTERFACE -- flagging for confirmation with Person A:
        `models/signal_processing.py` doesn't exist in this codebase yet,
        so the exact function names/signatures below are a guess based on
        the task description, not confirmed:

            from models.signal_processing import compute_rms
            compute_rms(window: np.ndarray) -> np.ndarray

            from models.signal_processing import apply_filter
            apply_filter(window: np.ndarray) -> np.ndarray

        Both are assumed to take one (32, 18) window and return one
        (32, 18) array (same shape, just RMS'd/filtered in place of the
        raw values) -- so that nothing downstream (the rolling buffers'
        shape checks in particular) needs to change. If RMS is actually
        meant to collapse each window down to one value per channel
        (e.g. shape (32,) or (32, 1)) rather than keep 18 samples,
        that's a plausible alternative reading of "RMS" and would need
        the two call sites below (and the views' rolling buffers) updated
        together -- confirm which one Person A implements.

        If the import fails because the module doesn't exist yet (or
        raises while running), this falls back to the raw window and
        prints a one-time-per-mode console warning via
        `_warn_processing_unavailable()`, rather than crashing or silently
        showing nothing.
        """
        mode = self._selected_mode

        if mode == "original":
            return window

        if mode == "rms":
            try:
                from models.signal_processing import compute_rms
            except ImportError:
                self._warn_processing_unavailable("rms", "models.signal_processing.compute_rms")
                return window
            try:
                return compute_rms(window)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
                self._warn_processing_unavailable("rms", f"compute_rms() (raised {exc!r})")
                return window

        if mode == "filtered":
            try:
                from models.signal_processing import apply_filter
            except ImportError:
                self._warn_processing_unavailable("filtered", "models.signal_processing.apply_filter")
                return window
            try:
                return apply_filter(window)
            except Exception as exc:  # noqa: BLE001 -- deliberately broad, see docstring
                self._warn_processing_unavailable("filtered", f"apply_filter() (raised {exc!r})")
                return window

        # set_mode() validates against VALID_MODES, so this shouldn't be
        # reachable -- but fail safe (show raw data) rather than silently
        # drop a window if it somehow ever is.
        return window

    def _warn_processing_unavailable(self, mode_name: str, missing_symbol: str) -> None:
        """
        Print a console warning the first time a mode's processing turns
        out to be unavailable (module missing, or the call itself raised),
        then stay quiet about it for the rest of this mode selection.

        Windows can arrive many times a second, and the underlying reason
        a mode isn't working isn't going to change from one window to the
        next -- printing this on every single window would flood the
        console without adding information, which risks the warning
        getting lost/ignored rather than noticed. Once per mode keeps it
        loud enough to be impossible to miss during integration testing,
        without becoming noise.
        """
        if mode_name in self._processing_warned_modes:
            return
        self._processing_warned_modes.add(mode_name)
        print(
            f"WARNING: '{mode_name}' mode is selected, but {missing_symbol} "
            "is not available (module not found, not merged in yet, or it "
            "raised an error). Falling back to unprocessed/original data "
            "for now."
        )

    def _on_connection_state_changed(self, connected: bool) -> None:
        """Track connection state locally, then forward it on to the View."""
        self._is_connected = connected
        self.connection_state_changed.emit(connected)

    def _on_thread_finished(self) -> None:
        """
        Runs once the QThread has fully stopped (whether that was triggered
        by a user-initiated disconnect() or the server dropping the
        connection on its own). Clears our references so a later
        connect() call is free to start a brand new thread/worker.
        """
        if self._thread is not None:
            self._thread.deleteLater()
        self._thread = None
        self._worker = None