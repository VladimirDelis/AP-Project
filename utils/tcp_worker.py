"""
models/tcp_worker.py

TCPWorker -- a QObject designed to live inside its own QThread, wrapping the
blocking EMGTCPClient so its blocking socket calls never run on the GUI
thread.

Why this exists
----------------
EMGTCPClient.connect() / receive_window() / stream() all block on raw
socket.recv() calls. If any of that ran on the main/GUI thread, the whole
UI would freeze every time it had to wait on the network for the next
window. TCPWorker never runs on the GUI thread: LiveViewModel moves this
object to a dedicated QThread via `moveToThread()`, so all the blocking
work happens off the GUI thread, and only lightweight Qt signals cross
back over to the GUI side.

Threading contract
-------------------
- `run()` is connected to `QThread.started` by whoever owns this worker
  (see LiveViewModel), so it executes on the worker thread once that
  thread starts. It performs the (blocking) connect + streaming loop and
  does not return until the stream ends, the connection drops, or
  `stop()` has interrupted it.
- `stop()` is the one deliberate exception to "only touch this object from
  its own thread": it's designed to be called directly from another
  thread (the GUI thread) while `run()` is blocked inside
  `EMGTCPClient.stream()`. See its docstring for why that's safe here.
- Every other method/attribute on this class should be treated as owned by
  the worker thread and only touched from `run()` / its helpers.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, Signal


class TCPWorker(QObject):
    """
    Runs EMGTCPClient's blocking connect/stream loop on a background thread.

    Signals
    -------
    data_received : Signal(np.ndarray)
        Emitted once per received window, shape (32, 18), dtype float64.
    status_updated : Signal(str)
        Human-readable status/error messages for the UI.
    connection_state_changed : Signal(bool)
        True once connected, False once disconnected/failed.
    finished : Signal()
        Emitted when `run()` returns, for any reason: a clean user-requested
        stop, a dropped connection, or a failed initial connect. LiveViewModel
        listens for this to know when it's safe to quit the QThread.
    """

    data_received = Signal(np.ndarray)
    status_updated = Signal(str)
    connection_state_changed = Signal(bool)
    finished = Signal()

    def __init__(self, client, parent=None) -> None:
        """
        Parameters
        ----------
        client : EMGTCPClient
            A not-yet-connected client instance this worker will drive.
        parent : QObject, optional
        """
        super().__init__(parent)
        self._client = client
        self._stop_requested = False

    def run(self) -> None:
        try:
            self._client.connect()
        except OSError as exc:
            self.status_updated.emit(f"Connection failed: {exc}")
            self.connection_state_changed.emit(False)
            self.finished.emit()
            return

        self.status_updated.emit(f"Connected to {self._client.host}:{self._client.port}")
        self.connection_state_changed.emit(True)

        try:
            self._client.stream(on_window=self._handle_window)
        except ConnectionError as exc:
            # Server closed the connection / connection lost mid-stream.
            self.status_updated.emit(f"Disconnected: {exc}")
        except OSError as exc:
            # Also the path taken when we deliberately close the socket in
            # stop() to interrupt a blocked receive_window() call.
            if self._stop_requested:
                self.status_updated.emit("Disconnected")
            else:
                self.status_updated.emit(f"Connection error: {exc}")
        finally:
            self.connection_state_changed.emit(False)
            self.finished.emit()

    def _handle_window(self, window: np.ndarray, index: int) -> None:
        """
        `stream()`'s per-window callback. Still runs on the worker thread;
        emitting a Qt signal here is the safe, correct way to hand the data
        off toward the GUI thread -- Qt marshals it across via the event
        loop for us, we don't need our own locking.
        """
        self.data_received.emit(window)

    def stop(self) -> None:
        """
        Ask the streaming loop to stop.

        Unlike the rest of this class, this method is meant to be called
        directly from the GUI thread *while* `run()` is blocked inside
        `EMGTCPClient.stream()` -> `receive_window()` -> a raw
        `socket.recv()`. A normal (queued) Qt signal connection wouldn't
        help here: the worker thread's event loop can't process a queued
        call while it's stuck inside that blocking call, so a signal-based
        "please stop" would just sit in the queue until the loop already
        happened to return on its own.

        The one thing that *does* reliably interrupt a blocked `recv()`
        from another thread is closing the socket out from under it. So
        that's what this does: close the client's socket, which makes the
        in-flight recv() (inside `receive_window()` / `_recv_exact`) raise,
        which unwinds `stream()`, which lets `run()` reach its `finally`
        block and emit `finished`. Setting `_stop_requested` first lets
        `run()` report a clean "Disconnected" instead of a scary-looking
        error message once that exception surfaces.
        """
        self._stop_requested = True
        self._client.close()