import socket

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal


class TcpClientModel(QObject):
    """
    TCP client model for receiving streamed EMG data.

    Owns the raw socket connection, reconstructs packets from the raw byte
    stream, and keeps two buffers:
    - rolling_buffer: only the newest `rolling_window_seconds` of data (for the live plot)
    - full_buffer: everything received since connecting (for offline inspection)

    Polls the socket on its own QTimer and emits Qt signals when new data
    arrives, so a ViewModel only needs to connect() and listen.
    """

    data_received = Signal(object)  # newly parsed chunk: np.ndarray, shape (channels, new_samples)
    status_updated = Signal(str)
    connection_changed = Signal(bool)

    def __init__(
        self,
        host="localhost",
        port=12345,
        channels=32,
        samples_per_packet=18,
        sampling_rate=2000,
        rolling_window_seconds=10,
        poll_interval_ms=10,
        parent=None,
    ):
        super().__init__(parent)

        self.host = host
        self.port = port
        self.channels = channels
        self.samples_per_packet = samples_per_packet
        self.sampling_rate = sampling_rate

        # Must match the dtype the server uses before calling .tobytes().
        self.dtype = np.float64

        self._packet_size_bytes = (
            channels * samples_per_packet * np.dtype(self.dtype).itemsize
        )

        self.rolling_window_samples = int(sampling_rate * rolling_window_seconds)

        self._byte_buffer = bytearray()
        self.rolling_buffer = np.empty((channels, 0), dtype=self.dtype)
        self.full_buffer = np.empty((channels, 0), dtype=self.dtype)

        self.total_samples_received = 0

        self._socket = None
        self.is_connected = False

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_interval_ms = poll_interval_ms

    def connect(self):
        """Open the TCP connection and start polling for data."""
        if self.is_connected:
            return

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(3)
            self._socket.connect((self.host, self.port))
            self._socket.setblocking(False)
        except OSError as error:
            self.status_updated.emit(f"Could not connect to {self.host}:{self.port} ({error})")
            self._socket = None
            return

        self.is_connected = True
        self.connection_changed.emit(True)
        self.status_updated.emit(f"Connected to {self.host}:{self.port}")
        self._poll_timer.start(self._poll_interval_ms)

    def disconnect(self):
        """Stop polling and close the socket."""
        if not self.is_connected:
            return

        self._poll_timer.stop()
        self.is_connected = False

        if self._socket is not None:
            self._socket.close()
            self._socket = None

        self.connection_changed.emit(False)
        self.status_updated.emit("Disconnected.")

    def _poll(self):
        """Timer callback: read whatever bytes are available and emit new packets."""
        new_bytes = self._read_available_bytes()
        if new_bytes is None:
            return  # disconnected during read

        if new_bytes:
            self._byte_buffer.extend(new_bytes)

        new_chunk = self._extract_packets()
        if new_chunk is not None:
            self._append_to_buffers(new_chunk)
            self.data_received.emit(new_chunk)

    def _read_available_bytes(self):
        """Drain everything currently available on the non-blocking socket."""
        chunks = []
        while True:
            try:
                data = self._socket.recv(4096)
            except BlockingIOError:
                break  # no more data available right now
            except OSError as error:
                self.status_updated.emit(f"Connection lost: {error}")
                self.disconnect()
                return None

            if not data:
                self.status_updated.emit("Server closed the connection.")
                self.disconnect()
                return None

            chunks.append(data)

        return b"".join(chunks)

    def _extract_packets(self):
        """
        TCP is a byte stream: one recv() does not necessarily line up with one
        packet boundary. Pull out only complete packets, leave partial bytes
        in the buffer for next time.
        """
        packets = []
        while len(self._byte_buffer) >= self._packet_size_bytes:
            packet_bytes = self._byte_buffer[: self._packet_size_bytes]
            del self._byte_buffer[: self._packet_size_bytes]

            packet = np.frombuffer(packet_bytes, dtype=self.dtype).reshape(
                self.channels, self.samples_per_packet
            )
            packets.append(packet)

        if not packets:
            return None

        return np.concatenate(packets, axis=1)

    def _append_to_buffers(self, new_chunk):
        self.total_samples_received += new_chunk.shape[1]

        self.full_buffer = np.concatenate((self.full_buffer, new_chunk), axis=1)

        self.rolling_buffer = np.concatenate((self.rolling_buffer, new_chunk), axis=1)
        if self.rolling_buffer.shape[1] > self.rolling_window_samples:
            self.rolling_buffer = self.rolling_buffer[:, -self.rolling_window_samples:]

    def get_signal_time_seconds(self):
        """Elapsed recording time based on samples received, not wall-clock time."""
        return self.total_samples_received / self.sampling_rate
