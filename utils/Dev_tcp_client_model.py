import socket
import numpy as np


class EMGTCPClient:
    def __init__(self, host='localhost', port=12345,
                 channels=32, samples_per_packet=18, dtype=np.float64):
        self.host = host
        self.port = port
        self.channels = channels
        self.samples_per_packet = samples_per_packet
        self.dtype = dtype
        self.bytes_per_packet = channels * samples_per_packet * np.dtype(dtype).itemsize
        self.sock = None

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        print(f"Connected to {self.host}:{self.port}")
        print(f"Expecting packets of {self.bytes_per_packet} bytes "
              f"({self.channels} ch x {self.samples_per_packet} samples, {self.dtype})")

    def _recv_exact(self, num_bytes):
        """Block until exactly num_bytes have been received.
        A single recv() call is NOT guaranteed to return a full packet's
        worth of data over TCP -- it can arrive split across multiple reads.
        This loop accumulates until we have exactly what we asked for."""
        buf = bytearray()
        while len(buf) < num_bytes:
            chunk = self.sock.recv(num_bytes - len(buf))
            if not chunk:
                raise ConnectionError("Server closed the connection")
            buf.extend(chunk)
        return bytes(buf)

    def receive_window(self):
        """Receive one window and return it as a (channels, samples) array."""
        raw = self._recv_exact(self.bytes_per_packet)
        window = np.frombuffer(raw, dtype=self.dtype)
        window = window.reshape((self.channels, self.samples_per_packet), order="C")
        return window

    def stream(self, on_window=None, max_windows=None):
        """Continuously receive windows, calling on_window(window, index) for each.
        If on_window is None, just prints a summary per window."""
        index = 0
        try:
            while max_windows is None or index < max_windows:
                window = self.receive_window()
                if on_window is not None:
                    on_window(window, index)
                else:
                    print(f"Window {index}: shape={window.shape}, "
                          f"mean={window.mean():.4f}")
                index += 1
        except ConnectionError as e:
            print(f"Stream ended: {e}")

    def close(self):
        if self.sock:
            self.sock.close()


if __name__ == "__main__":
    client = EMGTCPClient(host='localhost', port=12345)
    client.connect()
    try:
        client.stream(max_windows=50)  # remove max_windows for continuous streaming
    except KeyboardInterrupt:
        print("\nStopping client...")
    finally:
        client.close()