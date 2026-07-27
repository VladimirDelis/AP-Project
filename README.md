# AP-Project

TCP signal visualization application for the Applied Programming 2026 final project.

## Installation

```bash
pip install -r requirements.txt
```

## Model layer (`models/`)

### TCP connection

`TcpClientModel` (`models/tcp_client_model.py`) connects to the course-provided TCP server over a plain socket.

- Configure `host`/`port` when constructing the model (defaults: `localhost`, `12345`).
- Call `connect()` / `disconnect()` to start/stop streaming. Once connected, the model polls the socket on an internal `QTimer` (every 10 ms by default) — no manual polling needed from the ViewModel.
- Connection errors (server not running, wrong port, dropped connection) are caught and reported via the `status_updated` signal instead of raising/crashing the app.

### Data format

32 channels x 18 samples per packet, `float64`, raw bytes (4608 bytes/packet). The model reassembles packets from the raw byte stream since TCP does not preserve message boundaries (one `recv()` may contain a partial or multiple packets).

### Buffering

Two buffers are maintained from the same incoming data:
- `rolling_buffer`: trimmed to the newest `rolling_window_seconds` (default 10s) — feeds the live view.
- `full_buffer`: grows for the entire session, never trimmed — feeds the offline view after disconnecting.

### Signals exposed (for ViewModel integration)

- `data_received(object)`: emitted with each newly parsed chunk, `np.ndarray` shape `(channels, new_samples)`.
- `status_updated(str)`: human-readable status/error messages (connect success, connection lost, disconnect, etc.).
- `connection_changed(bool)`: emitted on connect/disconnect.

### Signal processing (`models/signal_processing.py`)

Pure, GUI-agnostic functions operating on `(channels, samples)` arrays — usable identically for live (on `rolling_buffer`) and offline (on `full_buffer`) data:

- `bandpass_filter(data, sampling_rate, low_cut=20, high_cut=450, order=4)`: zero-phase Butterworth bandpass (`scipy.signal.butter` + `filtfilt`). Cutoffs (20-450 Hz) are standard EMG values. Raises `ValueError` on invalid parameters or insufficient samples to filter, for the ViewModel to catch and display.
- `compute_rms(data, sampling_rate, window_ms=100)`: sliding-window RMS via vectorized convolution (100 ms window). Vectorized (rather than a per-sample loop) so it's fast enough to recompute on every live-view tick.

### Error handling

The model never crashes on expected failure modes — it catches `OSError` (bad port, connection reset, server closing the connection) around all socket operations and reports them through `status_updated`. `bandpass_filter`/`compute_rms` raise `ValueError` with a clear message for invalid parameters or too little data, for the ViewModel to catch similarly.

## ViewModel layer (`viewmodels/`)

TODO

## View layer (`views/`)

TODO
