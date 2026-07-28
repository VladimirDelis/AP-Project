# AP-Project

TCP signal visualization application for the Applied Programming 2026 final project.

## Team

**Group:** 26

**Team members and responsibilities:**

- Mahir Rafi Kasim, 23797021, af76aqul — **Person A:** TCP client and signal processing (Model layer) — `models/tcp_client_model.py`, `models/signal_processing.py`
- Vladimir Delis, 23882224, qo21topo — **Person B:** live view, connection controls, and VisPy visualization — `views/connection_widget.py`, `views/live_plot_view.py`, `views/all_channels_plot_view.py`, `views/channel_selector_widget.py`, `views/mode_selector_widget.py`
- Hamza Ali, 23656618, te30tozo — **Person C:** main window and offline Matplotlib view — `main.py`, `views/offline_view.py`, `viewmodels/offline_viewmodel.py`

The `SharedSessionModel` adapter (`models/model_adapter.py`), which bridges `TcpClientModel` to both ViewModels' expected interfaces, was a collaborative effort by the whole team during integration rather than any one person's task.

*This README was written and reviewed collaboratively by the team, with formatting/drafting assistance from Claude.*

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Running the application

From the project root, with dependencies installed:

```bash
python main.py
```

This opens the main window with two tabs: **Live** and **Offline Inspection**.

### Connecting to the TCP server

1. Start the course-provided TCP server (Exercise 5) first.
2. In the **Live** tab, enter the server's **Host** (default `127.0.0.1`) and **Port** (default `12345`) in the connection controls at the top.
3. Click **Connect**. The status label and the window's status bar show the result — e.g. "Connected to ..." on success, or a clear error message if the server isn't running or the port is wrong.
4. Streaming starts automatically as soon as the connection succeeds — there's no separate "start" step.
5. Click **Disconnect** at any time to stop streaming and close the connection. The app automatically switches to the **Offline Inspection** tab so you can immediately inspect what was just recorded.

### Using the live plot

- While connected, the single-channel plot (VisPy) scrolls in real time, showing the last few seconds of the currently selected channel.
- Use the **Channel** spinbox to pick which of the 32 channels (0-31) is displayed.
- Use the **Mode** dropdown to switch between **Original**, **RMS**, and **Filtered** views of the signal (see "Signal processing" below for the parameters used).
- Click **Plot All Channels** to switch to an overview showing all 32 channels at once, stacked with a small vertical offset so they stay readable; click **Show Single Channel** to switch back. Both plots keep receiving data in the background regardless of which one is visible, so switching never shows stale data.

### Opening the offline plot

- Switch to the **Offline Inspection** tab at any time (the app also switches to it automatically when you disconnect).
- It shows the full recorded session — everything received since connecting — not just the rolling live window.
- If no data has been recorded yet, it shows a "No data recorded yet." message instead of a blank or broken plot.
- The offline plot does not update live — use the **Refresh** button, or change the channel/mode, to redraw it with the latest data.

### Switching channels and signal modes

Both the Live and Offline views have independent channel and mode selections, so you can watch one channel live while inspecting a different one offline:

- **Live tab:** the **Channel** spinbox and **Mode** dropdown above the plot.
- **Offline tab:** the **Channel** spinbox and **Mode** dropdown above the offline plot — changing either immediately redraws the plot.

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

- `compute_rms(signal, window=50)`: moving RMS via vectorized convolution over a 50-sample window. Vectorized (rather than a per-sample loop) so it's fast enough to recompute on every live-view tick.
- `compute_filtered(signal, sample_rate_hz, low_hz=1.0, high_hz=40.0, order=4)`: zero-phase 4th-order Butterworth band-pass filter (`scipy.signal.butter` + `filtfilt`), 1-40 Hz. Falls back to returning the input unchanged if there are too few samples for stable filtering, rather than raising.
- `apply_mode(signal, mode, sample_rate_hz)`: dispatch helper selecting between `"original"`, `"rms"`, and `"filtered"` — the single integration point both ViewModels call into.

### Error handling

The model never crashes on expected failure modes — it catches `OSError` (bad port, connection reset, server closing the connection) around all socket operations and reports them through `status_updated`. `compute_filtered` returns the input unchanged (rather than raising) when there isn't enough data to filter stably.

## ViewModel layer (`viewmodels/`)

Sits between the Views and the shared Model, holding all UI-facing state and business logic. Views never touch the Model directly — only ViewModel methods/signals.

- **`LiveViewModel`** (`viewmodels/live_viewmodel.py`): wraps the shared session model (constructor-injected), forwards its `status_updated`/`connection_state_changed` signals, and owns the live view's "current channel" and "current mode" selections. Every incoming raw chunk is run through `apply_mode()` and re-emitted as `processed_data_ready` — the only signal the plot Views should connect to.
- **`OfflineViewModel`** (`viewmodels/offline_viewmodel.py`): pull-based. Owns the offline view's own independent "current channel"/"current mode" selections, pulls the full recorded session from the shared model on demand, and applies `apply_mode()` before handing data to the View. Emits `data_changed` whenever the View should re-read and redraw (channel/mode change, or new data arriving while still connected).

Both ViewModels are constructed around the *same* shared model instance (built once in `main.py`), so the Live and Offline tabs always reflect one underlying session.

## View layer (`views/`)

Plain Qt widgets — no TCP or signal-processing logic. Each either binds to a ViewModel (calling its methods, reacting to its signals) or, for the plot widgets, takes plain config values and is wired up externally.

- **`ConnectionWidget`**: host/port entry and connect/disconnect buttons, bound to `LiveViewModel`.
- **`ChannelSelectorWidget`** / **`ModeSelectorWidget`**: small controls that call `set_channel()`/`set_mode()` on `LiveViewModel` and reflect its `channel_changed`/`mode_changed` signals back into the widget — the ViewModel remains the single source of truth for both selections.
- **`LivePlotView`**: scrolling single-channel plot (VisPy), fed via `append_window()` and switched via `set_channel()`.
- **`AllChannelsPlotView`**: scrolling overview of all 32 channels stacked vertically (VisPy), fed via `append_window()`; independent of the channel selector — it always shows every channel.
- **`OfflineView`**: Matplotlib plot of the full recorded session, bound to `OfflineViewModel`; channel/mode spinboxes call into the ViewModel, and the plot redraws on `data_changed`.
- **`_plot_common.py`**: shared rolling-buffer implementations and sample-rate resolution helper used by both plot widgets.
