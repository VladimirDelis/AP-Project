"""
views/live_plot_view.py

LivePlotView -- the "V" in MVVM for the live scrolling signal plot.

Like ConnectionWidget, this stays a "dumb" View: it never imports or
touches TCPWorker/EMGTCPClient, and it never decides *which* channel is
selected -- it only displays whatever channel index it's told to display
(via `set_channel()`) and whatever windows it's handed (via
`append_window()`). The ViewModel remains the single source of truth for
"current channel"; see LiveViewModel.set_channel() / live_plot_view wiring
in dev_test_step2.py for how that stays consistent.

VisPy backend choice
---------------------
This uses `vispy.scene` (a `SceneCanvas` with a grid of a `ViewBox` +
two `AxisWidget`s) rather than the higher-level `vispy.plot` API.
`vispy.plot.Fig`/`PlotWidget` is convenient for one-off, mostly-static
plots, but it's built around re-creating/re-styling plot items through a
matplotlib-like convenience layer that isn't designed to have a single
Line's data mutated tens of times a second. `vispy.scene` lets us build the
scene graph (line + axes) once and then just call `Line.set_data(...)` and
`camera.set_range(...)` on every redraw tick, which is the right shape for
a real-time rolling plot and avoids rebuilding plot objects every frame.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import vispy.app

vispy.app.use_app("pyside6")  # must happen before any SceneCanvas is created
from vispy import scene

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QVBoxLayout, QWidget

from views._plot_common import (
    N_CHANNELS,
    N_SAMPLES_PER_WINDOW,
    RollingBuffer,
    resolve_sample_rate,
)

DEFAULT_WINDOW_SECONDS = 5.0
DEFAULT_REFRESH_FPS = 30
# How often (in redraw ticks) we recompute the y-axis auto-scale. Doing this
# less often than every redraw is what keeps the axis from jittering.
Y_RESCALE_EVERY_N_TICKS = 10


class LivePlotView(QWidget):
    """
    Scrolling single-channel line plot, backed by a VisPy scene canvas.

    Data flow (all off the View's own decision-making):
        LiveViewModel.data_received -> append_window(window)
            -> pulls out the row for the current channel, pushes it into a
               fixed-size rolling buffer (cheap, happens as fast as data
               arrives)
        internal QTimer (fixed FPS) -> _on_redraw_tick()
            -> reads the buffer, updates the VisPy Line + axis ranges
               (independent of how fast data is arriving)

    Parameters
    ----------
    sample_rate_hz : float, optional
        Samples per second, used purely to convert the rolling buffer's
        sample positions into second labels on the x-axis. If omitted, a
        placeholder default is used and a warning is printed to the
        console (see DEFAULT_SAMPLE_RATE_HZ above) -- confirm the real
        value with the team / exercise material and pass it explicitly
        once known.
    window_seconds : float
        How many seconds of trailing data the plot shows at once.
    refresh_fps : int
        How often the canvas redraws per second. This is intentionally
        decoupled from the data arrival rate: `append_window()` only
        touches the buffer, it never triggers a draw directly.
    parent : QWidget, optional
    """

    def __init__(
        self,
        sample_rate_hz: Optional[float] = None,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        refresh_fps: int = DEFAULT_REFRESH_FPS,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        sample_rate_hz = resolve_sample_rate(sample_rate_hz, widget_name="LivePlotView")

        self._sample_rate_hz = float(sample_rate_hz)
        self._window_seconds = float(window_seconds)
        self._channel = 0

        capacity = max(1, int(self._sample_rate_hz * self._window_seconds))
        self._buffer = RollingBuffer(capacity)

        # Running count of samples ever appended (since the last reset).
        # Used to compute each visible sample's absolute time position, so
        # the x-axis reads as continuously advancing elapsed seconds rather
        # than restarting at 0 every redraw.
        self._total_samples_appended = 0

        self._redraw_tick_count = 0
        self._y_range = (-1.0, 1.0)  # current smoothed y-axis range

        self._build_vispy_scene()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas.native)

        # Redraw timer: fixed UI refresh rate, independent of data rate.
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setInterval(int(1000 / refresh_fps))
        self._redraw_timer.timeout.connect(self._on_redraw_tick)
        self._redraw_timer.start()

    # ------------------------------------------------------------------
    # VisPy scene setup
    # ------------------------------------------------------------------
    def _build_vispy_scene(self) -> None:
        self._canvas = scene.SceneCanvas(keys=None, show=False, bgcolor="white")
        grid = self._canvas.central_widget.add_grid()

        view = grid.add_view(row=0, col=1)
        view.camera = scene.PanZoomCamera()
        view.camera.set_range(x=(0, self._window_seconds), y=self._y_range)
        # Users dragging/zooming the canvas by accident would fight with
        # our own auto-ranging every redraw tick, so we disable interaction
        # for this step; panning/zooming isn't part of the spec here.
        view.camera.interactive = False
        self._view = view

        self._line = scene.Line(pos=np.zeros((1, 2)), color="#1f77b4", parent=view.scene, width=1.5)

        y_axis = scene.AxisWidget(orientation="left", axis_label="Amplitude")
        y_axis.width_max = 60
        grid.add_widget(y_axis, row=0, col=0)
        y_axis.link_view(view)

        x_axis = scene.AxisWidget(orientation="bottom", axis_label="Time (s)")
        x_axis.height_max = 40
        grid.add_widget(x_axis, row=1, col=1)
        x_axis.link_view(view)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def set_active(self, active: bool) -> None:
        """
        Pause or resume this widget's own redraw timer.

        `append_window()` keeps updating the rolling buffer regardless of
        this setting -- data is never lost or skipped while inactive.
        This only controls whether the (comparatively expensive) redraw
        tick -- reassembling the buffer, updating the VisPy Line, issuing
        a GL redraw -- keeps running. Intended to be called by whatever
        container widget toggles this view's visibility (e.g. a
        QStackedWidget alongside AllChannelsPlotView), so a hidden plot
        isn't still spending CPU/GPU time drawing frames nobody sees.

        Parameters
        ----------
        active : bool
            True to (re)start redrawing, False to pause it.
        """
        if active:
            if not self._redraw_timer.isActive():
                self._redraw_timer.start()
        else:
            self._redraw_timer.stop()

    def set_channel(self, index: int) -> None:
        """
        Switch which channel's data this widget displays.

        The rolling buffer is cleared on a channel switch: the samples
        already in it belong to the *previous* channel, and continuing to
        scroll them alongside newly-arriving data from the new channel
        would draw a single line that's a discontinuous mix of two
        unrelated signals. Starting the buffer (and its time axis) fresh
        makes the switch visually unambiguous, at the cost of a brief
        empty-plot moment while it refills.

        Parameters
        ----------
        index : int
            Channel index, expected to be in range [0, 31].
        """
        if not (0 <= index < N_CHANNELS):
            raise ValueError(f"Channel index must be between 0 and {N_CHANNELS - 1}, got {index}")
        self._channel = index
        self._buffer.reset()
        self._total_samples_appended = 0

    def append_window(self, window: np.ndarray) -> None:
        """
        Feed one new (32, 18) window into the rolling buffer.

        Only the row for the currently selected channel is kept; this is
        the "internally selects the row for the current channel" behavior
        described in the task. This method does not touch VisPy or trigger
        a redraw -- it only updates the buffer. The redraw timer picks up
        the new data on its own schedule.

        Parameters
        ----------
        window : np.ndarray
            Shape (32, 18), dtype float64.
        """
        if window.shape != (N_CHANNELS, N_SAMPLES_PER_WINDOW):
            raise ValueError(f"Expected window shape (32, 18), got {window.shape}")

        channel_samples = window[self._channel]
        self._buffer.append(channel_samples)
        self._total_samples_appended += len(channel_samples)

    # ------------------------------------------------------------------
    # Redraw timer -- the only place that touches the VisPy scene
    # ------------------------------------------------------------------
    def _on_redraw_tick(self) -> None:
        """
        Runs at a fixed rate (e.g. 30fps), regardless of how fast data is
        arriving. Reassembles the buffer into chronological order, updates
        the line's data and the visible x-range (which continuously
        advances with elapsed time), widens the y-range immediately if the
        current data would otherwise clip, and periodically checks whether
        the y-axis can be tightened back up.
        """
        y_values = self._buffer.ordered()
        n = len(y_values)
        if n == 0:
            return  # nothing received yet -- nothing to draw

        # Compute each visible sample's absolute elapsed time, so the most
        # recent sample lands at "now" and older ones trail off to its left.
        now = self._total_samples_appended / self._sample_rate_hz
        oldest_visible = now - (n - 1) / self._sample_rate_hz
        x_values = np.linspace(oldest_visible, now, n)

        self._line.set_data(pos=np.column_stack((x_values, y_values)))

        # Expand (never shrink) the y-range every tick, before it's applied
        # to the camera below. This runs unconditionally -- it's what fixes
        # clipping, since a range that's late to widen is exactly what cuts
        # off real signal peaks.
        self._ensure_y_range_contains(y_values)

        # Keep the visible x-range scrolling forward with the data instead
        # of resetting to (0, window_seconds) every tick.
        x_min = max(0.0, now - self._window_seconds)
        self._view.camera.set_range(x=(x_min, max(x_min + self._window_seconds, now)), y=self._y_range, margin=0)

        self._redraw_tick_count += 1
        if self._redraw_tick_count % Y_RESCALE_EVERY_N_TICKS == 0:
            self._maybe_shrink_y_range(y_values)

    @staticmethod
    def _padded_range(data_min: float, data_max: float) -> tuple:
        """Add ~10% padding around [data_min, data_max], handling a flat signal."""
        if data_min == data_max:
            pad = 1.0 if data_min == 0 else abs(data_min) * 0.1
            return data_min - pad, data_max + pad
        span = data_max - data_min
        pad = span * 0.1
        return data_min - pad, data_max + pad

    def _ensure_y_range_contains(self, y_values: np.ndarray) -> None:
        """
        Immediately widen (never narrow) the y-range so it always contains
        the currently-visible data, with padding.

        This is the actual fix for the clipping bug: the previous version
        only ever adjusted the range from `_maybe_rescale_y` (now
        `_maybe_shrink_y_range`, see below), which ran every
        Y_RESCALE_EVERY_N_TICKS ticks *and* smoothed toward its target by
        only 30% per check. Against a real signal much bigger than the
        (-1, 1) starting range, that meant the trace stayed visibly clipped
        for many ticks (measured: still clipping a full second-plus into a
        realistic-amplitude stream) while smoothing slowly caught up.

        This method runs on every tick and only ever grows the range, so
        by the time `camera.set_range()` is called just after this, the
        range is already guaranteed to contain the data about to be drawn.
        Narrowing the range back down when the signal quiets is handled
        separately, and more cautiously, by `_maybe_shrink_y_range()`.
        """
        data_min = float(np.min(y_values))
        data_max = float(np.max(y_values))
        padded_min, padded_max = self._padded_range(data_min, data_max)

        old_min, old_max = self._y_range
        new_min = min(old_min, padded_min)
        new_max = max(old_max, padded_max)
        if new_min != old_min or new_max != old_max:
            self._y_range = (new_min, new_max)

    def _maybe_shrink_y_range(self, y_values: np.ndarray) -> None:
        """
        Periodically ease the y-range back down if it's become wider than
        the visible data actually needs.

        This only ever narrows the range -- widening is `_ensure_y_range_
        contains()`'s job above, which already ran this same tick before
        the camera was updated, so this method can never be the thing that
        causes a clip: even if it narrows too aggressively, the next tick's
        `_ensure_y_range_contains()` call will widen it again before
        anything is drawn. That's what makes it safe to keep this step
        slow and smoothed (same 10-tick cadence and 30% easing as before),
        preserving the original "don't jitter on tiny fluctuations" intent
        without that caution being able to cause clipping anymore.
        """
        data_min = float(np.min(y_values))
        data_max = float(np.max(y_values))
        padded_min, padded_max = self._padded_range(data_min, data_max)

        alpha = 0.3
        old_min, old_max = self._y_range
        new_min = old_min + alpha * (padded_min - old_min)
        new_max = old_max + alpha * (padded_max - old_max)
        self._y_range = (new_min, new_max)