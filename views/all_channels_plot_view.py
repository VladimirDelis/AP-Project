"""
views/all_channels_plot_view.py

AllChannelsPlotView -- the "V" in MVVM for the "quick overview" of all 32
channels at once, each vertically offset so traces are stacked and
readable instead of overlapping.

Like LivePlotView, this stays a "dumb" View: it never imports or touches
TCPWorker/EMGTCPClient, and it doesn't respond to the channel selector at
all -- it always shows every channel, regardless of which one is currently
selected for the single-channel view. See LiveViewModel /
dev_test_step3.py for how the two plot widgets and the channel selector
are wired independently of each other.

This is a separate, self-contained widget with its own VisPy canvas --
not the same canvas as LivePlotView with 32 lines bolted on. The two views
solve different problems (one focused single trace vs. a broad overview)
and keeping them independent means either one's internals can change
without risking the other.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import vispy.app

vispy.app.use_app("pyside6")  # no-op if LivePlotView already called this
from vispy import scene

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QVBoxLayout, QWidget

from views._plot_common import (
    N_CHANNELS,
    MultiChannelRollingBuffer,
    resolve_sample_rate,
)

DEFAULT_WINDOW_SECONDS = 5.0
DEFAULT_REFRESH_FPS = 30

# Multiplier applied to the observed per-channel peak-to-peak amplitude to
# get a spacing that leaves a visible gap between neighboring channels.
OFFSET_SAFETY_FACTOR = 2.5
# Floor so a momentarily flat/zero signal can't collapse spacing to 0.
MIN_OFFSET_SPACING = 1e-3
# Same cadence/smoothing idea as LivePlotView's y-axis fix: widen the
# spacing immediately if channels would otherwise start overlapping, but
# only ever tighten it back up periodically and smoothed.
OFFSET_ADAPT_EVERY_N_TICKS = 10
OFFSET_SHRINK_ALPHA = 0.3
# Used only to build the scene before any real data has arrived (in
# auto-spacing mode); overwritten by the first tick that has data.
_PLACEHOLDER_OFFSET_SPACING = 1.0

# Extra clearance added beyond the OUTERMOST channels' (Ch 0 at the
# bottom, Ch N_CHANNELS-1 at the top) own baselines, so their traces have
# room to render fully without being clipped by the camera's outer edge.
# See `_compute_outer_y_range()`.
#
# EDGE_PADDING_FRACTION * offset_spacing -- deliberately a PROPORTION of
# offset_spacing, not a flat data-space constant. offset_spacing is itself
# auto-scaled from the real signal's own amplitude (see
# `_update_offset_spacing()`), and confirmed via live debug logging
# against the real server to range from ~929 up into the 5900s (and still
# climbing) within a single session. The widget's PIXEL height, however,
# is fixed regardless of that (see MIN_PX_PER_CHANNEL below) -- so a flat
# padding constant becomes a shrinking, eventually sub-pixel fraction of
# the rendered height as offset_spacing grows, which is exactly why a
# flat EDGE_PADDING had to keep growing indefinitely to stay visible, and
# why doubling it produced no visible change once it was already
# negligible. Expressing it as a fraction of offset_spacing instead keeps
# it a roughly CONSTANT share of the rendered pixel height, regardless of
# how large the real signal's amplitude turns out to be.
#
# EDGE_PADDING_FLOOR covers the opposite extreme: a flat/near-silent
# signal where offset_spacing collapses toward MIN_OFFSET_SPACING, where
# a pure fraction would round to ~nothing -- the floor guarantees the
# outer channels still get reasonable breathing room even with
# essentially no signal amplitude to scale from.
#
# This covers trace excursion only -- see LABEL_EDGE_PADDING_FRACTION
# below for the outer channels' text labels, a separate concern.
EDGE_PADDING_FRACTION = 0.35
EDGE_PADDING_FLOOR = 600.0

# Same reasoning as EDGE_PADDING_FRACTION, but reserved specifically for
# the outermost channels' own "Ch 0"/"Ch 31" TEXT LABELS' vertical extent,
# on top of EDGE_PADDING_FRACTION's trace-excursion margin.
#
# The "Ch N" labels are scene.Text glyphs rendered at a fixed PIXEL size
# (LABEL_FONT_SIZE), anchored at each channel's baseline (anchor_y=
# "center") -- their vertical extent never scales with the data-space
# y-range the way a trace's amplitude does, so this needs its own margin
# separate from EDGE_PADDING_FRACTION's trace-only one. It has to scale
# with offset_spacing for the exact same reason EDGE_PADDING does (a flat
# value is equally negligible against a large real offset_spacing) -- just
# with a smaller fraction, since a label glyph's own height is much
# smaller than a full trace's realistic peak-to-peak swing.
LABEL_EDGE_PADDING_FRACTION = 0.20
LABEL_EDGE_PADDING_FLOOR = 350.0

# Rough floor on legible vertical pixels per channel row, given
# LABEL_FONT_SIZE below -- used to pick a sensible default widget height so
# 32 "Ch N" labels don't overlap each other regardless of what offset
# spacing (a data-space unit, not a pixel unit) ends up being chosen.
MIN_PX_PER_CHANNEL = 18
LABEL_FONT_SIZE = 7


class AllChannelsPlotView(QWidget):
    """
    Scrolling overview plot showing all 32 channels stacked vertically.

    Data flow (mirrors LivePlotView's, just across all channels at once):
        LiveViewModel.data_received -> append_window(window)
            -> pushes all 32 rows into one shared rolling buffer (cheap,
               happens as fast as data arrives)
        internal QTimer (fixed FPS) -> _on_redraw_tick()
            -> reads the buffer, updates one multi-segment VisPy Line +
               the channel label positions (independent of data rate)

    Parameters
    ----------
    sample_rate_hz : float, optional
        Same meaning and same placeholder-default/startup-warning behavior
        as LivePlotView's parameter (both widgets share the warning logic
        via `views._plot_common.resolve_sample_rate`).
    window_seconds : float
        How many seconds of trailing data are shown at once.
    refresh_fps : int
        Redraw rate, decoupled from data arrival rate exactly as in
        LivePlotView.
    offset_spacing : float, optional
        Vertical distance between channels' zero-lines. Channel `i` is
        drawn at y = raw_signal + i * offset_spacing, with channel 0 at
        the bottom and channel 31 at the top. If omitted (the default),
        this is auto-scaled from the incoming data's own observed
        peak-to-peak amplitude (see `_update_offset_spacing()`) instead of
        a fixed guessed constant -- a hardcoded value that happens not to
        match the real signal's amplitude is exactly what caused traces to
        visually collide into a solid block. Pass an explicit value here
        to opt out of auto-scaling and pin it yourself.
    parent : QWidget, optional
    """

    def __init__(
        self,
        sample_rate_hz: Optional[float] = None,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        refresh_fps: int = DEFAULT_REFRESH_FPS,
        offset_spacing: Optional[float] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        sample_rate_hz = resolve_sample_rate(sample_rate_hz, widget_name="AllChannelsPlotView")

        self._sample_rate_hz = float(sample_rate_hz)
        self._window_seconds = float(window_seconds)

        # Auto-scaling mode (the default) vs. a manually pinned spacing.
        # `_offset_spacing` starts at a placeholder in auto mode -- nothing
        # is actually drawn with it until the first real window arrives
        # (see the early-return in _on_redraw_tick()), at which point
        # _update_offset_spacing() replaces it with a data-driven value
        # before anything is rendered.
        self._auto_offset_spacing = offset_spacing is None
        self._offset_spacing = float(offset_spacing) if offset_spacing is not None else _PLACEHOLDER_OFFSET_SPACING
        self._offset_tick_count = 0

        capacity = max(1, int(self._sample_rate_hz * self._window_seconds))
        self._buffer = MultiChannelRollingBuffer(N_CHANNELS, capacity)

        # Running count of samples ever appended. Same role as in
        # LivePlotView: converts buffer positions into continuously
        # advancing elapsed-time x values. This view is never reset by a
        # channel switch (it always shows all channels), so this only
        # resets if the widget itself is recreated.
        self._total_samples_appended = 0

        # Enough vertical room for 32 "Ch N" labels to stay legible: this
        # is independent of offset_spacing (a data-space unit) -- however
        # spacing is chosen, the camera always fits the full data range
        # into whatever pixel height this widget actually has, so *pixels
        # per channel* is really governed by widget height / N_CHANNELS.
        # Too little of that, combined with a fixed-size font, is what
        # made every "Ch N" label visually overlap its neighbors.
        self.setMinimumHeight(N_CHANNELS * MIN_PX_PER_CHANNEL + 60)  # +60 for the x-axis strip/margins

        self._build_vispy_scene()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas.native)

        # Redraw timer: fixed UI refresh rate, independent of data rate --
        # same decoupling rationale as LivePlotView.
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

        # Only a shared x (time) axis is meaningful here -- the y-axis has
        # no real units once traces are artificially offset, so we skip a
        # y-AxisWidget entirely rather than show numbers that would be
        # misleading (per the "y-axis doesn't need meaningful units"
        # requirement). Channel identity is instead carried by the "Ch N"
        # text labels drawn at each trace's own baseline.
        view = grid.add_view(row=0, col=0)
        view.camera = scene.PanZoomCamera()
        y_min, y_max = self._compute_outer_y_range()
        view.camera.set_range(x=(0, self._window_seconds), y=(y_min, y_max))
        view.camera.interactive = False  # same reasoning as LivePlotView
        self._view = view

        # All 32 traces are drawn as ONE Line visual using a `connect`
        # boolean mask to break the line between channels, rather than 32
        # separate Line objects. One draw call for all channels is both
        # simpler to update every tick (one set_data call) and cheaper
        # than managing 32 visuals' worth of GL state.
        self._line = scene.Line(pos=np.zeros((1, 2)), color="#1f77b4", parent=view.scene, width=1.0)

        # All channels share one plain color: with 32 stacked traces,
        # identity is carried by vertical position + the "Ch N" labels,
        # not by color -- adding 32 distinct colors would need a
        # per-vertex color array and add visual noise without actually
        # making it easier to tell traces apart (this is meant to be a
        # quick overview, not a polished dashboard).
        baselines = np.arange(N_CHANNELS) * self._offset_spacing
        label_x = 0.0  # updated every redraw tick to track the left edge
        initial_positions = np.column_stack((np.full(N_CHANNELS, label_x), baselines))
        # Labels are 0-indexed ("Ch 0".."Ch 31") to match the channel
        # selector's own 0-31 indexing elsewhere in the app -- using
        # 1-indexed labels here would make "channel 5" in the spinbox and
        # "the 5th trace from the bottom" refer to different channels.
        channel_labels = [f"Ch {i}" for i in range(N_CHANNELS)]
        self._labels = scene.Text(
            text=channel_labels,
            pos=initial_positions,
            color="black",
            anchor_x="left",
            anchor_y="center",
            font_size=LABEL_FONT_SIZE,
            parent=view.scene,
        )

        x_axis = scene.AxisWidget(orientation="bottom", axis_label="Time (s)")
        x_axis.height_max = 40
        grid.add_widget(x_axis, row=1, col=0)
        x_axis.link_view(view)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def set_active(self, active: bool) -> None:
        """
        Pause or resume this widget's own redraw timer.

        Same rationale as LivePlotView.set_active(): `append_window()`
        keeps filling the buffer regardless, only the (comparatively
        expensive) redraw is paused while this view is hidden.
        """
        if active:
            if not self._redraw_timer.isActive():
                self._redraw_timer.start()
        else:
            self._redraw_timer.stop()

    def append_window(self, window: np.ndarray) -> None:
        """
        Feed one new (32, n_samples) chunk into the shared rolling buffer.

        Unlike LivePlotView, every row is kept -- this view always shows
        all channels and has no notion of "current channel" at all.

        n_samples is NOT assumed fixed -- see LivePlotView.append_window()
        for why (the real TcpClientModel emits variable-length chunks).

        Parameters
        ----------
        window : np.ndarray
            Shape (32, n_samples), dtype float64. n_samples may vary
            between calls.
        """
        if window.shape[0] != N_CHANNELS:
            raise ValueError(f"Expected {N_CHANNELS} channels, got shape {window.shape}")

        self._buffer.append(window)
        self._total_samples_appended += window.shape[1]

    # ------------------------------------------------------------------
    # Redraw timer -- the only place that touches the VisPy scene
    # ------------------------------------------------------------------
    def _update_offset_spacing(self, channel_rows: np.ndarray) -> None:
        """
        Recompute offset_spacing from the currently visible buffer, instead
        of relying on one fixed guessed constant.

        This is the actual fix for the "32 traces collapse into one solid
        block" bug: the previous hardcoded DEFAULT_OFFSET_SPACING (4.0) had
        no relationship to the real signal's amplitude. Measured against a
        realistic ~80-unit-amplitude EMG-like signal, each channel's trace
        extended roughly 25+ channel-slots above and below its own
        baseline -- so at any given moment on screen, all 32 channels'
        traces were overlapping almost entirely, which is what "looks like
        one solid filled block" actually was.

        Uses each channel's peak-to-peak range over the visible window,
        summarized with the median (not max) across channels so one
        unusually noisy/large-amplitude channel doesn't blow the spacing
        out for the other 31 -- a max-based version was tried, but with a
        real outlier channel present it made every other channel's trace
        look visually flat by comparison, since the shared spacing became
        much larger than their actual amplitude. Follows the same "expand
        immediately, shrink only occasionally/smoothed" policy as
        LivePlotView's y-axis fix, for the same reason: a spacing that's
        late to widen means channels visibly collide first and separate
        afterward, which is worse than occasionally being a bit more
        spaced out than strictly necessary.

        KNOWN LIMITATION (intentionally deferred): because this sizes
        spacing from the median, a genuine outlier channel (e.g. one
        channel swinging much wider than the rest) can still spill past
        its own lane into a neighbor's. That's a separate, later fix --
        this method is not the place `Ch 0 clipped at the canvas edge` is
        addressed either; see `_compute_outer_y_range()` for that.

        Note: this only affects trace-vs-trace overlap. It does NOT
        change how many actual screen pixels each channel gets (that's
        fixed by the widget's height / N_CHANNELS, regardless of what
        data-space spacing is chosen) -- the separate label-overlap
        problem is fixed by `MIN_PX_PER_CHANNEL`/`LABEL_FONT_SIZE` above
        instead.
        """
        peak_to_peak = np.ptp(channel_rows, axis=1)  # shape (32,)
        typical_amplitude = float(np.median(peak_to_peak))
        target_spacing = max(typical_amplitude * OFFSET_SAFETY_FACTOR, MIN_OFFSET_SPACING)

        if target_spacing > self._offset_spacing:
            # Widen immediately -- never let real signal amplitude catch up
            # to the gap between channels before the display reacts.
            self._offset_spacing = target_spacing
            return

        # Only ever tighten the spacing periodically and smoothed, so a
        # brief quiet moment doesn't make all 32 traces visibly jump closer
        # together and then back apart a moment later.
        self._offset_tick_count += 1
        if self._offset_tick_count % OFFSET_ADAPT_EVERY_N_TICKS == 0:
            self._offset_spacing += OFFSET_SHRINK_ALPHA * (target_spacing - self._offset_spacing)
            self._offset_spacing = max(self._offset_spacing, MIN_OFFSET_SPACING)

    def _compute_outer_y_range(self) -> tuple:
        """
        Compute the overall camera y-range's two OUTER edges.

        The gap BETWEEN neighboring channels is entirely offset_spacing's
        job (see `_update_offset_spacing()`) and is untouched here. This
        method only decides how far below Ch 0's baseline (0) and above
        Ch N_CHANNELS-1's baseline the camera's outer bounds extend, which
        is a different problem from a channel spilling into a *neighbor's*
        lane: the outer edge is the actual canvas/camera boundary, so
        anything past it is completely clipped (invisible), not just
        visually overlapping another trace.

        Padding is computed as a PROPORTION of offset_spacing (plus a
        floor for near-zero offset_spacing) rather than a flat constant --
        see EDGE_PADDING_FRACTION/LABEL_EDGE_PADDING_FRACTION's
        definitions near the top of this file for why: the widget's pixel
        height is fixed (see MIN_PX_PER_CHANNEL) regardless of how large
        offset_spacing gets, so a flat padding value becomes an
        increasingly negligible, eventually sub-pixel fraction of that
        fixed height once offset_spacing grows large against a
        real-amplitude signal -- confirmed via live debug logging showing
        offset_spacing climbing from ~929 into the 5900s+ within one
        session while a flat combined padding of 650 stayed put at well
        under 1% of the total range. A proportional value keeps padding a
        roughly CONSTANT share of the rendered height instead. (An even
        earlier version scaled padding from each outer channel's own noisy
        observed min/max directly, which felt too tight during quiet
        moments -- this is different: it scales from offset_spacing, which
        is itself already smoothed and safety-margined in
        `_update_offset_spacing()`, so it doesn't inherit that noisiness.)

        The two paddings are added together (both applied identically to
        y_min and y_max, so this stays symmetric): EDGE_PADDING_FRACTION
        covers the outer channels' own trace excursion,
        LABEL_EDGE_PADDING_FRACTION separately covers their "Ch N" text
        labels' own vertical extent -- see that constant's docstring for
        why it had to be a second, distinct margin rather than folded into
        the trace one.
        """
        edge_padding = max(self._offset_spacing * EDGE_PADDING_FRACTION, EDGE_PADDING_FLOOR)
        label_edge_padding = max(self._offset_spacing * LABEL_EDGE_PADDING_FRACTION, LABEL_EDGE_PADDING_FLOOR)
        total_padding = edge_padding + label_edge_padding
        y_min = 0.0 - total_padding
        y_max = (N_CHANNELS - 1) * self._offset_spacing + total_padding
        return y_min, y_max

    def _on_redraw_tick(self) -> None:
        """
        Runs at a fixed rate, decoupled from data arrival, exactly like
        LivePlotView. Reassembles all 32 channels into chronological
        order in one vectorized step, builds one (n_channels * n_samples,
        2) position array with each channel's row shifted up by
        `i * offset_spacing`, and updates the single Line visual plus the
        channel labels' x position (so labels stay pinned to the current
        left edge of the scrolling view).
        """
        channel_rows = self._buffer.ordered()  # shape (32, n_samples)
        n_samples = channel_rows.shape[1]
        if n_samples == 0:
            return  # nothing received yet -- nothing to draw

        if self._auto_offset_spacing:
            self._update_offset_spacing(channel_rows)

        now = self._total_samples_appended / self._sample_rate_hz
        oldest_visible = now - (n_samples - 1) / self._sample_rate_hz
        x_values = np.linspace(oldest_visible, now, n_samples)

        # Stack all channels into one flat position array:
        #   - x repeats the same time axis once per channel
        #   - y is each channel's samples shifted up by i * offset_spacing
        x_tiled = np.tile(x_values, N_CHANNELS)
        offsets = np.arange(N_CHANNELS)[:, None] * self._offset_spacing
        y_flat = (channel_rows + offsets).reshape(-1)
        pos = np.column_stack((x_tiled, y_flat))

        # Break the line between the end of one channel's run and the
        # start of the next channel's, so VisPy draws 32 separate traces
        # instead of one continuous zigzag connecting channel 31's last
        # point to channel 0's first point (etc.) across the gaps.
        connect = np.ones(len(pos) - 1, dtype=bool)
        if n_samples > 0 and N_CHANNELS > 1:
            break_indices = np.arange(1, N_CHANNELS) * n_samples - 1
            connect[break_indices] = False

        self._line.set_data(pos=pos, connect=connect)

        # Labels track the left edge of the current view so they stay
        # readable next to each trace as the plot scrolls forward.
        x_min = max(0.0, now - self._window_seconds)
        baselines = np.arange(N_CHANNELS) * self._offset_spacing
        self._labels.pos = np.column_stack((np.full(N_CHANNELS, x_min), baselines))

        y_min, y_max = self._compute_outer_y_range()
        self._view.camera.set_range(
            x=(x_min, max(x_min + self._window_seconds, now)),
            y=(y_min, y_max),
            margin=0,
        )