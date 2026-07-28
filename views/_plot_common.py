"""
views/_plot_common.py

Shared building blocks for the live plotting widgets (LivePlotView,
AllChannelsPlotView). Factored out here so both widgets use the exact same
rolling-buffer semantics and the exact same "placeholder sample rate"
warning behavior, instead of two copies that could quietly drift apart.
"""

from __future__ import annotations

import numpy as np

N_CHANNELS = 32
N_SAMPLES_PER_WINDOW = 18

# Placeholder default -- NOT confirmed against the real hardware/exercise
# material. Anything that uses this must warn loudly on startup (see
# resolve_sample_rate() below) so it's impossible to miss during testing
# that this value is still a guess.
DEFAULT_SAMPLE_RATE_HZ = 1000.0


def resolve_sample_rate(sample_rate_hz, widget_name: str) -> float:
    """
    Resolve a possibly-omitted sample rate, warning on the console if it's
    falling back to the shared placeholder default.

    Parameters
    ----------
    sample_rate_hz : float or None
        The value passed in by the caller, or None if omitted.
    widget_name : str
        Name to mention in the warning (e.g. "LivePlotView"), so it's
        obvious in the console which widget is still using a placeholder.

    Returns
    -------
    float
        `sample_rate_hz` if provided, otherwise DEFAULT_SAMPLE_RATE_HZ.
    """
    if sample_rate_hz is not None:
        return float(sample_rate_hz)

    print(
        f"WARNING: {widget_name} is using a PLACEHOLDER sample rate of "
        f"{DEFAULT_SAMPLE_RATE_HZ} Hz because none was provided. This has not "
        "been confirmed against the real hardware/exercise material -- pass "
        "sample_rate_hz explicitly once it's known, otherwise the x-axis time "
        "labels will be wrong."
    )
    return DEFAULT_SAMPLE_RATE_HZ


class RollingBuffer:
    """
    Fixed-size circular buffer of float64 samples for a single channel.

    `append()` writes new samples in O(len(samples)) with no shifting/copy
    of existing data -- it just advances a write pointer, wrapping around
    the end of a pre-allocated array. This is what makes it safe to call
    on every incoming window (potentially >10x/sec) without the cost
    growing over time the way `np.append`/`np.roll`-based approaches would.

    `ordered()` is the only place that costs O(capacity): it assembles the
    buffer's contents back into chronological order (oldest -> newest) for
    display. That's fine because it's only called from a redraw timer, at
    a fixed, low rate (e.g. 30fps), decoupled from how fast data arrives.
    """

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._data = np.zeros(capacity, dtype=np.float64)
        self._write_pos = 0
        self._filled = 0

    def append(self, samples: np.ndarray) -> None:
        n = len(samples)
        if n == 0:
            return
        if n >= self._capacity:
            self._data[:] = samples[-self._capacity :]
            self._write_pos = 0
            self._filled = self._capacity
            return

        end = self._write_pos + n
        if end <= self._capacity:
            self._data[self._write_pos : end] = samples
        else:
            first_part = self._capacity - self._write_pos
            self._data[self._write_pos :] = samples[:first_part]
            self._data[: end - self._capacity] = samples[first_part:]

        self._write_pos = end % self._capacity
        self._filled = min(self._capacity, self._filled + n)

    def ordered(self) -> np.ndarray:
        """Return the buffer's valid contents, oldest sample first."""
        if self._filled < self._capacity:
            return self._data[: self._filled].copy()
        return np.concatenate((self._data[self._write_pos :], self._data[: self._write_pos]))

    def reset(self) -> None:
        self._data[:] = 0
        self._write_pos = 0
        self._filled = 0

    @property
    def filled(self) -> int:
        return self._filled

    @property
    def capacity(self) -> int:
        return self._capacity


class MultiChannelRollingBuffer:
    """
    Fixed-size circular buffer holding ALL channels at once, backed by a
    single (n_channels, capacity) array rather than n_channels separate
    RollingBuffer instances.

    Why one shared array instead of 32 independent buffers:
    - Every window arrives as all 32 channels together, one row each, so
      all 32 channels are always appended in lockstep -- there's never a
      case where one channel's buffer should be at a different write
      position than another's. A single shared write pointer makes that
      already-true invariant automatic instead of something we'd have to
      keep 32 separate objects in sync on by hand.
    - The "reassemble to chronological order" step needed every redraw
      tick becomes one vectorized numpy operation across all channels at
      once (see `ordered()`), instead of looping over 32 small buffers and
      concatenating each individually.
    - One contiguous (32, capacity) block is simpler to reason about and
      touches memory more predictably than 32 separate allocations.

    The append/ordered semantics are otherwise identical to RollingBuffer,
    just with an extra leading channel axis.
    """

    def __init__(self, n_channels: int, capacity: int) -> None:
        self._n_channels = n_channels
        self._capacity = capacity
        self._data = np.zeros((n_channels, capacity), dtype=np.float64)
        self._write_pos = 0
        self._filled = 0

    def append(self, columns: np.ndarray) -> None:
        """
        Parameters
        ----------
        columns : np.ndarray
            Shape (n_channels, n_new_samples) -- one row per channel, new
            samples in time order along the second axis.
        """
        n = columns.shape[1]
        if n == 0:
            return
        if n >= self._capacity:
            self._data[:, :] = columns[:, -self._capacity :]
            self._write_pos = 0
            self._filled = self._capacity
            return

        end = self._write_pos + n
        if end <= self._capacity:
            self._data[:, self._write_pos : end] = columns
        else:
            first_part = self._capacity - self._write_pos
            self._data[:, self._write_pos :] = columns[:, :first_part]
            self._data[:, : end - self._capacity] = columns[:, first_part:]

        self._write_pos = end % self._capacity
        self._filled = min(self._capacity, self._filled + n)

    def ordered(self) -> np.ndarray:
        """Return shape (n_channels, filled), oldest sample first per row."""
        if self._filled < self._capacity:
            return self._data[:, : self._filled].copy()
        return np.concatenate((self._data[:, self._write_pos :], self._data[:, : self._write_pos]), axis=1)

    def reset(self) -> None:
        self._data[:] = 0
        self._write_pos = 0
        self._filled = 0

    @property
    def filled(self) -> int:
        return self._filled

    @property
    def capacity(self) -> int:
        return self._capacity