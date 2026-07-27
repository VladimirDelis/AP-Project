import numpy as np
from scipy import signal


def bandpass_filter(data, sampling_rate, low_cut=20, high_cut=450, order=4):
    """
    Zero-phase Butterworth bandpass filter, applied independently per channel.

    data: np.ndarray, shape (channels, samples)
    Raises ValueError if the parameters or the input are invalid, so the
    caller (ViewModel) can catch it and show a status message instead of
    crashing.
    """
    nyquist = sampling_rate / 2

    if low_cut <= 0:
        raise ValueError("Low cutoff frequency must be greater than 0 Hz.")
    if high_cut >= nyquist:
        raise ValueError(
            f"High cutoff frequency ({high_cut} Hz) exceeds the Nyquist frequency ({nyquist} Hz)."
        )
    if low_cut >= high_cut:
        raise ValueError("Low cutoff frequency must be smaller than the high cutoff frequency.")

    b, a = signal.butter(order, [low_cut / nyquist, high_cut / nyquist], btype="band")

    # filtfilt needs a minimum number of samples relative to the filter order,
    # otherwise it raises a cryptic error. Fail clearly instead.
    min_length = 3 * (max(len(a), len(b)) - 1)
    if data.shape[1] <= min_length:
        raise ValueError(
            f"Not enough samples ({data.shape[1]}) to filter; need more than {min_length}."
        )

    filtered = np.zeros_like(data)
    for channel in range(data.shape[0]):
        filtered[channel, :] = signal.filtfilt(b, a, data[channel, :])

    return filtered


def compute_rms(data, sampling_rate, window_ms=100):
    """
    Sliding-window RMS per channel, same shape as input.

    Vectorized with convolution (instead of a per-sample Python loop) so it
    stays fast enough to recompute on every live-view update tick.

    Caveat: np.convolve(..., mode="same") implicitly zero-pads past the
    edges of the array. For the *offline* full recording this only affects
    the very first/last few samples. For the *live* rolling buffer, it means
    the most recent half-window of samples (the newest arrivals) will read
    as an underestimate until more data arrives to fill the window - this is
    expected, not a bug.
    """
    window_size = max(1, int((window_ms / 1000) * sampling_rate))
    kernel = np.ones(window_size) / window_size

    mean_squared = np.array(
        [np.convolve(channel**2, kernel, mode="same") for channel in data]
    )

    return np.sqrt(mean_squared)
