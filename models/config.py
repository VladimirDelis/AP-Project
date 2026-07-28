"""
Shared configuration constants.

Adjust these to match the actual Exercise 5 server you connect to.
"""

N_CHANNELS = 32
SAMPLES_PER_PACKET = 18

# TODO: this MUST match the server's recording.pkl metadata
# (self.data['device_information']['sampling_frequency']), printed by the
# server on startup as "Sampling rate: {rate} Hz". It is NOT sent over the
# socket, so it has to be hardcoded here to match. Getting this wrong will
# throw off the time axis, RMS, and filter cutoffs.
SAMPLING_RATE_HZ = 250.0

# How many seconds of history the live rolling-window plot keeps.
ROLLING_WINDOW_SECONDS = 10.0

# How often the ViewModel polls the socket for new data (milliseconds).
NETWORK_POLL_INTERVAL_MS = 20

# How often the GUI redraws the live plot from the current buffer (milliseconds).
GUI_REFRESH_INTERVAL_MS = 50
