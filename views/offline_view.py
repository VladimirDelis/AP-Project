"""
Offline plot View (Matplotlib) — Person C's responsibility.

Embeddable QWidget (not a modal dialog) so it can live in the main
window's tab layout. Pure display + user-input widgets: reads data
through the OfflineViewModel and draws it. Contains no TCP or signal
processing logic (MVVM: View layer).
"""
from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QLabel,
    QSpinBox,
    QPushButton,
)

from models.config import N_CHANNELS
from viewmodels.offline_viewmodel import OfflineViewModel


class OfflineView(QWidget):
    def __init__(self, view_model: OfflineViewModel, parent=None):
        super().__init__(parent)
        self.view_model = view_model

        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Channel:"))
        self.channel_spin = QSpinBox()
        self.channel_spin.setRange(1, N_CHANNELS)
        self.channel_spin.valueChanged.connect(
            lambda v: self.view_model.set_channel(v - 1)
        )
        controls.addWidget(self.channel_spin)

        controls.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["original", "rms", "filtered"])
        self.mode_combo.currentTextChanged.connect(self.view_model.set_mode)
        controls.addWidget(self.mode_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self._redraw)
        controls.addWidget(self.refresh_btn)

        controls.addStretch()
        layout.addLayout(controls)

        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout.addWidget(self.canvas)

        self.view_model.data_changed.connect(self._redraw)
        self._redraw()

    def _redraw(self) -> None:
        t, values = self.view_model.get_current_channel_data()

        self.ax.clear()
        if t.size > 0:
            self.ax.plot(t, values, linewidth=1.0)
            self.ax.set_title(
                f"Channel {self.view_model.selected_channel + 1} — {self.view_model.signal_mode}"
            )
            self.ax.set_xlabel("Time (s)")
            self.ax.set_ylabel("Amplitude")
            self.ax.grid(True, alpha=0.3)
        else:
            self.ax.text(
                0.5, 0.5, "No data recorded yet.", ha="center", va="center",
                transform=self.ax.transAxes,
            )
        self.canvas.draw_idle()
