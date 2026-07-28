"""
views/connection_widget.py

ConnectionWidget -- the "V" in MVVM for the connection controls.

This widget is deliberately "dumb": it only
  (a) calls methods on the ViewModel in response to user actions, and
  (b) updates its own display in response to ViewModel signals.
It never imports or touches TCPWorker/EMGTCPClient, never processes signal
data, and holds no business logic of its own -- all decisions live in
LiveViewModel.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

DEFAULT_PORT = 12345  # matches EMGTCPClient's default


class ConnectionWidget(QWidget):
    """
    UI for entering a host/TCP port and connecting/disconnecting.

    Talks only to a LiveViewModel: calls its methods on user actions, and
    reacts to its signals to keep the display in sync.

    Parameters
    ----------
    viewmodel : LiveViewModel
        The ViewModel this widget is bound to.
    parent : QWidget, optional
        Standard Qt parent widget.
    """

    def __init__(self, viewmodel, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._viewmodel = viewmodel

        # --- widgets ---
        self._host_edit = QLineEdit("127.0.0.1")
        self._host_edit.setMaximumWidth(110)

        self._port_spinbox = QSpinBox()
        self._port_spinbox.setRange(1, 65535)
        self._port_spinbox.setValue(DEFAULT_PORT)
        self._port_spinbox.setMaximumWidth(110)

        self._connect_button = QPushButton("Connect")
        self._disconnect_button = QPushButton("Disconnect")
        self._disconnect_button.setEnabled(False)  # disabled until connected

        self._status_label = QLabel("Disconnected")

        # --- layout ---
        # Compact 2-column grid: host/port label+field pairs on the left,
        # Connect/Disconnect stacked to their right (each button lined up
        # with its own row), status spanning the full width below both.
        grid = QGridLayout(self)
        grid.setContentsMargins(6, 2, 6, 2)  # narrow top/bottom margins
        grid.setVerticalSpacing(4)
        # Pin contents to the top-left instead of stretching to fill
        # whatever extra space this widget is given by its container, and
        # give any leftover horizontal space to a trailing empty column
        # rather than letting it spread the fields/buttons apart.
        grid.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        grid.setColumnStretch(3, 1)

        grid.addWidget(QLabel("Host:"), 0, 0)
        grid.addWidget(self._host_edit, 0, 1)
        grid.addWidget(self._connect_button, 0, 2)

        grid.addWidget(QLabel("Port:"), 1, 0)
        grid.addWidget(self._port_spinbox, 1, 1)
        grid.addWidget(self._disconnect_button, 1, 2)

        grid.addWidget(self._status_label, 2, 0, 1, 3)  # span all 3 columns

        # --- wiring: View -> ViewModel (user actions become VM calls) ---
        self._connect_button.clicked.connect(self._on_connect_clicked)
        self._disconnect_button.clicked.connect(self._on_disconnect_clicked)

        # --- wiring: ViewModel -> View (VM signals update the display) ---
        self._viewmodel.status_updated.connect(self._on_status_updated)
        self._viewmodel.connection_state_changed.connect(self._on_connection_state_changed)

    # ------------------------------------------------------------------
    # View -> ViewModel
    # ------------------------------------------------------------------
    def _on_connect_clicked(self) -> None:
        host = self._host_edit.text().strip()
        port = self._port_spinbox.value()
        self._viewmodel.connect(host, port)

    def _on_disconnect_clicked(self) -> None:
        self._viewmodel.disconnect()

    # ------------------------------------------------------------------
    # ViewModel -> View
    # ------------------------------------------------------------------
    def _on_status_updated(self, message: str) -> None:
        self._status_label.setText(message)

    def _on_connection_state_changed(self, connected: bool) -> None:
        self._connect_button.setEnabled(not connected)
        self._disconnect_button.setEnabled(connected)
        self._host_edit.setEnabled(not connected)
        self._port_spinbox.setEnabled(not connected)