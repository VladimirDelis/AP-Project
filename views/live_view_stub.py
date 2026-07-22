"""
STUB — placeholder for Person B's real live VisPy view + connection
controls (per the plan, B owns connection controls since they're coupled
to the live view's lifecycle).

REPLACE this whole file with B's real live_view.py. The only thing
main.py (Person C) needs from it is:
    - a QWidget you can drop into a tab
    - connection_state_changed / status_updated signals coming from the
      Model, forwarded up so main.py can update the shared status bar and
      auto-switch tabs on disconnect
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSpinBox,
)


class LiveViewStub(QWidget):
    """Bare-bones connect/disconnect controls, no real plotting.
    Person B will replace this with the actual VisPy live view."""

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model

        layout = QVBoxLayout(self)

        conn_row = QHBoxLayout()
        conn_row.addWidget(QLabel("Port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1, 65535)
        self.port_spin.setValue(12345)  # matches EMGTCPServer default
        conn_row.addWidget(self.port_spin)

        self.connect_btn = QPushButton("Connect")
        self.connect_btn.clicked.connect(self._on_connect)
        conn_row.addWidget(self.connect_btn)

        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.clicked.connect(self.model.disconnect)
        self.disconnect_btn.setEnabled(False)
        conn_row.addWidget(self.disconnect_btn)
        conn_row.addStretch()
        layout.addLayout(conn_row)

        layout.addWidget(QLabel(
            "[Live VisPy plot placeholder — Person B's real view goes here]"
        ))

        self.model.connection_state_changed.connect(self._on_connection_state_changed)

    def _on_connect(self) -> None:
        self.model.connect("127.0.0.1", self.port_spin.value())

    def _on_connection_state_changed(self, connected: bool) -> None:
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)
        self.port_spin.setEnabled(not connected)
