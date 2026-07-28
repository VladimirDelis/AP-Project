"""
views/channel_selector_widget.py

ChannelSelectorWidget -- a small, dumb View for picking which channel is
currently displayed.

Like ConnectionWidget, it never decides state on its own: a user changing
the spinbox calls `viewmodel.set_channel(index)`, and the ViewModel is the
only thing that actually owns "current channel". This widget listens to
`viewmodel.channel_changed` and reflects it back into the spinbox display,
so if the channel is ever changed some other way (e.g. programmatically,
or by another View), this widget stays in sync instead of just diverging.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QHBoxLayout, QLabel, QSpinBox, QWidget

N_CHANNELS = 32


class ChannelSelectorWidget(QWidget):
    """
    A labeled QSpinBox (0-31) bound to a LiveViewModel's channel selection.

    Parameters
    ----------
    viewmodel : LiveViewModel
        The ViewModel this widget is bound to.
    parent : QWidget, optional
    """

    def __init__(self, viewmodel, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        self._viewmodel = viewmodel

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Channel:"))

        self._spinbox = QSpinBox()
        self._spinbox.setRange(0, N_CHANNELS - 1)
        self._spinbox.setValue(viewmodel.selected_channel)
        layout.addWidget(self._spinbox)
        layout.addStretch(1)

        # View -> ViewModel
        self._spinbox.valueChanged.connect(self._on_spinbox_changed)

        # ViewModel -> View
        self._viewmodel.channel_changed.connect(self._on_channel_changed)

    def _on_spinbox_changed(self, value: int) -> None:
        self._viewmodel.set_channel(value)

    def _on_channel_changed(self, index: int) -> None:
        # Guard against feedback: setValue() would otherwise re-trigger
        # valueChanged -> _on_spinbox_changed -> set_channel(), which is
        # harmless here (set_channel() no-ops on an unchanged value) but
        # unnecessary and worth avoiding on principle.
        if self._spinbox.value() != index:
            self._spinbox.blockSignals(True)
            self._spinbox.setValue(index)
            self._spinbox.blockSignals(False)