"""
views/mode_selector_widget.py

ModeSelectorWidget -- a small, dumb View for picking the display mode
(original / RMS / filtered).

Follows the exact same pattern as ChannelSelectorWidget: it never decides
or stores the mode itself. Choosing an item calls `viewmodel.set_mode()`,
and the ViewModel is the only thing that actually owns "current mode" --
this widget just reflects `viewmodel.mode_changed` back into its own
display, the same way ChannelSelectorWidget reflects `channel_changed`.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

# (mode string the ViewModel expects, human-readable label for the combo box)
MODE_OPTIONS = (
    ("original", "Original"),
    ("rms", "RMS"),
    ("filtered", "Filtered"),
)


class ModeSelectorWidget(QWidget):
    """
    A labeled QComboBox bound to a LiveViewModel's mode selection.

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
        layout.addWidget(QLabel("Mode:"))

        self._combo = QComboBox()
        for mode_value, label in MODE_OPTIONS:
            self._combo.addItem(label, userData=mode_value)

        initial_index = self._combo.findData(viewmodel.selected_mode)
        if initial_index >= 0:
            self._combo.setCurrentIndex(initial_index)

        layout.addWidget(self._combo)
        layout.addStretch(1)

        # View -> ViewModel
        self._combo.currentIndexChanged.connect(self._on_combo_changed)

        # ViewModel -> View
        self._viewmodel.mode_changed.connect(self._on_mode_changed)

    def _on_combo_changed(self, index: int) -> None:
        mode_value = self._combo.itemData(index)
        self._viewmodel.set_mode(mode_value)

    def _on_mode_changed(self, mode: str) -> None:
        # Guard against feedback, same reasoning as ChannelSelectorWidget:
        # setCurrentIndex() would otherwise re-trigger currentIndexChanged
        # -> _on_combo_changed -> set_mode(), which is harmless (set_mode()
        # no-ops on an unchanged value) but unnecessary.
        index = self._combo.findData(mode)
        if index >= 0 and self._combo.currentIndex() != index:
            self._combo.blockSignals(True)
            self._combo.setCurrentIndex(index)
            self._combo.blockSignals(False)