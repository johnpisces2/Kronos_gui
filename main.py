#!/usr/bin/env python3
"""Main entry point for Kronos GUI."""

import os
import sys

if sys.platform == "darwin":
    os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kronos_gui import KronosGUI

try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt6.QtWidgets import QApplication


def main():
    app = QApplication(sys.argv)
    window = KronosGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
