from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from UserInterface import GuiWindow
import MouseDetectObject
import time


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = GuiWindow()
    app.exec_()
