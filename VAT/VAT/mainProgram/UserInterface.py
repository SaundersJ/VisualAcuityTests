import sys
from PyQt5.QtWidgets import QProgressBar, QTextEdit, QCheckBox, QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot, QThread
import MouseDetectObject
import LogUtil

class GuiWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.title = 'Mouse Detection'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 400

        self.folderName = ""
        self.viewVideo = False
        self.clickedButton = False
        self.initUI()
        self.workerThread = WorkerThread(None, self)
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        # Folder TextBox
        self.textbox = QLineEdit(r"C:\Users\Jack\Desktop\python\Videos", self)
        self.textbox.move(20, 20)
        self.textbox.resize(280,20)

        # Boolean show video
        self.viewVideoBox = QCheckBox("Show Video", self)
        self.viewVideoBox.move(20, 50)

        # Create a button in the window
        self.button = QPushButton('Start Detection', self)
        self.button.move(20,80)

        # Log Of How The program is doing
        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.move(320, 20)
        self.log.resize(300, 300)

        self.currentStatusBar = QProgressBar(self)
        self.currentStatusBar.setGeometry(20, 130, 280, 20)

        self.totalStatusBar = QProgressBar(self)
        self.totalStatusBar.setGeometry(20, 160, 280, 20)

        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()
        self.initMessage()
 
    @pyqtSlot()
    def on_click(self):
        self.logMessage("Started application")
        self.logMessage("Settings:")
        self.folderName = self.textbox.text()
        self.logMessage(" - {}".format(self.folderName))
        self.viewVideo = self.viewVideoBox.isChecked()
        self.logMessage(" - {}".format(self.viewVideo))
        self.clickedButton = True

        self.workerThread.start()

    def logMessage(self, msg):
        self.log.append(msg)

    def getClickedButton(self):
        return self.clickedButton

    def initMessage(self):
        self.logMessage("Mouse Detection CUROP Project ...")
        self.logMessage("Fill out the settings and then click start.")
 
class WorkerThread(QThread):

    def __init__(self, parent=None, guiWindow=None):
        super(WorkerThread, self).__init__(parent)
        self.guiWindow = guiWindow

    def run(self):
        self.guiWindow.logMessage("NewThread")
        #def getTrackingTimes(fileName, showVideo, gui):
        track, videos = MouseDetectObject.getTrackingTimes(self.guiWindow.folderName, self.guiWindow.viewVideo, self.guiWindow)