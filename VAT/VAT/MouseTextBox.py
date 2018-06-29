import sys
from PyQt5.QtWidgets import QTextEdit, QCheckBox, QMainWindow, QApplication, QWidget, QPushButton, QAction, QLineEdit, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
 
class TextBox(QMainWindow):
 
    def __init__(self):
        super().__init__()
        self.title = 'Settings'
        self.left = 10
        self.top = 10
        self.width = 800
        self.height = 400
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        # Folder TextBox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280,20)

        # Boolean show video
        self.viewVideo = QCheckBox(self)
        self.viewVideo.move(20, 50)

        # Create a button in the window
        self.button = QPushButton('Start', self)
        self.button.move(20,80)

        # Log Of How The program is doing
        self.log = QTextEdit(self)
        self.log.setReadOnly(True)
        self.log.move(320, 20)
        self.log.resize(300, 300)
        
 
        # connect button to function on_click
        self.button.clicked.connect(self.on_click)
        self.show()
 
    @pyqtSlot()
    def on_click(self):
        self.log.append("Started application")
    
    def logMessage(self, msg):
        self.log.append(msg)
 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())