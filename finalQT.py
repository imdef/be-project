import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtCore import Qt

import time

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Set the window size
        self.resize(500, 500)

        # Create a QLabel to show the video
        self.label = QLabel(self)
        self.label.setGeometry(QtCore.QRect(0, 0, 500, 500))
        # Create the first button
        self.button1 = QPushButton("Background", self)
        self.button1.setCheckable(True)
        # self.button1.setChecked(True)
        # self.button1.setEnabled(False)
        self.button1.move(10, 10)

        # Create the second button
        self.button2 = QPushButton("Process", self)
        self.button2.setCheckable(True)
        self.button2.setChecked(True)                
        self.button2.move(100, 10)

        self.button1.clicked.connect(lambda: self.handle_button("button1"))
        self.button2.clicked.connect(lambda: self.handle_button("button2"))
        
        # Load the video
        self.capture = cv2.VideoCapture(2)

        # Start the timer to update the video
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(5) # 5ms

    def keyPressEvent(self, event):
        if (event.key() == Qt.Key_R and self.button1.isChecked()):
            self.timer.stop()
            ret, self.frame = self.capture.read()
            if ret:
                # Convert the OpenCV image to a QImage
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)

                # Show the QImage in the QLabel
                self.label.setPixmap(QPixmap.fromImage(image))
                
        if (event.key() == Qt.Key_S and self.button1.isChecked()):
            cv2.imwrite("bg.jpg", self.frame)
            self.button1.setChecked(False)
            self.button2.setChecked(True)            
            self.timer.start()            

    def handle_button(self, butt):
        if butt == "button1":
            self.button2.setChecked(False)
        elif butt == "button2":
            self.button1.setChecked(False)
            self.timer.start()            
                    
        
    
    
    def update(self):
        
        ret, self.frame = self.capture.read()
        if ret:
            # Convert the OpenCV image to a QImage
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            image = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)

            # Show the QImage in the QLabel
            self.label.setPixmap(QPixmap.fromImage(image))
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
