import sys
import cv2
import os
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton
from PyQt5.QtCore import Qt
import tensorflow as tf

import time

#model load
model = tf.keras.models.load_model('model.h5')

#labels load
labels = []
for i in range(10):
    labels.append(chr(48+i))
for i in range(26):
    labels.append(chr(65+i))
labels = sorted(labels)

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        # Set the window size
        self.resize(1050, 500)

        # Create a QLabel to show the video
        self.label = QLabel(self)
        self.label.setGeometry(QtCore.QRect(12, 0, 500, 500))
        
        self.label_mask = QLabel(self)
        self.label_mask.setGeometry(QtCore.QRect(525, 0, 500, 500))
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
        camera_indexes = []
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_indexes.append(i)
                cap.release()

        camera_indexes = sorted(camera_indexes, reverse=True)
        print("List of available cameras:", camera_indexes)
        
        if not os.path.isfile("bg.jpg"):
            cap = cv2.VideoCapture(camera_indexes[0])
            ret, frame = cap.read()
            if ret:
                cv2.imwrite("bg.jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cap.release()
            else:
                print("Initialization Error")
                sys.exit(1)

        self.capture = cv2.VideoCapture(camera_indexes[0])
        
        self.greyBackground = cv2.resize(cv2.imread("bg.jpg"), (128, 128))
        self.greyBackground = cv2.cvtColor(self.greyBackground, cv2.COLOR_BGR2GRAY) #bg.jpg has to be already present, add check for that in here and in handle_button
        
        
        # Start the timer to update the video
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(200) # 5ms
        
        
    def keyPressEvent(self, event):
        if (event.key() == Qt.Key_R and self.button1.isChecked()):
            
            ret, self.frame = self.capture.read()
            if ret:
                # Convert the OpenCV image to a QImage
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                image = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)

                # Show the QImage in the QLabel
                self.label.setPixmap(QPixmap.fromImage(image))
                
        if (event.key() == Qt.Key_S and self.button1.isChecked()):
            cv2.imwrite("bg.jpg", cv2.cvtColor(self.frame, cv2.COLOR_RGB2BGR))
            self.button1.setChecked(False)
            self.button2.setChecked(True)            
            self.timer.start()
            self.greyBackground = cv2.cvtColor(cv2.imread("bg.jpg"), cv2.COLOR_BGR2GRAY)
            self.greyBackground = cv2.resize(self.greyBackground, (128, 128)) 

    def handle_button(self, butt):
        if butt == "button1":
            self.button2.setChecked(False)
            self.timer.stop()
        elif butt == "button2":
            self.button1.setChecked(False)
            self.timer.start()            
                    
        
    
    
    def update(self):
        
        ret, self.frame = self.capture.read()
        if ret:
            self.frame = cv2.flip(self.frame, 1)
            # Convert the OpenCV image to a QImage
            self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            frame = cv2.resize(self.frame, (128, 128))
            greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            dframe = cv2.absdiff(self.greyBackground, greyFrame)

            # Realtime Hyperparameter
            _, mask = cv2.threshold(dframe, 50, 255, cv2.THRESH_BINARY)
            
            foreground = cv2.bitwise_and(frame, frame, mask=mask)
            
            img = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
            # cv2.imshow("hey",img)
            img = np.expand_dims(img, 0)
            pred = model.predict(img) # This will limit the speed which is fine?
            # print(labels[np.argmax(pred, axis=-1)[0]])
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = "Symbol: " + labels[np.argmax(pred, axis=-1)[0]]
            text_scale = 0.7
            text_thickness = 1
            text_color = (0, 255, 0)
            text_org = (350, 470)
            cv2.putText(self.frame, text, text_org, font, text_scale, text_color, text_thickness, cv2.LINE_AA)

            # maskSh = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            # image = QImage(maskSh, maskSh.shape[1], maskSh.shape[0], QImage.Format_RGB888)
            displayMask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            displayMask = cv2.resize(displayMask, (self.frame.shape[1],self.frame.shape[0]))
            displayMask_image = QImage(displayMask, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)            
            image = QImage(self.frame, self.frame.shape[1], self.frame.shape[0], QImage.Format_RGB888)
                        
            # Show the QImage in the QLabel
            self.label.setPixmap(QPixmap.fromImage(image))
            self.label_mask.setPixmap(QPixmap.fromImage(displayMask_image))
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
