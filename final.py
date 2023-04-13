import tensorflow as tf
import cv2
import numpy as np

#model load
model = tf.keras.models.load_model('main/model1.h5')

#labels load
labels = []
for i in range(10):
    labels.append(chr(48+i))
for i in range(26):
    labels.append(chr(65+i))
labels = sorted(labels)

#background load
greyBackground = cv2.cvtColor(cv2.imread("bg.jpg"), cv2.COLOR_BGR2GRAY)
greyBackground = cv2.resize(greyBackground, (128, 128))

#camera load
capture = cv2.VideoCapture(2)

if not capture.isOpened():
    print('Unable to open: ')
    exit(0)

#process
while True:
    ret, frame = capture.read()
    if frame is None:
        break
    
    showww = cv2.resize(frame,(500,500)) 
    frame = cv2.resize(frame, (128, 128))
    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    dframe = cv2.absdiff(greyBackground, greyFrame)

    _, mask = cv2.threshold(dframe, 50, 255, cv2.THRESH_BINARY)
    
    foreground = cv2.bitwise_and(frame, frame, mask=mask)
    
    img = cv2.cvtColor(foreground, cv2.COLOR_BGR2RGB)
    # cv2.imshow("hey",img)
    img = np.expand_dims(img, 0)
    pred = model.predict(img) # This will limit the speed which is fine?
    print(labels[np.argmax(pred, axis=-1)[0]])
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = "Symbol: " + labels[np.argmax(pred, axis=-1)[0]]
    text_scale = 0.5
    text_thickness = 1
    text_color = (0, 255, 0)
    text_org = (400, 470)
    cv2.putText(showww, text, text_org, font, text_scale, text_color, text_thickness, cv2.LINE_AA)
    cv2.imshow("hey", showww)
    
    
    keyboard = cv2.waitKey(1)