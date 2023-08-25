import math
import time
import numpy as np
from string import ascii_uppercase as alc

import cv2
import tensorflow
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5","Model/labels.txt")

offset = 20
imgSize = 300

while True:
    success,img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize,3), np.uint8)*255
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        #size image and center
        aspectRatio = h/w
        if aspectRatio > 1:
            k = imgSize/h
            newWidth = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(newWidth,imgSize))

            wGap = math.ceil((imgSize - newWidth)/2)

            imgWhite[:, wGap:wGap+newWidth] = imgResize
        else:
            k = imgSize/w
            newHeight = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,newHeight))

            hGap = math.ceil((imgSize - newHeight)/2)

            imgWhite[hGap:hGap+newHeight,:] = imgResize

        prediction,index = classifier.getPrediction(imgWhite, draw=False)
        print(prediction[index])
        accurary = round(prediction[index],3)
        cv2.putText(imgOutput, alc[index],(x,y-40),cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,255),2)
        cv2.putText(imgOutput, str(accurary) + "%",(x+30,y-40),cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,0,255),1)
        cv2.rectangle(imgOutput, (x-offset,y-offset), (x+w+offset, y+h+offset), (255,0,255), 4)
    
    cv2.imshow("Image", imgOutput)
    key = cv2.waitKey(1)
