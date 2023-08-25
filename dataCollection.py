import math
import time
import numpy as np

import cv2
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase as alc

# video feed
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
counter = 0
index = 0

while True:
    success,img = cap.read()
    hands, img = detector.findHands(img)

    # if hand detected on camera
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

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
    
    # change folder for data collection
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("n"):
        folder = f"Data/{alc[index]}"
        print("Saving to ", folder)
        index+=1
        counter = 0

    #hit s to save image 
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)