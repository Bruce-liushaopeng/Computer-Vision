import cv2
import time
import mediapipe as mp
cap = cv2.VideoCapture(1) # run video capture

mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB first because our package only work with RGB

    results = hands.process(imgRGB) # after this, all we need is to extract the results and use them

    print(results.multi_hand_landmarks) # will print nothing, but when we put our hand in cam, it will print something


    cv2.imshow("Image", img)
    cv2.waitKey(1)