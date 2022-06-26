import cv2
import mediapipe as mp
import time

myPose = mp.solutions.pose
pose = myPose.Pose()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture('PoseVideos/2.mp4')
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB) # send our image to being process
    # print(result.pose_landmarks)   gives x y z values as well as visibility

    if results.pose_landmarks :
        mpDraw.draw_landmarks(img, results.pose_landmarks, myPose.POSE_CONNECTIONS) # points and connection lines
        for id, lm in enumerate(results.pose_landmarks.landmark): # by using enumerate, we get the loop count, which is the id
            h, w, c = img.shape # we need the h, w because we want to multiply them by the x and y ratio to get the pixel
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy),6, (255, 0, 0), cv2.FILLED)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
    img = cv2.resize(img, (1440, 810))
    cv2.imshow("Image", img)
    cv2.waitKey(2)