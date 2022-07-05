import cv2
import mediapipe as mp
import time

class PoseDetector:

    def __init__(self, mode=False, upBody=False, smooth=True,
                 detectionCon=0.5, trackingCon=0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, 1 , self.upBody, self.smooth, self.detectionCon, self.trackingCon)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)  # send our image to being process
        if self.results.pose_landmarks and draw:
            self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                  self.mpPose.POSE_CONNECTIONS)  # points and connection lines
            return img

    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):  # by using enumerate, we get the loop count, which is the id
                h, w, c = img.shape # we need the h, w because we want to multiply them by the x and y ratio to get the pixel
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy),6, (255, 0, 0), cv2.FILLED)
        return lmList




        # success, img = cap.read()
        # imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # results = pose.process(imgRGB) # send our image to being process
        # # print(result.pose_landmarks)   gives x y z values as well as visibility
        #
        # if results.pose_landmarks :
        #     mpDraw.draw_landmarks(img, results.pose_landmarks, myPose.POSE_CONNECTIONS) # points and connection lines
        #     for id, lm in enumerate(results.pose_landmarks.landmark): # by using enumerate, we get the loop count, which is the id
        #         h, w, c = img.shape # we need the h, w because we want to multiply them by the x and y ratio to get the pixel
        #         cx, cy = int(lm.x * w), int(lm.y * h)
        #         cv2.circle(img, (cx, cy),6, (255, 0, 0), cv2.FILLED)




def main():
    cap = cv2.VideoCapture('PoseVideos/2.mp4')
    pTime = 0
    detector = PoseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        lmList = detector.findPosition(img,draw=False)
        cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,0), 3)
        print(lmList[14])
        cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (255,0 ,0), cv2.FILLED)
        img = cv2.resize(img, (1440, 810))
        cv2.imshow("Image", img)
        cv2.waitKey(2)

if __name__ == "__main__":
    main()