import cv2
import time
import mediapipe as mp

class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,1 , self.detectionCon, self.trackCon) # model_complexity=1,
        self.mpDraw = mp.solutions.drawing_utils # we use this to draw lines and dots

    # return the image with hand point and hand connections
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to RGB first because our package only work with RGB

        self.results = self.hands.process(imgRGB)  # after this, all we need is to extract the results and use them

        if self.results.multi_hand_landmarks:  # if hands are found
            for handLms in self.results.multi_hand_landmarks:  # we loop through the land landmark

                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                      self.mpHands.HAND_CONNECTIONS)  # we are displaying the BGR image, not the RGB one, and here we are drawing the hand dows
        return img

    # return the position of 21 points in the hand as a list. such as [0, 25, 265].... as index 0 is at x=25, y=265
    def findPosition(self, img, handNo=0, draw=True):

        lmList = []

        if self.results.multi_hand_landmarks:  # if hands are found
            myHand = self.results.multi_hand_landmarks[handNo] #get the desired hand number's landmark

            for id, lm in enumerate(myHand.landmark):  # within that hand, get all the landmarks
                h, w, c = img.shape  # get width, height, and channel.
                cx, cy = int(lm.x * w), int(lm.y * h)  # find the pisition of the center ( for all 21 values )
                # print(id, cx, cy)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)  # 25 is the size of the circle

        return lmList


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(1)  # run video capture
    detector = handDetector()
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()  # current time
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255),
                    3)  # (img, fps integer, position, font, blue, thick

        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__": # we are going to write in main to show what this module can do
    main()