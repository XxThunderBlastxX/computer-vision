import time

import cv2 as cv
import mediapipe as mp


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectCon = detectCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLmr in results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLmr, self.mpHands.HAND_CONNECTIONS)

        return img

        # for id, lm in enumerate(handLmr.landmark):
        #     print(id, lm)
        #     h, w, c = img.shape
        #     cx, cy = int(lm.x * w), int(lm.y * h)
        #
        #     if id == 12:
        #         cv.circle(img, (cx, cy), 18, (255, 0, 255), cv.FILLED)


def main():
    pTime = 0
    cTime = 0
    cap = cv.VideoCapture(1)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        cv.imshow("Image", img)
        cv.waitKey(0)


if __name__ == "__main__":
    main()
