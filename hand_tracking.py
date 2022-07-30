import time

import cv2 as cv
import mediapipe as mp

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLmr in results.multi_hand_landmarks:
            for id, lm in enumerate(handLmr.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 12:
                    cv.circle(img, (cx, cy), 18, (255, 0, 255), cv.FILLED)
            mpDraw.draw_landmarks(img, handLmr, mpHands.HAND_CONNECTIONS)

    cTime = time.time()

    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

    cv.imshow("Image", img)

    cv.waitKey(1)
