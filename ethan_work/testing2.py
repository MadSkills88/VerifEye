from scipy.spatial import distance
from imutils import face_utils
from matplotlib import pyplot as plt
import imutils
import dlib
import cv2
import numpy as np
import datetime

import tkinter as tk

# Button set up
top = tk.Tk()
buttonPressed = False


def start():
    buttonPressed = True


def stop():
    buttonPressed = False


thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("[INFO] starting video stream thread...")
# video_capture = cv2.VideoCapture('http://10.19.187.92:8080/video')
video_capture = cv2.VideoCapture(0)


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

flag = 0
success = False

startTime = datetime.datetime.now()
TIME_CAP = 30

n = 0
x_coords = 100
negative = False

# Timer inialization
startTest = False
nSeconds = 0
SECCAP = 30

# Samples y coordinates
samples_y_total = 0
samples_y_avg = 0
samples_y_counter = 0
doneCalibrating = False

while not success:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    src = imutils.resize(frame, width=1200)
    x, y, channels = src.shape
    crop_img = src

    # src = cv2.imread('aaron_paul.jpg')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        print("lefthexhull")
        leftEyeHull = cv2.convexHull(leftEye)
        print(leftEyeHull)
        print(type(leftEyeHull))
        rightEyeHull = cv2.convexHull(rightEye)
        # cv2.drawContours(src, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(src, [rightEyeHull], -1, (0, 255, 0), 1)

        # left_center = ((max(leftEye[:,0]) + min(leftEye[:,0]))//2,
        #     (max(leftEye[:,1]) + min(leftEye[:,1]))//2)
        # right_center = ((max(rightEye[:,0])+  min(rightEye[:,0]))//2,
        #     (max(rightEye[:,1]) + min(rightEye[:,1]))//2)
        # cv2.circle(src, left_center, 3, (255, 0, 255), 3)
        # cv2.circle(src, right_center, 3, (255, 0, 255), 3)
        left_max = (np.max(leftEye[:, 0] + 10), np.max(leftEye[:, 1]) + 10)
        left_min = (min(leftEye[:, 0]) - 10, min(leftEye[:, 1]) - 10)

        right_max = (max(rightEye[:, 0]) + 10, max(rightEye[:, 1]) + 10)
        right_min = (min(rightEye[:, 0]) - 10, min(rightEye[:, 1]) - 10)
        #     (max(leftEye[:,1]) + min(leftEye[:,1]))//2)
        left_range = cv2.rectangle(
            src, left_min, left_max, (0, 128, 255), 1)
        right_range = cv2.rectangle(
            src, right_min, right_max, (0, 128, 255), 1)
        crop_img = gray[min(left_min[1], right_min[1]): max(left_max[1], right_max[1]),
        min(left_min[0], right_min[0]): max(left_max[0], right_max[0])]

        maxradius = max(leftEye[:, 1]) - min(leftEye[:, 1])

        samples_y_counter += 2
        samples_y_total = samples_y_total + leftEyeHull[0][0][1] + rightEyeHull[0][0][1]

        if crop_img is None:
            continue
        gray = cv2.medianBlur(crop_img, 5)

        rows = gray.shape[0]
        # Detect pupils
        pupils = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows // 16,
                                      param1=100, param2=30,
                                      minRadius=1, maxRadius=maxradius // 2)

        iris = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows // 16,
                                      param1=100, param2=30,
                                      minRadius=maxradius // 2,
                                      maxRadius=maxradius + 10)
        print(pupils, iris)
        if iris is not None:
            # print(iris)
            circles = np.uint16(np.around(iris))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(crop_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                # print("radius: ", radius)
                cv2.circle(crop_img, center, radius, (255, 0, 255), 3)
                cv2.rectangle(
                    crop_img, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)

        if pupils is not None:
            # print(iris)
            # UNCOMMENT OUT WHEN PUPILS ARE DETECTED LMAO
            # samples_y_counter += 1

            circles = np.uint16(np.around(pupils))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # UNCOMMENT OUT WHEN PUPILS ARE DETECTED LMAO
                # samples_y_total += i[1]

                # circle center
                cv2.circle(crop_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                # print("radius: ", radius)
                cv2.circle(crop_img, center, radius, (255, 0, 255), 3)
                cv2.rectangle(
                    crop_img, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)

        cv2.imshow("cropped", crop_img)
    # timer: https://stackoverflow.com/questions/28421559/python-opencv-display-time-countdown-using-puttext-in-webcam-vide
    # Check if it has been over 5 minutes
    # Display timer rn

    # Press start => starts timer and movement of eyes
    if not doneCalibrating and samples_y_counter >= 20:
        samples_y_avg = int(samples_y_total / samples_y_counter)
        doneCalibrating = True

    timeElapsed = (datetime.datetime.now() - startTime).total_seconds()
    cv2.putText(img = src,
                text = "Press 2 to Start/Restart",
                org = (0,int(y-y/4)),
                fontFace = cv2.FONT_HERSHEY_COMPLEX,
                fontScale = 3,
                color = (255,255,255),
                thickness = 3,
                lineType = cv2.LINE_AA)

    if (cv2.waitKey(1) & 0xFF == ord('2')):
        startTime = datetime.datetime.now()
        startTest = True


    if startTest and samples_y_avg > 0:
        if timeElapsed < TIME_CAP:
            cv2.putText(img = src,
                        text = "Time Elapsed: " + str(timeElapsed) + " s",
                        org = (0,int(y/4)),
                        fontFace = cv2.FONT_HERSHEY_COMPLEX,
                        fontScale = 3,
                        color = (0,0,0),
                        thickness = 3,
                        lineType = cv2.LINE_AA)

            # Circle stuff
            # Need to align with pupils
            if x_coords >= x-50:
                negative = True
            elif x_coords < 50:
                negative = False

            cv2.circle(src, (x_coords, samples_y_avg), 5, (0,0,255), -1)
            if negative:
                x_coords-=5
            else:
                x_coords+=5
        else:
            startTest = False
            cv2.putText(img = src,
                        text = "Test Done",
                        org = (0,int(y/4)),
                        fontFace = cv2.FONT_HERSHEY_COMPLEX,
                        fontScale = 3,
                        color = (0,0,0),
                        thickness = 3,
                        lineType = cv2.LINE_AA)






    # move circle across screen

    # show the output src
    cv2.imshow("output", src)

    cv2.imshow("detected circles", src)
    # cv2.imshow('Video', frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break



video_capture.release()
cv2.destroyAllWindows()
