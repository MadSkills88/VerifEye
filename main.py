from scipy.spatial import distance
from imutils import face_utils
import matplotlib.pyplot as plt
import imutils
import dlib
import cv2
import numpy as np
from collections import deque
import time
import pandas as pd
import datetime

import tkinter as tk

# Tracking Balls Lists for Pandas Dataframe
# Forward
forward_timestamps = []
forward_x = []
forward_y = []

#Backward
backward_timestamps = []
backward_x = []
backward_y = []

TIME_CAP = 30

n = 0
x_coords_1 = 600
negative = False

# Timer inialization
startTest = False
nSeconds = 0
SECCAP = 30

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


# Samples y coordinates
samples_y_total = 0
samples_y_avg = 0
samples_y_counter = 0
doneCalibrating = False



print("[INFO] starting video stream thread...")

video_capture=cv2.VideoCapture(0)
# video_capture.set(cv2.CAP_PROP_FPS, 60)

right_pts = []
left_pts= []

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

flag=0
success = False
startTime = datetime.datetime.now()

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

left_eye_times = []
right_eye_times = []
# initialize latest time

while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    src = imutils.resize(frame, width=1200)
    x, y, channels = src.shape
    blacked_image = np.zeros((512,512,3))
    blacked_image = cv2.resize(blacked_image, (y, x))

    # src = cv2.imread('aaron_paul.jpg')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    value = 10

    gray = cv2.multiply(gray, 1.7, gray)
    gray = np.where((255 - gray) < value, 255 , gray + value)

    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape) #converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        left_max = (np.max(leftEye[:, 0] + 10), np.max(leftEye[:, 1]) + 10)
        left_min = (min(leftEye[:, 0]) - 10, min(leftEye[:, 1]) - 10)

        right_max = (max(rightEye[:,0]) + 10, max(rightEye[:,1]) + 10)
        right_min = (min(rightEye[:,0]) - 10, min(rightEye[:,1]) - 10)
        #     (max(leftEye[:,1]) + min(leftEye[:,1]))//2)
        left_range = cv2.rectangle(src, left_min, left_max, (0, 128, 255), 1)
        right_range = cv2.rectangle(src, right_min, right_max, (0, 128, 255), 1)
        crop_left = gray[left_min[1]: left_max[1], left_min[0] : left_max[0]]
        crop_right = gray[right_min[1]: right_max[1], right_min[0] : right_max[0]]


        max_left_radius = max(leftEye[:,1]) - min(leftEye[:,1])
        max_right_radius = max(rightEye[:,1]) - min(rightEye[:,1])
        samples_y_counter += 2
        samples_y_total = samples_y_total + leftEyeHull[0][0][1] + rightEyeHull[0][0][1]

        if startTest and doneCalibrating:
            timeElapsed = (datetime.datetime.now() - startTime).total_seconds()
            print("timeElapsed", timeElapsed)
        else:
            timeElapsed = 0

        if crop_left is not None:
            gray_left = cv2.medianBlur(crop_left, 5)
            if gray_left is not None:

                rows = gray_left.shape[0]
                # Detect pupils
                left_iris = cv2.HoughCircles(gray_left, cv2.HOUGH_GRADIENT, 1, rows//16,
                                              param1=100, param2=30,
                                              minRadius=1,
                                              maxRadius=max_left_radius)
                # Get time of left eye measurement
                left_time = timeElapsed

        if crop_right is not None:

            gray_right = cv2.medianBlur(crop_right, 5)
            if gray_right is not None:

                rows = gray_right.shape[0]

                right_iris = cv2.HoughCircles(gray_right, cv2.HOUGH_GRADIENT, 1, rows//16,
                                              param1=100, param2=30,
                                              minRadius=1,
                                              maxRadius=max_right_radius)
                # Get time of right eye measurement
                right_time = timeElapsed
        # print(pupils, iris)
        if right_iris is not None and left_iris is not None:
            # print(iris)
            circles = np.uint16(np.around(right_iris))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # print(center, left_max)
                if center != (0,0) and startTest == True:
                    right_pts.append(center)
                    right_eye_times.append(timeElapsed)

                # circle center
                cv2.circle(crop_right, center, 2, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(crop_right, center, radius, (255, 0, 255), 3)
                cv2.rectangle(crop_right, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)
                break
            # print(iris)
            circles = np.uint16(np.around(left_iris))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # print(center, left_max)
                if center != (0,0) and startTest == True:
                    left_pts.append(center)
                    left_eye_times.append(timeElapsed)
                # circle center
                cv2.circle(crop_left, center, 2, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(crop_left, center, radius, (255, 0, 255), 3)
                cv2.rectangle(crop_left, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)
                break

        cv2.imshow("cropped", crop_left)

        if not doneCalibrating and samples_y_counter >= 20:
            print("samples_y_total: ", samples_y_total)
            print("samples_y_counter: ", samples_y_counter)
            samples_y_avg = int(samples_y_total / samples_y_counter)
            doneCalibrating = True

        print("left_pts", left_pts)
        print("right_pts", right_pts)

    # Make sure it is the same

    cv2.putText(img = blacked_image,
                text = "Press 2 to Start/Restart",
                org = (0,int(y-y/2)),
                fontFace = cv2.FONT_HERSHEY_COMPLEX,
                fontScale = 3,
                color = (255,255,255),
                thickness = 3,
                lineType = cv2.LINE_AA)

    if (cv2.waitKey(1) & 0xFF == ord('2')):
        startTime = datetime.datetime.now()
        startTest = True

        # and samples_x_distance_avg > 0:
    if doneCalibrating:
        if startTest:
            print(TIME_CAP, timeElapsed)
            if timeElapsed < TIME_CAP:
                if x_coords_1 >= y - 50:
                    negative = True
                elif x_coords_1 < 50:
                    negative = False

                # Draw right pupil to the right
                print("samples_y_avg: ", samples_y_avg)
                cv2.circle(blacked_image, (x_coords_1, 20), 15, (0,0,255), -1)

                # To account for lag
                cv2.circle(blacked_image, (x_coords_1-1, 20), 15, (0,0,255), -1)
                cv2.circle(blacked_image, (x_coords_1-2, 20), 15, (0,0,255), -1)
                cv2.circle(blacked_image, (x_coords_1-3, 20), 15, (0,0,255), -1)
                cv2.circle(blacked_image, (x_coords_1-4, 20), 15, (0,0,255), -1)
                cv2.circle(blacked_image, (x_coords_1+1, 20), 15, (0,0,255), -1)
                cv2.circle(blacked_image, (x_coords_1+2, 20), 15, (0,0,255), -1)
                cv2.circle(blacked_image, (x_coords_1+3, 20), 15, (0,0,255), -1)
                cv2.circle(blacked_image, (x_coords_1+4, 20), 15, (0,0,255), -1)

                # cv2.line(blacked_image, (x?_coords_1, samples_y_avg), (x_coords_1, samples_y_avg), (0,0,255), 3, -1)

                if negative:
                    backward_timestamps.append(timeElapsed)
                    backward_x.append(x_coords_1)
                    backward_y.append(samples_y_avg)
                    x_coords_1-=15
                else:
                    forward_timestamps.append(timeElapsed)
                    forward_x.append(x_coords_1)
                    forward_y.append(samples_y_avg)
                    x_coords_1+=15
            else:
                print("timeElapsed: ", timeElapsed)
                startTest = False
                cv2.putText(img = blacked_image,
                            text = "Test Done",
                            org = (0,int(y/4)),
                            fontFace = cv2.FONT_HERSHEY_COMPLEX,
                            fontScale = 3,
                            color = (255,255,255),
                            thickness = 3,
                            lineType = cv2.LINE_AA)

                forward = pd.DataFrame(
                    {'timestamp': forward_timestamps,
                     'x': forward_x,
                     'y': forward_y
                    })

                backward = pd.DataFrame(
                    {'timestamp': backward_timestamps,
                     'x': backward_x,
                     'y': backward_y
                    })

                forward.to_csv("data/forward.csv")
                backward.to_csv("data/backward.csv")

    cv2.imshow("black overlay", blacked_image)


    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()

print(right_pts)
print(left_pts)

print(right_eye_times)
print(left_eye_times)

right_eye_data = []
for i in range(len(right_pts)):
    time_step = right_eye_times[i]
    right_eye_x = right_pts[i][0]
    right_eye_y = right_pts[i][1]
    entry = [time_step, right_eye_x, right_eye_y]
    right_eye_data.append(entry)

right_eye_data = np.array(right_eye_data)

print(right_eye_data)

left_eye_data = []
for i in range(len(left_pts)):
    time_step = left_eye_times[i]
    left_eye_x = left_pts[i][0]
    left_eye_y = left_pts[i][1]
    entry = [time_step, left_eye_x, left_eye_y]
    left_eye_data.append(entry)

left_eye_data = np.array(left_eye_data)

print(left_eye_data)

right_pd = pd.DataFrame(right_eye_data, columns=['t', 'x', 'y'])

left_pd = pd.DataFrame(left_eye_data, columns=['t', 'x', 'y'])

right_pd.to_csv("data/right_eye.csv")
left_pd.to_csv("data/left_eye.csv")
