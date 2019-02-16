from scipy.spatial import distance
from imutils import face_utils
import matplotlib.pyplot as plt
import imutils
import dlib
import cv2
import numpy as np
from collections import deque
import time

t_end = time.time() + 10

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

print("[INFO] starting video stream thread...")
# video_capture = cv2.VideoCapture('http://10.19.187.92:8080/video')
video_capture=cv2.VideoCapture(0)

right_pt = []
left_pt= []

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

while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    src = imutils.resize(frame, width=1200)

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

        right_max = (max(rightEye[:,0]) + 10, max(rightEye[:,1]) + 10)
        right_min = (min(rightEye[:,0]) - 10, min(rightEye[:,1]) - 10)
        #     (max(leftEye[:,1]) + min(leftEye[:,1]))//2)
        left_range = cv2.rectangle(src, left_min, left_max, (0, 128, 255), 1)
        right_range = cv2.rectangle(src, right_min, right_max, (0, 128, 255), 1)
        crop_left = gray[left_min[1]: left_max[1], left_min[0] : left_max[0]]
        crop_right = gray[right_min[1]: right_max[1], right_min[0] : right_max[0]]
    

        max_left_radius = max(leftEye[:,1]) - min(leftEye[:,1])
        max_right_radius = max(rightEye[:,1]) - min(rightEye[:,1])

        if crop_left is not None:
            gray_left = cv2.medianBlur(crop_left, 5)
            if gray_left is not None:

                rows = gray_left.shape[0]
                # Detect pupils
                # pupils = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows//16,
                #                               param1=100, param2=30,
                #                               minRadius=1, maxRadius=maxradius//2)

                left_iris = cv2.HoughCircles(gray_left, cv2.HOUGH_GRADIENT, 1, rows//16,
                                              param1=100, param2=30,
                                              minRadius=1, 
                                              maxRadius=max_left_radius)

        if crop_right is not None:
            
            gray_right = cv2.medianBlur(crop_right, 5)
            if gray_right is not None:

                rows = gray_right.shape[0]
                # Detect pupils
                # pupils = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows//16,
                #                               param1=100, param2=30,
                #                               minRadius=1, maxRadius=maxradius//2)

                right_iris = cv2.HoughCircles(gray_right, cv2.HOUGH_GRADIENT, 1, rows//16,
                                              param1=100, param2=30,
                                              minRadius=1, 
                                              maxRadius=max_right_radius)
            # print(pupils, iris)
        if right_iris is not None:
            # print(iris)
            circles = np.uint16(np.around(right_iris))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # print(center, left_max)
                if center != (0,0):
                    right_pt.append(center)
                # circle center
                cv2.circle(crop_right, center, 2, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(crop_right, center, radius, (255, 0, 255), 3)
                cv2.rectangle(crop_right, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)

        if left_iris is not None:
            # print(iris)
            circles = np.uint16(np.around(left_iris))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # print(center, left_max)
                if center != (0,0):
                    left_pt.append(center)
                # circle center
                cv2.circle(crop_left, center, 2, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(crop_left, center, radius, (255, 0, 255), 3)
                cv2.rectangle(crop_left, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)
        '''
        pupils = None
        if pupils is not None:
            # print(iris)
            circles = np.uint16(np.around(pupils))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                if i[1] > x_max/2:
                    right_pt.append(center)
                else:
                    left_pt.append(center)
                cv2.circle(crop_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                # print("radius: ", radius)
                cv2.circle(crop_img, center, radius, (255, 0, 255), 3)
                cv2.rectangle(crop_img, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)
        '''
        cv2.imshow("cropped", crop_left)
        cv2.imshow("output", crop_right)
        
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()     
cv2.destroyAllWindows()  

print(right_pt)
print(left_pt)
# plt.imshow(crop_img)
#for i in range(len(right_pt)):
#    plt.scatter(right_pt[i][0], right_pt[i][1])
#
#for i in range(len(left_pt)):
#    plt.scatter(left_pt[i][0], left_pt[i][1])

# plt.savefig('test.jpg')
#plt.show()


