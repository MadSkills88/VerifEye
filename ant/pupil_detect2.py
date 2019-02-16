from scipy.spatial import distance
from imutils import face_utils
from matplotlib import pyplot as plt
import imutils
import dlib
import cv2
import numpy as np


thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("../sunny/shape_predictor_68_face_landmarks.dat")

print("[INFO] starting video stream thread...")
# video_capture = cv2.VideoCapture('http://10.19.187.92:8080/video')
video_capture=cv2.VideoCapture(0)


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

while not success:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    src = imutils.resize(frame, width=1200)
    x, y, channels = src.shape
    crop_img = src

    # src = cv2.imread('../images/aaron_paul.jpg')
    src = cv2.imread('../images/sunny1.jpg')
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

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
        crop_img = gray[min(left_min[1], right_min[1]): max(left_max[1], right_max[1]),
        min(left_min[0], right_min[0]): max(left_max[0], right_max[0])]

        # Binary thresholding
        ret, binary_crop = cv2.threshold(crop_img, 1, 255, cv2.THRESH_BINARY_INV)
        # cs,_ = cv2.findContours(binary_img.astype('uint8'), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
        # crop_img = binary_img

        maxradius = max(leftEye[:,1]) - min(leftEye[:,1])

        if crop_img is None:
            continue

        # Play with gaussian blurr filter size
        # binary_img = cv2.medianBlur(crop_img, 5)

        rows = gray.shape[0]
        # Detect pupils
        pupils = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows//16,
                                      param1=100, param2=30,
                                      minRadius=1, maxRadius=maxradius//2)

        iris = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows//16,
                                      param1=100, param2=30,
                                      minRadius=maxradius//2,
                                      maxRadius=maxradius + 10)


        ## good
        # if iris is not None:
        #     circles = np.uint16(np.around(iris))
        #     for i in circles[0, :]:
        #         center = (i[0], i[1])
        #         # circle center
        #         cv2.circle(src, center, 1, (0, 100, 100), 3)
        #         # circle outline
        #         radius = i[2]
        #         print("radius: ", radius)
        #         cv2.circle(src, center, radius, (255, 0, 255), 3)
        #         cv2.rectangle(src, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)
        #         print(i[0])
        #         print(i[1])
        #         print(x)
        #         print(y)
        #         print(x - (i[0] - 5))
        #         print(y-(i[1] + 5))
        #         print(src.shape)
        #         crop_iris = src[y:(y-(i[1] + 5)), x:(x - (i[0] - 5))]
        #         cv2.imshow("iris", crop_iris)

        ## bad
        # for i in iris:
        #     print(type(i))
        #     print(i)
        #     print("iris")
            # (ix, iy, iw, ih) = i
            # crop_iris = src[y:y + iy + ih, x:x + ix + iw]
            # cv2.imshow("iris", crop_iris)
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
                cv2.rectangle(crop_img, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)

        if pupils is not None:
            # print(iris)
            circles = np.uint16(np.around(pupils))
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(crop_img, center, 1, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                # print("radius: ", radius)
                cv2.circle(crop_img, center, radius, (255, 0, 255), 3)
                cv2.rectangle(crop_img, (i[0] - 5, i[1] - 5), (i[0] + 5, i[1] + 5), (0, 128, 255), -1)

        cv2.imshow("cropped", crop_img)
        cv2.imshow("binary crop", binary_crop)
        # cv2.imshow("binary", binary_img)
    # show the output src
    # cv2.imshow("output", src)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()
