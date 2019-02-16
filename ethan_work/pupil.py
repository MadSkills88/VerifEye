import numpy as np
import cv2
from matplotlib import pyplot as plt

# Modelled after recent research: https://arxiv.org/pdf/1712.08900.pdf

# load image and get eye
# Read as grayscale bro0
img = cv2.imread("base.jpg")


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eyeglasses_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eyeglasses_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
# print(faces)
# for (x,y,w,h) in faces:
#     print("asdf")
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#     roi_gray = img[y:y+h, x:x+w]
#     roi_color = img[y:y+h, x:x+w]
#     roi_gray = img[y:y+h, x:x+w]
#     eyes = eyeglasses_cascade.detectMultiScale(roi_gray)
#     for (ex,ey,ew,eh) in eyes:
#         cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#
#     edges = cv2.Canny(eyes,100,200)
#     print(edges)
#     plt.subplot(122),plt.imshow(edges,cmap = 'gray')
#     plt.title("Edge version"), plt.xticks([]), plt.yticks([])
#     plt.show()
#
# # Canny edge detection
#
# # cv2.imshow('dst_rt', img)
# # cv2.waitKey(0)
# cv2.destroyAllWindows()
