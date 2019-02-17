import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

def read_csv_file(filename):
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        csv_data = list(reader)
    return csv_data

right_eye_filename = "../data/right_eye.csv"
left_eye_filename = "../data/left_eye.csv"

right_eye_data = np.array(read_csv_file(right_eye_filename)[1:]).astype(float)
left_eye_data = np.array(read_csv_file(left_eye_filename)[1:]).astype(float)

right_eye_x = []
right_eye_y = []
right_eye_t = []

left_eye_x = []
left_eye_y = []
left_eye_t = []

for i in range(1, len(right_eye_data)):
    r_t = right_eye_data[i][1]
    r_x = right_eye_data[i][2]
    r_y = right_eye_data[i][3]
    right_eye_t.append(r_t)
    right_eye_x.append(r_x)
    right_eye_y.append(r_y)

for i in range(1, len(left_eye_data)):
    l_t = left_eye_data[i][1]
    l_x = left_eye_data[i][2]
    l_y = left_eye_data[i][3]
    left_eye_t.append(l_t)
    left_eye_x.append(l_x)
    left_eye_y.append(l_y)

diff_right_eye_x = [abs(j-i) for i, j in zip(right_eye_x[:-1], right_eye_x[1:])]
diff_right_eye_y = [abs(j-i) for i, j in zip(right_eye_y[:-1], right_eye_y[1:])]
diff_left_eye_x = [abs(j-i) for i, j in zip(left_eye_x[:-1], left_eye_x[1:])]
diff_left_eye_y = [abs(j-i) for i, j in zip(left_eye_y[:-1], left_eye_y[1:])]

diff_right_eye_t = [j-i for i, j in zip(right_eye_t[:-1], right_eye_t[1:])]
diff_left_eye_t = [j-i for i, j in zip(left_eye_t[:-1], left_eye_t[1:])]

# line_lx, = plt.plot(left_eye_t, left_eye_x, 'g', label="left eye x")
# line_ly, = plt.plot(left_eye_t, left_eye_y, 'b', label="left eye y")
#
# line_rx, = plt.plot(right_eye_t, right_eye_x, 'r', label="right eye x")
# line_ry, = plt.plot(right_eye_t, right_eye_y, 'orange', label="right eye y")

line_lx, = plt.plot(diff_left_eye_t, diff_left_eye_x, 'g', label="left eye x")
line_ly, = plt.plot(diff_left_eye_t, diff_left_eye_y, 'b', label="left eye y")

line_rx, = plt.plot(diff_right_eye_t, diff_right_eye_x, 'r', label="right eye x")
line_ry, = plt.plot(diff_right_eye_t, diff_right_eye_y, 'orange', label="right eye y")


print(left_eye_t)
print(left_eye_x)

print(diff_left_eye_t)
print(diff_left_eye_x)

# plt.scatter(left_eye_t, left_eye_x, c='g')
# plt.scatter(left_eye_t, left_eye_y, c='b')
#
# plt.scatter(right_eye_t, right_eye_x, c='r')
# plt.scatter(right_eye_t, right_eye_y, c='orange')

plt.ylabel('pixels')
plt.xlabel('time')
plt.show()
