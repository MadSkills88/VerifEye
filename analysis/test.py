import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

def read_csv_file(filename):
    with open(filename) as csv_file:
        reader = csv.reader(csv_file)
        csv_data = list(reader)
    return csv_data

distance_filename = "data/distances.csv"

distance_data = np.array(read_csv_file(distance_filename)[1:]).astype(float)

plt.plot(distance_data[:,0], distance_data[:,1])
plt.show()