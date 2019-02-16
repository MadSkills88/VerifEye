import matplotlib.pyplot as plt
import numpy as np

pts = [[636 ,658], [678, 611], [733, 608], [771,644], [737, 667] ,[683 ,667]]


pts = np.array(pts)
print(max(pts[:, 1] - min(pts[:,1])))
print(pts[:, 0])
plt.plot(pts[:,0], pts[:,1])
plt.show()