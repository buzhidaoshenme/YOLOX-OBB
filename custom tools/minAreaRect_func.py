import cv2
import numpy as np
cnt = np.array([[20, 20], [20, 40], [40, 20], [40, 40]])
cnt = np.array([[20, 20], [20, 80], [21, 20], [21, 80]])
cnt = np.array([[20, 20], [20, 21], [40, 20], [40, 21]])

rect = cv2.minAreaRect(cnt)
print(rect)
