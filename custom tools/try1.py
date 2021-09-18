import cv2
import numpy as np
from DAL_utils.overlaps_cuda.rbbox_overlaps import rbbx_overlaps
from DAL_utils.overlaps.rbox_overlaps import rbox_overlaps

a = np.array([[10, 10, 20, 10, 0]], dtype=np.float32)
b = np.array([[10, 10, 20, 10, 0]], dtype=np.float32)
c = rbbx_overlaps(a, b)
d = rbox_overlaps(a, b)
print(c)
print(d)

# def iou_rotate_calculate(boxes1, boxes2):
#     area1 = boxes1[2] * boxes1[3]
#     area2 = boxes2[2] * boxes2[3]
#     r1 = ((boxes1[0], boxes1[1]), (boxes1[2], boxes1[3]), boxes1[4])
#     r2 = ((boxes2[0], boxes2[1]), (boxes2[2], boxes2[3]), boxes2[4])
#
#     int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
#     if int_pts is not None:
#         order_pts = cv2.convexHull(int_pts, returnPoints=True)
#
#         int_area = cv2.contourArea(order_pts)
#
#         inter = int_area * 1.0 / (area1 + area2 - int_area)
#         return inter
#     else:
#         return 0.0
#
# for i in range(100):
#     a = np.array([np.random.randn()])
#     a = np.array([1, 0.5, 2, 1, 0])
#     a = np.array([1, 0.5, 1, 2, 90])
#     b = np.array([0.5, 1, 2, 1, -90])
#
#     iou = iou_rotate_calculate(a, b)
#     print(iou)

# cnt = np.array([[0, 0], [2, 0], [2, 2], [0, 2]])
# cnt = np.array([[0, 2], [2, 0], [3, 1], [1, 3]])
# cnt = np.array([[0, 1], [1, 0], [3, 2], [2, 3]])
#
# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# print(rect)
# print(box)

# rect = ((1.5, 1.5), (2.8284271, 1.4142135), -45)
# box = cv2.boxPoints(rect)
# print(box)

# a = np.array([[10, 10, 20, 10, 0]], dtype=np.float32)
# b = np.array([[10, 10, 40, 10, 0]], dtype=np.float32)
# c = rbbx_overlaps(a, b)
# d = rbox_overlaps(a, b)
# print(c)
# print(d)
