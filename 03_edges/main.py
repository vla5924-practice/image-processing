import cv2
import canny
import watershed
from time import time

gs_image = cv2.cvtColor(cv2.imread("image.png"), cv2.COLOR_BGR2GRAY)

edges = canny.do_edge_detection(gs_image, 100, 200)
cv2.imwrite('edges_image.png', edges)
cv2edges = cv2.Canny(gs_image, 100, 200)
cv2.imwrite('cv2edges_image.png', cv2edges)

time_start = time()
labels = watershed.do_watershed(gs_image)
time1 = time() - time_start

time_start = time()
labels = cv2.watershed(gs_image)
time2 = time() - time_start

print("watershed:", time1, "watershedcv2:", time2)
