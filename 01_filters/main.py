import cv2
import metrics
import grayscale
import color_models
import time


bgr_image = cv2.imread("image.png")


# Grayscale
gs_image = grayscale.luminosity(bgr_image)
cv2.imwrite('gs_image.png', gs_image)
gscv2_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gscv2_image.png', gscv2_image)


# Color models
time_start1 = time.clock()
image1 = color_models.bgr_to_hsv(bgr_image)
image1 = color_models.hsv_to_bgr(image1)
time_end1 = time.clock()
cv2.imwrite('image1.png', image1)
time_start2 = time.clock()
image2 = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
image2 = cv2.cvtColor(image2, cv2.COLOR_HSV2BGR)
time_end2 = time.clock()
cv2.imwrite('image2.png', image2)
time_delta1 = time_end1 - time_start1
time_delta2 = time_end2 - time_start2
print(time_delta1, time_delta2)
mse_image1 = metrics.mse(bgr_image, image1)
mse_image2 = metrics.mse(bgr_image, image2)
print(mse_image1, mse_image2)
image3 = color_models.bgr_brighten(bgr_image, 1.05)
cv2.imwrite('image3.png', image3)
