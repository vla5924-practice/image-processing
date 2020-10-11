import cv2
from time import time
import metrics
import grayscale
import color_models


bgr_image = cv2.imread("image.png")

# Grayscale
gs_image = grayscale.luminosity(bgr_image)
cv2.imwrite('gs_image.png', gs_image)

gscv2_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gscv2_image.png', gscv2_image)


# Color models
hsv_image = color_models.bgr_to_hsv(bgr_image)
cv2.imwrite('hsv_image.png', hsv_image)

converted_image = color_models.hsv_to_bgr(hsv_image)
cv2.imwrite('converted_image.png', converted_image)

hsvcv2_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsvcv2_image.png', hsvcv2_image)

convertedcv2_image = cv2.cvtColor(hsvcv2_image, cv2.COLOR_HSV2BGR)
cv2.imwrite('convertedcv2_image.png', convertedcv2_image)

iterations = 1
time_start = time()
for i in range(iterations):
    image1 = color_models.bgr_brighten(bgr_image, 1.2)
    image2 = color_models.hsv_to_bgr(color_models.hsv_brighten(color_models.bgr_to_hsv(bgr_image), 1.2))
time1 = time() - time_start
mse1 = metrics.mse(image1, image2)
cv2.imwrite('bgr_brighten.png', image1)
cv2.imwrite('converted_brighten.png', image2)

time_start = time()
for i in range(iterations):
    image1 = color_models.bgr_brighten(bgr_image, 1.2)
    image2 = cv2.cvtColor(color_models.hsv_brighten(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV), 1.2), cv2.COLOR_HSV2BGR)
time2 = time() - time_start
mse2 = metrics.mse(image1, image2)
cv2.imwrite('convertedcv2_brighten.png', image2)

print(time1, time2)
print(mse1, mse2)
