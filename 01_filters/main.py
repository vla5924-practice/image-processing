import cv2
import metrics
import grayscale
import color_models


bgr_image = cv2.imread("image.png")

# 1. Metrics
pass

# 2. Grayscale
gs_image = grayscale.luminosity(bgr_image)
cv2.imwrite('gs_image.png', gs_image)
gscv2_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gscv2_image.png', gs_image)

# 3. Color models
hsv_image = color_models.bgr_to_hsv(bgr_image)
cv2.imwrite('hsv_image.png', hsv_image)
hsvcv2_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsvcv2_image.png', hsv_image)
