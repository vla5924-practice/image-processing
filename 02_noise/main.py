import cv2
import noise
import filters
from time import time

gs_image = cv2.imread("image.png")
gs_image = cv2.cvtColor(gs_image, cv2.COLOR_BGR2GRAY)

noise_image = noise.salt_and_pepper(gs_image)
cv2.imwrite('noise_image.png', noise_image)

avg_image = filters.averaging(noise_image, 1)
cv2.imwrite('avg_image.png', avg_image)


time_start = time()
median_image = filters.median(noise_image, 1)
time1 = time() - time_start
cv2.imwrite('median_image.png', median_image)

time_start = time()
mediancv2_image = noise_image.copy()
cv2.medianBlur(noise_image, 3, mediancv2_image)
time2 = time() - time_start
cv2.imwrite('mediancv2_image.png', mediancv2_image)

print(time1, time2)

