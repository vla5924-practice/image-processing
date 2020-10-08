import numpy
import cv2


def mse(bgr_image1, bgr_image2):
    if bgr_image2.shape != bgr_image1.shape:
        raise ValueError("Images have different sizes")
    width, height, dim = bgr_image1.shape
    sum_r = 0
    sum_g = 0
    sum_b = 0
    for x in range(width):
        for y in range(height):
            sum_b += (bgr_image1[x, y, 0] - bgr_image2[x, y, 0]) ** 2
            sum_g += (bgr_image1[x, y, 1] - bgr_image2[x, y, 1]) ** 2
            sum_r += (bgr_image1[x, y, 2] - bgr_image2[x, y, 2]) ** 2
    sum_avg = (sum_b + sum_g + sum_r) / 3
    return sum_avg / (width * height)
