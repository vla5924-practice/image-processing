import numpy
import cv2


def mse(bgr_image1: numpy.ndarray, bgr_image2: numpy.ndarray) -> float:
    if bgr_image1.shape != bgr_image2.shape:
        raise ValueError("Images have different sizes")
    width, height, dim = bgr_image1.shape
    if dim != 3:
        raise ValueError("Images must have 3 color channels")
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
