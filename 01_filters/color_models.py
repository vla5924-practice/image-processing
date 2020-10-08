import numpy
import cv2


def clamp(value, minimal, maximum):
    if value < minimal:
        return minimal
    if value > maximum:
        return maximum
    return value


def calculate_h(r, g, b, c_min=None, c_max=None):
    h = 0
    c_min = min(r, g, b) if c_min is None else c_min
    c_max = max(r, g, b) if c_max is None else c_max
    delta = c_max - c_min
    if delta == 0:
        h = 0
    elif c_max == r:
        h = (60 * (g - b) / delta) % 6
    elif c_max == g:
        h = 60 * ((b - r) / delta + 2)
    elif c_max == b:
        h = 60 * ((r - g) / delta + 4)
    return clamp(h, 0, 360)


def calculate_s(r, g, b, c_min=None, c_max=None):
    c_min = min(r, g, b) if c_min is None else c_min
    c_max = max(r, g, b) if c_max is None else c_max
    delta = c_max - c_min
    s = delta / c_max if c_max != 0 else 0
    return clamp(s, 0, 1)


def bgr_to_hsv(image: numpy.ndarray):
    width, height, dim = image.shape
    if dim != 3:
        raise ValueError("Image must have 3 color channels")
    r = numpy.zeros((width, height), "float32")
    g = numpy.zeros((width, height), "float32")
    b = numpy.zeros((width, height), "float32")
    c_max = numpy.zeros((width, height), "float32")
    c_min = numpy.zeros((width, height), "float32")
    for x in range(width):
        for y in range(height):
            r[x, y] = image[x, y, 2] / 255
            g[x, y] = image[x, y, 1] / 255
            b[x, y] = image[x, y, 0] / 255
            c_max[x, y] = max(r[x, y], g[x, y], b[x, y])
            c_min[x, y] = min(r[x, y], g[x, y], b[x, y])
    hsv_image = numpy.zeros((width, height, 3), "float32")
    for x in range(width):
        for y in range(height):
            hsv_image[x, y, 0] = calculate_h(r[x, y], g[x, y], b[x, y], c_min[x, y], c_max[x, y])
            hsv_image[x, y, 1] = calculate_s(r[x, y], g[x, y], b[x, y], c_min[x, y], c_max[x, y])
            hsv_image[x, y, 2] = clamp(c_max[x, y], 0, 1)
    return hsv_image


def hsv_to_bgr(image: numpy.ndarray):
    width, height, dim = image.shape
    if dim != 3:
        raise ValueError("Image must have 3 colors channel")
    r = numpy.zeros((width, height), "float32")
    g = numpy.zeros((width, height), "float32")
    b = numpy.zeros((width, height), "float32")
    m = numpy.zeros((width, height), "float32")
    for x in range(width):
        for y in range(height):
            c = image[x, y, 2] * image[x, y, 1]
            d = c * (1 - abs((image[x, y, 0] / 60) % 2 - 1))
            m[x, y] = image[x, y, 2] - c
            h = image[x, y, 0]
            if 0 <= h < 60:
                r[x, y] = c
                g[x, y] = d
                b[x, y] = 0
            elif 60 <= h < 120:
                r[x, y] = d
                g[x, y] = c
                b[x, y] = 0
            elif 120 <= h < 180:
                r[x, y] = 0
                g[x, y] = c
                b[x, y] = d
            elif 180 <= h < 240:
                r[x, y] = 0
                g[x, y] = d
                b[x, y] = c
            elif 240 <= h < 300:
                r[x, y] = d
                g[x, y] = 0
                b[x, y] = c
            elif 300 <= h < 360:
                r[x, y] = c
                g[x, y] = 0
                b[x, y] = d
    bgr_image = numpy.zeros((width, height, 3), "float32")
    for x in range(width):
        for y in range(height):
            bgr_image[x, y, 0] = clamp((b[x, y] + m[x, y]) * 255, 0, 255)
            bgr_image[x, y, 1] = clamp((g[x, y] + m[x, y]) * 255, 0, 255)
            bgr_image[x, y, 2] = clamp((r[x, y] + m[x, y]) * 255, 0, 255)
    return bgr_image


def bgr_brighten(image: numpy.ndarray, bright_coef):
    width, height, dim = image.shape
    if dim != 3:
        raise ValueError("Image must have 3 colors channel")
    bgr_image = numpy.zeros((width, height, 3), "float32")
    for x in range(width):
        for y in range(height):
            bgr_image[x, y, 0] = clamp(image[x, y, 0] * bright_coef, 0, 255)
            bgr_image[x, y, 1] = clamp(image[x, y, 1] * bright_coef, 0, 255)
            bgr_image[x, y, 2] = clamp(image[x, y, 2] * bright_coef, 0, 255)
    return bgr_image


def hsv_brighten(image: numpy.ndarray, bright_coef):
    width, height, dim = image.shape
    if dim != 3:
        raise ValueError("Image must have 3 colors channel")
    hsv_image = image.copy()
    for x in range(width):
        for y in range(height):
            hsv_image[x, y, 2] = clamp(hsv_image[x, y, 2] * bright_coef, 0, 1)
    return hsv_image
