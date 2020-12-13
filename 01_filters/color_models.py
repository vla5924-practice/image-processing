import numpy


def clamp(value, minimal, maximal):
    if value < minimal:
        return minimal
    if value > maximal:
        return maximal
    return value


def bgr_to_hsv(image: numpy.ndarray) -> numpy.ndarray:
    width, height, dim = image.shape
    if dim != 3:
        raise ValueError("Image must have 3 color channels")
    hsv_image = numpy.zeros((width, height, 3), "int32")
    coef = 360 / 255
    for x in range(width):
        for y in range(height):
            r = image[x, y, 2]
            g = image[x, y, 1]
            b = image[x, y, 0]
            c_max = max(r, g, b)
            c_min = min(r, g, b)
            delta = c_max - c_min
            h = 0
            s = 0
            v = c_max
            if v != 0:
                s = 255 * delta / v
                if s != 0:
                    if c_max == r:
                        h = 43 * (g - b) / delta + 0
                    elif c_max == g:
                        h = 43 * (b - r) / delta + 85
                    else:
                        safe_r_minus_g = r - g if r > g else 0
                        h = 43 * safe_r_minus_g / delta + 171
            hsv_image[x, y, 0] = clamp(h / coef, 0, 255)
            hsv_image[x, y, 1] = clamp(s, 0, 255)
            hsv_image[x, y, 2] = clamp(v, 0, 255)
    return hsv_image


def hsv_to_bgr(image: numpy.ndarray) -> numpy.ndarray:
    """
    Formula from Wikipedia https://en.wikipedia.org/wiki/HSV_color_space
    """
    width, height, dim = image.shape
    if dim != 3:
        raise ValueError("Image must have 3 color channels")
    bgr_image = numpy.zeros((width, height, 3), "int32")
    coef = 360 / 255
    for x in range(width):
        for y in range(height):
            h = image[x, y, 0] * coef
            s = image[x, y, 1] / 2.55
            v = image[x, y, 2] / 2.55
            h_i = numpy.floor(h / 60) % 6
            v_min = (100 - s) * v / 100
            a = (v - v_min) * ((h % 60) / 60)
            v_inc = v_min + a
            v_dec = v - a
            r = 0
            g = 0
            b = 0
            if h_i == 0:
                r = v
                g = v_inc
                b = v_min
            elif h_i == 1:
                r = v_dec
                g = v
                b = v_min
            elif h_i == 2:
                r = v_min
                g = v
                b = v_inc
            elif h_i == 3:
                r = v_min
                g = v_dec
                b = v
            elif h_i == 4:
                r = v_min
                g = v_inc
                b = v
            elif h_i == 5:
                r = v
                g = v_min
                b = v_dec
            bgr_image[x, y, 0] = clamp(b * 2.55, 0, 255)
            bgr_image[x, y, 1] = clamp(g * 2.55, 0, 255)
            bgr_image[x, y, 2] = clamp(r * 2.55, 0, 255)
    return bgr_image


def bgr_brighten(image: numpy.ndarray, bright_coef) -> numpy.ndarray:
    width, height, dim = image.shape
    if dim != 3:
        raise ValueError("Image must have 3 colors channel")
    bgr_image = numpy.zeros((width, height, 3), "float32")
    for x in range(width):
        for y in range(height):
            bgr_image[x, y, 0] = clamp(image[x, y, 0] * bright_coef, 0, 255)
            bgr_image[x, y, 1] = clamp(image[x, y, 1] * bright_coef, 0, 255)
            bgr_image[x, y, 2] = clamp(image[x, y, 2] * bright_coef, 0, 255)
    return bgr_image.astype("int32")


def hsv_brighten(image: numpy.ndarray, bright_coef) -> numpy.ndarray:
    width, height, dim = image.shape
    if dim != 3:
        raise ValueError("Image must have 3 colors channel")
    hsv_image = image.copy()
    for x in range(width):
        for y in range(height):
            hsv_image[x, y, 2] = clamp(image[x, y, 2] * bright_coef, 0, 255)
    return hsv_image
