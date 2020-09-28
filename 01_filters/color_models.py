import numpy
import cv2


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
    return h


def calculate_s(r, g, b, c_min=None, c_max=None):
    c_min = min(r, g, b) if c_min is None else c_min
    c_max = max(r, g, b) if c_max is None else c_max
    delta = c_max - c_min
    s = delta / c_max if c_max != 0 else 0
    return s


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
            hsv_image[x, y, 0] = calculate_h(r[x,y], g[x, y], b[x, y], c_min[x, y], c_max[x, y])
            hsv_image[x, y, 1] = calculate_s(r[x,y], g[x, y], b[x, y], c_min[x, y], c_max[x, y])
            hsv_image[x, y, 2] = c_max[x, y]
    return hsv_image


def test(filename):
    image = cv2.imread(filename)
    bgr = cv2.cvtColor(bgr_to_hsv(image), cv2.COLOR_HSV2BGR)
    cv2.imshow("Image", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test("image.png")
