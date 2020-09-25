import numpy
import cv2


def bgr_to_hsv(image: numpy.ndarray):
    width, height, _ = image.shape
    r = numpy.zeros((width, height), "float")
    g = numpy.zeros((width, height), "float")
    b = numpy.zeros((width, height), "float")
    c_max = numpy.zeros((width, height), "float")
    c_min = numpy.zeros((width, height), "float")
    delta = numpy.zeros((width, height), "float")
    for x in range(width):
        for y in range(height):
            r[x, y] = image[x, y, 2] / 255
            g[x, y] = image[x, y, 1] / 255
            b[x, y] = image[x, y, 0] / 255
            c_max[x, y] = max(r[x, y], g[x, y], b[x, y])
            c_min[x, y] = min(r[x, y], g[x, y], b[x, y])
            delta[x, y] = c_max[x, y] - c_min[x, y]
    h = numpy.zeros((width, height), "float")
    s = numpy.zeros((width, height), "float")
    v = numpy.zeros((width, height), "float")
    for x in range(width):
        for y in range(height):
            if delta[x, y] == 0:
                h[x, y] = 0
            elif c_max[x, y] == r[x, y]:
                h[x, y] = (60 * (g[x, y] - b[x, y]) / delta[x, y]) % 6
            elif c_max[x, y] == g[x, y]:
                h[x, y] = 60 * ((b[x, y] - r[x, y]) / delta[x, y] + 2)
            elif c_max[x, y] == b[x, y]:
                h[x, y] = 60 * ((r[x, y] - g[x, y]) / delta[x, y] + 4)
            if c_max[x, y] == 0:
                s[x, y] = 0
            else:
                s[x, y] = delta[x, y] / c_max[x, y]
            v[x, y] = c_max[x, y]
    hsv_image = numpy.zeros((width, height, 3), "float")
    for x in range(width):
        for y in range(height):
            hsv_image[x, y, 0] = h[x, y]
            hsv_image[x, y, 1] = s[x, y]
            hsv_image[x, y, 2] = v[x, y]
    return hsv_image


def test(filename):
    image = cv2.imread(filename)
    result = numpy.float32(bgr_to_hsv(image))
    bgr = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    cv2.imshow("Image", bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
