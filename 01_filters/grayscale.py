import numpy
import cv2


def luminosity(bgr_image: numpy.ndarray):
    width, height, _ = bgr_image.shape
    gs_image = numpy.zeros((width, height), "ubyte")
    for x in range(width):
        for y in range(height):
            gs_image[x, y] = 0.21 * bgr_image[x, y, 2] + 0.72 * bgr_image[x, y, 1] + 0.07 * bgr_image[x, y, 0]
    return gs_image


def test(filename):
    image = cv2.imread(filename)
    gs_image = luminosity(image)
    cv2.imshow("Image", gs_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
