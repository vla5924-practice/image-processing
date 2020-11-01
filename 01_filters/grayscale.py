import numpy


def luminosity(bgr_image: numpy.ndarray) -> numpy.ndarray:
    width, height, dim = bgr_image.shape
    if dim != 3:
        raise ValueError("Image must have 3 color channels")
    gs_image = numpy.zeros((width, height), "ubyte")
    for x in range(width):
        for y in range(height):
            gs_image[x, y] = 0.21 * bgr_image[x, y, 2] + 0.72 * bgr_image[x, y, 1] + 0.07 * bgr_image[x, y, 0]
    return gs_image
