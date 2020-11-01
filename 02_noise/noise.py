import numpy


def salt_and_pepper(gs_image: numpy.ndarray) -> numpy.ndarray:
    if gs_image.ndim != 2:
        raise ValueError("Image must have 1 color channel")
    width, height = gs_image.shape
    salt_pepper_noise = numpy.random.uniform(0, 255, (width, height))
    result_image = gs_image.copy()
    for x in range(width):
        for y in range(height):
            if salt_pepper_noise[x, y] < 30:
                result_image[x, y] = 0
            elif salt_pepper_noise[x, y] > 225:
                result_image[x, y] = 255
    return result_image
