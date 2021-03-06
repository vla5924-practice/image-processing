import numpy


def clamp(value, min, max):
    if value < min:
        return min
    if value > max:
        return max
    return value


def averaging(gs_image: numpy.ndarray, radius: int) -> numpy.ndarray:
    if gs_image.ndim != 2:
        raise ValueError("Image must have 1 color channel")
    if radius < 1:
        raise ValueError("Radius must be positive")
    width, height = gs_image.shape
    result_image = numpy.zeros((width, height), "ubyte")
    near_count = (2 * radius + 1) ** 2
    for x in range(width):
        for y in range(height):
            # Collecting near values sum
            near_sum = 0
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    near_x = clamp(x + j, 0, width - 1)
                    near_y = clamp(y + i, 0, height - 1)
                    near_sum += gs_image[near_x, near_y]
            # Counting avg value
            avg = near_sum / near_count
            # Set avg value
            result_image[x, y] = avg
    return result_image


def median(gs_image: numpy.ndarray, radius: int) -> numpy.ndarray:
    if gs_image.ndim != 2:
        raise ValueError("Image must have 1 color channel")
    if radius < 1:
        raise ValueError("Radius must be positive")
    width, height = gs_image.shape
    result_image = numpy.zeros((width, height), "ubyte")
    near_count = (2 * radius + 1) ** 2
    near = numpy.zeros(near_count, "ubyte")
    for x in range(width):
        for y in range(height):
            # Collecting near values array
            k = 0
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    near_x = clamp(x + j, 0, width - 1)
                    near_y = clamp(y + i, 0, height - 1)
                    near[k] = gs_image[near_x, near_y]
                    k += 1
            # Near values array is now ready, we can sort it
            near.sort()
            # Set median value
            result_image[x, y] = near[int(near_count / 2)]
    return result_image
