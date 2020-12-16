import cv2
import numpy

MASK = -2
WSHD = 0
INIT = -1
INQE = -3

def get_neighbors(pixels, height, width):
    t_pixels = numpy.mgrid[
        max(0, pixel[0] - 1):min(height, pixel[0] + 2),
        max(0, pixel[1] - 1):min(width, pixel[1] + 2)
    ]
    return t_pixels.reshape(2, -1).T

def do_watershed(image):
    current_label = 0
    flag = False
    fifo = deque()

    height, width = image.shape
    total = height * width
    labels = numpy.full((height, width), INIT, "int32")

    reshaped_image = image.reshape(total)
    pixels = numpy.mgrid[0:height, 0:width].reshape(2, -1).T
    neighbours = numpy.array([get_neighbors(height, width, p) for p in pixels])
    if len(neighbours.shape) == 3:
        neighbours = neighbours.reshape(height, width, -1, 2)
    else:
        neighbours = neighbours.reshape(height, width)

    indices = numpy.argsort(reshaped_image)
    sorted_image = reshaped_image[indices]
    sorted_pixels = pixels[indices]

    levels = numpy.linspace(sorted_image[0], sorted_image[-1], 256)
    level_indices = []
    current_level = 0

    for i in xrange(total):
        if sorted_image[i] > levels[current_level]:
            while sorted_image[i] > levels[current_level]:
                current_level += 1
        level_indices.append(i)
    level_indices.append(total)

    start_index = 0
    for stop_index in level_indices:
        for p in sorted_pixels[start_index:stop_index]:
            labels[p[0], p[1]] = MASK
            for q in neighbours[p[0], p[1]]:
                if labels[q[0], q[1]] >= WSHD:
                    labels[p[0], p[1]] = INQE
                    fifo.append(p)
                    break
            while fifo:
            p = fifo.popleft()
            for q in neighbours[p[0], p[1]]:
                lab_p = labels[p[0], p[1]]
                lab_q = labels[q[0], q[1]]
                if lab_q > 0:
                    if lab_p == INQE or (lab_p == WSHD and flag):
                        labels[p[0], p[1]] = lab_q
                    elif lab_p > 0 and lab_p != lab_q:
                        labels[p[0], p[1]] = WSHD
                        flag = False
                elif lab_q == WSHD:
                    if lab_p == INQE:
                        labels[p[0], p[1]] = WSHD
                        flag = True
                elif lab_q == MASK:
                    labels[q[0], q[1]] = INQE
                    fifo.append(q)
        for p in sorted_pixels[start_index:stop_index]:
        if labels[p[0], p[1]] == MASK:
            current_label += 1
            fifo.append(p)
            labels[p[0], p[1]] = current_label
            while fifo:
                q = fifo.popleft()
                for r in neighbours[q[0], q[1]]:
                    if labels[r[0], r[1]] == MASK:
                    fifo.append(r)
                    labels[r[0], r[1]] = current_label

        start_index = stop_index
    return labels
