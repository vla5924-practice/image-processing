import cv2
import numpy
from math import atan, pi, sqrt


def follow_edges(x: int, y: int, magnitude: numpy.ndarray, t_upper: int, t_lower: int, edges: numpy.ndarray):
    edges[y, x] = 255
    rows, cols = magnitude.shape
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i != 0 and j != 0 and x + i >= 0 and y + j >= 0 and x + i < cols and y + j < rows:
                if magnitude[y + j, x + i] > t_lower and edges[y + j, x + i] != 255:
                    follow_edges(x + i, y + j, magnitude, t_upper, t_lower, edges)

def detect_edge(magnitude: numpy.ndarray, t_upper: int, t_lower: int, edges: numpy.ndarray):
    rows, cols = magnitude.shape
    edges = numpy.zeros(magnitude.shape, "float32")
    for x in range(cols):
        for y in range(rows):
            if magnitude[y, x] >= t_upper:
                follow_edges(x, y, magnitude, t_upper, t_lower, edges)

def non_maximum_suppression(magnitude: numpy.ndarray, direction: numpy.ndarray):
    check = numpy.zeros(magnitude.shape, "ubyte")
    rows, cols = magnitude.shape
    for x in range(cols):
        for y in range(rows):
            current = atan(direction[x, y]) * (180 / pi)
            while current < 0:
                current += 180
            direction[x, y] = current
            if 22.5 < current <= 67.5:
                if y > 0 and x > 0 and magnitude[y, x] <= magnitude[y - 1, x - 1]:
                    magnitude[y, x] = 0
                if y < rows - 1 and x < cols - 1 and magnitude[x, y] <= magnitude[y - 1, x - 1]:
                    magnitude[y, x] = 0
            elif 67.5 < current <= 112.5:
                if y > 0 and magnitude[y, x] <= magnitude[y-1, x]:
                    magnitude[y, x] = 0
                if y < rows - 1 and magnitude[y, x] <= magnitude[y + 1, x]:
                    magnitude[y, x] = 0
            elif 112.5 < current <= 157.5:
                if y > 0 and x < cols - 1 and magnitude[y, x] <= magnitude[y - 1, x + 1]:
                    magnitude[y, x] = 0
                if y < rows - 1 and x > 0 and magnitude[y, x] <= magnitude[y + 1, x - 1]:
                    magnitude[y, x] = 0
            else:
                if x > 0 and magnitude[y, x] <= magnitude[y, x - 1]:
                    magnitude[y, x] = 0
                if x < cols - 1 and magnitude[y, x] <= magnitude[y, x + 1]:
                    magnitude[y, x] = 0

def do_edge_detection(gs_image: numpy.ndarray, edges: numpy.ndarray, t_upper: int, t_lower: int)-> numpy.ndarray:
    image = gs_image.copy()
    image = cv2.GaussianBlur(image, (3, 3), 1.5)
    mag_x = cv2.Sobel(image, "float32", 1, 0, ksize=3)
    mag_y = cv2.Sobel(image, "float32", 0, 1, ksize=3)
    slopes = mag_y / mag_x
    sum = sqrt(mag_x * mag_x + mag_y * mag_y)
    magnitude = sum.copy()
    non_maximum_suppression(magnitude, slopes)
    detect_edge(magnitude, t_upper, t_lower, edges)
    return edges
