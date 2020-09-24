import cv2
import numpy

# Load image
image = cv2.imread('image.png')

# Convert to something about grayscale
gs_image = numpy.dot(image, [0.5, 0.2, 0.3])
gs_image = gs_image.astype('ubyte')

# Iterating over pixels
for x in range(len(image)):
    for y in range(len(image[x])):
        image[x, y] = [255, 0, 0]

# Save image
cv2.imwrite('new_image.png', image)

# Display image
cv2.imshow("Image", gs_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
