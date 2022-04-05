# Import library modules
import cv2
import numpy as np

img_color = cv2.imread("wheel.png", cv2.IMREAD_COLOR)
cv2.imshow(img_color)

print ('\n')
img_grayscale = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imshow(img_grayscale)

def display_rotated_images(img_in):
    degree_array = np.array([15, 30, 45, 60, 75, 90])

    for degree in degree_array:

    rotation = (-degree) * np.pi / 180.0

    row,col = img_in.shape

    rotated_image = np.zeros((row,col))

    mid_row = int((row)/2)
    mid_col = int((col)/2)

    for r in range(row):
        for c in range(col):
            x = (r-mid_row)*np.cos(rotation) -(c-mid_col)*np.sin(rotation)
            y = (r-mid_row)*np.sin(rotation) + (c-mid_col)*np.cos(rotation)

            x += mid_row
            y += mid_col

            x = round(x)
            y = round(y)

            if (x >= 0 and y >= 0 and x < row and y <  col):
                rotated_image[r][c] = img_in[x][y]

    cv2_imshow(rotated_image)
    print('\n')

    return None

# Usage
cv2.imshow(img_grayscale)
print('\n')

display_rotated_images(img_grayscale)
