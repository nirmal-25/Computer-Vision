# Import library modules
import cv2
import numpy as np

img_color = cv2.imread("wheel.png", cv2.IMREAD_COLOR)
cv2.imshow(img_color)

print ('\n')
img_grayscale = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
cv2.imshow(img_grayscale)

def lin_filter(img_in, kernel):
    img_in = img_in.astype(float)
    img_row, img_col = img_in.shape
    img_out = np.zeros((img_row, img_col), dtype=np.float32)

    k_size = kernel.shape[0]
    k = int((k_size - 1) / 2)

    for i in range(k, img_row - k):
        for j in range(k, img_col - k):
            sum = 0
            for u in range(-k, k + 1, 1):
                for v in range(-k, k + 1, 1):
                    sum += img_in[i + u, j + v] * kernel[u + k, v + k]

            if sum > 0.0:
                if sum > 255.0:
                    sum = 255.0
                img_out[i, j] = sum
            if sum < 0.0:
                sum = np.abs(sum)
                if sum > 255.0:
                    sum = 255.0
                img_out[i, j] = sum

    return img_out


# Here is an example smoothing filter,
# approximating a 2D Gaussian function with sigma = 1.
kernel = np.array([
          [1, 4, 7, 4, 1],
          [4, 16, 26, 16, 4],
          [7, 26, 41, 26, 7],
          [4, 16, 26, 16, 4],
          [1, 4, 7, 4, 1,]], dtype=np.float32) / 273.0

# Apply the smoothing filter
img_result = lin_filter(img_grayscale, kernel)

cv2.imshow(img_grayscale)
cv2.imshow(img_result)

# Sobel Filters
sobel_horizontal = 0.25 * np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
sobel_vertical = 0.25 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)

img_result_sobel1 = lin_filter(img_grayscale, sobel_horizontal)
cv2.imshow(img_result_sobel1)
print('\n')

img_result_sobel2 = lin_filter(img_grayscale, sobel_vertical)
cv2.imshow(img_result_sobel2)
print('\n')

# Merging the two outputs using a pixelwise maximum operation
row, col = img_result_sobel1.shape
img_merged = np.zeros((row, col), dtype=np.float32)

for i in range(row):
    for j in range(col):
        img_merged[i,j] = max(img_result_sobel1[i,j], img_result_sobel2[i,j])

cv2.imshow(img_merged)
