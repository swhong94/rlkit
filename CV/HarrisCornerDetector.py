''' Mount Google Drive '''
from google.colab import drive
drive.mount('/content/drive')

''' Import libraries '''
import cv2
import numpy as np
from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt

''' Load image using OpenCV '''
# Upload the LMS image to Google Drive and point to its location.
# Note OpenCV reads image as BGR.
img_bgr = cv2.imread("/content/drive/MyDrive/CV/checkerboards.png")
# Normalize image to between 0 and 1.
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(float) / 255.0

# Show output
plt.figure(figsize=(10, 10))
plt.imshow(img, cmap='gray')
plt.show()

# Perform Sobel filtering along the x-axis.
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
Ix = cv2.filter2D(img, -1, sobel_x)
plt.figure(figsize=(10, 10))
plt.imshow(Ix, cmap='gray')
plt.show()

# Perform Sobel filtering along the y-axis.
''' DO IT YOURSELF for sobel_y and Iy '''
soble_y = np.array([[-1, -2, -1],
                    [ 0,  0,  0],
                    [ 1,  2,  1]])
Iy=cv2.filter2D(img,-1,soble_y)
plt.figure(figsize=(10, 10))
plt.imshow(Iy, cmap='gray')
plt.show()

''' Approximate local error surface '''
window_size = 3
offset = int(np.floor(window_size/2))

det = np.zeros(img.shape)
trace = np.zeros(img.shape)

# For each pixel in image
for y in range(offset, img.shape[0]-offset):
  for x in range(offset, img.shape[1]-offset):

    # Build ROI window around the current pixel
    # Note numpy uses height-by-width convention (row x column)
    window_x = Ix[y-offset:y+offset+1, x-offset:x+offset+1]
    window_y = Iy[y-offset:y+offset+1, x-offset:x+offset+1]

    # Estimate elements of matrix M.
    Sxx = np.sum(window_x * window_x)
    Syy = np.sum(window_y * window_y)
    Sxy = np.sum(window_x * window_y)

    # Compute determinant of M and trace of M.
    # Note numpy uses height-by-width convention (row x column)
    trace[y,x] = Sxx + Syy
    det[y, x] = Sxx * Syy - Sxy * Sxy

    # Set hyperparameters
alpha = 0.05
beta = 0.1

# Compute response map
R = det - alpha * (trace ** 2)


# Use thresholding to discard responses with low amplitude
R[R < beta * np.max(R)] = 0

# Smooth the response map using Gaussian filter
gaussian_2d = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]], dtype=np.float32)
gaussian_2d /= np.sum(gaussian_2d)  # normalize

R = cv2.filter2D(R, -1, gaussian_2d)

# Show the response map
plt.figure(figsize=(10, 10))
plt.imshow(R, cmap='gray')
plt.show()

# Set NMS window size
window_size = 3
offset = int(np.floor(window_size/2))

output_img = np.zeros(img.shape)

# For each pixel, perform non-maximal suppression around it in 3x3 block.
for y in range(offset, img.shape[0]-offset):
  for x in range(offset, img.shape[1]-offset):

    if R[y,x] == 0.0:
      # If the response map value is 0, then we can skip
      continue

    center_value = R[y,x]
    # Get max_value of the 3x3 block
    local_block = R[y-offset:y+offset+1, x-offset:x+offset+1]
    max_value = np.max(local_block)

    # If the center value is not the same as the maximum value of the 3x3 block,
    # then it's not maximum, so suppress.
    # Otherwise, let the pixel survive.
    if center_value != max_value:
      output_img[y,x] = 0
    else:
      output_img[y,x] = center_value

''' Extract feature points and draw on the image '''
y, x = np.where(output_img > 1.0)

output_vis = img_bgr.copy()

for i in range(0, len(x)):
    cv2.circle(output_vis, (x[i], y[i]), 3, (0, 0, 255), -1)

cv2_imshow(output_vis)