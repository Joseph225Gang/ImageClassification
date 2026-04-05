import skimage
from skimage import data
from skimage import transform

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
bird=mpimg.imread("datasets/images/bird.jpeg")
plt.title("Original Image")
plt.imshow(bird)
bird.shape
bird[200: 250, 200:250]
bord_reshape = bird.reshape(bird.shape[0], -1)
bird.reshape
plt.figure (figsize = (6, 6))
plt.title("Reshaped Image")
plt.imshow(bord_reshape)
bird_resized = skimage.transform.resize(bird, (500, 500))
bird_resized.shape

plt.figure(figsize = (6,6))
plt.title("Resized Image")
plt.imshow(bird_resized)

aspect_ratio_original = bird.shape[1] / float(bird.shape[0])
aspect_ratio_resized = bird_resized.shape[1] / float(bird_resized.shape[0])

print("Original aspect ratio: ", aspect_ratio_original)
print("Resized aspect ratio: ", aspect_ratio_resized)

# 假設你想要固定高度為 500，寬度根據原始比例自動計算
target_h = 500
target_w = int(target_h * aspect_ratio_original)

bird_rescaled = transform.resize(bird_resized, (target_h, target_w))
bird_rescaled.shape
plt.figure(figsize=(6,6))
plt.title("Rescaled Image")
plt.imshow(bird_rescaled)
aspect_ratio_rescaled = bird_rescaled.shape[1] / float(bird_rescaled.shape[0])
print("Rescaled aspect ratio: ", aspect_ratio_rescaled)

bird_BGR = bird[:, :, (2,1,0)]

plt.figure(figsize=(6,6))
plt.title("BGR Image")
plt.imshow(bird_BGR)

bird_gray = skimage.color.rgb2gray(bird)

plt.figure (figsize = (6,6))
plt.title("Gray Image")
plt.imshow(bird_gray, cmap = 'gray')
plt.show()
bird_gray.shape