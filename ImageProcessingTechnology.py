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
bird_gray.shape

giraffes = skimage.img_as_float(skimage.io.imread('datasets/images/giraffes.jpg')).astype(np.float32)
plt.figure(figsize = (6,6))
plt.title("Original Image")
plt.imshow(giraffes)

def crop(image, cropx, cropy):
    y, x, c = image.shape

    startx = x//2 - (cropx // 8)
    starty = x//3 - (cropx // 4)

    stopx = startx + cropx
    stopy = starty + 2*cropy

    return image[starty:stopy, startx:stopx]

giraffes_cropped = crop (giraffes, 256, 256)

plt.title("Cropped Image")
plt.imshow(giraffes_cropped)

from skimage.util import random_noise

sigma = 0.155
noisy_giraffes = random_noise(giraffes, var=sigma**2)

plt.figure (figsize = (6,6))
plt.title("Image with added noise")
plt.imshow(noisy_giraffes)

from skimage.restoration import denoise_tv_chambolle, denoise_bilateral, denoise_wavelet, estimate_sigma

sigma_est = estimate_sigma(
    noisy_giraffes,
    channel_axis=-1,   
    average_sigmas=True
)

plt.imshow(
    denoise_tv_chambolle(
        noisy_giraffes,
        weight=0.1,
        channel_axis=-1
    )
)

from skimage.restoration import denoise_bilateral

plt.imshow(
    denoise_bilateral(
        noisy_giraffes,
        sigma_color=0.05,
        sigma_spatial=15,
        channel_axis=-1
    )
)
plt.imshow(
    denoise_wavelet(
        noisy_giraffes,
        channel_axis=-1
    )
)


monkeys = skimage.img_as_float(skimage.io.imread('datasets/images/monkeys.jpeg')).astype(np.float32)

plt.figure(figsize = (6,6))
plt.title("Orininal Image")
plt.imshow(monkeys)

monkeys_flip = np.fliplr(monkeys)

plt.figure(figsize = (6,6))
plt.title("Horizontal Flip")
plt.imshow(monkeys_flip)

mirror = skimage.img_as_float(skimage.io.imread('datasets/images/book-mirrored.jpg')).astype(np.float32)

plt.figure(figsize = (6,6))
plt.title("Orininal Image")
plt.imshow(mirror)

mirror_flip = np.fliplr(mirror)

plt.figure(figsize = (6,6))
plt.title("Horizontal Flip")
plt.imshow(mirror_flip)

plt.show()

