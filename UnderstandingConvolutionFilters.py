import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("datasets/images/street.jpg").convert('RGB')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tf

transforms = tf.Compose([tf.Resize(400), tf.ToTensor()])
img_tensor = transforms(img)

sharpen_kernel = [[[[0,-1,0]], [[-1,5,-1]],[[0,-1,0]]]]
horizontal_line_kernel = [[[[1,0,-1]],[[0,0,0]],[[-1,0,1]]]]
vertical_line_kernel = [[[[0,1,0]],[[1,-4,1]],[[0,1,0]]]]

conv_filter = torch.Tensor(horizontal_line_kernel)

img_tensor = img_tensor.unsqueeze(0)

conv_tensor = F.conv2d(img_tensor, conv_filter, padding=0)
conv_img = conv_tensor[0, :, :, :]
conv_img = conv_img.numpy().squeeze()
plt.figure(figsize=(20,10))
plt.imshow(conv_img)

pool = nn.MaxPool2d(2,2)
pool_tensor = pool(conv_tensor)
pool_img = pool_tensor[0,:,:,:]
pool_img = pool_img.numpy().squeeze()
plt.figure(figsize=(20,10))
plt.imshow(pool_img)
plt.show()