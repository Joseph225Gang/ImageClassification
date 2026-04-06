import pickle
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import os
import tarfile

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
output_path = "datasets/cifar-10-python.tar.gz"

os.makedirs("datasets", exist_ok=True)

urllib.request.urlretrieve(url, output_path)
tar_path = "datasets/cifar-10-python.tar.gz"
extract_path = "datasets"

print("Download complete!")

with tarfile.open(tar_path, "r:gz") as tar:
    tar.extractall(path=extract_path)

print("Extract complete!")

with open('datasets/cifar-10-batches-py/data_batch_1', 'rb') as input_file:
   X = pickle.load(input_file, encoding='latin1')
  
X = X['data']

X = X.reshape(-1, 3, 32, 32)

X = X.transpose(0, 2, 3, 1)

X = X.reshape(-1, 3 * 32 * 32)

plt.imshow(X[6].reshape(32, 32, 3))
plt.show()

X = X - X.mean(axis=0)

X = X / np.std(X, axis=0)

def show(i):
   i = i.reshape((32, 32, 3))
   m, M = i.min(), i.max()

   plt.imshow((i - m) / (M - m))
   plt.show()


show(X[6])

X_subset = X[:1000]

cov = np.cov(X_subset, rowvar=True)

U, S, V = np.linalg.svd(cov)

print(U.shape)
print(S.shape)
print(V.shape)

epsilon = 1e-5
zca_matrix = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))

zca = np.dot(zca_matrix, X_subset)

show(zca[6])

import torch
import torchvision
import torchvision.transforms as transforms

dir(transforms)

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
])

dataset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/train', download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(dataset, 
                                         batch_size=16,
                                         shuffle=True, 
                                         num_workers=0)

images_batch, labels_batch = next(iter(dataloader))

img=torchvision.utils.make_grid(images_batch)
img = np.transpose(img, (1,2,0))

plt.figure(figsize = (16, 12))

plt.imshow(img)
plt.axis('off')
plt.show()

trainset = torchvision.datasets.CIFAR10(root='./datasets/cifar10/train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)

images_batch, labels_batch = next(iter(trainloader))
img = torchvision.utils.make_grid(images_batch)

img = np.transpose(img, (1, 2, 0))

m, M = img.min(), img.max()

img = (1/(abs(m) * M)) * img + 0.5

plt.figure(figsize = (16, 12))
plt.imshow(img)
plt.show()