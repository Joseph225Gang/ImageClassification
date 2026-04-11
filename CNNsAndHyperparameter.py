import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torchvision

base_dir = os.path.dirname(__file__)
file_train_path = os.path.join(base_dir, 'datasets', 'mnist-in-csv', 'mnist_train.csv')
file_test_path = os.path.join(base_dir, 'datasets', 'mnist-in-csv', 'mnist_test.csv')


mnist_train = pd.read_csv(file_train_path)
mnist_test = pd.read_csv(file_test_path)

mnist_train = mnist_train.dropna()
mnist_test = mnist_test.dropna()
random_sel = mnist_train.sample(8)

image_features = random_sel.drop('label', axis=1)
image_batch = torch.Tensor(image_features.to_numpy() / 255.).reshape((-1, 28, 28))

grid = torchvision.utils.make_grid(image_batch.unsqueeze(1), nrow=8)

plt.figure(figsize = (12,12))
plt.imshow(grid.numpy().transpose((1,2,0)))
plt.axis('off')
mnist_train_features = mnist_train.drop('label', axis=1)
mnist_train_target = mnist_train['label']

mnist_test_features = mnist_test.drop('label',axis=1)
mnist_test_target = mnist_test['label']

X_train_tensor = torch.tensor(mnist_train_features.values, dtype=torch.float) 
x_test_tensor = torch.tensor(mnist_test_features.values, dtype=torch.float) 

Y_train_tensor = torch.tensor(mnist_train_target.values, dtype=torch.long) 
y_test_tensor = torch.tensor(mnist_test_target.values, dtype=torch.long) 

X_train_tensor = X_train_tensor.reshape(-1, 1, 28, 28)
x_test_tensor = x_test_tensor.reshape(-1, 1, 28, 28)

import torch.nn as nn
import torch.nn.functional as F

in_size = 1

hid1_size = 8  #16
hid2_size = 16

out_size = 10

k_conv_size = 5

class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_size, hid1_size, k_conv_size),
            nn.BatchNorm2d(hid1_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(hid1_size, hid2_size, k_conv_size),
            nn.BatchNorm2d(hid2_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        
        self.fc = nn.Linear(256, out_size)
        
 
    def forward(self, x):
        out = self.layer1(x)
        print(out.shape)
        
        out = self.layer2(out)
        print(out.shape)
        
        out = out.reshape(out.size(0), -1)
        print(out.shape)
        
        out = self.fc(out)
        print(out.shape)
        
        ## F.log_softmax(out, dim=-1)
        
        return out
    
model = ConvNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

X_train_tensor = X_train_tensor.to(device)
x_test_tensor = x_test_tensor.to(device)

Y_train_tensor = Y_train_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #SGD

num_epochs = 2
loss_values = list()

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True
)

for epoch in range(num_epochs):
    
    epoch_loss = 0.0

    for images, labels in train_loader:   # ⭐ 重點：一定要 DataLoader

        images = images.to(device).float()
        labels = labels.to(device).long()

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    loss_values.append(avg_loss)

    print(f"Epoch {epoch}, loss: {avg_loss:.5f}")
    
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()  #nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


x = range(len(loss_values))

plt.figure(figsize = (8, 8))
plt.plot(x, loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')

model.eval()

from sklearn.metrics import accuracy_score, precision_score, recall_score

with torch.no_grad():

      correct = 0
      total = 0

      outputs = model(x_test_tensor)
      _, predicted = torch.max(outputs.data, 1)
 
      y_test = y_test_tensor.cpu().numpy()
      predicted = predicted.cpu()

      print("Accuracy: ", accuracy_score(predicted, y_test))
      print("Precision ", precision_score(predicted, y_test, average='weighted'))
      print("Recall: ", recall_score(predicted, y_test, average='weighted'))
