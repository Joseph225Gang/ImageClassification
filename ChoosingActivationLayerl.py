import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

mean = [0.49159187, 0.48234594, 0.44671956]
std = [0.23834434, 0.23486559, 0.25264624]

train_transform = transforms.Compose([
	transforms.Resize(32),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
])

test_transform = transforms.Compose([
	transforms.Resize(32),
	transforms.ToTensor(),
	transforms.Normalize(mean,std)
])

trainset = torchvision.datasets.CIFAR10(root='datasets/cifar10/train',
					train=True,
					download=True,
					transform=train_transform)

testset = torchvision.datasets.CIFAR10(root='datasets/cifar10/train',
					train=True,
					download=True,
					transform=test_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=False, num_workers=0)
class_names = trainset.classes

print(class_names)
img, label = next(iter(trainloader))

in_size = 3

hid1_size = 16
hid2_size = 32

out1_size = 400
out2_size = 10

k_conv_size = 5

class ConvNet(nn.Module):
    
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_size, hid1_size, k_conv_size),
            nn.BatchNorm2d(hid1_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(hid1_size, hid2_size, k_conv_size),
            nn.BatchNorm2d(hid2_size),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2))
        
        self.layer3 = nn.Sequential(
            nn.Linear(hid2_size * k_conv_size * k_conv_size, out1_size),
            nn.ReLU(),
	        nn.Dropout(0.5),
            nn.Linear(out1_size, out2_size))
        
 
    def forward(self, x):
        out = self.layer1(x)      
        out = self.layer2(out)
        
        out = out.reshape(out.size(0), -1)
        print(out.shape)
        
        out = self.layer3(out)
        
        return F.log_softmax(out, dim=1)
    

model = ConvNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
learning_rate = 0.001

criterion = nn.NLLLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_step = len(trainloader)
num_epochs = 10
loss_values = list()

for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(trainloader):

		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if (i+1) % 2000 == 0:
                 loss_values.append(loss.item())

print('Finished Training')
X = (range(1,11))

plt.figure(figsize=(12,10))

plt.plot(X, loss_values)
plt.xlabel('Step')
plt.ylabel('Loss')

with torch.no_grad():

    correct = 0
    total = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the 1000 test images: {}%'.format(100 * correct / total))

sample_img, _= testset[23]
sample_img = np.transpose(sample_img, (1,2,0))
m, M = sample_img.min(), sample_img.max()
sample_img = (1/(abs(m) * M)) * sample_img + 0.5

plt.figure(figsize=(6,6))
plt.imshow(sample_img)
plt.show()

test_img, test_label = testset[23]
test_img = test_img.reshape(-1,3,32,32)

out_predict = model(test_img.to(device))
_,predicted = torch.max(out_predict.data, 1)
print("Actual Label : ", test_label)
print("Predicted Label : ", predicted.item())
print("Class name for {} : {}".format(predicted.item(), class_names[predicted.item()]))