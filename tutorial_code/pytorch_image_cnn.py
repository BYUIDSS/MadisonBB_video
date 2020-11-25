# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
import time
# %%


# define our transform pipeline to transform our data into tensors and normalize the images to values between -1 and 1
transform = transforms.Compose(
    [transforms.ToTensor(), 
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]) # we put 0.5 3 times because we are normalizing 3 different channels


# Download our data we will use CIFAR10 images to train on
trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform) 

# we want a batch size of 4 images, randomly shuffled
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True) # get our training data in a specific way

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform)

# we want a batch size of 4 images
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False) # get our testset in a specific way

# names of our classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
# show a few of our training images

# make a helper function that will show an image after it un-normalizes it
def imshow(img):
    img = img / 2 + 0.5  # un-normalize the image
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images if we want to show them
dataiter = iter(trainloader)
images, labels = dataiter.next()

# %%

# define our convolutional neural network class
class Net(nn.Module):
    # defining the needed layers in our CNN
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.fc1 = nn.Linear(in_features=256 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    # defining how data (x) will move forward throught the network
    # in here we also include our activation functions
    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = x.view(-1, 256 * 5 * 5) # this flattens our last convolutional layer so we can put it into fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# make an instance of our neural network
net = Net()

# define a loss function and optimizer
# we will use catagorical cross entropy because this is a classification problem
# we will use stocastic gradient descent with momentum

cross_entropy = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)



# now we define a function to train the network
def cpu_train(trainloader):
    running_loss = 0.0 # keep tabs on our loss
    # run over the training data twice
    for epoch in range(2):

        for i, data in enumerate(trainloader, 0):
             # get the inputs; data is a list of [inputs, labels] and put it on our gpu
             ## inputs, labels = data[0].to(gpu), data[1].to(gpu)
             inputs, labels = data

             # zero the paramater gradients 
             optimizer.zero_grad()

             # forward pass through the network
             outputs = net(inputs)

             # calulate the loss on this forward pass
             loss = cross_entropy(outputs, labels)

             # backpropogate the loss
             loss.backward()

             # take a step down the gradient
             optimizer.step()

             # print our statistics and progress
             running_loss += loss.item() # the item method extracts the loss's value as a python float

             if i % 2000 == 0:
                  # every 2000 mini-batches print the epoch number training number and our loss per batch
                  print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                  running_loss = 0.0

    print('finished training')


# set up to train the network on gpu
gpu = torch.device("cuda:0")


def gpu_train(trainloader):
    running_loss = 0.0 # keep tabs on our loss
    # run over the training data twice
    for epoch in range(2):

        for i, data in enumerate(trainloader, 0):
             # get the inputs; data is a list of [inputs, labels] and put it on our gpu
            inputs, labels = data[0].to(gpu), data[1].to(gpu)
             

             # zero the paramater gradients 
            optimizer.zero_grad()

             # forward pass through the network
            outputs = net(inputs)

             # calulate the loss on this forward pass
            loss = cross_entropy(outputs, labels)

             # backpropogate the loss
            loss.backward()

             # take a step down the gradient
            optimizer.step()

             # print our statistics and progress
            running_loss += loss.item() # the item method extracts the loss's value as a python float

            if i % 2000 == 0:
                  # every 2000 mini-batches print the epoch number training number and our loss per batch
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('finished training')


'''
# time training on cpu
start_time = time.time()
cpu_train(trainloader)
end_time = time.time()
seconds = end_time - start_time

# see how fast our model trains on the cpu
print('Training on CPU took: %.3f seconds' % (seconds))

'''
net.to(gpu) # put our neural network on our gpu
# time training on the gpu
start_time = time.time()
gpu_train(trainloader)
end_time = time.time()
seconds = end_time - start_time 

# see how fast our model trains on the gpu
print('Training on GPU took: %.3f seconds' % (seconds))

# save our model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

