import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms

positive_test = os.listdir("Test/Positive")
print("Number of Positive Images:", len(positive_test), "loaded from directory Test")

negative_test = os.listdir("Test/Negative")
print("Number of Negative Images:", len(negative_test), "loaded from directory Test")


########################################################################################################################
# NOT WORKING
def neural_network():
    # Hyperparameters for our network
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10

    # Build a feed-forward network
    from collections import OrderedDict
    model = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('output', nn.Linear(hidden_sizes[1], output_size)),
        ('softmax', nn.Softmax(dim=1))]))

    print(model)
    return model


########################################################################################################################


# class Network(nn.Module): means we are inheriting from nn.Module.
# This is combined with super().__init__() and creates a class that tracks the architecture
# and provides a lot of useful methods and attributes.
# The name of the class can be anything.
class Network(nn.Module):
    def __init__(self):
        super().__init__()

        # A module is created for linear transformation with 784 inputs and 256 outputs and assigns it to self.hidden.
        self.hidden = nn.Linear(784, 256)

        # Another linear transformation is created with 256 inputs and 10 outputs.
        self.output = nn.Linear(256, 10)

        # The sigmoid activation and softmax output operations are defined here.
        # dim=1 calculates softmax across the columns.
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        # PyTorch networks created with nn.Module must have a forward method defined.
        # It takes in a tensor x and passes it through the operations defined in the __init__ method.
        def forward(self, x):
            x = self.hidden(x)
            x = self.sigmoid(x)
            x = self.output(x)
            x = self.softmax(x)

            return x


network_model = Network()

print(network_model)

########################################################################################################################


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), ])

positive_train = os.listdir("Train/Positive")
print("Number of Positive Images:", len(positive_train), "loaded from directory Train")

negative_train = os.listdir("Train/Negative")
print("Number of Negative Images:", len(negative_train), "loaded from directory Train")

trainloader_neg = torch.utils.data.DataLoader(negative_train, batch_size=64, shuffle=True)

trainloader_pos = torch.utils.data.DataLoader(positive_train, batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU(),
                      nn.Linear(128, 64),
                      nn.ReLU(),
                      nn.Linear(64, 10),
                      nn.LogSoftmax(dim=1))

# Define the loss
criterion = nn.NLLLoss()

# Optimizers require the parameters to optimize and a learning rate
optimizer = optim.SGD(model.parameters(), lr=0.003)
epochs = 5
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader_neg:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)

        # Training pass
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        print(f"Training loss: {running_loss / len(trainloader_neg)}")

