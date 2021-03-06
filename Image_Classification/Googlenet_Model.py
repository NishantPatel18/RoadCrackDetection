import os
import time

import torch
from torch import nn, optim
from torchvision import datasets, transforms, models


def get_images_from_folder(folder):
    images = os.listdir(folder)
    print("Number of Images: ", len(images), " is loaded from directory", folder)
    return images


def loading_data(data_dir, batch_size):
    train_dir = data_dir + '/Train'
    # valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/Test'

    # Define your transforms for the training, validation, and testing sets
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    TrainSet = datasets.ImageFolder(train_dir, transform=image_transforms)
    TestSet = datasets.ImageFolder(test_dir, transform=image_transforms)
    TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=batch_size, shuffle=True)
    TestLoader = torch.utils.data.DataLoader(TestSet, batch_size=batch_size, shuffle=True)

    print(len(TrainSet), ' train image and ', len(TestSet), ' test images are successfully loaded.')
    print('Train classes are', TrainSet.classes)
    print('Test classes are', TestSet.classes)

    return TrainLoader, TestLoader, TrainSet, TestSet


def make_model():
    # Hyperparameters for our network
    input_size = 256
    hidden_sizes = [128, 64]
    output_size = 2
    model = models.googlenet(pretrained=True)
    # for param in model.parameters():
    #     param.requires_grad = False
    # Build a feed-forward network
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
        ('dropout', nn.Dropout(0.5)),
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(hidden_sizes[1], output_size)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    # To print specific layer, you can print index or name of layer like this
    # print(model[0]) or print(model.fc1)
    print('Classifier is successfully created.')
    return model


def train_model(model, epoch, Loader, device):
    model.to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    batch = 1
    start = time.time()
    for epoch in range(epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(Loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            print('Running loss at epoch ', epoch+1, ', batch ', batch, ' = ', running_loss)
            batch += 1
        batch = 1

        end = time.time()
    print('Finished Training in %0.2f minutes' % ((end - start) / 60))
    return model


def test_model(model, TestLoader, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in TestLoader:

            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
            # Add loss to the validation set's running loss
            val_loss += valloss.item() * inputs.size(0)

            # Since our model outputs a LogSoftmax, find the real
            # percentages by reversing the log function
            output = torch.exp(output)
            # Get the top class of the output
            top_p, top_class = output.topk(1, dim=1)
            # See how many of the classes were correct?
            equals = top_class == labels.view(*top_class.shape)
            # Calculate the mean (get the accuracy for this batch)
            # and add it to the running accuracy for this epoch
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            # print('output data', outputs.data)
            # _, prediction = torch.max(outputs.data, 1)
            # predict_class = []
            # for number in prediction:
            #     if number < 500:
            #         predict_class.append(0)
            #         # number == 1
            #     else:
            #         predict_class.append(1)
            #         # number == 1
            # # print(predict_class)
            # predict_tensor = torch.Tensor(predict_class)
            # # print(predict_tensor)
            # total += labels.size(0)
            # # print('total', total)
            # # print('label size', labels.size(0))
            # # print('prediction', prediction)
            # # print('label data', labels.data)
            # correct += (predict_tensor == labels).sum().item()

        print('Accuracy: %0.3f %%' % (accuracy / len(TestLoader) * 100))

epoch = 5
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
batch_size = 140

TrainLoader, TestLoader, TrainSet, TestSet = loading_data("RoadData", batch_size)

model = make_model()

TrainedModel = train_model(model, epoch, TrainLoader, device)

# test_model(model, Testloader, device)
test_model(TrainedModel, TestLoader, device)