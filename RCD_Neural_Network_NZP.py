# Import Libraries
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time


def get_all_preds(model, loader):
    counter = 0;
    all_preds = torch.tensor([])

    for batch in loader:
        counter += 1
        print('Batch: ', counter)
        images, labels = batch
        preds = model(images)
        all_preds = torch.cat((all_preds, preds), dim=0)

    return all_preds


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


# Specify transforms using torchvision.transforms as transforms library
transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load in each dataset and apply transformations using the torchvision.datasets as datasets library
train_set = datasets.ImageFolder("Train", transform=transformations)
val_set = datasets.ImageFolder("Test", transform=transformations)

print('The class labels are:', train_set.classes, '\n')

# Put into a Dataloader using torch library
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

# Get pretrained model using torchvision.models as models library
model = models.densenet161(pretrained=True)
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False

# Create new classifier for model using torch.nn as nn library
classifier_input = model.classifier.in_features
num_labels = 2
classifier = nn.Sequential(nn.Linear(classifier_input, 64),
                           nn.ReLU(),
                           nn.Linear(64, 32),
                           nn.ReLU(),
                           nn.Linear(32, num_labels),
                           nn.LogSoftmax(dim=1))
# Replace default classifier with new classifier
model.classifier = classifier

# Find the device available to use using torch library
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Move model to the device specified above
model.to(device)

# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
# Set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.classifier.parameters())

# Training the Model
epochs = 1
start_train = time.time()

for epoch in range(epochs):
    print('Epoch:', epoch + 1)
    train_loss = 0
    val_loss = 0
    accuracy = 0

    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
        # Clear optimizers
        optimizer.zero_grad()
        # Forward pass
        output = model.forward(inputs)
        # Loss
        loss = criterion(output, labels)
        # Calculate gradients (backpropogation)
        loss.backward()
        # Adjust parameters based on gradients
        optimizer.step()
        # Add the loss to the training set's rnning loss
        train_loss += loss.item() * inputs.size(0)

        # Print the progress of our training
        counter += 1
        print("Batch:", counter, "out of", len(train_loader))

    end_train = time.time()
    print('Finished Epoch', epoch + 1, 'Training in %0.2f minutes' % ((end_train - start_train) / 60))

    # Validating the Model

    # Evaluating the model
    start_valid = time.time()
    model.eval()
    counter = 0
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move to device
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

            # Print the progress of our evaluation
            counter += 1
            print("Batch:", counter, "out of", len(val_loader))

    end_valid = time.time()
    print('Finished Epoch', epoch + 1, 'Validating in %0.2f minutes' % ((end_valid - start_valid) / 60))

    # Prediction
    with torch.no_grad():
        prediction_loader = torch.utils.data.DataLoader(train_set, batch_size=1)
        train_preds = get_all_preds(model, prediction_loader)

    print(train_preds.shape)

    preds_correct = get_num_correct(train_preds, train_set.targets)

    print('total correct:', preds_correct)
    print('accuracy:', preds_correct / len(train_set))

    # Get the average loss for the entire epoch
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = val_loss / len(val_loader.dataset)
    # Print out the information
    print('Accuracy: %0.3f %%' % (accuracy / len(val_loader) * 100))
    print('Training Loss: {:.6f} ' '\tValidation Loss: {:.6f}'.format(train_loss, valid_loss), '\n')

print('Total Time is %0.2f minutes' % ((end_valid - start_train) / 60))

