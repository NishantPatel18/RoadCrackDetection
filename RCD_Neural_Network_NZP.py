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
<<<<<<< HEAD
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
=======
>>>>>>> master

# Specify transforms using torchvision.transforms as transforms library
transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load in each dataset and apply transformations using the torchvision.datasets as datasets library
<<<<<<< HEAD
train_set = datasets.ImageFolder("Train", transform=transformations)
val_set = datasets.ImageFolder("Test", transform=transformations)

print('The class labels are:', train_set.classes, '\n')

###################################
print(len(train_set))
print(len(val_set))
###################################

# Put into a Dataloader using torch library
batch_size=1
=======
train_set = datasets.ImageFolder("/content/RoadCrackDetection/AUG_RDDC_Train", transform=transformations)
val_set = datasets.ImageFolder("/content/RoadCrackDetection/AUG_RDDC_Test", transform=transformations)
# train_set = datasets.ImageFolder("/content/RoadCrackDetection/Small_Train", transform=transformations)
# val_set = datasets.ImageFolder("/content/RoadCrackDetection/Small_Test", transform=transformations)

print('The class labels are:', train_set.classes, '\n')

# Put into a Dataloader using torch library
batch_size = 32
>>>>>>> master
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=True)

# Get pretrained model using torchvision.models as models library
model = models.densenet161(pretrained=True)
<<<<<<< HEAD
=======

>>>>>>> master
# Turn off training for their parameters
for param in model.parameters():
    param.requires_grad = False

# Create new classifier for model using torch.nn as nn library
classifier_input = model.classifier.in_features
<<<<<<< HEAD
num_labels = 2
=======
num_labels = 8
>>>>>>> master
classifier = nn.Sequential(nn.Linear(classifier_input, 64),
                           nn.ReLU(),
                           nn.Linear(64, 32),
                           nn.ReLU(),
                           nn.Linear(32, num_labels),
                           nn.LogSoftmax(dim=1))
<<<<<<< HEAD
=======

>>>>>>> master
# Replace default classifier with new classifier
model.classifier = classifier

# Find the device available to use using torch library
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Move model to the device specified above
model.to(device)

<<<<<<< HEAD
###################################
# print(model)
###################################

=======
>>>>>>> master
# Set the error function using torch.nn as nn library
criterion = nn.NLLLoss()
# Set the optimizer function using torch.optim as optim library
optimizer = optim.Adam(model.classifier.parameters())

# Training the Model
<<<<<<< HEAD
epochs = 2
=======
epochs = 10
>>>>>>> master
start_train = time.time()

for epoch in range(epochs):
    print('Epoch:', epoch + 1)
    train_loss = 0
    val_loss = 0
    accuracy = 0

<<<<<<< HEAD
    ###################################
    # y_pred = model(torch.tensor(train_loader))
    # y_pred = model(train_loader)
    # y_pred = model(train_set)
    ###################################

=======
>>>>>>> master
    # Training the model
    model.train()
    counter = 0
    for inputs, labels in train_loader:
        # Move to device
        inputs, labels = inputs.to(device), labels.to(device)
<<<<<<< HEAD

        ###################################
        y_pred = model(torch.tensor(inputs))
        ###################################

=======
>>>>>>> master
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

<<<<<<< HEAD
    # Validating the Model

=======
>>>>>>> master
    # Evaluating the model
    start_valid = time.time()
    model.eval()
    counter = 0
<<<<<<< HEAD
=======

    total_classes = 8
    output = torch.randn(batch_size, total_classes)  # refer to output after softmax
    target = torch.randint(0, total_classes, (batch_size,))  # labels
    confusion_matrix = torch.zeros(total_classes, total_classes)

>>>>>>> master
    # Tell torch not to calculate gradients
    with torch.no_grad():
        for inputs, labels in val_loader:
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
<<<<<<< HEAD

            ###################################
            y_val = model(torch.tensor(inputs))
            ###################################

=======
>>>>>>> master
            # Forward pass
            output = model.forward(inputs)
            # Calculate Loss
            valloss = criterion(output, labels)
<<<<<<< HEAD
            ###################################
            # loss = criterion(y_val, val_set)
            ###################################
=======
>>>>>>> master
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
<<<<<<< HEAD
            ###################################
            print(output[:5])
            ###################################
=======

            _, preds = torch.max(output, 1)

            for p, t in zip(preds.view(-1), labels.view(-1)):
                confusion_matrix[p.long(), t.long()] += 1
                # confusion_matrix[p.int(), t.int()] += 1
                # confusion_matrix[p, t] += 1

        print(confusion_matrix)

        TP = confusion_matrix.diag()

        for c in range(total_classes):
            idx = torch.ones(total_classes).byte()
            idx[c] = 0
            TN = confusion_matrix[idx.nonzero()[:, None], idx.nonzero()].sum()
            FP = confusion_matrix[c, idx].sum()
            FN = confusion_matrix[idx, c].sum()

            sensitivity = (TP[c] / (TP[c] + FN))
            specificity = (TN / (TN + FP))
            re_call = (TP[c] / (TP[c] + FP))
            pre_cision = (TP[c] / (TP[c] + FN))
            f1_score = 2 * ((pre_cision * re_call) / (pre_cision + re_call))

            print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(c+1, TP[c], TN, FP, FN))
            print('Sensitivity = {}'.format(sensitivity))
            print('Specificity = {}'.format(specificity))
            print('Recall = {}'.format(re_call))
            print('Precision = {}'.format(pre_cision))
            print('F1 Score = {}'.format(f1_score))
>>>>>>> master

    end_valid = time.time()
    print('Finished Epoch', epoch + 1, 'Validating in %0.2f minutes' % ((end_valid - start_valid) / 60))

    # Get the average loss for the entire epoch
    train_loss = train_loss / len(train_loader.dataset)
    valid_loss = val_loss / len(val_loader.dataset)
<<<<<<< HEAD
=======

>>>>>>> master
    # Print out the information
    print('Accuracy: %0.3f %%' % (accuracy / len(val_loader) * 100))
    print('Training Loss: {:.6f} ' '\tValidation Loss: {:.6f}'.format(train_loss, valid_loss), '\n')

<<<<<<< HEAD
    ###################################
    print(len(val_loader.dataset))
    print(confusion_matrix(output, y_val))
    # print(classification_report(test_outputs, y_val))
    # print(accuracy_score(test_outputs, y_val))
    ###################################

print('Total Time is %0.2f minutes' % ((end_valid - start_train) / 60))
=======
    print('Total Time is %0.2f minutes' % ((end_valid - start_train) / 60))
>>>>>>> master
