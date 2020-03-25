import os
import torch
from torch import nn
from torchvision import datasets, transforms, models


def get_images_from_folder(folder):
    images = os.listdir(folder)
    print("Number of Images: ", len(images), " is loaded from directory", folder)
    return images


def make_model():
    # Hyperparameters for our network
    input_size = 784
    hidden_sizes = [128, 64]
    output_size = 10
    # Build a feed-forward network
    from collections import OrderedDict
    model = nn.Sequential(OrderedDict([
        # Fully Connected Layer One
        ('fc1', nn.Linear(input_size, hidden_sizes[0])),
        # Rectified Linear Unit One
        ('rlu1', nn.ReLU()),
        # Fully Connected Layer Two
        ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
        # Rectified Linear Unit Two
        ('rlu2', nn.ReLU()),
        # Output layer
        ('output', nn.Linear(hidden_sizes[1], output_size)),
        # Softmax layer
        ('softmax', nn.Softmax(dim=1))]))
    # To print specific layer, you can print index or name of layer like this
    # print(model[0]) or print(model.fc1)
    print(model)
    return model
