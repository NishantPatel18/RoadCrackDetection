import torch
import torch.nn as nn
import os

positive_train = os.listdir("Train/Positive")
print("Number of Positive Images:", len(positive_train), "loaded from directory Train")

negative_train = os.listdir("Train/Negative")
print("Number of Negative Images:", len(negative_train), "loaded from directory Train")

positive_test = os.listdir("Test/Positive")
print("Number of Positive Images:", len(positive_test), "loaded from directory Test")

negative_test = os.listdir("Test/Negative")
print("Number of Negative Images:", len(negative_test), "loaded from directory Test")


