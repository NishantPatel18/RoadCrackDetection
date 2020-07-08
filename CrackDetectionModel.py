import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import torchvision
import numpy as np
import os
cwd = os.getcwd()
from PIL import Image
import time
import copy
import random
# import cv2
import re
import shutil
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline

crack_train = os.listdir("Train/Positive")
print("Number of Crack Images:", len(crack_train))

non_crack_train = os.listdir("Train/Negative")
print("Number of Non-Crack Images:", len(non_crack_train))

crack_test = os.listdir("Test/Positive")
print("Number of Crack Images:", len(crack_test))

non_crack_test = os.listdir("Test/Negative")
print("Number of Non-Crack Images:", len(non_crack_test))
