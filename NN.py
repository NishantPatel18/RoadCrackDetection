import torch
import torch.nn as nn

import cv2
import os
import glob
import Function

train_po = Function.get_images_from_folder("Train/Po")
train_ne = Function.get_images_from_folder("Train/Ne")
test_po = Function.get_images_from_folder("Test/Po")
test_ne = Function.get_images_from_folder("Test/Ne")
valid_po = Function.get_images_from_folder("Valid/ValidHunPo")
valid_ne = Function.get_images_from_folder("Valid/ValidHunNe")

model = Function.make_model()
