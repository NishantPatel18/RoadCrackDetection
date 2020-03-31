import torch.nn as nn
from torch import optim
import torch
from torch.utils import data
import torch.nn.functional as F
from SetFunctionUp import loading_data, train_model, test_model, make_model
from SetFunctionUp import get_images_from_folder
from torchvision import datasets, transforms, models

Trainloader, Testloader, Trainset, Testset = loading_data("RoadData")


model = make_model()

epoch = 5

TrainedModel = train_model(model, epoch, Trainloader)

# test_model(model, Testloader)
test_model(TrainedModel, Testloader)
