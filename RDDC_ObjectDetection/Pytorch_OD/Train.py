import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import pandas as pd

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def parse_one_annot(path_to_data_file, filename):
    data = pd.read_csv(path_to_data_file)
    boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
    class_name = data[data["filename"] == filename][["class"]].values

    return boxes_array, class_name


def get_class_number(class_input):
    class_number = 0
    if (class_input == ['D00']):
        class_number = 1
    elif (class_input == ['D01']):
        class_number = 2
    elif (class_input == ['D10']):
        class_number = 3
    elif (class_input == ['D11']):
        class_number = 4
    elif (class_input == ['D20']):
        class_number = 5
    elif (class_input == ['D30']):
        class_number = 6
    elif (class_input == ['D40']):
        class_number = 7
    elif (class_input == ['D43']):
        class_number = 8
    elif (class_input == ['D44']):
        class_number = 9
    else:
        class_number = 0

    return class_number


class RoadDamageDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_file, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root)))
        self.path_to_data_file = data_file

    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        box_list, class_name = parse_one_annot(self.path_to_data_file,
                                               self.imgs[idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)
        class_list = []
        # print(class_name)

        for name in class_name:
            # print(name)
            number = get_class_number(name)
            # print(number)
            class_list.append(number)

        # print(class_list)
        class_list_numpy = np.array(class_list)
        labels = torch.as_tensor(class_list_numpy, dtype=torch.int64)
        # labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        # print(target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


dataset = RoadDamageDataset(root="/content/RoadCrackDetection/RDDC_ObjectDetection/RoadDamageDataset/All_cities/Images",
                         data_file="/content/RoadCrackDetection/RDDC_ObjectDetection/Pytorch_OD/all_cities.csv")
dataset.__getitem__(0)


def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


# use our dataset and defined transformations
dataset = RoadDamageDataset(root="/content/RoadCrackDetection/RDDC_ObjectDetection/RoadDamageDataset/All_cities/Images",
                         data_file="/content/RoadCrackDetection/RDDC_ObjectDetection/Pytorch_OD/all_cities.csv",
                         transforms=get_transform(train=True))

dataset_test = RoadDamageDataset(
    root="/content/RoadCrackDetection/RDDC_ObjectDetection/RoadDamageDataset/All_cities/Images",
    data_file="/content/RoadCrackDetection/RDDC_ObjectDetection/Pytorch_OD/all_cities.csv",
    transforms=get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
num_total_for_train_test = 9892
# 30%
num_test = 2968
# num_test = 10
num_train = num_total_for_train_test - num_test
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:num_train])
dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(dataset, batch_size=14, shuffle=True, num_workers=4,
                                          collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=14, shuffle=False, num_workers=4,
                                               collate_fn=utils.collate_fn)

print("We have: {} examples, {} are training and {} are testing".format(len(indices), len(dataset), len(dataset_test)))

device = torch.device('cuda')

# our dataset has 10 classes which are D00, D01, D10, D11, D20, D30, D40, D43, D44 and NO(negative class)
num_classes = 10

# get the model using our helper function
model = get_model(num_classes)

# move model to the right device
model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# let's train it for 10 epochs
num_epochs = 10

for epoch in range(num_epochs):
    # train for one epoch, printing every 10 iterations
    # change print freq to display 1 by 1
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

    # update the learning rate
    lr_scheduler.step()

    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

# to save model
# torch.save(model.state_dict(), "/content/RoadCrackDetection/RDDC_ObjectDetection/Pytorch_OD/road_crack/model")