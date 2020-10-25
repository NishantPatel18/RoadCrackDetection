# RoadCrackDetection
Our industrial project is on 'Intelligent Road Crack Detection using Deep Learning' where we create an artificial intelligence (AI) system that is able to classify and detect road cracks on New Zealand roads.

This repositories contains 3 main folders which are:
1.  Faster_R-CNN_Object_Detection
2.  Image_Classification
3.  YOLO_v3_Object_Detection

Faster_R-CNN_Object_Detection
This folder contains Pytorch_OD and RoadDamageDataset folders.
Pytorch_OD folder contains all python code files.
RoadDamageDataset folder contains 7 japanese cities road damage image folders of RDDC dataset.

Image_Classification
This folder contains 5 CNN models and 2 main datasets.
2 main datasets are CCIC (COncrete Crack for Image Classification) dataset and RDDC (Road Damage Detection and Classification) dataset.
Each dataset is split into 70/30 ratio.
There are 2 extra train and test folders which are AUG_RDDC and Balanced_RDDC.
AUG_RDD_Test and AUG_RDD_Train contains augmented images.
Balanced_RDDC_Test and Balanced_RDDC_Train contains equal amount of images for each class.

YOLO_v3_Object_Detection
This folder contains python code files for yolo v3 object detection and configuration files.


