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

--------------------------------------------------------------------------------------------------------------

Manual guide to run the object detection model
--------------------------------------------------------------------------------------------------------------

Guide to "RoadCrackDetection/RDDC_ObjectDetection/Pytorch_OD" directory.

Step 1: If annotation of training is xml, there is a python file called "xml_to_csv.py" to change xml to csv.
--------------------------------------------------------------------------------------------------------------

Step 2: Training
--------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/49043498/97121898-8d620580-1786-11eb-970e-867d7930053d.png)

Provide the directories of image folders and csv files.

--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97122237-4de8e880-1789-11eb-845a-d962f43eade7.png)

Set the number of total and test here. That will also calculate the number of train.

--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97122142-699fbf00-1788-11eb-9492-a5134974b512.png)

Define the "num_classes" according to your number of classes.
Adjust the learning rate in "lr" and "num_epochs" as you need.

--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97122320-f1d29400-1789-11eb-9b0a-14b14dac8a48.png)

Give the directory to save the trained model and name the model. After that, run the train.

--------------------------------------------------------------------------------------------------------------

Step 3: Testing
--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97122450-c3a18400-178a-11eb-825b-6e7f38ddef00.png)

Define the number of classes and load the trained model that is saved in the training.
After that, run the test.

--------------------------------------------------------------------------------------------------------------



