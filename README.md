# RoadCrackDetection
Our industrial project is called *Intelligent Road Crack Detection using Deep Learning* where we create an artificial intelligence (AI) system that is able to classify and detect road cracks on New Zealand roads.

Description
--------------------------------------------------------------------------------------------------------------

This repositories contains *three* main folders and *two* Google Colab files.

Main Folders:
1.  Faster_R-CNN_Object_Detection
2.  Image_Classification
3.  YOLO_v3_Object_Detection

Google Colab Files:
1.  DenseNet161_Image_Classification.ipynb
2.  Faster_R_CNN_Object_Detection.ipynb

Faster_R-CNN_Object_Detection
--------------------------------------------------------------------------------------------------------------

The Faster_R-CNN_Object Detection folder is the main folder that was used to create the object detection model for our project.
Inside this folder, there are two folders which are called Pytorch_OD and RoadDamageDataset.
The Pytorch_OD folder contains all the Python code files.
The RoadDamageDataset folder contains the road damage image folders of seven Japanese cities from the Road Damage Detection and Classification [(RDDC)](https://github.com/sekilab/RoadDamageDetector) dataset.

Image_Classification
--------------------------------------------------------------------------------------------------------------

The Image_Classification folder is the main folder that was used to create the image classification model for our project.
This folder contains five Convolutional Neural Network (CNN) models and two main datasets.
The two main datasets are the Concrete Crack for Image Classification [(CCIC)](https://data.mendeley.com/datasets/5y9wdsg2zt/2) dataset and the [RDDC](https://github.com/sekilab/RoadDamageDetector) dataset.
Each dataset is split into a 70/30 ratio for training and testing.
There are two extra training and testing folders which are named AUG_RDDC and Balanced_RDDC.
AUG_RDD_Train and AUG_RDD_Test contains augmented images that was used to try and improve the performance of our model.
Balanced_RDDC_Test and Balanced_RDDC_Train contains an equal amount of images for each class in the RDDC dataset.

YOLO_v3_Object_Detection
--------------------------------------------------------------------------------------------------------------

This folder contains Python code files for YOLO v3 object detection and configuration files.
The YOLO v3 object detection model was only used for upskilling purposes during the object detection phase in our project.
Therefore, it is only provided to show evidence of upskilling.

Options to run our image classification and object detection models
--------------------------------------------------------------------------------------------------------------

Option 1: Using the code in the "Image_Classification" and "Faster_R-CNN_Object_Detection" folders.

Option 2: Directly using the Google Colab files which are "DenseNet161_Image_Classification.ipynb" and "Faster_R_CNN_Object_Detection.ipynb".

--------------------------------------------------------------------------------------------------------------

Manual guide to run the object detection model
--------------------------------------------------------------------------------------------------------------
Step 1:
--------------------------------------------------------------------------------------------------------------
Proceed to "RoadCrackDetection/Faster_R-CNN_Object_Detection/Pytorch_OD" directory.

--------------------------------------------------------------------------------------------------------------

Step 2: 
--------------------------------------------------------------------------------------------------------------
If the annotation file is of type XML, there is a Python file called "xml_to_csv.py" to change it to CSV.

--------------------------------------------------------------------------------------------------------------

Step 3: Training
--------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/49043498/97121898-8d620580-1786-11eb-970e-867d7930053d.png)

Provide the directories of image folders and CSV files.

--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97122237-4de8e880-1789-11eb-845a-d962f43eade7.png)

Define the total number of images in "num_total_for_train_test".

Define the number of test images in "num_test" that will be used for testing.

Once total and test images have been specified, the number of images that will be used for training is defined in the formula:

"num_train = num_total_for_train_test - num_test"

--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97122142-699fbf00-1788-11eb-9492-a5134974b512.png)

Define the "num_classes" according to your number of classes.

Adjust the learning rate in "lr" and number of epochs in "num_epochs" as you need.

--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97122320-f1d29400-1789-11eb-9b0a-14b14dac8a48.png)

Provide a directory to save and name the trained model. After that, run the model for training.

--------------------------------------------------------------------------------------------------------------

Step 4: Testing
--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97122450-c3a18400-178a-11eb-825b-6e7f38ddef00.png)

Define the number of classes and load the trained model that was saved after training.

After that, run the model for testing.

--------------------------------------------------------------------------------------------------------------

Manual guide to run the image classification model
--------------------------------------------------------------------------------------------------------------
Step 1:
--------------------------------------------------------------------------------------------------------------
Proceed to "RoadCrackDetection/Image_Classification" directory.

--------------------------------------------------------------------------------------------------------------

Step 2: Training and Testing
--------------------------------------------------------------------------------------------------------------
![image](https://user-images.githubusercontent.com/49043498/97136449-02016800-17b8-11eb-89e4-922c127149fa.png)

Provide the directories of image folders for training and testing.

--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97136564-51e02f00-17b8-11eb-92a0-787f536bd8cd.png)

(Optional) Change the DenseNet model if you want to use DenseNet-121 or DenseNet-201.

--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97136746-d59a1b80-17b8-11eb-97c1-b73bbb12e412.png)

Define the "num_labels" according to your number of classes.

(Optional) Modify the layers of neural network.

--------------------------------------------------------------------------------------------------------------

![image](https://user-images.githubusercontent.com/49043498/97136846-214cc500-17b9-11eb-9426-bc40bbb249df.png)

Define the number of epoch.

After that, run the model for training and testing.

--------------------------------------------------------------------------------------------------------------
