from PIL import Image, ImageFont, ImageDraw
from IOU import get_iou
import matplotlib.pyplot as plt
import time

# Define the number of class
loaded_model = get_model(num_classes=10)
# Load the model
loaded_model.load_state_dict(torch.load("/content/drive/My Drive/Models/model_10_class"))


def getTPR_FPR(TP, TN, FP, FN):
    TPR = 0
    FPR = 0
    if (TP == 0 and FN == 0):
        pass
    else:
        TPR = TP / (TP + FN)

    if (TN == 0 and FP == 0):
        pass
    else:
        FPR = FP / (TN + FP)

    return TPR, FPR


# calculate recall, precision, accuracy and F1 score
def getEval(TP, TN, FP, FN):
    REC = 0
    PRE = 0
    ACU = 0
    FSO = 0

    if (TP == 0 and FN == 0):
        pass
    else:
        REC = TP / (TP + FN)

    if (TP == 0 and FP == 0):
        pass
    else:
        PRE = TP / (TP + FP)

    if (TP == 0 and TN == 0 and FP == 0 and FN == 0):
        pass
    else:
        ACU = (TP + TN) / (TP + TN + FP + FN)

    if (TP == 0 and FP == 0 and FN == 0):
        pass
    else:
        FSO = (2 * TP) / ((2 * TP) + FP + FN)

    return REC, PRE, ACU, FSO


# remove the same element from the list
# to be used to check false negative
def remove_items_from_list(my_list, temp):
    dup_list = [item for item in temp if item in my_list]

    for ele in dup_list:
        my_list.remove(ele)

    return my_list


# convert class number to class name
def get_class_name(class_number_input):
    class_name = ''
    if (class_number_input == 1):
        class_name = 'D00'
    elif (class_number_input == 2):
        class_name = 'D01'
    elif (class_number_input == 3):
        class_name = 'D10'
    elif (class_number_input == 4):
        class_name = 'D11'
    elif (class_number_input == 5):
        class_name = 'D20'
    elif (class_number_input == 6):
        class_name = 'D30'
    elif (class_number_input == 7):
        class_name = 'D40'
    elif (class_number_input == 8):
        class_name = 'D43'
    elif (class_number_input == 9):
        class_name = 'D44'
    else:
        class_name = 'NO'

    return class_name


# visualising image and getting data for confusion matrix
def visual_image(index_of_image):
    num_total_cracks = 0
    num_cracks = 0
    num_passed_iou = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    Precision = 0
    Recall = 0
    FPR = 0
    F1_score = 0
    iou_array = []
    iou_passed_class_array = []
    groundtruth_class_array = []
    extra_FP = 0
    D00_TP = 0
    D01_TP = 0
    D10_TP = 0
    D11_TP = 0
    D20_TP = 0
    D30_TP = 0
    D40_TP = 0
    D43_TP = 0
    D44_TP = 0
    NO_TP = 0
    D00_TN = 0
    D01_TN = 0
    D10_TN = 0
    D11_TN = 0
    D20_TN = 0
    D30_TN = 0
    D40_TN = 0
    D43_TN = 0
    D44_TN = 0
    NO_TN = 0
    D00_FP = 0
    D01_FP = 0
    D10_FP = 0
    D11_FP = 0
    D20_FP = 0
    D30_FP = 0
    D40_FP = 0
    D43_FP = 0
    D44_FP = 0
    NO_FP = 0
    D00_FN = 0
    D01_FN = 0
    D10_FN = 0
    D11_FN = 0
    D20_FN = 0
    D30_FN = 0
    D40_FN = 0
    D43_FN = 0
    D44_FN = 0
    NO_FN = 0
    D00_TPR = 0
    D01_TPR = 0
    D10_TPR = 0
    D11_TPR = 0
    D20_TPR = 0
    D30_TPR = 0
    D40_TPR = 0
    D43_TPR = 0
    D44_TPR = 0
    NO_TPR = 0
    D00_FPR = 0
    D01_FPR = 0
    D10_FPR = 0
    D11_FPR = 0
    D20_FPR = 0
    D30_FPR = 0
    D40_FPR = 0
    D43_FPR = 0
    D44_FPR = 0
    NO_FPR = 0

    # to calculate inference time
    start_test = time.time()
    img, _ = dataset_test[index_of_image]
    label_boxes = np.array(dataset_test[index_of_image][1]["boxes"])
    label_classes = np.array(dataset_test[index_of_image][1]["labels"])
    # print(label_classes)
    class_list = label_classes.tolist()

    # put the model in evaluation mode
    loaded_model.eval()

    with torch.no_grad():
        prediction = loaded_model([img])

    end_test = time.time()

    # print(prediction)
    pred_class_list = prediction[0]["labels"].tolist()
    # print(pred_class_list)

    image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(image)

    num_groundtruth_obj = len(label_boxes)

    # draw groundtruth
    for elem in range(len(label_boxes)):
        # print(label_boxes)
        # print('Ground Truth Class', label_classes[elem])
        draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]), (label_boxes[elem][2], label_boxes[elem][3])],
                       outline="green", width=3)
        draw.text((label_boxes[elem][0], label_boxes[elem][1]), text=str(get_class_name(class_list[elem])))
        groundtruth_class_array.append(class_list[elem])

    for element in range(len(prediction[0]["boxes"])):
        # print(prediction[0]["boxes"])
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        confidence = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)
        # print(prediction[0]["boxes"][element])

        # checking confidence
        if confidence >= 0.7:
            # draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="blue", width=3)
            # draw.text((boxes[0], boxes[1]), text=(get_class_name(pred_class_list[element])))
            # print(get_class_name(pred_class_list[element]))
            prediction_class_element = pred_class_list[element]
            # print('Prediction Class', prediction_class_element)
            num_cracks += 1
            num_total_cracks += 1

            max_iou = 0
            class_name_checking = ''

            # checking IOU
            for box in range(len(label_boxes)):
                draft_cal_iou = get_iou(prediction[0]["boxes"][element], label_boxes[box])
                cal_iou = draft_cal_iou.data.cpu().numpy()
                if (cal_iou > max_iou):
                    max_iou = cal_iou
                    class_name_checking = class_list[box]
                    # print('max_iou', max_iou)

            # checking class label to meet all criteria to be a true positive
            if (max_iou >= 0.5 and prediction_class_element == class_name_checking):
                num_passed_iou += 1
                iou_array.append(max_iou)
                iou_passed_class_array.append(pred_class_list[element])
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=3)
                draw.text((boxes[0], boxes[1]), text=(get_class_name(pred_class_list[element])))
                if (pred_class_list[element] == 1):
                    D00_TP += 1
                elif (pred_class_list[element] == 2):
                    D01_TP += 1
                elif (pred_class_list[element] == 3):
                    D10_TP += 1
                elif (pred_class_list[element] == 4):
                    D11_TP += 1
                elif (pred_class_list[element] == 5):
                    D20_TP += 1
                elif (pred_class_list[element] == 6):
                    D30_TP += 1
                elif (pred_class_list[element] == 7):
                    D40_TP += 1
                elif (pred_class_list[element] == 8):
                    D43_TP += 1
                elif (pred_class_list[element] == 9):
                    D44_TP += 1
                else:
                    NO_TP += 1
            else:
                # confidence is above 0.7 but it did not pass IOU and class label check
                if (pred_class_list[element] == 1):
                    D00_FP += 1
                elif (pred_class_list[element] == 2):
                    D01_FP += 1
                elif (pred_class_list[element] == 3):
                    D10_FP += 1
                elif (pred_class_list[element] == 4):
                    D11_FP += 1
                elif (pred_class_list[element] == 5):
                    D20_FP += 1
                elif (pred_class_list[element] == 6):
                    D30_FP += 1
                elif (pred_class_list[element] == 7):
                    D40_FP += 1
                elif (pred_class_list[element] == 8):
                    D43_FP += 1
                elif (pred_class_list[element] == 9):
                    D44_FP += 1
                else:
                    NO_FP += 1
        else:
            # confidence below 0.7
            TN += 1
            if (pred_class_list[element] == 1):
                D00_TN += 1
            elif (pred_class_list[element] == 2):
                D01_TN += 1
            elif (pred_class_list[element] == 3):
                D10_TN += 1
            elif (pred_class_list[element] == 4):
                D11_TN += 1
            elif (pred_class_list[element] == 5):
                D20_TN += 1
            elif (pred_class_list[element] == 6):
                D30_TN += 1
            elif (pred_class_list[element] == 7):
                D40_TN += 1
            elif (pred_class_list[element] == 8):
                D43_TN += 1
            elif (pred_class_list[element] == 9):
                D44_TN += 1
            else:
                NO_TN += 1

    # to check the possible extra false postive from true postive where the number of ground truch bounding box is lower than true positive
    if (num_passed_iou > num_groundtruth_obj):
        extra_FP = num_passed_iou - num_groundtruth_obj
        FN = 0
    else:
        FN = num_groundtruth_obj - num_passed_iou
    if (num_passed_iou > num_groundtruth_obj):
        extra_FP = num_passed_iou - num_groundtruth_obj
        TP = num_passed_iou - extra_FP
    else:
        TP = num_passed_iou

    FP = (num_cracks - num_passed_iou) + extra_FP

    if (extra_FP > 0):
        extra_FP_array = remove_items_from_list(iou_passed_class_array, groundtruth_class_array)
        # print('extra FP')
        # print(extra_FP_array)
        for element in range(len(extra_FP_array)):
            if (extra_FP_array[element] == 1):
                D00_FP += 1
                D00_TP -= 1
            elif (extra_FP_array[element] == 2):
                D01_FP += 1
                D01_TP -= 1
            elif (extra_FP_array[element] == 3):
                D10_FP += 1
                D10_TP -= 1
            elif (extra_FP_array[element] == 4):
                D11_FP += 1
                D11_TP -= 1
            elif (extra_FP_array[element] == 5):
                D20_FP += 1
                D20_TP -= 1
            elif (extra_FP_array[element] == 6):
                D30_FP += 1
                D30_TP -= 1
            elif (extra_FP_array[element] == 7):
                D40_FP += 1
                D40_TP -= 1
            elif (extra_FP_array[element] == 8):
                D43_FP += 1
                D43_TP -= 1
            elif (extra_FP_array[element] == 9):
                D44_FP += 1
                D44_TP -= 1
            else:
                NO_FP += 1
                NO_TP -= 1

    # to get class label of false negative and record it into individual class false negative
    if (FN > 0):
        FN_array = remove_items_from_list(groundtruth_class_array, iou_passed_class_array)
        # print('FN is there')
        # print(FN_array)
        for element in range(len(FN_array)):
            if (FN_array[element] == 1):
                D00_FN += 1
            elif (FN_array[element] == 2):
                D01_FN += 1
            elif (FN_array[element] == 3):
                D10_FN += 1
            elif (FN_array[element] == 4):
                D11_FN += 1
            elif (FN_array[element] == 5):
                D20_FN += 1
            elif (FN_array[element] == 6):
                D30_FN += 1
            elif (FN_array[element] == 7):
                D40_FN += 1
            elif (FN_array[element] == 8):
                D43_FN += 1
            elif (FN_array[element] == 9):
                D44_FN += 1
            else:
                NO_FN += 1

    if ((TP == 0 and FP == 0) or (TP == 0 and FN == 0) or (TN == 0 and FP == 0)):
        pass
    else:
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        FPR = FP / (TN + FP)

    if (Precision == 0 and Recall == 0):
        pass
    else:
        F1_score = 2 * ((Precision * Recall) / (Precision + Recall))

    # calculate true positive rate and false positive rate for each class
    D00_TPR, D00_FPR = getTPR_FPR(D00_TP, D00_TN, D00_FP, D00_FN)
    D01_TPR, D01_FPR = getTPR_FPR(D01_TP, D01_TN, D01_FP, D01_FN)
    D10_TPR, D10_FPR = getTPR_FPR(D10_TP, D10_TN, D10_FP, D10_FN)
    D11_TPR, D11_FPR = getTPR_FPR(D11_TP, D11_TN, D11_FP, D11_FN)
    D20_TPR, D20_FPR = getTPR_FPR(D20_TP, D20_TN, D20_FP, D20_FN)
    D30_TPR, D30_FPR = getTPR_FPR(D30_TP, D30_TN, D30_FP, D30_FN)
    D40_TPR, D40_FPR = getTPR_FPR(D40_TP, D40_TN, D40_FP, D40_FN)
    D43_TPR, D43_FPR = getTPR_FPR(D43_TP, D43_TN, D43_FP, D43_FN)
    D44_TPR, D44_FPR = getTPR_FPR(D44_TP, D44_TN, D44_FP, D44_FN)
    NO_TPR, NO_FPR = getTPR_FPR(NO_TP, NO_TN, NO_FP, NO_FN)

    # display the image
    display(image)

    # output message
    if num_cracks == 1:
        print('There is', num_cracks, 'road crack in this image')
    elif num_cracks > 1:
        print('There are', num_cracks, 'road cracks in this image')
    else:
        print('There is no crack')

    infer_time = end_test - start_test

    print('TP', TP)
    print('FP', FP)
    print('FN', FN)
    print('TN', TN)
    print('Infer Time is %0.2f second' % infer_time)

    return infer_time, Recall, FPR, num_total_cracks, iou_array, TP, FP, FN, TN, D00_TPR, D01_TPR, D10_TPR, D11_TPR, D20_TPR, D30_TPR, D40_TPR, D43_TPR, D44_TPR, NO_TPR, D00_FPR, D01_FPR, D10_FPR, D11_FPR, D20_FPR, D30_FPR, D40_FPR, D43_FPR, D44_FPR, NO_FPR, D00_TP, D01_TP, D10_TP, D11_TP, D20_TP, D30_TP, D40_TP, D43_TP, D44_TP, NO_TP, D00_TN, D01_TN, D10_TN, D11_TN, D20_TN, D30_TN, D40_TN, D43_TN, D44_TN, NO_TN, D00_FP, D01_FP, D10_FP, D11_FP, D20_FP, D30_FP, D40_FP, D43_FP, D44_FP, NO_FP, D00_FN, D01_FN, D10_FN, D11_FN, D20_FN, D30_FN, D40_FN, D43_FN, D44_FN, NO_FN


def main():
    total = 0
    Precision = 0
    TPR_Recall = 0
    TPR_array = []
    FPR_array = []
    D00_TPR_array = []
    D00_FPR_array = []
    D01_TPR_array = []
    D01_FPR_array = []
    D10_TPR_array = []
    D10_FPR_array = []
    D11_TPR_array = []
    D11_FPR_array = []
    D20_TPR_array = []
    D20_FPR_array = []
    D30_TPR_array = []
    D30_FPR_array = []
    D40_TPR_array = []
    D40_FPR_array = []
    D43_TPR_array = []
    D43_FPR_array = []
    D44_TPR_array = []
    D44_FPR_array = []
    F1_score = 0
    input_Recall = 0
    input_FPR = 0
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    Total_D00_TP = 0
    Total_D01_TP = 0
    Total_D10_TP = 0
    Total_D11_TP = 0
    Total_D20_TP = 0
    Total_D30_TP = 0
    Total_D40_TP = 0
    Total_D43_TP = 0
    Total_D44_TP = 0
    Total_NO_TP = 0
    Total_D00_TN = 0
    Total_D01_TN = 0
    Total_D10_TN = 0
    Total_D11_TN = 0
    Total_D20_TN = 0
    Total_D30_TN = 0
    Total_D40_TN = 0
    Total_D43_TN = 0
    Total_D44_TN = 0
    Total_NO_TN = 0
    Total_D00_FP = 0
    Total_D01_FP = 0
    Total_D10_FP = 0
    Total_D11_FP = 0
    Total_D20_FP = 0
    Total_D30_FP = 0
    Total_D40_FP = 0
    Total_D43_FP = 0
    Total_D44_FP = 0
    Total_NO_FP = 0
    Total_D00_FN = 0
    Total_D01_FN = 0
    Total_D10_FN = 0
    Total_D11_FN = 0
    Total_D20_FN = 0
    Total_D30_FN = 0
    Total_D40_FN = 0
    Total_D43_FN = 0
    Total_D44_FN = 0
    Total_NO_FN = 0
    total_infer_time = 0
    all_tests = len(dataset_test)
    # some_tests = 20

    for index in range(all_tests):

        # get all recorded information and combine all values for each
        infer_time, each_Recall, each_FPR, num_total_cracks, iou_array, each_TP, each_FP, each_FN, each_TN, D00_TPR, D01_TPR, D10_TPR, D11_TPR, D20_TPR, D30_TPR, D40_TPR, D43_TPR, D44_TPR, NO_TPR, D00_FPR, D01_FPR, D10_FPR, D11_FPR, D20_FPR, D30_FPR, D40_FPR, D43_FPR, D44_FPR, NO_FPR, D00_TP, D01_TP, D10_TP, D11_TP, D20_TP, D30_TP, D40_TP, D43_TP, D44_TP, NO_TP, D00_TN, D01_TN, D10_TN, D11_TN, D20_TN, D30_TN, D40_TN, D43_TN, D44_TN, NO_TN, D00_FP, D01_FP, D10_FP, D11_FP, D20_FP, D30_FP, D40_FP, D43_FP, D44_FP, NO_FP, D00_FN, D01_FN, D10_FN, D11_FN, D20_FN, D30_FN, D40_FN, D43_FN, D44_FN, NO_FN = visual_image(
            index)
        total_infer_time += infer_time
        total += num_total_cracks
        input_Recall += each_Recall
        input_FPR += each_FPR
        TPR_array.append(each_Recall)
        FPR_array.append(each_FPR)
        D00_TPR_array.append(D00_TPR)
        D00_FPR_array.append(D00_FPR)
        D01_TPR_array.append(D01_TPR + 0.01)
        D01_FPR_array.append(D01_FPR + 0.01)
        D10_TPR_array.append(D10_TPR + 0.02)
        D10_FPR_array.append(D10_FPR + 0.02)
        D11_TPR_array.append(D11_TPR + 0.03)
        D11_FPR_array.append(D11_FPR + 0.03)
        D20_TPR_array.append(D20_TPR + 0.04)
        D20_FPR_array.append(D20_FPR + 0.04)
        D30_TPR_array.append(D30_TPR + 0.05)
        D30_FPR_array.append(D30_FPR + 0.05)
        D40_TPR_array.append(D40_TPR + 0.06)
        D40_FPR_array.append(D40_FPR + 0.06)
        D43_TPR_array.append(D43_TPR + 0.07)
        D43_FPR_array.append(D43_FPR + 0.07)
        D44_TPR_array.append(D44_TPR + 0.08)
        D44_FPR_array.append(D44_FPR + 0.08)
        TP += each_TP
        FP += each_FP
        FN += each_FN
        TN += each_TN
        Total_D00_TP += D00_TP
        Total_D01_TP += D01_TP
        Total_D10_TP += D10_TP
        Total_D11_TP += D11_TP
        Total_D20_TP += D20_TP
        Total_D30_TP += D30_TP
        Total_D40_TP += D40_TP
        Total_D43_TP += D43_TP
        Total_D44_TP += D44_TP
        Total_NO_TP += NO_TP
        Total_D00_TN += D00_TN
        Total_D01_TN += D01_TN
        Total_D10_TN += D10_TN
        Total_D11_TN += D11_TN
        Total_D20_TN += D20_TN
        Total_D30_TN += D30_TN
        Total_D40_TN += D40_TN
        Total_D43_TN += D43_TN
        Total_D44_TN += D44_TN
        Total_NO_TN += NO_TN
        Total_D00_FP += D00_FP
        Total_D01_FP += D01_FP
        Total_D10_FP += D10_FP
        Total_D11_FP += D11_FP
        Total_D20_FP += D20_FP
        Total_D30_FP += D30_FP
        Total_D40_FP += D40_FP
        Total_D43_FP += D43_FP
        Total_D44_FP += D44_FP
        Total_NO_FP += NO_FP
        Total_D00_FN += D00_FN
        Total_D01_FN += D01_FN
        Total_D10_FN += D10_FN
        Total_D11_FN += D11_FN
        Total_D20_FN += D20_FN
        Total_D30_FN += D30_FN
        Total_D40_FN += D40_FN
        Total_D43_FN += D43_FN
        Total_D44_FN += D44_FN
        Total_NO_FN += NO_FN

        # calculate recall, precision, accuracy and F1 score for each class
        D00_Recall, D00_Precision, D00_Accuracy, D00_F1_Score = getEval(Total_D00_TP, Total_D00_TN, Total_D00_FP,
                                                                        Total_D00_FN)
        D01_Recall, D01_Precision, D01_Accuracy, D01_F1_Score = getEval(Total_D01_TP, Total_D01_TN, Total_D01_FP,
                                                                        Total_D01_FN)
        D10_Recall, D10_Precision, D10_Accuracy, D10_F1_Score = getEval(Total_D10_TP, Total_D10_TN, Total_D10_FP,
                                                                        Total_D10_FN)
        D11_Recall, D11_Precision, D11_Accuracy, D11_F1_Score = getEval(Total_D11_TP, Total_D11_TN, Total_D11_FP,
                                                                        Total_D11_FN)
        D20_Recall, D20_Precision, D20_Accuracy, D20_F1_Score = getEval(Total_D20_TP, Total_D20_TN, Total_D20_FP,
                                                                        Total_D20_FN)
        D30_Recall, D30_Precision, D30_Accuracy, D30_F1_Score = getEval(Total_D30_TP, Total_D30_TN, Total_D30_FP,
                                                                        Total_D30_FN)
        D40_Recall, D40_Precision, D40_Accuracy, D40_F1_Score = getEval(Total_D40_TP, Total_D40_TN, Total_D40_FP,
                                                                        Total_D40_FN)
        D43_Recall, D43_Precision, D43_Accuracy, D43_F1_Score = getEval(Total_D43_TP, Total_D43_TN, Total_D43_FP,
                                                                        Total_D43_FN)
        D44_Recall, D44_Precision, D44_Accuracy, D44_F1_Score = getEval(Total_D44_TP, Total_D44_TN, Total_D44_FP,
                                                                        Total_D44_FN)

        # output message
        if index > 0:
            print(index + 1, 'predictions are done')
        else:
            print(index + 1, 'prediction is done')

        if len(iou_array) <= 0:
            pass
        elif len(iou_array) == 1:
            print('IOU of the bounding box of detected crack is')
        else:
            print('IOUs of the bounding boxes of detected cracks are')

        for ind in range(len(iou_array)):
            iou = float(iou_array[ind])
            print('IOU:', round(iou, 4))

        print('-----------------------------------------------------------------------------')

    if ((TP == 0 and FP == 0) or (TP == 0 and FN == 0) or (TN == 0 and FP == 0)):
        pass
    else:
        Precision = TP / (TP + FP)
        TPR_Recall = TP / (TP + FN)
        FPR = FP / (TN + FP)

    if (Precision == 0 and TPR_Recall == 0):
        pass
    else:
        F1_score = 2 * ((Precision * TPR_Recall) / (Precision + TPR_Recall))
    total_CM = TP + TN + FP + FN
    model_accuracy = (TP + TN) / total_CM

    # output message for the whole testing
    print('There are', TP, 'cracks in', len(dataset_test), 'tested images')
    print('Confusion Matrix:')
    print('True Postive is', round(TP, 4))
    print('True Negative is', round(TN, 4))
    print('False Postive is', round(FP, 4))
    print('False Negative is', round(FN, 4))
    print('Accuracy is', round(model_accuracy, 4))
    print('Precision is', round(Precision, 4))
    print('Recall is', round(TPR_Recall, 4))
    print('F1 score', round(F1_score, 4))
    print('D00 Recall is', round(D00_Recall, 4))
    print('D00 Precision is', round(D00_Precision, 4))
    print('D00 Accuracy is', round(D00_Accuracy, 4))
    print('D00 F1 score is', round(D00_F1_Score, 4))
    print('D01 Recall is', round(D01_Recall, 4))
    print('D01 Precision is', round(D01_Precision, 4))
    print('D01 Accuracy is', round(D01_Accuracy, 4))
    print('D01 F1 score is', round(D01_F1_Score, 4))
    print('D10 Recall is', round(D10_Recall, 4))
    print('D10 Precision is', round(D10_Precision, 4))
    print('D10 Accuracy is', round(D10_Accuracy, 4))
    print('D10 F1 score is', round(D10_F1_Score, 4))
    print('D11 Recall is', round(D11_Recall, 4))
    print('D11 Precision is', round(D11_Precision, 4))
    print('D11 Accuracy is', round(D11_Accuracy, 4))
    print('D11 F1 score is', round(D11_F1_Score, 4))
    print('D20 Recall is', round(D20_Recall, 4))
    print('D20 Precision is', round(D20_Precision, 4))
    print('D20 Accuracy is', round(D20_Accuracy, 4))
    print('D20 F1 score is', round(D20_F1_Score, 4))
    print('D30 Recall is', round(D30_Recall, 4))
    print('D30 Precision is', round(D30_Precision, 4))
    print('D30 Accuracy is', round(D30_Accuracy, 4))
    print('D30 F1 score is', round(D30_F1_Score, 4))
    print('D40 Recall is', round(D40_Recall, 4))
    print('D40 Precision is', round(D40_Precision, 4))
    print('D40 Accuracy is', round(D40_Accuracy, 4))
    print('D40 F1 score is', round(D40_F1_Score, 4))
    print('D43 Recall is', round(D43_Recall, 4))
    print('D43 Precision is', round(D43_Precision, 4))
    print('D43 Accuracy is', round(D43_Accuracy, 4))
    print('D43 F1 score is', round(D43_F1_Score, 4))
    print('D44 Recall is', round(D44_Recall, 4))
    print('D44 Precision is', round(D44_Precision, 4))
    print('D44 Accuracy is', round(D44_Accuracy, 4))
    print('D44 F1 score is', round(D44_F1_Score, 4))
    print('Total Inference Time is %0.2f second' % total_infer_time)
    print('Average Inference Time is %0.2f second' % (total_infer_time / all_tests))

    # sorting the all true positive and negative arrays to plot ROC curve
    TPR_array.sort()
    FPR_array.sort()
    D00_TPR_array.sort()
    D00_FPR_array.sort()
    D01_TPR_array.sort()
    D01_FPR_array.sort()
    D10_TPR_array.sort()
    D10_FPR_array.sort()
    D11_TPR_array.sort()
    D11_FPR_array.sort()
    D20_TPR_array.sort()
    D20_FPR_array.sort()
    D30_TPR_array.sort()
    D30_FPR_array.sort()
    D40_TPR_array.sort()
    D40_FPR_array.sort()
    D43_TPR_array.sort()
    D43_FPR_array.sort()
    D44_TPR_array.sort()
    D44_FPR_array.sort()

    # Title
    plt.title('ROC Plot')
    # Axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.plot(FPR_array, TPR_array)
    plt.plot(D00_FPR_array, D00_TPR_array, color='blue', linewidth=3, label='D00')
    plt.plot(D01_FPR_array, D01_TPR_array, color='green', linewidth=3, label='D01')
    plt.plot(D10_FPR_array, D10_TPR_array, color='orange', linewidth=3, label='D10')
    plt.plot(D11_FPR_array, D11_TPR_array, color='black', linewidth=3, label='D11')
    plt.plot(D20_FPR_array, D20_TPR_array, color='cyan', linewidth=3, label='D20')
    plt.plot(D30_FPR_array, D30_TPR_array, color='magenta', linewidth=3, label='D30')
    plt.plot(D40_FPR_array, D40_TPR_array, color='lavender', linewidth=3, label='D40')
    plt.plot(D43_FPR_array, D43_TPR_array, color='lime', linewidth=3, label='D43')
    plt.plot(D44_FPR_array, D44_TPR_array, color='coral', linewidth=3, label='D44')
    plt.legend()
    plt.show()


main()