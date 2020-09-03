from PIL import Image
from PIL import ImageDraw
from IOU import get_iou

loaded_model = get_model(num_classes=8)
loaded_model.load_state_dict(torch.load("/content/drive/My Drive/Models/model_all_cities_BS14"))


def visual_image(index_of_image):
    num_total_cracks = 0
    num_cracks = 0
    num_passed_iou = 0
    TP = 0
    FP = 0
    FN = 0
    Precision = 0
    Recall = 0
    F1_score = 0
    iou_array = []

    img, _ = dataset_test[index_of_image]
    # print(dataset_test[index_of_image])
    label_boxes = np.array(dataset_test[index_of_image][1]["boxes"])

    # put the model in evaluation mode
    loaded_model.eval()

    with torch.no_grad():
        prediction = loaded_model([img])

    # print(prediction)

    image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(image)

    num_groundtruth_obj = len(label_boxes)

    # draw groundtruth
    for elem in range(len(label_boxes)):
        # print(label_boxes[elem])
        draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]), (label_boxes[elem][2], label_boxes[elem][3])],
                       outline="green", width=3)
        # draw.text((label_boxes[elem][0], label_boxes[elem][1]), text=str('Hi'))

    for element in range(len(prediction[0]["boxes"])):
        # print(prediction[0]["boxes"])
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        confidence = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)
        # print(prediction[0]["boxes"][element])

        if confidence >= 0.7:
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=3)
            draw.text((boxes[0], boxes[1]), text=str(confidence))
            num_cracks += 1
            num_total_cracks += 1

            max_iou = 0

            for box in range(len(label_boxes)):
                draft_cal_iou = get_iou(prediction[0]["boxes"][element], label_boxes[box])
                cal_iou = draft_cal_iou.data.cpu().numpy()
                if (cal_iou > max_iou):
                    max_iou = cal_iou
                    # print('max_iou', max_iou)

            if (max_iou >= 0.5):
                num_passed_iou += 1
                iou_array.append(max_iou)

    # print('****************************************************************')
    # print(num_groundtruth_obj)
    # print(num_cracks)
    # print(num_passed_iou)
    TP = num_passed_iou
    FP = num_cracks - num_passed_iou
    FN = num_groundtruth_obj - num_passed_iou

    if ((TP == 0 and FP == 0) or (TP == 0 and FN == 0)):
        pass
    else:
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)

    if (Precision == 0 and Recall == 0):
        pass
    else:
        F1_score = 2 * ((Precision * Recall) / (Precision + Recall))

    display(image)
    if num_cracks == 1:
        print('There is', num_cracks, 'road crack in this image')
    elif num_cracks > 1:
        print('There are', num_cracks, 'road cracks in this image')
    else:
        print('There is no crack')

    print('TP', TP)
    print('FP', FP)
    print('FN', FN)
    print('Precision', round(Precision, 4))
    print('Recall', round(Recall, 4))
    print('F1 score', round(F1_score, 4))

    return num_total_cracks, Precision, Recall, F1_score, iou_array


def main():
    total = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_average_precision = 0
    total_average_recall = 0
    total_average_f1 = 0
    for index in range(len(dataset_test)):
        num_total_cracks, Precision, Recall, F1_score, iou_array = visual_image(index)
        total += num_total_cracks
        total_precision += Precision
        total_recall += Recall
        total_f1 += F1_score
        num_of_images = len(dataset_test)
        total_average_precision = total_precision / num_of_images
        total_average_recall = total_recall / num_of_images
        total_average_f1 = total_f1 / num_of_images
        if index > 0:
            print(index + 1, 'predictions are done')
        else:
            print(index + 1, 'prediction is done')

        for ind in range(len(iou_array)):
            iou = float(iou_array[ind])
            print('IOU:', round(iou, 4))

    print('There are', total, 'cracks in', len(dataset_test), 'tested images')
    print('Total average precision is', round(total_average_precision, 4))
    print('Total average recall is', round(total_average_recall, 4))
    print('Total average F1 score', round(total_average_f1, 4))


main()
