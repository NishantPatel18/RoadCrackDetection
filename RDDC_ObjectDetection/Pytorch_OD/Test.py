from PIL import Image
from PIL import ImageDraw

loaded_model = get_model(num_classes=8)
loaded_model.load_state_dict(torch.load("/content/RoadCrackDetection/RDDC_ObjectDetection/Pytorch_OD/road_crack/model"))


def visual_image(index_of_image):
    num_total_cracks = 0
    num_cracks = 0
    img, _ = dataset_test[index_of_image]
    label_boxes = np.array(dataset_test[index_of_image][1]["boxes"])

    # put the model in evaluation mode
    loaded_model.eval()

    with torch.no_grad():
        prediction = loaded_model([img])

    image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
    draw = ImageDraw.Draw(image)

    # draw groundtruth
    for elem in range(len(label_boxes)):
        draw.rectangle([(label_boxes[elem][0], label_boxes[elem][1]), (label_boxes[elem][2], label_boxes[elem][3])],
                       outline="green", width=3)

    for element in range(len(prediction[0]["boxes"])):
        boxes = prediction[0]["boxes"][element].cpu().numpy()
        score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)

        if score > 0.75:
            draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=3)
            draw.text((boxes[0], boxes[1]), text=str(score))
            num_cracks += 1
            num_total_cracks += 1

    display(image)
    if num_cracks == 1:
        print('There is', num_cracks, 'road crack in this image')
    elif num_cracks > 1:
        print('There are', num_cracks, 'road cracks in this image')
    else:
        print('There is no crack')

    return num_total_cracks


def main():
    total = 0
    for index in range(len(dataset_test)):
        indi = visual_image(index)
        total += indi
        if index > 0:
            print(index + 1, 'predictions are done')
        else:
            print(index + 1, 'prediction is done')

    print('There are', total, 'cracks in', len(dataset_test), 'tested images')


main()
