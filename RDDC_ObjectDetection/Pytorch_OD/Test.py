loaded_model = get_model(num_classes=2)
loaded_model.load_state_dict(torch.load("/content/Pytorch_OD/raccoon/model"))

idx = 0
img, _ = dataset_test[idx]
label_boxes = np.array(dataset_test[idx][1]["boxes"])

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

    if score > 0.8:
        draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=3)
        draw.text((boxes[0], boxes[1]), text=str(score))

image