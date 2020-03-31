import torch
from SetFunctionUp import loading_data, train_model, test_model, make_model


epoch = 5
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
# device = 'cpu'
batch_size = 140

TrainLoader, TestLoader, TrainSet, TestSet = loading_data("RoadData", batch_size)

model = make_model()

TrainedModel = train_model(model, epoch, TrainLoader, device)

# test_model(model, Testloader, device)
test_model(TrainedModel, TestLoader, device)
