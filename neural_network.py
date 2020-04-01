from torch import nn


# create_nn() creates a model architecture
def create_nn():
    model = nn.Sequential(
        nn.Linear(16, 12),
        nn.ReLU(),
        nn.Linear(12, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    return model


# Testing model architecture
# print(create_nn())
