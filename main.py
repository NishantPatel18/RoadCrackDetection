import file_handler as fh
import neural_network as nn


# Getting content from Positive and Negative inside of Train
train_positive = fh.get_content('Train/Positive')
train_negative = fh.get_content('Train/Negative')

# Getting content from Positive and Negative inside of Test
test_positive = fh.get_content('Test/Positive')
test_negative = fh.get_content('Test/Negative')

# creating a neural network
print(nn.create_nn())
