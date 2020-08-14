import file_handler as fh
import neural_network as nn


# Getting content from Crack and NonCrack inside of Train
train_positive = fh.get_content('Train/Crack')
train_negative = fh.get_content('Train/NonCrack')

# Getting content from Crack and NonCrack inside of Test
test_positive = fh.get_content('Test/Crack')
test_negative = fh.get_content('Test/NonCrack')

# creating a neural network
print(nn.create_nn())
