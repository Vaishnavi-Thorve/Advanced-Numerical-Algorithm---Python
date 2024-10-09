import gzip
import pickle
import numpy as np

input_size = 784
hidden_size = 30
output_size = 10
epochs = 15
mini_batch_size = 30
learning_rate = 0.1

################ Load the data #############

# reads the file as binary in read mode
# r ensures raw data to avoid unicode error
# encoding latin1 represents what internal language is used in the data

def load_data(file: str):
    with gzip.open(r'C:\Users\91884\Downloads/mnist.pkl.gz', 'rb') as file_contents:
        train_set, validation_set, test_set = pickle.load(file_contents, encoding='latin1')

        # x input and y gives output of the corresponding data
        train_X, train_Y = train_set
        test_X, test_Y = test_set
        validation_X, validation_Y = validation_set

        print(f"Train data shape: {train_X.shape}, Train labels shape: {train_Y.shape}")
        print(f"Sample label: {train_Y[2]}")  # This will show the integer label (0-9)

        return train_X, train_Y, test_X, test_Y, validation_X, validation_Y


######### FNN ####################

# Definition of sigmoid function
def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))


# Deriative of the activation function (sigmoid)
def sigmoid_derivative(x):
    return sigmoid_function(x) * (1 - sigmoid_function(x))


class FNN:
    def _init_(self, input_size, hidden_size, output_size):
        # Xavier/Glorot Initialization
        # ensures that the weights are scaled based on the number of neurons in the layers
        self.hidden_weights = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
        self.hidden_bias = np.zeros((1, hidden_size))

        self.output_weights = np.random.randn(output_size, hidden_size) * np.sqrt(1.0 / output_size)
        self.output_bias = np.zeros((1, output_size))