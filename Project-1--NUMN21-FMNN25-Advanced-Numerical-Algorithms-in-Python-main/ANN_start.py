import numpy as np
import matplotlib.pyplot as plt
import pickle
import gzip

class ANN:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size  # The layer sizes do not count the +1 (Bias I think??)
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size

    def build_network(self):
        input_layer = np.zeros(self.input_layer_size + 1)  # Not sure if we need to initialise this, we might just create it from the data
        hidden_layer = np.zeros(self.hidden_layer_size + 1)  # This might need to be initialised with random numbers instead of zeros
        output_layer = np.zeros(self.output_layer_size)  # Output layer doesn't have a bias neuron

        input_layer[-1] = 1
        hidden_layer[-1] = 1



    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


class PreProcessing:
    def __init__(self, data_path):
        with gzip.open(r'C:\Users\91884\Downloads/mnist.pkl.gz', 'rb') as file_contents:
            train_set, validation_set, test_set = pickle.load(file_contents, encoding='latin1')

            # x input and y gives output of the corresponding data
            train_X, train_Y = train_set
            test_X, test_Y = test_set
            validation_X, validation_Y = validation_set

            print(f"Train data shape: {train_X.shape}, Train labels shape: {train_Y.shape}")
            print(f"Sample label: {train_Y[2]}")  # This will show the integer label (0-9)

            return train_X, train_Y, test_X, test_Y, validation_X, validation_Y


if __name__ == '__main__':
    test = ANN(784, 30, 10)
    # digits = np.arange(-15, 15, step=1)
    # plt.plot(digits, test.sigmoid(digits))
    # plt.show()

    test.build_network()
