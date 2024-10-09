# %% Main Class and helper functions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
import pickle
import gzip


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative of activation function
def sigmoid_derivative(x):
    return np.exp(-x) / (np.exp(-x) + 1) ** 2


# loss function
def square_loss(prediction, Y):
    # Predictions should be only the final ones (predictions[-1])
    # Calculate loss function from difference between predictions and true labels
    return np.sum(np.linalg.norm(prediction - Y) ** 2) / (2 * len(Y))


# derivation of loss function
def square_loss_derivative(activation, Y):
    return (activation - Y)


# Main neural network class
class NeuralNetwork:
    def __init__(self, layer_sizes, activation_function, activation_function_derivative, loss_function,
                 loss_derivative):
        self.layer_sizes = layer_sizes  # List of all layer sizes, e.g., [784, 64, 32, 10]
        self.num_layers = len(layer_sizes)
        # creates random biases between every layer
        self.biases = [
            np.random.rand() * 0.01
            for i in range(self.num_layers - 1)
        ]
        # creates random weights between every layer and saves it in weights.
        self.weights = [
            np.random.rand(layer_sizes[i], layer_sizes[i + 1]) * 0.01
            for i in range(self.num_layers - 1)
        ]
        print(f"Weights between: {self.weights[1].shape}")

        # Custom activation functions and custom loss functions
        self.activation_function = activation_function
        self.activation_function_derivative = activation_function_derivative
        self.loss_function = loss_function
        self.loss_derivative = loss_derivative

    def run_neural_network(self, X):
        # create matrices with the activated output and the unactivated output of each layer (so size: layer number-1)
        activation_output = [X]
        pre_activation_output = []

        for i in range(len(self.layer_sizes) - 1):
            a_hidden = np.dot(activation_output[i], self.weights[i]) + self.biases[i]
            o_hidden = sigmoid(a_hidden)
            activation_output.append(o_hidden)
            pre_activation_output.append(a_hidden)

        return activation_output, pre_activation_output

    def lossFunction(self, predictions, Y):
        return self.loss_function(predictions, Y)

    def loss_derivativeFunction(self, activation, Y):
        return self.loss_derivative(activation, Y)

    def backPropagation(self, X, Y):
        # calculation of final output of the neural network
        activation_output, pre_activation_output = self.run_neural_network(X)
        activation = activation_output[-1]
        loss_derivative = self.loss_derivativeFunction(activation, Y)
        # declaration of all necessary variables
        error = []
        delta = []
        gradient = []
        # loss of last layer initialized
        error_output = loss_derivative
        error.append(error_output)
        delta_output = error[-1] * self.activation_function_derivative(pre_activation_output[-1])
        delta.append(delta_output)

        # calculating all errors and subsequently all deltas for the number of hidden layers
        for i in range(len(self.layer_sizes) - 2):
            error_output = np.dot(delta[i], self.weights[(len(self.layer_sizes) - 1) - (i + 1)].T)
            error.append(error_output)

            delta_output = error[i + 1] * self.activation_function_derivative(
                pre_activation_output[(len(self.layer_sizes) - 1) - (i + 2)])
            delta.append(delta_output)

        # stores all the gradient values from input to output
        for i in range(len(self.layer_sizes) - 1):
            grad_layers = np.outer(activation_output[i].T, delta[len(self.layer_sizes) - (i + 2)])
            gradient.append(grad_layers)

        # for the update of the biases, the delta is used because of the chain rule, the multiplication with the output o does not appear
        # reverse delta so its also stored from input to output
        delta.reverse()

        return error, delta, gradient

    def SGD(self, X, Y, epochs, miniBatchSize, learningRate):  # Stochastic Gradient Descent
        training_history = []  # List to store the loss at every epoch to plot later

        for epoch in range(epochs):
            XY = list(zip(X, Y))  # keep label and data together
            random.shuffle(XY)  # randomly shuffle data

            # Create the minibatches with given minibatchsize
            mini_batches = [XY[k:k + miniBatchSize] for k in range(0, len(XY), miniBatchSize)]

            for mini_batch in mini_batches:
                X_mini, Y_mini = zip(*mini_batch)
                X_mini = np.array(X_mini)
                Y_mini = np.array(Y_mini)

                # Initialize gradient accumulators

                gradient = [np.zeros_like(self.weights[i]) for i in range(len(self.layer_sizes) - 1)]
                bias_gradient = [np.zeros_like(self.biases[i]) for i in range(len(self.layer_sizes) - 1)]

                for i in range(len(X_mini)):
                    # Backpropagation to compute gradients
                    _, delta, grad_ih = self.backPropagation(X_mini[i], Y_mini[i])

                    for j in range(len(self.weights)):
                        gradient[j] += grad_ih[j]  # Updating the gradient accumulators

                    for j in range(len(self.biases)):
                        bias_gradient[j] += np.sum(delta[j])

                # When minibatch is finished, update weights with accumulated gradients
                # a replacement for update_parameters
                for j in range(len(self.weights)):  # Update for all layers
                    self.weights[j] -= learningRate * (gradient[j] / len(X_mini))

                # update of the biases to train them as well
                for j in range(len(self.biases)):  # Update for all layers
                    self.biases[j] -= learningRate * (bias_gradient[j] / len(X_mini))

            # Print the loss at every epoch
            predictions, _ = self.run_neural_network(X)  # Run predictions
            loss = self.lossFunction(predictions[-1], Y)
            print(f"Epoch {epoch}: Loss = {loss}, biases = {self.biases}")
            training_history.append(loss)  # Append the loss to the training history

        return training_history  # Return training history to be plotted

    def plot_confusion_matrix(self, true_label, prediction_label):
        # Create the confusion matrix
        cm = confusion_matrix(true_label, prediction_label)

        # Initialising plots
        _, ax = plt.subplots(figsize=(8, 6))

        # Display the confusion matrix
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f'Confusion Matrix')
        plt.show()

    def predict(self, X_test, Y_test):
        # function to run the neural network on X and compare the predictions to Y including outputting a confusion matrix
        prediction, _ = self.run_neural_network(X_test)
        prediction = prediction[-1]

        # Convert both labels back from one hot encoding
        prediction_label = convert_back(prediction)
        true_label = convert_back(Y_test)

        self.plot_confusion_matrix(true_label, prediction_label)

    def predict_bitwise(self, X_test, Y_test):
        # function to run the neural network on X and compare the predictions to Y including outputting a confusion matrix
        # this time in bitwise form
        prediction, _ = self.run_neural_network(X_test)
        prediction = prediction[-1]

        # Convert the values from the float values returned by the sigmoid activation to bits
        prediction_bits = np.where(prediction >= 0.5, 1, 0)

        prediction_label = convert_bit_back(prediction_bits)
        true_label = convert_bit_back(Y_test)

        self.plot_confusion_matrix(true_label, prediction_label)

    def fgsm_attack(self, epsilon, X, Y, miniBatchSize=128):
        error = [None] * (len(self.layer_sizes) - 1)
        delta = [None] * (len(self.layer_sizes) - 1)
        # Method to implement an FGSM attack on the neural network
        XY = list(zip(X, Y))  # keep label and data together

        # Split into mini batches
        mini_batches = [XY[k:k + miniBatchSize] for k in range(0, len(XY), miniBatchSize)]

        perturbed_images = np.zeros_like(X)  # To store all perturbed images

        batch_idx = 0  # Initialise a batch index counter

        for mini_batch in mini_batches:
            X_mini, Y_mini = zip(*mini_batch)
            X_mini = np.array(X_mini)
            Y_mini = np.array(Y_mini)

            # Forward pass through the whole network
            activation_output, pre_activation_output = self.run_neural_network(X_mini)

            # Backpropagation to get gradients w.r.t. inputs for this mini-batch
            error_output = activation_output[-1] - Y_mini
            error[0] = error_output
            delta_output = error[0] * self.activation_function_derivative(pre_activation_output[-1])
            delta[0] = delta_output

            # Propagate the gradient to the hidden layer
            for i in range(len(self.layer_sizes) - 2):
                error_output = np.dot(delta[i], self.weights[(len(self.layer_sizes) - 1) - (i + 1)].T)
                error[i + 1] = error_output

                delta_output = error[i + 1] * self.activation_function_derivative(
                    pre_activation_output[(len(self.layer_sizes) - 1) - (i + 2)])
                delta[i + 1] = delta_output

            # Compute gradient w.r.t. input using delta_hidden and input_hidden_weights
            grad_input = np.dot(delta[-1], self.weights[0].T)

            # Ensure that grad_input matches the shape of X_mini
            if grad_input.shape != X_mini.shape:
                print(f"Shape mismatch: grad_input shape {grad_input.shape} does not match X_mini shape {X_mini.shape}")
                continue

            # For FGSM, we use the gradient with respect to input data (grad_input)
            data_grad = grad_input

            # Apply the FGSM attack by adding the sign of the gradient scaled by epsilon
            perturbed_batch = X_mini + epsilon * np.sign(data_grad)

            # Store the perturbed batch in the full perturbed images array
            start_idx = batch_idx * miniBatchSize
            perturbed_images[start_idx:start_idx + len(perturbed_batch)] = perturbed_batch

            # Increment the batch index
            batch_idx += 1

        # Ensure that the datatype of the perturbed images is the same as the original input
        assert perturbed_images.dtype == X.dtype, "Datatype mismatch between original and perturbed images."

        # If epsilon is 0, check if the output is the same as the input
        if epsilon == 0:
            assert np.array_equal(perturbed_images,
                                  X), "With epsilon=0, perturbed images should be identical to original input."

        return perturbed_images

    def augment_training_data(self, X_train, Y_train, epsilon=0.1, miniBatchSize=128):

        # Generating adversarial examples from the training data and combining them with the original data.

        print("Generating adversarial examples...")
        X_adv = self.fgsm_attack(epsilon, X_train, Y_train, miniBatchSize)

        # Combine the original and adversarial data
        X_augmented = np.vstack((X_train, X_adv))
        Y_augmented = np.vstack((Y_train, Y_train))  # Labels remain the same

        return X_augmented, Y_augmented


# %% Data loading and preprocessing
def load_data(path, train_size=50000):
    # Data loading, with custom train_size if needed
    with gzip.open(path, 'rb') as file_contents:  # unzip the data
        train_set, validation_set, test_set = pickle.load(file_contents, encoding='latin1')  # load the sets

        # x input and y gives output of the corresponding data
        train_X, train_Y = train_set
        test_X, test_Y = test_set
        validation_X, validation_Y = validation_set

        # Randomly sample from the training set
        indices = np.random.choice(len(train_X), train_size, replace=False)
        train_X = train_X[indices]
        train_Y = train_Y[indices]

        print(f"Train data shape: {train_X.shape}, Train labels shape: {train_Y.shape}")
        print(f"Sample label: {train_Y[2]}")  # This will show the integer label (0-9)

        return train_X, train_Y, test_X, test_Y, validation_X, validation_Y


def convert_to_one_hot(numbers):
    # Create an array of zeros with shape (len(numbers), 10)
    one_hot_array = np.zeros((len(numbers), 10), dtype=int)

    # Set the position of each number in the array to 1
    for i, number in enumerate(numbers):
        one_hot_array[i][number] = 1

    return one_hot_array


def convert_back(one_hot_array):
    # Convert back from one hot encoding to a list of labels, automatically takes the biggest number as the label
    numbers = np.argmax(one_hot_array, axis=1)
    return numbers


def convert_to_bitwise(numbers):
    # Create a list to store the labels in 4 digit bitwise format
    bitwise_labels = []

    for number in numbers:
        # Convert the label to a 4-bit binary representation, strip the '0b' prefix and pad with leading zeros
        binary_rep = format(number, '04b')
        # Convert the string binary representation into a list of integers
        bitwise_vector = [int(bit) for bit in binary_rep]
        # Append the bitwise vector to the list
        bitwise_labels.append(bitwise_vector)

    return bitwise_labels


def convert_bit_back(bitwise_list):
    # Convert back from bitwise encoding to a list of labels
    labels = []

    for bitwise_vector in bitwise_list:
        # Convert the list of bits back into a binary string
        binary_str = ''.join(map(str, bitwise_vector))
        # Convert the binary string to its decimal equivalent
        label = int(binary_str, 2)
        # Append the result to the labels list
        labels.append(label)

    return labels


def plot_learning_success(learning_history):
    # Plot the learning success against epochs
    plt.plot(learning_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Average Error')
    plt.title('Learning Success per Epoch')
    plt.show()


def display_attacked_images(attacked_set):
    # Select 10 random indices from attacked_set to display
    indices = np.random.choice(attacked_set.shape[0], 5, replace=False)

    # Create a figure with 10 subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 10))

    for i, index in enumerate(indices):
        # Reshape the flattened image into a 2D array
        image = attacked_set[index].reshape(28, 28)

        # Plot the image in a subplot
        axes[i].imshow(image, cmap='gray')
        axes[i].axis('off')  # Hide the axis

    plt.show()