import numpy as np
from matplotlib import pyplot, pyplot as plt
import random
import pickle
import gzip

# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return np.exp(-x) / (np.exp(-x) + 1) ** 2


# neural network
class NeuralNetwork:
    def __init__(self, input_size, input_bias, hidden_size, hidden_bias, output_size, path_to_data):
        # constructor
        self.input_size = input_size
        self.input_bias = input_bias
        self.hidden_size = hidden_size
        self.hidden_bias = hidden_bias
        self.output_size = output_size
        self.path_to_data = path_to_data

        self.input_hidden_weights = np.random.rand(input_size, hidden_size) * 0.01          # randomly set weights
        self.hidden_output_weights = np.random.rand(hidden_size, output_size) * 0.01

        self.train_X, self.train_Y, self.test_X, self.test_Y, self.validation_X, self.validation_Y = load_data(path_to_data)

        self.train_Y = convert_to_one_hot(self.train_Y)
        self.test_Y = convert_to_one_hot(self.test_Y)
        self.validation_Y = convert_to_one_hot(self.validation_Y)


    def hiddenOutputNeuralNetwork(self, X):
        # output o_tj (after activation function), output a_tj (without)
        a_hidden = np.dot(X, self.input_hidden_weights) + self.input_bias
        o_hidden = sigmoid(a_hidden)
        return o_hidden, a_hidden

    def outputNeuralNetwork(self, hidden_output):
        # output o_tj (after activation function), output a_tj (without)
        a_output = np.dot(hidden_output, self.hidden_output_weights) + self.hidden_bias
        o_output = sigmoid(a_output)
        return o_output, a_output

    def run_neural_network(self, X):
        hidden_output, _ = self.hiddenOutputNeuralNetwork(X)
        output, _ = self.outputNeuralNetwork(hidden_output)
        return output

    def lossFunction(self, X, Y):
        loss = 0
        for i in range(len(X)):
            loss += np.linalg.norm(self.run_neural_network(X[i]) - Y[i]) ** 2
        loss = loss / (2*len(X))
        return loss


    def backPropagation(self, X, Y):
        # Forward pass
        o_hidden, a_hidden = self.hiddenOutputNeuralNetwork(X)
        o_output, a_output = self.outputNeuralNetwork(o_hidden)

        # Calculate error at output layer
        error_output = o_output - Y  # Derivative of loss w.r.t. output
        delta_output = error_output * sigmoid_derivative(a_output)  # Delta for output layer

        # Calculate error at hidden layer
        error_hidden = np.dot(delta_output, self.hidden_output_weights.T)
        delta_hidden = error_hidden * sigmoid_derivative(a_hidden)  # Delta for hidden layer

        # Gradients
        grad_hidden_output_weights = np.outer(o_hidden.T, delta_output)         # outer product to create matrix of gradients (with size of input x output)
        grad_input_hidden_weights = np.outer(X.T, delta_hidden)

        return grad_input_hidden_weights, grad_hidden_output_weights, delta_hidden, delta_output


    def update_parameters(self, grad_input_hidden_weights, grad_hidden_output_weights, learning_rate):
        self.input_hidden_weights -= learning_rate * grad_input_hidden_weights
        self.hidden_output_weights -= learning_rate * grad_hidden_output_weights



    def SGD(self, X, Y, epochs, miniBatchSize, learningRate):
        training_history = []
        for epoch in range(epochs):
            XY = list(zip(X, Y))  # keep label and data together
            random.shuffle(XY)  # randomly shuffle data

            mini_batches = [XY[k:k + miniBatchSize] for k in range(0, len(XY), miniBatchSize)]

            for mini_batch in mini_batches:
                X_mini, Y_mini = zip(*mini_batch)
                X_mini = np.array(X_mini)
                Y_mini = np.array(Y_mini)

                # Initialize gradient accumulators
                grad_input_hidden_weights = np.zeros_like(self.input_hidden_weights)
                grad_hidden_output_weights = np.zeros_like(self.hidden_output_weights)

                for i in range(len(X_mini)):
                    # Backpropagation to compute gradients
                    grad_ih, grad_ho, _, _ = self.backPropagation(X_mini[i], Y_mini[i])
                    grad_input_hidden_weights += grad_ih
                    grad_hidden_output_weights += grad_ho

                # Update weights with accumulated gradients
                self.update_parameters(grad_input_hidden_weights / len(X_mini), grad_hidden_output_weights / len(X_mini), learningRate)

            # Optionally, print the loss at every epoch
            loss = self.lossFunction(X, Y)
            print(f"Epoch {epoch}: Loss = {loss}")
            training_history.append(loss)
        return training_history

    def predict(self, X_test, Y_test):
        prediction = self.run_neural_network(X_test)

        # Get the indices of the maximum values in each row
        prediction_label = np.argmax(prediction, axis=1)

        true_label = convert_back(Y_test)

        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        cm = confusion_matrix(true_label, prediction_label)

        _, ax = plt.subplots(figsize=(8, 6))

        # Display the normalized confusion matrix
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title(f'Confusion Matrix')
        # plt.savefig(f'Confusion Matrix {name}')
        plt.show()

    def fgsm_attack(self, epsilon, miniBatchSize=128):
        # Ensure the images are of float type
        X = self.train_X.astype(np.float32)
        Y = self.train_Y.astype(np.float32)

        # Batch processing similar to your logic
        XY = list(zip(X, Y))  # keep label and data together
        random.shuffle(XY)  # randomly shuffle data

        mini_batches = [XY[k:k + miniBatchSize] for k in range(0, len(XY), miniBatchSize)]
        perturbed_images = np.zeros_like(X)  # To store all perturbed images

        batch_idx = 0  # Initialize a batch index counter

        for mini_batch in mini_batches:
            X_mini, Y_mini = zip(*mini_batch)
            X_mini = np.array(X_mini)
            Y_mini = np.array(Y_mini)

            # Forward pass
            o_hidden, a_hidden = self.hiddenOutputNeuralNetwork(X_mini)
            o_output, a_output = self.outputNeuralNetwork(o_hidden)

            # Calculate the loss for the mini-batch (MSE loss)
            loss = self.lossFunction(X_mini, Y_mini)

            # Backpropagation to get gradients w.r.t. inputs for this mini-batch
            error_output = o_output - Y_mini
            delta_output = error_output * sigmoid_derivative(a_output)

            # Propagate the gradient to the hidden layer
            error_hidden = np.dot(delta_output, self.hidden_output_weights.T)
            delta_hidden = error_hidden * sigmoid_derivative(a_hidden)

            # Compute gradient w.r.t. input using delta_hidden and input_hidden_weights
            grad_input = np.dot(delta_hidden, self.input_hidden_weights.T)

            # Ensure that grad_input matches the shape of X_mini
            if grad_input.shape != X_mini.shape:
                print(f"Shape mismatch: grad_input shape {grad_input.shape} does not match X_mini shape {X_mini.shape}")
                continue

            # For FGSM, we use the gradient with respect to input data (grad_input)
            data_grad = grad_input

            # Apply the FGSM attack by adding the sign of the gradient scaled by epsilon
            perturbed_batch = X_mini + epsilon * np.sign(data_grad)

            # Clip the perturbed images to keep valid image values (between 0 and 1)
            perturbed_batch = np.clip(perturbed_batch, 0, 1)

            # Store the perturbed batch in the full perturbed images array
            start_idx = batch_idx * miniBatchSize
            perturbed_images[start_idx:start_idx + len(perturbed_batch)] = perturbed_batch

            # Increment the batch index
            batch_idx += 1

        return perturbed_images





    def attack(self):
        return 0

def load_data(path):  # Data loading by Vaishnavi
    with gzip.open(path, 'rb') as file_contents:
        train_set, validation_set, test_set = pickle.load(file_contents, encoding='latin1')

        # x input and y gives output of the corresponding data
        train_X, train_Y = train_set
        test_X, test_Y = test_set
        validation_X, validation_Y = validation_set

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
    # Get the index of the maximum value in each row
    numbers = np.argmax(one_hot_array, axis=1)
    return numbers

def plot_learning_success(learning_history):
    plt.plot(learning_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss / Average Error')
    plt.title('Learning Success per Epoch')
    plt.show()

# if __name__ == "__main__":
#     # just for testing the existing code
#     testNetwork = NeuralNetwork(784, 1, 30, 1, 10, r'mnist.pkl.gz')
#     # loss of the random distribution of weights
#     print(testNetwork.lossFunction(testNetwork.train_X, testNetwork.train_Y))
    
#     # actual training of the neural network
#     history = testNetwork.SGD(testNetwork.train_X,testNetwork.train_Y,10,1,0.01)
#     plot_learning_success(history)
#     testNetwork.predict(testNetwork.validation_X, testNetwork.validation_Y)
    

if __name__ == "__main__":
    # Initialize the network
    testNetwork = NeuralNetwork(784, 1, 30, 1, 10, r'mnist.pkl.gz')
    
    # Perform FGSM attack with epsilon = 0.1
    epsilon = 0.1
    attacked_X = testNetwork.fgsm_attack(epsilon)

    
    
    # Evaluate the network on the adversarial examples
    print("Loss on adversarial examples:", testNetwork.lossFunction(attacked_X, testNetwork.train_Y))
    
    # Train the neural network on the original data
    history = testNetwork.SGD(testNetwork.train_X, testNetwork.train_Y, 10, 1, 0.01)
    
    # Optionally, plot the learning curve
    plot_learning_success(history)
    
    # Test the network on adversarial examples
    testNetwork.predict(attacked_X, testNetwork.train_Y)
