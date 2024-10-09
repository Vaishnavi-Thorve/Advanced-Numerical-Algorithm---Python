from project1 import NeuralNetwork, sigmoid, sigmoid_derivative, square_loss, square_loss_derivative, load_data
import numpy as np
import json

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

def save_results(filename, results):
    with open(filename, 'w') as f:
        json.dump(results, f)

if __name__ == '__main__':
    train_X, train_Y, test_X, test_Y, validation_X, validation_Y = load_data(r'mnist.pkl.gz')

    # Convert the labels to one hot encoding
    train_Y = convert_to_one_hot(train_Y)
    validation_Y = convert_to_one_hot(validation_Y)
    test_Y = convert_to_one_hot(test_Y)

    filename = r"search_output.json"

    results = []

    for batch_size in [1, 2, 8, 32, 128]: # laptop needs to restart at 1, 0.25, 30, new needs to restart at 2, 0.1, 45
        for learning_rate in range(5, 35, 5):
            learning_rate = learning_rate / 100
            for hidden_size in range(20,50,5):
                print(f"Testing batch_size={batch_size}, learning_rate={learning_rate}, hidden_size={hidden_size}")
                currNetwork = NeuralNetwork([784,hidden_size,10],[0.0,0.0],sigmoid,sigmoid_derivative, square_loss,square_loss_derivative)
                currNetwork.SGD(train_X, train_Y, 20, batch_size, learning_rate)
                prediction, _ = currNetwork.run_neural_network(validation_X)
                prediction = prediction[-1]

                # Convert both labels back from one hot encoding
                prediction_label = convert_back(prediction)
                true_label = convert_back(validation_Y)
                val_loss = currNetwork.lossFunction(prediction, validation_Y)

                result = {
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'hidden_size': hidden_size,
                    'val_loss': val_loss
                }

                results.append(result)
                save_results(filename, results)
