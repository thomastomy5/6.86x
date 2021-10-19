import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    # return np.where(x > 0, x, 0)
    if x>0:
        return x
    else:
        return 0


def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    # return np.where(x > 0, 1, 0)
    if x>0:
        return 1
    else:
        return 0


# x = np.matrix([[1],[2],[3],[-5]])
# print(rectified_linear_unit_derivative(x))


def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x


def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1


class NeuralNetwork():
    """
        Contains the following functions:
            -train: tunes parameters of the neural network based on error obtained from forward propagation.
            -predict: predicts the label of a feature vector based on the class's parameters.
            -train_neural_network: trains a neural network over all the data points for the specified number of epochs during initialization of the class.
            -test_neural_network: uses the parameters specified at the time in order to test that the neural network classifies the points given in testing_points within a margin of error.
    """

    def __init__(self):

        # DO NOT CHANGE PARAMETERS (Initialized to floats instead of ints)
        self.input_to_hidden_weights = np.matrix('1. 1.; 1. 1.; 1. 1.')
        self.hidden_to_output_weights = np.matrix('1. 1. 1.')
        self.biases = np.matrix('0.; 0.; 0.')
        self.learning_rate = .001
        self.epochs_to_train = 10
        self.training_points = [((2, 1), 10), ((3, 3), 21), ((4, 5), 32), ((6, 6), 42)]
        # self.training_points = [((-3, 5), 2), ((-2, -8), -10), ((-6, -3), -9), ((-1, 6), 5), ((-10, 5), -5), ((-6, -7), -13), ((-7, -10), -17),((8, 1), 9), ((6, -5), 1), ((7, -7), 0)]
        self.testing_points = [(1, 1), (2, 2), (3, 3), (5, 5), (10, 10)]

    def train(self, x1, x2, y):

        ### Forward propagation ###
        input_values = np.matrix([[x1], [x2]])  # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.dot(input_values.T, self.input_to_hidden_weights.T) +  self.biases.T # TODO (3 by 1 matrix)

        rectified_linear_unit_v = np.vectorize(rectified_linear_unit)
        hidden_layer_activation = rectified_linear_unit_v(hidden_layer_weighted_input)  # TODO (3 by 1 matrix)

        output = np.dot(hidden_layer_activation, self.hidden_to_output_weights.T)
        activated_output = output_layer_activation(output)

        ### Backpropagation ###

        # Compute gradients
        output_layer_activation_derivative_v = np.vectorize(output_layer_activation_derivative)

        output_layer_error = (y - activated_output) * output_layer_activation_derivative_v(activated_output)
        a = self.hidden_to_output_weights*np.asscalar(output_layer_error)

        rectified_linear_unit_derivative_v = np.vectorize(rectified_linear_unit_derivative)
        b = rectified_linear_unit_derivative_v(hidden_layer_weighted_input)
        hidden_layer_error = np.multiply(a, b)


        bias_gradients = hidden_layer_error
        hidden_to_output_weight_gradients = hidden_layer_activation * np.asscalar(output_layer_error)
        input_to_hidden_weight_gradients = (input_values * hidden_layer_error).T

        # Use gradients to adjust weights and biases using gradient descent
        # for _ in range(self.epochs_to_train):
        self.biases += self.learning_rate * bias_gradients.T
        self.input_to_hidden_weights += self.learning_rate * input_to_hidden_weight_gradients
        self.hidden_to_output_weights += self.learning_rate * hidden_to_output_weight_gradients

        # return self.biases, self.input_to_hidden_weights, self.hidden_to_output_weights

    def predict(self, x1, x2):                                 

        input_values = np.matrix([[x1], [x2]])

        # Compute output for a single input(should be same as the forward propagation in training)
        hidden_layer_weighted_input =  np.dot(input_values.T, self.input_to_hidden_weights.T)

        rectified_linear_unit_v = np.vectorize(rectified_linear_unit)
        hidden_layer_activation =  rectified_linear_unit_v(hidden_layer_weighted_input)

        output = np.dot(hidden_layer_activation, self.hidden_to_output_weights.T)
        activated_output =  output_layer_activation(output)

        return activated_output.item()

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x, y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7 * point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return


x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
x.test_neural_network()
