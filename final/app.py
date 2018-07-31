import numpy as np
from sklearn.datasets import make_moons
import neuralnet as nnet
from matplotlib import pyplot as plt
from utilities import *

"""Loads dataset for training and testing a simple neural network. Currently 
the data set is generated using the moon distribution provided by scikit-learn.
The data set has two features, with data classified into two groups based on these
features.

Arguments:
    num_training {int} -- the number of training examples to generate
    num_testing {int} -- the number of testing examples to generate
    noise {float} -- the amount of noise (from 0 to 1) to introduce into each moon distribution

Returns:
    tuple -- A tuple containing the training and test data sets respectively. 
    Each data set is another tuple consisting of the (2 x num_examples) independent
    features array and the (num_examples x 1) dependent classifications array.
"""

def load_data(num_training, num_test, noise):
    X_train, Y_train = make_moons(num_training, noise=noise)
    X_test, Y_test = make_moons(num_test, noise=noise)
    return (X_train.T, Y_train), (X_test.T, Y_test)

"""Initializes and trains a neural network based on the provided specification and 
training data set. To help monitor the training process the average cost is printed
and eventually plotted as a function of the number of iterations.

Arguments:
    X {(num_features x num_train) array} -- The independent variables for the training data set
    Y {(num_train x 1) array} -- The dependent variable for the training data set
    activations {list[Activation]} -- the activation functions to use on each hidden and output layer
    layer_sizes {list[int]} -- The size of the input, all hidden, and output layers, respectively.
    cost_function {Cost} -- The cost function used to measure performance
    num_iterations {int} -- The number of training passes to perform
    delta {int} -- The number of iterations to perform between evaluatiions of the cost function
    learning_rate {float} -- The learning rate for gradient descent

Returns:
    Network -- The trained neural network
"""
def train_model(X, Y, activations, layer_sizes, cost_function, num_iterations, delta, learning_rate):
    network = nnet.initialize_network(activations, layer_sizes, cost_function)

    iterations = []
    costs = []
    for i in range(num_iterations):
        predicted = nnet.full_forward_propagation(X, network)
        nnet.full_backward_propagation(predicted, Y, network, learning_rate)
        
        if i%delta == 0:
            iterations.append(i)
            cost = cost_function.f(predicted, Y)
            costs.append(cost)
            print("Cost for iteration %i is %e" % (i, cost))

    plot_cost(iterations, costs, learning_rate)
    return network

"""Utility method for plotting the cost as a function of the iteration number.

Arguments:
    iterations {list[int]} -- The iteration numbers that the cost function is evaluated at.
    costs {list[float]} -- The cost of the neural network on the training set at each iteration.
    learning_rate {float} -- The learning rate used for the training session being plotted.
"""
def plot_cost(iterations, costs, learning_rate):   
    plt.figure(0)
    plt.plot(iterations, costs)
    plt.ylabel('Cost')
    plt.xlabel('Iteration #')
    plt.title('Learning Rate = %.6f' % learning_rate)
    plt.show()

"""Creates a utility function used to predict the output for a given input using a trained neural network.
Note, the output of the utility function should always correspond to the most likely result, not the probability
of one result vs another.

Arguments:
    network {Network} -- the trained neural network to make predictions for

Returns:
    predictor {function} -- A function that accepts an input array (num_input_features x num_examples) and
    outputs the neural networks prediction.
"""
def create_predictor(network):
    def predictor(inputs):
        return np.round(nnet.full_forward_propagation(inputs, network))
    return predictor

"""Calculates the percentage of predictions that match expectations.

Arguments:
    X {(num_features x num_examples) array} -- the independent variables of the dataset
    Y {(num_examples x 1) array} -- the dependent variable of the dataset (i.e. the expected values)
    predictor {function} -- A function that makes predictions based on X

Returns:
    accuracy {float} -- The percentage of predictions that match expectations
"""
def get_accuracy(X, Y, predictor):
    return np.sum((predictor(X) == Y)/X.shape[1])

"""Utility method for plotting the decision boundary formed by a predictor function over a two feature
 dataset's domain. Plots the decision boundaries as a countour plot with the original data overlayed as
a scattor plot.

Arguments:
    X {(num_features x num_examples) array} -- the independent variables of the dataset
    Y {(num_examples x 1) array} -- the dependent variable of the dataset
    predictor {function} -- A function that makes predictions based on X
"""
def plot_boundaries(X, Y, predictor):
    x_min, x_max = X[0].min() - .5, X[0].max() + .5
    y_min, y_max = X[1].min() - .5, X[1].max() + .5
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = predictor(np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[0], X[1], c=Y, cmap=plt.cm.Spectral)

"""Main method of this module. The application loads a training and test set, train's the
neural network, measures it's accuracy on both sets, and plots the decision boundary against
 the training and test sets.
"""
def main():
    training, test = load_data(200, 100, 0.2)
    network = train_model(training[0], training[1], 
        [relu_activation, relu_activation, sigmoid_activation], [2,5,5,1],
        cross_entropy_cost, 40000, 500, 0.01)
    
    predictor = create_predictor(network)
    print("Training Accuracy: %.6f" % get_accuracy(training[0], training[1], predictor))
    print("    Test Accuracy: %.6f" % get_accuracy(test[0], test[1], predictor))

    plt.figure(1)
    plot_boundaries(training[0], training[1], predictor)
    plt.title("Training Boundary")

    plt.figure(2)
    plot_boundaries(test[0], test[1], predictor)
    plt.title("Test Boundary")

if __name__=="__main__":
    main()