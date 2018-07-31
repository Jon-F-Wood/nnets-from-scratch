import numpy as np
from components import Activation, Cost

"""The ReLU activation function and it's derivative. ReLU stands for
Rectified Linear Unit and is a function that returns x if x is positive,
and zero otherwise. Also, it's derivative is 1 if x is positive and 0 
otherwise.

Arguments:
    Z {(num_outputs x num_examples) array} -- The pre activation of a hidden or output layer

Returns:
    {(num_outputs x num_examples) array} -- The computed rectified linear unit for each cell of Z
"""
def relu(Z):
    return np.maximum(0, Z)
def d_relu(Z):
    return (Z > 0) * 1
relu_activation = Activation(relu, d_relu)

"""The sigmoid activation function and it's derivative. The sigmoid function
is characterized by a smooth, symmetric 'S' like curve that asymptotes to 0 for large negative inputs
and 1 for large positive inputs. The specific functional form is

f(x) = 1 / (1 + e^(-x))

f'(x) = e^(-x) / (1 + e^(-x))^2

Arguments:
    Z {(num_outputs x num_examples) array} -- The pre activation of a hidden or output layer

Returns:
    {(num_outputs x num_examples) array} -- The computed sigmoid for each cell of Z
"""
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))
def d_sigmoid(Z):
    S = sigmoid(Z)
    return S * (1-S)
sigmoid_activation = Activation(sigmoid, d_sigmoid)

"""The cross entropy cost function and it's derivative with respect to the
predicted value Y. Note: the derivative is technically missing an overall small
constant factor. This factor is often dropped to avoid small gradients and can
effectively be absorbed into the learning rate while performing training.

Arguments:
    A {(num_example, 1) array} -- the predicted values (last activation of the neural network)
    Y {(num_example, 1) array} -- the actual dependent values

Returns:
    float -- The calculated cross entropy (or its' derivative) over the entire training set
"""
def cross_entropy(A, Y):
    return -np.average(Y * np.log(A) + (1 - Y) * np.log(1 - A))
def d_cross_entropy(A, Y):
    return - (np.divide(Y, A) - np.divide(1 - Y, 1 - A))
cross_entropy_cost = Cost(cross_entropy, d_cross_entropy)