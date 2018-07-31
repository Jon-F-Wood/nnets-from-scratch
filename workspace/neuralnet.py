import numpy as np
from components import *

"""Sets the activation function and initializes the parameters and cache
for one layer of a neural network. The weights for each incoming connection are 
randomly chosen from a normal distribution with variance 1/num_output
where num_output is the number of nodes in this layer. The biases are all
initialized to zero. Finally, the cache is instantiated with the inputs and 
midsteps set to None.

Arguments: 
    activation {Activation} -- The activation function for this layer's output
    num_input (int) -- The number of nodes connecting to this layer from the prior layer
    num_output (int) -- The number of nodes in this layer

Returns:
    Layer -- The initialized layer
"""
def initialize_layer(activation, num_input, num_output):
    pass

"""Initializes the a complete neural network by iteratively initializing each layer
of the network. 

Arguments:
    activations {list[Activation]} -- The activation functions used for every hidden and output layer
    layer_sizes {list[int]} -- The number of nodes in each layer, starting with the input layer.
    cost {Cost} -- The cost function used to assess performance for this network.

Returns:
    [type] -- [description]
"""
def initialize_network(activations, layer_sizes, cost):
    pass

"""Forward propagation for a single layer of a neural network: the outputs for each 
node in the layer are computed from an activated linear combination of the
outputs from the previous layer. Specifically, the linear combination z_i for the 
ith node in this layer is 

    z_i = Sum_j (W_i,j * X_j) + b_i

Then, the linear combination is passed into the activation function g to compute the output
(also called activation a_i for node i)

    a_i = g(z_i)

When extended to multiple examples, the result for example k is given by replacing

    x_j -> X_j,k
    z_i -> Z_i,k

And the bias b_i is broadcast to size (num_i x num_k) in the linear combination.

While this computation is taking place, two pieces are cached for later: the inputs 
and the linear combinations (or pre-activations) Z_i,k.

Arguments:
    inputs {(num_input x num_examples) array} -- The inputs from the previous layer
    layer {Layer} -- the layer to perform forward propagation through

Returns:
    (num_output x num_examples) array -- This layer's activations (or outputs)
"""
def forward_propagation(inputs, layer):
    pass

"""Full forward propation of an initial input layer to the final activation layer.
Specifically, each layer performs forward propagation where the output of one layer 
is the input into the next.

Arguments:
    inputs {(num_input x num_examples) array} -- The input dependent variables
    layer {Layer} -- The network to perform forward propagation through

Returns:
    (num_output x num_examples) array -- The final layer's activations
"""

def full_forward_propagation(inputs, network):
    pass

"""Backward propagation through a single layer of a neural network: the slope
of the cost function with respect to each weight and bias is computed by applying
the chain rule. Starting with the slope of the cost function with respect to this
layer's outputs, the derivative with respect to the mid-activation, each parameter,
and the inputs are computed.

dL/dz = dL/da * da/dz; a = g(z)
dL/dw = dL/dz * dz/dw = dL/dz * x
dL/db = dL/dz * dz/db = dL/dz
dL/dx = dL/dz * dz/dx = dL/dz * w

The weights and biases are then updated using gradient descent:

param -= learning_rate * dL/dparam

Arguments:
    d_outputs {(num_output x num_examples) array} -- the derivative of the cost/loss with respect to this layers outputs
    layer {Layer} -- the layer to update with back prop
    rate {float} -- the learning rate for gradient descent

Returns:
    d_inputs {(num_inputs x num_examples) array} -- the derivative of the cost/loss with respect to this layer's inputs,
        which also serve as the previous layer's outputs.
"""
def backward_propagation(d_outputs, layer, rate):
    pass

"""Executes backward propagation through every layer of a neural network. The first
derivative is computed from the derivative of the cost function with respect to the
final activation layer. Then the layers are updated in reverse order, computing the
derivatives with respect to the prior layer's activations on each step.

Arguments:
    predicted {(num_outputs x num_examples) array} -- the final activation layer's predictions
    actual {(num_outputs x num_examples) array} -- the actual observed values to match predictions against
    network {Network} -- the network to update
    rate {float} -- the learning rate for gradient descent
"""
def full_backward_propagation(predicted, actual, network, rate):
    pass





