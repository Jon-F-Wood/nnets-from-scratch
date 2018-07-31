"""Pairs an activation function with it's derivative"""
class Activation:
    def __init__(self, f, df):
        self.f = f
        self.df = df

"""The set of parameters for one layer of a neural network"""
class Params:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

"""The last set of inputs and pre-activations used in forward propagation for a layer"""
class Cache:
    def __init__(self, inputs, midstep):
        self.inputs = inputs
        self.midstep = midstep

"""Encapsulates all the properties of a single layer in a deep neural network"""
class Layer:
    def __init__(self, activation, params, cache):
        self.activation = activation
        self.params = params
        self.cache = cache

"""Pairs a cost function with it's derivative with respect to the predicted outputs"""
class Cost:
    def __init__(self, f, df):
        self.f = f
        self.df = df

"""Encapsulates all the properties of a deep neural network"""
class Network:
    def __init__(self, layers, cost):
        self.layers = layers
        self.cost = cost