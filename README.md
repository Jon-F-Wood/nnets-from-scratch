# Building Neural Networks

The intention of this project is to serve as a workspace for learning about neural networks. This includes the before and after state of several exercises focused on forward and backward propagation, cost functions, vectorization, and evaluating trained models.

## Installation

The following python libraries must be installed using pip for all of the modules to function:

1) numpy: Th main computational workhorse of this project that focuses on n-dimensional arrays.
2) sklearn: A package containing convenience methods for machine learning. It's used to create the data sets for these exercises.
3) matplotlib: A plotting utility used to monitor performance of a neural network as and after it trains.

All of the above can be installed on a mac by running
```bash
install.sh
```
For Windows, the easiest option is to [install Anaconda](https://www.anaconda.com/download/) which will provide up to date versions of all of these libraries.

## Organization

The content is roughly split into two directories:
- *workspace*: <br>
The modules in this subdirectory have been gutted of their implementations, leaving behind the interface documentation and a few utilitiy scripts. <br><br>
- *final*: <br>
The modules in this subdirectory have been fully implemented and can act as a guide whenever one of the implementations in the workspace is to confusing or not behaving as expected.

In both sub directories there are four python modules:

- *app.py*: <br>
This module drives the program. The main method will load the data, train the model, and perform a very simple evaluation of the model's performance. 

- *neuralnet.py*: <br>
This module is the meat of the neural network algorithm. The methods within implement initialization of network parameters, forward propagation, and backward propagation.

- *components.py*: <br>
This module provides data classes for structuring the content of a neural network. This includes the various parameters, intermediary cached computations, and layers.

- *utilities.py* <br> 
This module contains utility methods for activation and cost functions. As a result, the module probably contains the most math while providing the least insight into a neural network to those not trained in mathematics.
