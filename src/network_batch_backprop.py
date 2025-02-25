from __future__ import absolute_import
"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library

import random

import src.graph_plot as graph
# Third-party libraries
import numpy as np
import sys
import pickle

class Network_batch(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, x_train, y_train, epochs, eta, x_test, y_test):
        "Train the network with batch backpropagation"
        "Batch Propagation updates weights when we run our first epoch on entire data"
        if x_test: n_test = len(x_test)
        n = len(x_train)
        # print(x_train[0])
        # print(y_train[0])
        # print("Length of Test Data : ", n_test)
        # print("Length of Train Data : ", n)
        accuracy = []
        loss = []
        for j in range(epochs):
            s = np.arange(len(x_train))
            random.shuffle(s)
            x_train = list(np.array(x_train)[s])
            y_train = list(np.array(y_train)[s])
            training_data = zip(x_train, y_train)
            self.update_batch(training_data, eta, n)
            if x_test:
                acc = self.evaluate(zip(x_test, y_test))
                accuracy.append(acc/100)
                loss.append(1-(acc/100))
                print("Epoch {0}: {1} / {2}".format(j, acc, n_test))
            else:
                print("Epoch {0} complete".format(j))
        self.save_weights()
        g = graph.Graph()
        g.accuracy_plot(accuracy, 'network_back_accuracy')
        g.loss_plot(loss, 'network_back_loss')

    def update_batch(self, training_data, eta, len_training_data):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # for i in range(len(nabla_w)):
        #
        #     print(i,len(nabla_b[i]), len(nabla_w[i]))
        count = 0
        for x, y in training_data:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            count+=1
        # print("Training of {} much data has completed".format(count))
        self.weights = [w-(eta/len_training_data)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len_training_data)*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            # print("Shape of z : ", z.shape)
            zs.append(z)
            activation = sigmoid(z)
            # print("Shape of activation : ", activation.shape)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        # print("Delta : ", delta)
        # print("Shape of sigmoid_prime : ",sigmoid(zs[-1]))
        nabla_b[-1] = delta
        # print("Shape of Delta : ", delta.shape)
        # print("Shape of activations[-2].transpose() ",activations[-2].transpose().shape)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # print("nabla_w weight :" ,nabla_w[-1].shape)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def save_weights(self):
        pickle_out = open(sys.path[0]+"/network_batch_weights/weights.pickle","wb")
        pickle.dump(self.weights, pickle_out)
        pickle_out.close()

        pickle_out = open(sys.path[0]+"/network_batch_weights/biases.pickle","wb")
        pickle.dump(self.biases, pickle_out)
        pickle_out.close()
        print("*"*10+" Weights Saved "+"*"*10)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
