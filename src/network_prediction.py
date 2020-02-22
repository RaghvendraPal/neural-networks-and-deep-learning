import cv2
import numpy as np
import sys
import pickle

class Prediction:
    # def __init__(self, path):
    #     self.image_path = path
    def image_resize(self, image_path):
        img = cv2.imread(image_path)
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        img = cv2.resize(img, (28, 28))
        img = np.reshape(img, (1,784,1))
        return img
    def feedforward(self, a):
        pickle_in = open(sys.path[0]+"/network_weights/weights.pickle","rb")
        weights = pickle.load(pickle_in)
        pickle_in.close()
        pickle_in = open(sys.path[0]+"/network_weights/biases.pickle","rb")
        biases = pickle.load(pickle_in)
        pickle_in.close()
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(biases, weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    def evaluate(self, image_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        return np.argmax(self.feedforward(image_data))

    def load_image(self, image_path):
        image = self.image_resize(image_path)
        return self.evaluate(image[0])
        # print(image.shape)
        # print(evaluate(image[0]))

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))
