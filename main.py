from __future__ import absolute_import
import src.network as network
import src.mnist_loader as mnist_loader

x_train, y_train, x_validation, y_validation, x_test, y_test = mnist_loader.load_data_wrapper()
net = network.Network([784,20,40,10])
net.SGD(x_train, y_train, 30, 10, 3.0, x_test, y_test)
