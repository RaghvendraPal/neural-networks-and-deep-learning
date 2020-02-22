from __future__ import absolute_import
import src.network as network
import src.network_batch_backprop as network_batch_b
import src.mnist_loader as mnist_loader
import src.cnn_network as cnn_network
import src.graph_plot as graph
# import src.mnist_data
x_train, y_train, x_validation, y_validation, x_test, y_test = mnist_loader.load_data_wrapper()
# # Using Mini Batch for Training the Model
print("*"*100)
print("Training Mini Batch BackPropagation")
print("*"*100)
net = network.Network([784,20,40,10])
net.SGD(x_train, y_train, 50, 10, 3.0, x_test, y_test)
# #
#Using Batch Propagation to train the network
print("*"*100)
print("Training Batch BackPropagation")
print("*"*100)
net = network_batch_b.Network_batch([784,20,40,10])
net.SGD(x_train, y_train, 500, 3.0, x_test, y_test)
print("*"*100)
#
#Using CNN Encoder with adding some noise to training data
print("*"*100)
print("Training CNN")
print("*"*100)
# parameters are epoch, batch_size, n_factor(what percentage of noise we need to add in data)
network = cnn_network.CNN_network(30, 128, 10)
print("Accuracy on Test Data : ",network.save_weights()[1]*100)
print("*"*100)
