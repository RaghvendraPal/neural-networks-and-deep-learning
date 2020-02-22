from __future__ import absolute_import
from optparse import OptionParser

import src.network_batch_prediction as network_batch_prediction
import src.network_prediction as network_prediction
import src.cnn_network as cnn_network

parser = OptionParser()
parser.add_option("-p", "--path", dest="image_path", help="Path to Image.")
(options, args) = parser.parse_args()
path = options.image_path

print("For Image Present : ", path, "Output is -")
print("*"*100)
print("Prediction using Mini Batch BackPropagation Model")
print("*"*100)
print("*"*100)
prediction = network_prediction.Prediction()
print("Predicted Image Value : ",prediction.load_image(path))
print("*"*100)
#
#Using Batch Propagation to train the network
print("*"*100)
print("Prediction Using Batch BackPropagation Model")
print("*"*100)
print("*"*100)
prediction = network_batch_prediction.Prediction()
print("Predicted Image Value : ",prediction.load_image(path))
print("*"*100)


#Using CNN Encoder with adding some noise to training data
print("*"*100)
print("Prediction Using CNN Model")
print("*"*100)
# parameters are epoch, batch_size, n_factor(what percentage of noise we need to add in data)
network = cnn_network.CNN_network(30, 128, 10)
print("*"*100)
print("Predicted Image Value : ",network.load_image(path))
print("*"*100)
