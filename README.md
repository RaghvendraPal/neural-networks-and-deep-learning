# Code samples for "Neural Networks and Deep Learning"

TO Predict the result on image:
user need to run models_prediction.py in cmd using following command:
python models_prediction.py -p image_path

like example python models_prediction.py -p G:\Machine-Learning\github\neural-networks-and-deep-learning\test_image\2.pngcode

TO RUN the assignment -
user needs to run models_training.py in cmd using command 'python models_training.py'.



network.py has the for mini batch BackPropagation
network_batch_backprop.py has the for batch BackPropagation
cnn_network.py has the for batch CNN Model

from the graph network_accuracy.png that shows accuracy for mini batch BackPropagation and network_back_accuracy.png that shows accuracy for Batch BackPropagation, we can conclude that batch Propagation takes lot of time to converge as compare to mini batch propagation.

because in batch propagation we are getting 62% accuracy i 200 epoch but in mini batch backpropagation we are getting 94% accuracy in only 30 epochs.
