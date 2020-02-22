"""
cnn_network
~~~~~~~~~~~~

A library to load the MNIST image data.
We have added noise to MNIST data and applied CNN on the data to predict the numbers
"""

#### Libraries
# Standard library
import pickle
import gzip
import keras
# Third-party libraries
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from random import randint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization

class CNN_network:
    def __init__(self, epochs, batch_size, n_factor):
        self.num_classes = 10
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_factor = n_factor
        # input image dimensions
        self.img_rows, self.img_cols = 28, 28
        self.input_shape = (self.img_rows, self.img_cols, 1)

    def load_data(self):
        """ Return MNIST Train and Test Data """

        f = gzip.open(sys.path[0]+'/data/mnist.pkl.gz', 'rb')
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
        f.close()
        print("Shape of training data : ",training_data[0].shape, training_data[1].shape)
        print("Shape of validation data : ",validation_data[0].shape, validation_data[1].shape)
        print("Shape of test data : ",test_data[0].shape, test_data[1].shape)
        x_train = training_data[0].reshape(training_data[0].shape[0], self.img_rows,self.img_cols,1)
        x_test = test_data[0].reshape(test_data[0].shape[0], self.img_rows,self.img_cols,1)

        # print(x_train.shape)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        # print('x_train shape:', x_train.shape)
        # print(x_train.shape[0], 'train samples')
        # print(x_test.shape[0], 'test samples')
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(training_data[1], self.num_classes)
        y_test = keras.utils.to_categorical(test_data[1], self.num_classes)
        # print('y_train shape:', y_train.shape)
        return x_train, y_train, x_test, y_test
        # return x_train, x_test, test_data)

    def add_noise(self):
        " Functio is used to add some amount of noise to the data"
        "n_factor will tells how much noise we need to add"
        x_train, y_train, x_test, y_test = self.load_data()
        # print(x_train[0].shape)
        # print(x_train[0])
        print("Adding "+str(self.n_factor)+" % noise to the MNIST data")
        self.n_factor = self.n_factor/100
        x_train_noisy = x_train + self.n_factor*np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
        x_test_noisy = x_test + self.n_factor*np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
        # print(x_train_noisy[0].shape)
        # print(x_train_noisy[0])
        x_train /= 255
        x_test /= 255
        x_train_noisy /= 255
        x_test_noisy /= 255
        value = randint(0, 100)
        plt.imshow(x_train[value,:,:,0])
        plt.title("Normal Number")
        plt.savefig(sys.path[0]+'/fig/Number.png')
        plt.show()
        plt.close()
        plt.title("Noise Number After adding "+str(self.n_factor)+" '%' of noise")
        plt.imshow(x_train_noisy[value,:,:,0])
        plt.savefig(sys.path[0]+'/fig/Noisy_number.png')
        plt.show()
        plt.close()
        return x_train_noisy, y_train, x_test_noisy, y_test

    def network(self):
        # Using CNN for MNIST data Training
        x_train_noisy, y_train, x_test_noisy, y_test = self.add_noise()
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])

        history = model.fit(x_train_noisy, y_train,
                  batch_size=self.batch_size,
                  epochs=self.epochs,
                  verbose=1,
                  validation_data=(x_test_noisy, y_test))
        score = model.evaluate(x_test_noisy, y_test, verbose=0)
        print("Accuracy of model on test data : ", score[1])
        return model, history, score

    def save_weights(self):
        # Storing the weights for further prediction
        model, history, score = self.network()
        pickle_out = open(sys.path[0]+"/model/model.pickle","wb")
        pickle.dump(model, pickle_out)
        pickle_out.close()

        pickle_out = open(sys.path[0]+"/model/history.pickle","wb")
        pickle.dump(history, pickle_out)
        pickle_out.close()
        print("*"*10+" Model Saved "+"*"*10)

        self.graph_plot()
        return score

    def graph_plot(self):
        #  "Accuracy"
        # functions generates Loss and Accuracy
        pickle_in = open(sys.path[0]+"/model/history.pickle","rb")
        history = pickle.load(pickle_in)
        pickle_in.close()
        # print(history)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('Accuracy Plot')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(sys.path[0]+"/fig/CNN_accuracy_plot_for_best_n_factor")
        # plt.show()
        plt.close()
        # "Loss"
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('Loss Plot')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(sys.path[0]+"/fig/CNN_loss_plot_for_best_n_factor")
        # plt.show()
        plt.close()

    def image_resize(self, image_path):
        img = cv2.imread(image_path)
        # print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        img = cv2.resize(img, (28, 28))
        img = np.reshape(img, (1,28,28,1))
        return img

    def evaluate(self, image_data):
        pickle_in = open(sys.path[0]+"/model/model.pickle","rb")
        model = pickle.load(pickle_in)
        pickle_in.close()

        " Function will return predicted number"
        return np.argmax(model.predict(image_data))

    def load_image(self, image_path):
        image = self.image_resize(image_path)
        return self.evaluate(image)
        # print(image.shape)
        # print(evaluate(image[0]))
