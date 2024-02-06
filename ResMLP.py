import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

class ResidualMLPClassifier:
    def __init__(self):
        self.model = None
        self.history = None

    def load_mnist_data(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train, y_train, x_test, y_test

    def ResMLP_get_input_val(self, x_train):
        width = x_train.shape[1]
        height = x_train.shape[2]
        input_val = width * height
        return input_val

    def ResMLP_scale_image_size(self, x_train, x_test, input_val):
        x_train = np.reshape(x_train, [-1, input_val])
        x_train = x_train.astype('float32') / 255

        x_test = np.reshape(x_test, [-1, input_val])
        x_test = x_test.astype('float32') / 255

        return x_train, x_test

    