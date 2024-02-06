import numpy as np
from tensorflow.keras.datasets import mnist

class MY_MLP_Model:
    def __init__(self):
        self.weights_input_hidden = None
        self.bias_hidden = None

    def load_model(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        return x_train, y_train, x_test, y_test
    
    def define_input_size(self, x_train):
        image_size = x_train.shape[1]
        # This is the total count of the features for each image. 
        # In MNIST Dataset, each image is 28 pixels wide and 28 pixels tall.
        input_size = image_size ** 2
        return input_size

    # By flattening the image pixels, this method converts a 2D structure into a 1D structure by taking all the pixel values 
    # in the 28x28 grid and arranging them in a single line which is ht input_size
    def resize_training_data(self, x_train, x_test):
        input_size = self.define_input_size(x_train)
        resized_x_train = np.reshape(x_train, [-1, input_size])
        resized_x_test = np.reshape(x_test, [-1, input_size])
        return resized_x_train, resized_x_test

    # the following method changes the data type to float32 which was before integer
    # it scales the image between 0 and 1, each pixel in the image contains the value between 0 and 255
    # 0 means being completely black (dark) and 255 means being completely white (bright)
    def normalize_training_data(self, x_train, x_test):
        resized_x_train, resized_x_test = self.resize_training_data(x_train, x_test)
        normalize_x_train = resized_x_train.astype('float32') / 255
        normalize_x_test = resized_x_test.astype('float32') / 255
        return normalize_x_train, normalize_x_test
    
    def preprocess_labels(self, y_train, y_test):
        num_labels = len(np.unique(y_train))
        one_hot_y_train = self.to_one_hot_encoding(y_train)
        one_hot_y_test = self.to_one_hot_encoding(y_test)
        return one_hot_y_train, one_hot_y_test, num_labels
    
    # The following method I write replaces all labeled values in y_train and y_test with one-hot encoding. 
    # For instance, it changes the value of 4 to [0, 0, 0, 0, 1 0, 0, 0, 0, 0].
    def to_one_hot_encoding(self, datasets):
        set_all_zeros = np.zeros(10, dtype=int)
        changed_to_one_hot = []

        for i in range(len(datasets)):
            for j in range(10):
                if datasets[i] == j:
                    set_all_zeros[j] = 1
                    changed_to_one_hot.append(set_all_zeros)
                    set_all_zeros = np.zeros(10, dtype=int)
                    break
        return changed_to_one_hot
    
    def relu(self, x):
        return np.maximum(0, x)

    def dropout(self, x, rate):
        mask = np.random.rand(*x.shape) < rate
        return x * mask / (1 - rate)
    
    def set_hyperparameters(self, x_train):
        input_size = self.define_input_size(x_train)
        hidden_units = 256
        self.weights_input_hidden = np.random.randn(input_size, hidden_units)
        self.bias_hidden = np.ones((hidden_units,), dtype=float)
        batch_size = 128
        learning_rate = 0.05
        epoch = 20
        return batch_size, epoch, learning_rate


    def train_model(self, x_train, y_train):
        batch, epoch, learning_rate = self.set_hyperparameters(x_train)
        train_features = x_train.shape[1]
        self.weights_input_hidden = np.zeros((train_features, 10))
        self.bias_hidden = 0

        for _ in range(epoch):
            activation = np.dot(x_train, self.weights_input_hidden) + self.bias_hidden      

            y_predict = self.step_function(activation)

            update_weight = learning_rate * np.dot(x_train.T, (y_train - y_predict))
            update_bias = learning_rate * np.sum(y_train - y_predict)

            self.weights_input_hidden += update_weight
            self.bias_hidden += update_bias
        return self.weights_input_hidden, self.bias_hidden

    def step_function(self, x_train):
        return np.eye(10)[np.argmax(x_train, axis=1)].reshape(-1, 10)

    def predict(self, x_train):
        activate = np.dot(x_train, self.weights_input_hidden) + self.bias_hidden
        return self.step_function(activate)

# instantiate the My_MLP_Model class
model_call = MY_MLP_Model()
x_train, y_train, x_test, y_test = model_call.load_model()
one_hot_y_train, one_hot_y_test, num_labels = model_call.preprocess_labels(y_train, y_test)
x_train_normalized, x_test_normalized = model_call.normalize_training_data(x_train, x_test)

w, b = model_call.train_model(x_train_normalized[:1000], one_hot_y_train[:1000])

y_p_trained = model_call.predict(x_train_normalized)
y_p_test = model_call.predict(x_test_normalized)

print("Training Accuracy:  %.2f%%" % (100 - np.mean(np.abs(y_p_trained - one_hot_y_train)) * 100))
print("Testing Accuracy: %.2f%%" %  (100 - np.mean(np.abs(y_p_test - one_hot_y_test)) * 100))

prediction = model_call.predict(x_train_normalized)
predicted_one_hot = model_call.step_function(prediction)

true_one_hot = one_hot_y_train

accuracy = np.mean(np.equal(predicted_one_hot, true_one_hot))
print("Accuracy: %.2f%%" % (accuracy * 100))
