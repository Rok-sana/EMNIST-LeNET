import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report
import tensorflow_datasets as tfds
import numpy as np


class DataLoader:
    def __init__(self, name):
        self.dataset_name = name
        self.ds_builder = tfds.builder(name)
        self.ds_train, self.ds_test = tfds.load(name, split=['train', 'test'], shuffle_files=True, as_supervised=True,
                                                with_info=False)

    def preprocess(self, image, label):
        # Cast image to float32 and normalize pixel values to range [0, 1]
        image = tf.cast(image, tf.float32) / 255.0

        # Reshape image to (height, width, num_channels)
        image = tf.reshape(image, [28, 28, 1])

        # One-hot encode label
        num_classes = self.ds_builder.info.features['label'].num_classes
        label = tf.one_hot(label, num_classes)

        return image, label

    def load_dataset(self):
        self.ds_train = self.ds_train.map(self.preprocess)
        self.ds_test = self.ds_test.map(self.preprocess)

        # Reshape train and test data to (num_samples, height, width, num_channels)
        trainX = np.array(list(self.ds_train.map(lambda x, y: x)))
        trainY = np.array(list(self.ds_train.map(lambda x, y: y)))
        testX = np.array(list(self.ds_test.map(lambda x, y: x)))
        testY = np.array(list(self.ds_test.map(lambda x, y: y)))

        return trainX, trainY, testX, testY

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if keras.backend.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        model.add(Conv2D(20, (5, 5), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model



# Load EMNIST dataset
data_loader = DataLoader('emnist/balanced')
trainX, trainY, testX, testY = data_loader.load_dataset()

# Build and compile LeNet model
num_clases = data_loader.ds_builder.info.features['label'].num_classes
model = LeNet.build(width=28, height=28, depth=1, classes=num_clases)
opt = SGD(learning_rate=0.01)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train model
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=20, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)


model.save("model.h5")
print("Model saved to disk.")
