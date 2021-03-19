from keras.layers import (
    Dense,
    Activation,
    Dropout,
    Flatten,
    MaxPooling2D,
    Convolution2D,
)
from keras.models import Sequential
from keras.optimizers import RMSprop


class ConvolutionNet:
    def __init__(self, input_dimension, loss_function, metrics):
        self._input_dim = input_dimension
        self._model = None
        self._loss_function = loss_function
        self._metrics = metrics
        self.create_network()

    def create_network(self):
        model = Sequential()

        model.add(
            Convolution2D(32, 3, 3, border_mode="same", input_shape=self._input_dim)
        )
        model.add(Activation("relu"))

        model.add(Convolution2D(32, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(64, 3, 3, border_mode="same"))
        model.add(Activation("relu"))

        model.add(Convolution2D(64, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(128, 3, 3, border_mode="same"))
        model.add(Activation("relu"))

        model.add(Convolution2D(128, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(256, 3, 3, border_mode="same"))
        model.add(Activation("relu"))

        model.add(Convolution2D(256, 3, 3, border_mode="same"))
        model.add(Activation("relu"))
        model.add(Dropout(0.25))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(256))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(1))
        model.add(Activation("sigmoid"))

        optimizer = RMSprop(lr=1e-4)

        model.compile(
            loss=self._loss_function, optimizer=optimizer, metrics=self._metrics
        )
        self._model = model

    def get_model(self):
        return self._model
