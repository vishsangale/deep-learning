from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, Flatten, Dense
from keras.models import Sequential

from Config import NUMBER_OF_FILTERS, KERNEL_SIZE, POOLING_SIZE, NUMBER_OF_CLASSES


class ConvolutionalNet:
    def __init__(self, dimension):
        self._input_dimension = dimension
        self._model = None

    def create_network(self):
        model = Sequential()
        model.add(Convolution2D(NUMBER_OF_FILTERS, KERNEL_SIZE[0], KERNEL_SIZE[1],
                                border_mode='valid',
                                input_shape=self._input_dimension))
        model.add(Activation('relu'))

        model.add(Convolution2D(NUMBER_OF_FILTERS, KERNEL_SIZE[0], KERNEL_SIZE[1]))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=POOLING_SIZE))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(NUMBER_OF_CLASSES))
        model.add(Activation('softmax'))

        self._model = model

    def get_model(self):
        return self._model
