from keras.engine import Input
from keras.engine import Model
from keras.engine import merge
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam


class ConvolutionNet:
    def __init__(self, input_dimension, loss_function, metrics):
        self._input_dim = Input(input_dimension)
        self._model = None
        self._loss_function = loss_function
        self._metrics = metrics
        self.create_network()

    def create_network(self):
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(self._input_dim)
        conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
        conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
        conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
        conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
        conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

        up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
        conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

        up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
        conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

        up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
        conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

        up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
        conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

        conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

        model = Model(input=self._input_dim, output=conv10)
        opt = Adam(lr=1e-5)
        model.compile(optimizer=opt, loss=self._loss_function, metrics=self._metrics)

        self._model = model

    def get_model(self):
        return self._model
