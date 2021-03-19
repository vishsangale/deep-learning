import os

import h5py
import numpy as np
from keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.models import Sequential

from Preprocess import (
    ROWS,
    COLS,
    TRAIN_DIR,
    TRAINING_SIZE,
    VALIDATION_SIZE,
    read_test_data,
    FEATURES_TEST_NPY,
    FEATURES_VALIDATION_NPY,
    FEATURES_TRAIN_NPY,
    BATCH_SIZE,
)


def get_trained_train_features():
    train_data = np.load(open(FEATURES_TRAIN_NPY))
    train_labels = np.array([0] * (TRAINING_SIZE / 2) + [1] * (TRAINING_SIZE / 2))
    return train_data, train_labels


def get_trained_validation_features():
    validation_data = np.load(open(FEATURES_VALIDATION_NPY))
    validation_labels = np.array(
        [0] * (VALIDATION_SIZE / 2) + [1] * (VALIDATION_SIZE / 2)
    )
    return validation_data, validation_labels


class VGG:
    def __init__(self, input_dimension):
        self._input_dimension = input_dimension
        self._model = None

    def create_VGG16(self):
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=self._input_dimension))

        model.add(Convolution2D(64, 3, 3, activation="relu", name="conv1_1"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, 3, 3, activation="relu", name="conv1_2"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation="relu", name="conv2_1"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, 3, 3, activation="relu", name="conv2_2"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation="relu", name="conv3_1"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation="relu", name="conv3_2"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(256, 3, 3, activation="relu", name="conv3_3"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation="relu", name="conv4_1"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation="relu", name="conv4_2"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation="relu", name="conv4_3"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation="relu", name="conv5_1"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation="relu", name="conv5_2"))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(512, 3, 3, activation="relu", name="conv5_3"))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self._model = model

    def load_model(self, model_type="VGG16"):
        if model_type == "VGG16":
            print "Creating VGG16 network..."
            self.create_VGG16()
            weights_path = "../weights/vgg16_weights.h5"
        assert os.path.exists(
            weights_path
        ), 'Model weights not found (see "weights_path" variable in script).'
        f = h5py.File(weights_path)
        for k in range(f.attrs["nb_layers"]):
            if k >= len(self._model.layers):
                # we don't look at the last (fully-connected) layers in the savefile
                break
            g = f["layer_{}".format(k)]
            weights = [g["param_{}".format(p)] for p in range(g.attrs["nb_params"])]
            self._model.layers[k].set_weights(weights)
        f.close()
        print ("Model loaded.")

    def create_bottleneck_features(self, imageGenerator):
        if not os.path.isfile(FEATURES_TRAIN_NPY):
            generator = imageGenerator.flow_from_directory(
                os.path.join(TRAIN_DIR, "train"),
                target_size=(ROWS, COLS),
                batch_size=BATCH_SIZE,
                class_mode=None,
                shuffle=False,
            )
            bottleneck_features_train = self._model.predict_generator(
                generator, TRAINING_SIZE
            )
            np.save(open(FEATURES_TRAIN_NPY, "w"), bottleneck_features_train)

        if not os.path.isfile(FEATURES_VALIDATION_NPY):
            generator = imageGenerator.flow_from_directory(
                os.path.join(TRAIN_DIR, "validation"),
                target_size=(ROWS, COLS),
                batch_size=BATCH_SIZE,
                class_mode=None,
                shuffle=False,
            )
            bottleneck_features_validation = self._model.predict_generator(
                generator, VALIDATION_SIZE
            )
            np.save(open(FEATURES_VALIDATION_NPY, "w"), bottleneck_features_validation)

    def get_model(self):
        return self._model

    def get_trained_test_features(self):
        if not os.path.isfile(FEATURES_TEST_NPY):
            test_data = read_test_data("X_test.npy")
            bottleneck_features_test = self._model.predict(test_data)
            np.save(open(FEATURES_TEST_NPY, "w"), bottleneck_features_test)
        return np.load(open(FEATURES_TEST_NPY))
