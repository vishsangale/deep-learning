import os

import numpy as np
from scipy.misc import imread, imresize

TRAIN_DIR = '../datasets/dogsvscats/_pre_train'
TEST_DIR = '../datasets/dogsvscats/test'

ROWS = 150
COLS = 150
CHANNELS = 3
TRAINING_SIZE = 20002
VALIDATION_SIZE = 4998

TESTING_SIZE = 12500

FEATURES_TEST_NPY = 'bottleneck_features_test.npy'
FEATURES_VALIDATION_NPY = 'bottleneck_features_validation.npy'
FEATURES_TRAIN_NPY = 'bottleneck_features_train.npy'
NUMBER_OF_EPOCHS = 20
BATCH_SIZE = 32
NUMBER_OF_CLASSES = 2
TOP_MODEL_WEIGHTS_PATH = 'fc_model.h5'
use_generator = True
use_pre_trained = True


def read_image(file_path):
    img = imread(file_path, mode='RGB')
    img = imresize(img, (ROWS, COLS), interp='cubic')
    img = img.astype('float32')
    img /= 255.0
    return img


def extract_train_data(path):
    print 'Extracting train data...'
    data = np.ndarray((TRAINING_SIZE+VALIDATION_SIZE, CHANNELS, ROWS, COLS), dtype=np.float32)
    y = np.ndarray(TRAINING_SIZE+VALIDATION_SIZE, dtype=np.uint8)
    count = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            image = read_image(os.path.join(root, name))
            data[count] = image.T
            if 'cat' in name:
                y[count] = 0
            else:
                y[count] = 1
            count += 1
    np.save('X_train', data)
    np.save('y_train', y)


def extract_test_data(path):
    print 'Extracting test data...'
    data = np.ndarray((TESTING_SIZE, CHANNELS, ROWS, COLS), dtype=np.float32)
    count = 0
    files = range(1, TESTING_SIZE + 1)
    for i in files:
        image = read_image(os.path.join(path, str(i) + '.jpg'))
        data[count] = image.T
        count += 1
    np.save('X_test', data)


def read_test_data(x_test_file):
    if not os.path.isfile('X_test.npy'):
        extract_test_data(TEST_DIR)
    return np.load(x_test_file)


def read_training_data(x_file, y_file):
    if not os.path.isfile('X_train.npy') or not os.path.isfile('y_train.npy'):
        extract_train_data(TRAIN_DIR)
    return np.load(x_file), np.load(y_file)


