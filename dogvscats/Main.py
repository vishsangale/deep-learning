import os
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Flatten, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils.visualize_util import plot

from ConvNet import ConvolutionNet
from Postprocess import create_output_file
from Preprocess import read_training_data, read_test_data, CHANNELS, ROWS, COLS, TRAIN_DIR, TRAINING_SIZE, \
    VALIDATION_SIZE, NUMBER_OF_EPOCHS, BATCH_SIZE, TOP_MODEL_WEIGHTS_PATH, use_generator, use_pre_trained
from VGG import VGG, get_trained_train_features, get_trained_validation_features


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


def add_top_model(model):
    top_model = Sequential()
    top_model.add(Flatten(input_shape=model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu', W_regularizer=l2(0.1)))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    return top_model


def show_graph(history):
    plt.plot(history.losses)
    plt.plot(history.val_losses)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    history = LossHistory()
    if not use_pre_trained:
        net = ConvolutionNet((CHANNELS, ROWS, COLS), 'binary_crossentropy', ['accuracy'])
        model = net.get_model()
        if use_generator:
            data_generator = ImageDataGenerator(
                rescale=1. / 255,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True)

            train_generator = data_generator.flow_from_directory(
                os.path.join(TRAIN_DIR, 'train'),
                target_size=(ROWS, COLS),
                batch_size=BATCH_SIZE,
                class_mode='binary')
            validation_generator = data_generator.flow_from_directory(
                os.path.join(TRAIN_DIR, 'validation'),
                target_size=(ROWS, COLS),
                batch_size=BATCH_SIZE,
                class_mode='binary')

            model.fit_generator(generator=train_generator,
                                samples_per_epoch=TRAINING_SIZE,
                                nb_epoch=NUMBER_OF_EPOCHS,
                                callbacks=[early_stopping, history],
                                validation_data=validation_generator,
                                nb_val_samples=VALIDATION_SIZE,
                                verbose=1)
        else:
            X_train, y_train = read_training_data('X_train.npy', 'y_train.npy')

            model.fit(X_train, y_train,
                      batch_size=BATCH_SIZE,
                      nb_epoch=NUMBER_OF_EPOCHS,
                      validation_split=0.2,
                      verbose=1,
                      shuffle=True,
                      callbacks=[history, early_stopping])

        X_test = read_test_data('X_test.npy')

    else:
        print 'Using pre-trained VGG16'
        image_generator = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
        vgg_net = VGG((CHANNELS, ROWS, COLS))
        vgg_net.load_model("VGG16")
        vgg_net.create_bottleneck_features(image_generator)
        model = vgg_net.get_model()

        X_train, y_train = get_trained_train_features()

        X_valid, y_valid = get_trained_validation_features()

        top_model = add_top_model(model)
        top_model.load_weights(TOP_MODEL_WEIGHTS_PATH)
        model.add(top_model)
        for layer in model.layers[:25]:
            layer.trainable = False

        optimizer = RMSprop()
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)

        train_generator = train_datagen.flow_from_directory(
            os.path.join(TRAIN_DIR, 'train'),
            target_size=(ROWS, COLS),
            batch_size=BATCH_SIZE,
            class_mode='binary')

        test_datagen = ImageDataGenerator(rescale=1. / 255)

        validation_generator = test_datagen.flow_from_directory(
            os.path.join(TRAIN_DIR, 'validation'),
            target_size=(ROWS, COLS),
            batch_size=BATCH_SIZE,
            class_mode='binary')

        model.fit_generator(train_generator,
                            samples_per_epoch=TRAINING_SIZE,
                            nb_epoch=NUMBER_OF_EPOCHS,
                            validation_data=validation_generator,
                            nb_val_samples=VALIDATION_SIZE,
                            callbacks=[early_stopping])

        X_test = read_test_data('X_test.npy')

    y_predict = model.predict(X_test, verbose=1)

    create_output_file(y_predict)
    show_graph(history)

    plot(model, to_file='model.png', show_shapes=True)
