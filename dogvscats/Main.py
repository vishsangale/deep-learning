import os

from keras.callbacks import EarlyStopping, Callback
from keras.preprocessing.image import ImageDataGenerator

from ConvNet import ConvolutionNet
from Postprocess import predict, create_output_file
from Preprocess import read_training_data, read_test_data, CHANNELS, ROWS, COLS, TRAIN_DIR, TRAINING_SIZE, \
    VALIDATION_SIZE

NUMBER_OF_EPOCHS = 20
BATCH_SIZE = 32
NUMBER_OF_CLASSES = 2
use_generator = True


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


if __name__ == '__main__':

    net = ConvolutionNet((CHANNELS, ROWS, COLS), 'binary_crossentropy', ['accuracy'])

    model = net.get_model()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    history = LossHistory()

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

    y_predict = predict(model, X_test)

    create_output_file(y_predict)
