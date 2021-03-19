import matplotlib.pyplot as plt
from keras.callbacks import Callback, EarlyStopping
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.utils.visualize_util import plot

from Config import IMG_ROWS, IMG_COLS, NUMBER_OF_CLASSES, NUMBER_OF_EPOCHS, BATCH_SIZE
from ConvNet import ConvolutionalNet


def reshape_data(train, test):
    train = train.reshape(train.shape[0], 1, IMG_ROWS, IMG_COLS)
    test = test.reshape(test.shape[0], 1, IMG_ROWS, IMG_COLS)
    return train, test


def normalize_training_data(train, test):
    train = train.astype("float32")
    test = test.astype("float32")
    train /= 255
    test /= 255
    return train, test


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get("loss"))
        self.val_losses.append(logs.get("val_loss"))


def show_graph(history):
    plt.plot(history.losses)
    plt.plot(history.val_losses)
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    print "Get MNIST dataset"
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    print "Reshape from (784,) to (28, 28)"
    X_train, X_test = reshape_data(X_train, X_test)

    print "Normalize data from 0-255 to 0-1"
    X_train, X_test = normalize_training_data(X_train, X_test)

    Y_train = np_utils.to_categorical(Y_train, NUMBER_OF_CLASSES)
    Y_test = np_utils.to_categorical(Y_test, NUMBER_OF_CLASSES)

    print "Shape of training images {0} and training labels {1}".format(
        X_train.shape, Y_train.shape
    )
    print "Shape of testing images {0} and testing labels {1}".format(
        X_test.shape, Y_test.shape
    )

    net = ConvolutionalNet((1, IMG_ROWS, IMG_COLS))

    net.create_network()

    model = net.get_model()

    history = LossHistory()

    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, verbose=1, mode="auto"
    )

    optimizer = Adam(1e-4)

    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )

    model.fit(
        X_train,
        Y_train,
        batch_size=BATCH_SIZE,
        nb_epoch=NUMBER_OF_EPOCHS,
        validation_data=(X_test, Y_test),
        callbacks=[history, early_stopping],
        verbose=1,
    )

    score = model.evaluate(X_test, Y_test, verbose=0)
    print ("Test score:", score[0])
    print ("Test accuracy:", score[1])

    show_graph(history)

    plot(model, to_file="model.png", show_shapes=True)
