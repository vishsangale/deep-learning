import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from ConvNet import ConvolutionNet
from Postprocess import create_submission, predict
from Preprocess import load_train_data, pre_process

img_rows, img_cols = 64, 80
smooth = 1.0


def dice_coefficient(y_true, y_predict):
    y_true_f = K.flatten(y_true)
    y_predict_f = K.flatten(y_predict)
    intersection = K.sum(y_true_f * y_predict_f)
    return (2.0 * intersection + smooth) / (
        K.sum(y_true_f) + K.sum(y_predict_f) + smooth
    )


def dice_coefficient_loss(y_true, y_predict):
    return -dice_coefficient(y_true, y_predict)


if __name__ == "__main__":
    images_train, images_mask_train = load_train_data()
    images_train = pre_process(images_train)
    images_mask_train = pre_process(images_mask_train)

    images_train = images_train.astype("float32")
    mean = np.mean(images_train)
    std = np.std(images_train)

    images_train -= mean
    images_train /= std

    images_mask_train = images_mask_train.astype("float32")
    images_mask_train /= 255.0

    net = ConvolutionNet(
        (1, img_rows, img_cols), dice_coefficient_loss, [dice_coefficient]
    )
    model = net.get_model()

    model_checkpoint = ModelCheckpoint(
        "unet.hdf5", monitor="val_loss", save_best_only=True, mode="auto"
    )

    model.fit(
        images_train,
        images_mask_train,
        batch_size=32,
        nb_epoch=20,
        verbose=1,
        shuffle=True,
        validation_split=0.1,
        callbacks=[model_checkpoint],
    )

    predict(mean, std, model)

    create_submission()
