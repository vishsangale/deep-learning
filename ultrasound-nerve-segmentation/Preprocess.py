import os
from scipy.misc import imresize

from scipy.ndimage import imread

import numpy as np

data_path = "../datasets/ultrasound-nerve-segmentation/"

image_rows = 420
image_cols = 580


def create_train_data():
    train_data_path = os.path.join(data_path, "train")
    images = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print("Creating training images...")
    for image_name in images:
        if "mask" in image_name:
            continue
        image_mask_name = image_name.split(".")[0] + "_mask.tif"
        img = imread(os.path.join(train_data_path, image_name), mode="L")
        img_mask = imread(os.path.join(train_data_path, image_mask_name), mode="L")

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print("Done: {0}/{1} images".format(i, total))
        i += 1
    print("Loading done.")

    np.save("imgs_train.npy", imgs)
    np.save("imgs_mask_train.npy", imgs_mask)
    print("Saving to .npy files done.")


def load_train_data():
    if not os.path.isfile("imgs_train.npy") or not os.path.isfile(
        "imgs_mask_train.npy"
    ):
        create_train_data()
    return np.load("imgs_train.npy"), np.load("imgs_mask_train.npy")


def create_test_data():
    train_data_path = os.path.join(data_path, "test")
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, 1, image_rows, image_cols), dtype=np.uint8)
    imgs_id = np.ndarray((total,), dtype=np.int32)

    i = 0
    print("Creating test images...")
    for image_name in images:
        img_id = int(image_name.split(".")[0])
        img = imread(os.path.join(train_data_path, image_name), mode="L")

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print("Done: {0}/{1} images".format(i, total))
        i += 1
    print("Loading done.")

    np.save("imgs_test.npy", imgs)
    np.save("imgs_id_test.npy", imgs_id)
    print("Saving to .npy files done.")


def load_test_data():
    if not os.path.isfile("imgs_test.npy") or not os.path.isfile("imgs_id_test.npy"):
        create_test_data()
    imgs_test = np.load("imgs_test.npy")
    imgs_id = np.load("imgs_id_test.npy")
    return imgs_test, imgs_id


def pre_process(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], 64, 80), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = imresize(imgs[i, 0], (64, 80), interp="cubic")
    return imgs_p
