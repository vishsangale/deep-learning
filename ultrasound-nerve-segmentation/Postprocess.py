from scipy.stats import threshold

from Preprocess import pre_process, load_test_data, image_cols, image_rows
import numpy as np


def predict(mean, std, model):
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = pre_process(imgs_test)

    imgs_test = imgs_test.astype("float32")
    imgs_test -= mean
    imgs_test /= std

    print("Loading saved weights...")
    model.load_weights("unet.hdf5")

    print("Predicting masks on test data...")
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save("imgs_mask_test.npy", imgs_mask_test)


def prep(img):
    img = img.astype("float32")
    img = threshold(img, 0.5, 1.0)[1].astype(np.uint8)
    img = np.resize(img, (image_cols, image_rows))
    return img


def run_length_enc(label):
    from itertools import chain

    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ""
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z + 1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s + 1, l + 1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return " ".join([str(r) for r in res])


def create_submission():
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = np.load("imgs_mask_test.npy")

    argsort = np.argsort(imgs_id_test)
    imgs_id_test = imgs_id_test[argsort]
    imgs_test = imgs_test[argsort]

    total = imgs_test.shape[0]
    ids = []
    rles = []
    for i in range(total):
        img = imgs_test[i, 0]
        img = prep(img)
        rle = run_length_enc(img)

        rles.append(rle)
        ids.append(imgs_id_test[i])

        if i % 100 == 0:
            print("{}/{}".format(i, total))

    first_row = "img,pixels"
    file_name = "submission.csv"

    with open(file_name, "w+") as f:
        f.write(first_row + "\n")
        for i in range(total):
            s = str(ids[i]) + "," + rles[i]
            f.write(s + "\n")
