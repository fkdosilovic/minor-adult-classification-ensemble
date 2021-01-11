import os

import yaml
import joblib

import numpy as np

import cv2
from skimage import transform
from skimage.color import rgb2gray


def preprocess(image, size, grayscale):
    """Resize, normalize and flatten an image."""
    if grayscale:
        image = rgb2gray(image)

    resized_image = transform.resize(image, size, anti_aliasing=True)
    return resized_image.flatten()


def read_dataset_and_labels(dataset_path, labels_path, train_dtype=np.float64):
    with open(dataset_path, "rb") as f:
        dataset = np.fromfile(f, dtype=train_dtype)
    with open(labels_path, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
    return dataset, labels


def read_experiment_setup(path):
    with open(path) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return args


def save_model(clf, path, model_name):
    joblib.dump(clf, os.path.join(path, model_name))


def stdout_experiment_setup(setup):
    print("\nExperiment setup:")
    for item, doc in setup.items():
        print(item, " : ", doc)
    print()


def extract_faces_from_image(image, boxes):
    faces = []
    for (top, right, bottom, left) in boxes:
        # width, height = right - left, bottom - top
        # assert width > 0 and height > 0
        faces.append(image[top:bottom, left:right])
    return faces


def label_image(image, boxes, labels):
    for ((top, right, bottom, left), label) in zip(boxes, labels):
        # draw the predicted face name on the image
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(
            image,
            label,
            (left, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )


# Taken from https://www.pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/.
def blur_image(image, boxes, labels, factor=3.0):
    for ((top, right, bottom, left), label) in zip(boxes, labels):
        if label == 0:
            face_image = image[top:bottom, left:right]
            w, h = right - left, bottom - top
            (h, w) = face_image.shape[:2]
            kW = int(w / factor)
            kH = int(h / factor)

            kW -= int(kW % 2 == 0)
            kH -= int(kH % 2 == 0)

            blurred = cv2.GaussianBlur(face_image, (kW, kH), 0)
            image[top:bottom, left:right] = blurred

    return image


def read_and_prepare_dataset(
    dataset,
    data_dir,
    params,
    train_dtype=np.float64,
    transform_labels=True,
):
    # Prepare training dataset.
    dataset_path = os.path.join(data_dir, params[f"{dataset}_dataset"])
    labels_path = os.path.join(data_dir, params[f"{dataset}_labels"])

    # Read train data and labels.
    X, y = read_dataset_and_labels(dataset_path, labels_path, train_dtype)
    X = X.reshape(params[f"{dataset}_size"], -1)

    assert X.shape[0] == params[f"{dataset}_size"]
    assert X.shape[1] == params["features"]

    # Transform the problem to
    if transform_labels:
        y = np.uint8(y >= 18)

    return X, y


# Taken from sklearn website.
# See: https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")
