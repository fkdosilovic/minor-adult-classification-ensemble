"""Script for reducing dimensionality of a dataset."""

import sys
import os

import utils

import numpy as np

from sklearn.decomposition import PCA

dirname = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(dirname)
experiments_dir = os.path.join(project_dir, "experiments")
data_dir = os.path.join(project_dir, "data")
models_dir = os.path.join(project_dir, "models")


def prepare_dataset(dataset, params, train_dtype=np.float64):
    # Prepare training dataset.
    dataset_path = os.path.join(data_dir, params[f"{dataset}_dataset"])
    labels_path = os.path.join(data_dir, params[f"{dataset}_labels"])

    # Read train data and labels.
    X, _ = utils.read_dataset_and_labels(
        dataset_path,
        labels_path,
        train_dtype,
    )
    X = X.reshape(params[f"{dataset}_size"], 200, 200, 3)

    size = (params["size"],) * 2
    grayscale = params["grayscale"]

    # Preprocess the whole dataset.
    X = np.stack([utils.preprocess(img, size, grayscale) for img in X])

    return X


def save_dataset(X, dataset, params):
    size = params["size"]
    # Save the reduced dataset.
    fn = os.path.join(
        data_dir, params[f"{dataset}_dataset_name"].format(size, X.shape[1])
    )

    with open(fn, "wb") as f:
        X.tofile(f)


def main(params_yaml_path):
    # Read and print experiment setup.
    params = utils.read_experiment_setup(params_yaml_path)
    utils.stdout_experiment_setup(params)

    X_train = prepare_dataset("train", params, np.uint8)

    # Train PCA.
    pca = PCA(n_components=params["variance"])
    pca.fit(X_train)

    components = pca.components_.shape[0]
    model_name = params["model_name"].format(params["size"], components)
    utils.save_model(pca, models_dir, model_name)

    # Reduce dimensionality of training set and save it.
    X_train_pca = pca.transform(X_train)
    save_dataset(X_train_pca, "train", params)

    # Reduce dimensionality of test set and save it.
    X_test = prepare_dataset("test", params, np.uint8)
    X_test_pca = pca.transform(X_test)
    save_dataset(X_test_pca, "test", params)


if __name__ == "__main__":
    main(os.path.join(experiments_dir, sys.argv[1]))
