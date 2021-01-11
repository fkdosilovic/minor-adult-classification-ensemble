"""Training and test script for bagging, with MLP as the base estimator, with
ADASYN for handling imbalanced dataset."""

import os
import sys

import utils

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    plot_confusion_matrix,
    accuracy_score,
)

import matplotlib.pyplot as plt

from imblearn.over_sampling import ADASYN

dirname = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(dirname)
experiments_dir = os.path.join(project_dir, "experiments")
data_dir = os.path.join(project_dir, "data")
models_dir = os.path.join(project_dir, "models")


def main(params_yaml_path):
    # Read and print experiment setup.
    params = utils.read_experiment_setup(params_yaml_path)
    utils.stdout_experiment_setup(params)

    X_train, y_train = utils.read_and_prepare_dataset(
        "train",
        data_dir,
        params,
    )

    # Perform over-sampling with ADASYN.
    resample = ADASYN(
        n_neighbors=params["n_neighbors"],
        n_jobs=params["n_jobs"],
    )

    X_train_resample, y_train_resample = resample.fit_resample(
        X_train,
        y_train,
    )

    base_estimator = MLPClassifier(
        hidden_layer_sizes=params["hidden_layers"],
        activation=params["activation"],
        alpha=params["alpha"],
        batch_size=params["batch_size"],
        max_iter=params["max_iter"],
    )

    clf = BaggingClassifier(
        base_estimator=base_estimator,
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        max_features=params["max_features"],
        n_jobs=params["n_jobs"],
    )

    # Train the classifier.
    clf.fit(X_train_resample, y_train_resample)

    # Save the model.
    utils.save_model(clf, models_dir, params["model_name"])

    X_test, y_test = utils.read_and_prepare_dataset(
        "test",
        data_dir,
        params,
    )

    y_pred = clf.predict(X_test)
    print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
    print(f"Balanced acc. score: {balanced_accuracy_score(y_test, y_pred)}")

    plot_confusion_matrix(
        clf,
        X_test,
        y_test,
        display_labels=["minor", "adult"],
    )
    plt.show()


if __name__ == "__main__":
    main(os.path.join(experiments_dir, sys.argv[1]))
