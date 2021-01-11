"""Hyperparameter optimization of Bagging (with MLP as base
estimator) with ADASYN for handling imbalanced dataset."""

import os
import sys

import utils

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from imblearn.over_sampling import ADASYN

dirname = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(dirname)
experiments_dir = os.path.join(project_dir, "experiments")
data_dir = os.path.join(project_dir, "data")
# models_dir = os.path.join(project_dir, "models")


def main(params_yaml_path):
    # Read and print experiment setup.
    params = utils.read_experiment_setup(params_yaml_path)
    utils.stdout_experiment_setup(params)

    X_train, y_train = utils.read_and_prepare_dataset(
        "train",
        data_dir,
        params,
    )

    # TODO: Not the best solution for finding optimal n_neighbors, but
    # it will work for now. Rewrite!
    param_n_neighbors = params["n_neighbors"]
    for n_neighbors in param_n_neighbors:
        # Perform over-sampling with ADASYN.
        print(f"Perform hyperparameter search for n_neighbors={n_neighbors}.")
        resample = ADASYN(
            n_neighbors=n_neighbors,
            n_jobs=params["n_jobs"],
        )

        X_train_resample, y_train_resample = resample.fit_resample(
            X_train,
            y_train,
        )

        # We perform hyperparam optimization only on a subset of resampled
        # dataset!
        X_train_small, _, y_train_small, _ = train_test_split(
            X_train_resample,
            y_train_resample,
            stratify=y_train_resample,
            train_size=X_train.shape[0] // 2,
        )

        # Prepare parameter distribution.
        param_dist = {
            "base_estimator__hidden_layer_sizes": params["hidden_layers"],
            "base_estimator__alpha": params["alpha"],
            "n_estimators": params["n_estimators"],
            "max_samples": params["max_samples"],
            "max_features": params["max_features"],
        }

        clf = RandomizedSearchCV(
            estimator=BaggingClassifier(
                base_estimator=MLPClassifier(
                    activation="relu",
                    batch_size=params["batch_size"],
                    max_iter=params["max_iter"],
                ),
                bootstrap=False,
            ),
            param_distributions=param_dist,
            n_iter=params["n_configs"],
            n_jobs=params["n_jobs"],
        )

        # Find best estimator.
        search = clf.fit(X_train_small, y_train_small)
        utils.report(search.cv_results_)


if __name__ == "__main__":
    main(os.path.join(experiments_dir, sys.argv[1]))
