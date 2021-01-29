from typing import Dict
from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import numpy as np


def split_dataset(dataset):
    """
    Split dataset into 60% train, 20% validation, 20% test sets
    """
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(dataset.data, dataset.target, test_size=0.4)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def fit_svm(X_train, Y_train, config):
    """
    Fit an SVM using a given hyperparameter configuration.
    """
    model = svm.SVC(C=config["C"], gamma=config["gamma"])
    model.fit(X_train, Y_train)
    return model


def fit_surrogates(history: dict, history_end: int):
    """
    For each hyperparameter, fit a surrogate that approximates
    the accuracy of the model as a function of the hyperparameter,
    using previous observations stored in ```history```.
    """
    surrogates = dict()
    for hyperparam in history["configs"].keys():
        std = np.std(history["accuracies"][:history_end])
        gp = GaussianProcessRegressor(kernel=RBF(length_scale=std)) # TODO fit 1 multidimensional GP for all HPs instead
        # Only fit the surrogate on non-None history entries
        if history_end > 0:
            gp.fit(history["configs"][hyperparam][:history_end], history["accuracies"][:history_end])
        surrogates[hyperparam] = gp
    return surrogates


def acquisition(surrogates: dict, config_space: dict, beta=0.1):
    """
    For each hyperparameter, return the best hyperparameter value
    to sample next, following the Upper Confidence Bound.
    """
    best_new_config = dict()
    for hp, sur in surrogates.items():
        # Find the bounds inside which the hyperparameter should be sampled
        bounds = config_space[hp]
        # Evaluate the surrogate between those bounds
        space = np.linspace(*bounds, num=100).reshape(-1, 1)
        # sur: GaussianProcessRegressor
        y_mean, y_stdev = sur.predict(space, return_std=True)
        # Find the best next value according to the upper confidence bound
        max_acquisition = np.argmax(y_mean + beta * y_stdev)
        best_new_config[hp] = space[max_acquisition]
    return best_new_config


def smbo(config_space: dict, iterations: int =50):
    """
    Run the Sequential Model-Based Optimizer algorithm.
    """
    history = {
        "configs": {
            "C": [None for _ in range(iterations)],
            "gamma": [None for _ in range(iterations)],
        },
        "accuracies": [None for _ in range(iterations)],
    }
    for i in range(iterations):
        # Fit surrogate to history
        surrogates = fit_surrogates(history, i)
        # Find best new configuration to try
        best_new_config = acquisition(surrogates, config_space)
        # Evaluate a model with this configurations
        model = fit_svm(X_train, Y_train, best_new_config)
        accuracy = model.score(X_val, Y_val)
        # Save the results to history
        history["configs"]["C"][i] = best_new_config["C"]
        history["configs"]["gamma"][i] = best_new_config["gamma"]
        history["accuracies"][i] = accuracy
        print(f"Iteration {i:3} of SMBO")
    best_config_overall_i = np.argmax(history["accuracies"])
    return history["configs"]["C"][best_config_overall_i], history["configs"]["gamma"][best_config_overall_i]


# config = {
#     "gamma": 1.0,
#     "C": 1.0
# }

config_space = {
    "gamma": (0.1, 2.),
    "C": (0.1, 2.)
}

dataset = fetch_openml(name="credit-g")
X_train, X_val, X_test, Y_train, Y_val, Y_test = split_dataset(dataset)
print(smbo(config_space))