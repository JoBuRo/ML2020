import numpy as np
from sklearn import datasets
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

np.random.seed(1)

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


def bootstrap(X: np.array, y: np.array, target_size: int) -> tuple:
    """
    Implement bootsrapping without using sklearn!

    Args:
        X (ndarray): The data in shape NxM
        y (ndarray): The target in shape N
        target_size (int): The size of the sample to return
    
    Returns:
        tuple(ndarray, ndarray): The bootstrapped sample where
            the data is at the first postion and the target at
            the second.
    """
    assert X.shape[0] == len(y)

    random_indices = np.random.randint(X.shape[0], size=target_size)

    return X[random_indices], y[random_indices]


class RandomForestClassifier():

    def __init__(self, num_trees: int=100, bootstrapping_size: int=50, feature_subset: int=3):
        """
        Initializes the RandomForestClassifier.

        Args:
            num_trees (int): Number of weak learner trees.
            bootstrapping_size (int): The size of the bootstrapped samples.
            feature_subset (int): The number of features to use for the
                splitting criteria.
        """

        self._forest = [DecisionTreeClassifier(max_features=feature_subset) for _ in range(num_trees) ]
        self.bootstrapping_size = bootstrapping_size


    def fit(self, X: np.array, y: np.array):
        """
        Fitting the random forest to the data given.

        Args:
            X (ndarray): The data in shape NxM.
            y (ndarray): The target in shape N.
        """
        assert X.ndim == 2 and y.ndim == 1, "Wrong dimensions for X!"
        assert X.shape[0] == len(y), "Size of X and y does not match!"
        
        for tree in self._forest:
            tree.fit(*bootstrap(X, y, self.bootstrapping_size))
    
    
    def predict(self, X: np.array) -> np.array:
        """
        Predict the target class for the data X using majority voting.

        Args:
            X (ndarray): The data in shape NxM.
        
        Returns:
            (ndarray): The predictions in shape N
        """
        assert X.ndim == 2

        all_predictions = []
        for tree in self._forest:
            all_predictions.append(tree.predict(X))
        all_predictions = np.array(all_predictions).T

        most_common_prediction = []
        for sample in all_predictions:
            most_common_prediction.append(np.bincount(sample).argmax())

        return most_common_prediction

def print_result():
    print('Init...', end='')
    rf = RandomForestClassifier(bootstrapping_size=50)
    print('OK')

    print('Fit...', end='')
    rf.fit(X_train, y_train)
    print('OK')
    print('Predict...', end='')
    y_hat = rf.predict(X_test)
    print(accuracy_score(y_test, y_hat))
    print('Done')

def plot_boostrapping_sizes():
    all_bootstrapping_sizes = np.arange(1, 15)
    # all_bootstrapping_sizes = np.arange(1, 200, step = 20)
    all_accuracies = []
    for bootstrapping_size in all_bootstrapping_sizes:
        rf = RandomForestClassifier(bootstrapping_size=bootstrapping_size)
        rf.fit(X_train, y_train)
        y_hat = rf.predict(X_test)
        all_accuracies.append(accuracy_score(y_test, y_hat))

    import matplotlib.pyplot as plt

    plt.plot(all_bootstrapping_sizes, all_accuracies)
    plt.xlabel('Bootstrapping size')
    plt.ylabel('Test accuracy')
    plt.show()

plot_boostrapping_sizes()
