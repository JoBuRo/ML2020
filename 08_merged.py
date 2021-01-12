import numpy as np
from sklearn import datasets
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from scipy import stats

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

    rng = np.random.default_rng()
    ransam = rng.choice(X.shape[0], target_size, replace=True)

    return X[ransam], y[ransam]


# Machine Learning is like a box of chocolates, you never know what you get
class RandomForrestClassifier():

    def __init__(self, num_trees: int=100, bootstrapping_size: int=50, feature_subset: int=3):
        """
        Initializes the RandomForrestClassifier.

        Args:
            num_trees (int): Number of weak learner trees.
            bootstrapping_size (int): The size of the bootstrapped samples.
            feature_subset (int): The number of features to use for the
                splitting criteria.
        """

        self._forest = [DecisionTreeClassifier(max_features=feature_subset) for _ in range(num_trees)]
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

        # Predictions for the whole Forrest, shape: KxN
        predictions = np.array([self._forest[k].predict(X) for k in range(len(self._forest))])

        # Perform majority vote on the predictions
        # Mode = Most common Element
        return np.array(stats.mode(predictions).mode).reshape(-1, 1)
    

def print_result():
    """
    Print the test accuracy for a boostrapping size of 50.
    """
    # Run Forrest, run!
    print('Init...', end='')
    rf = RandomForrestClassifier(bootstrapping_size=50)
    print('OK')

    print('Fit...', end='')
    rf.fit(X_train, y_train)
    print('OK')
    print('Predict...', end='')
    y_hat = rf.predict(X_test)
    print(accuracy_score(y_test, y_hat))
    print('Done')

def plot_boostrapping_sizes():
    """
    Plot the test accuracy for different boostrapping sizes
    """
    # Compute accuracies for different bootstrapping sizes
    all_bootstrapping_sizes = np.arange(1, 15)
    # all_bootstrapping_sizes = np.arange(1, 200, step = 20)
    all_accuracies = []
    for bootstrapping_size in all_bootstrapping_sizes:
        rf = RandomForrestClassifier(bootstrapping_size=bootstrapping_size)
        rf.fit(X_train, y_train)
        y_hat = rf.predict(X_test)
        all_accuracies.append(accuracy_score(y_test, y_hat))
    # Plot the accuracy in relation to the boostrapping size
    import matplotlib.pyplot as plt
    plt.plot(all_bootstrapping_sizes, all_accuracies)
    plt.xlabel('Bootstrapping size')
    plt.ylabel('Test accuracy')
    plt.show()

# print_result()
plot_boostrapping_sizes()