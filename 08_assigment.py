import numpy as np
from sklearn import datasets
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

    raise NotImplementedError()


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
        raise NotImplementedError()
    

    def fit(self, X: np.array, y: np.array):
        """
        Fitting the random forrest to the data given.

        Args:
            X (ndarray): The data in shape NxM.
            y (ndarray): The target in shape N.
        """
        assert X.ndim == 2 and y.ndim ==1, "Wrong dimensions for X!"
        assert X.shape[0] == len(y), "Size of X and y does not match!"
        
        # TODO: Implement the algorithm to learn a random forrest.
        # you may use sklearn.tree.DecisionThreeClassifier as the
        # implementation for the weak learner

        raise NotImplementedError()
    
    
    def predict(self, X: np.array) -> np.array:
        """
        Predict the target class for the data X using majority voting.

        Args:
            X (ndarray): The data in shape NxM.
        
        Returns:
            (ndarray): The predictions in shape N
        """
        assert X.ndim == 2
        raise NotImplementedError()


rf = RandomForrestClassifier()

rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)
print(accuracy_score(y_test, y_hat))
