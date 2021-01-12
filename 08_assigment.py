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

    #raise NotImplementedError()
    rand = np.random.randint(0,X.shape[0], target_size)
    return X[rand], y[rand]


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
        self._feat = feature_subset
        self._boot_samples = bootstrapping_size
        self._tree =  DecisionTreeClassifier()
        self._number_trees = num_trees
        #raise NotImplementedError()
    

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

        #raise NotImplementedError()

        for i in range(self._number_trees):
            subset = bootstrap(X, y, self._boot_samples)
            self._tree.fit(*subset)

       

    
    
    def predict(self, X: np.array) -> np.array:
        """
        Predict the target class for the data X using majority voting.

        Args:
            X (ndarray): The data in shape NxM.
        
        Returns:
            (ndarray): The predictions in shape N
        """
        assert X.ndim == 2
        #raise NotImplementedError()

        return self._tree.predict(X)


rf = RandomForrestClassifier()

rf.fit(X_train, y_train)
y_hat = rf.predict(X_test)
print(accuracy_score(y_test, y_hat))
