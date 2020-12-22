"""
   This class implements the OLS model. 
"""
import numpy as np

class OLS:
    def __init__(self):
        self.coeff = None

    def augment(self, X):
        """
            Augment the given matrix X

            >>> o = OLS()
            >>> X = np.arange(6).reshape(3, 2)
            >>> print(o.augment(X))
            [[0. 1. 1.]
             [2. 3. 1.]
             [4. 5. 1.]]

             >>> X = np.arange(3)
             >>> print(o.augment(X))
             [[0. 1.]
              [1. 1.]
              [2. 1.]]
        """
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        X_augmented = np.ones((X.shape[0], X.shape[1] + 1))
        X_augmented[:,:-1] = X
        return X_augmented
        

    def train(self, X, Y):
        """
            Train the model on the datapoints X and labels Y.
            This will set the member variables for the coefficients.

            >>> o = OLS()
            >>> X = np.arange(6)
            >>> Y = np.arange(6)
            >>> o.train(X, Y)
            >>> print(o.coeff)
            [1.0000000e+00 8.8817842e-16]
        """
        X = self.augment(X)
        xtx = np.dot(X.T, X)
        xty = np.dot(X.T, Y)
        self.coeff = np.dot(np.linalg.inv(xtx), xty)

    def predict(self, X):
        """
            Give a predictions for the datapoints X.
            
            >>> o = OLS()
            >>> X = np.arange(10)
            >>> Y = np.arange(10)
            >>> o.train(X[:6], Y[:6])
            >>> print(o.predict(X[6:]))
            [6. 7. 8.]
        """
        if self.coeff is None:
            print("ERROR: Run training function first")
            return
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1)
        return np.dot(X, self.coeff[:-1]) + self.coeff[-1]
