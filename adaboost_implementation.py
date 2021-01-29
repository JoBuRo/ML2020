import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

import math

np.random.seed(1)

X = np.array([[-1, 1, 1],[1,1,1],[-1,1,-1],[1,1,-1]])
y = np.array([-1,1,1,1])

print(X.shape)
print(y.shape)
print(len(y))

def indicator_function(target: np.array, prediction: np.array, is_positive: bool=1):

    #compare inversion of x feature if split within feature is done with < sign instead of >
    s=1
    if not is_positive:
        s=-1

    equal = np.zeros(len(target))

    for i in range(len(target)):
        equal[i] = target[i]*s*prediction[i]
    return equal # array which tells us which feature elements fullfill the split condition compared to y


class AdaboostClassifier():

    def __init__(self, num_rounds: int=3, num_samples: int=4):
        self._num_rounds = num_rounds
        #initialize weights: number of weights equals the number of training samples
        self._weights = [1/num_samples] * num_samples 
        #initialize alphas: Trustworthiness for every round
        self._alpha = [0]* num_rounds
        #initialize loss: output with respective split for every round
        self._loss = [0] * num_rounds

        self._splits = [0] * num_rounds


    def print_info(self):
        print("__________________")
        print("Weights: ", self._weights)
        print("Alphas: ", self._alpha)
        print("Losses: ", self._loss)
        print("__________________")

    def fit(self, X: np.array, y: np.array):
        print( y.ndim)
        assert X.ndim == 2 and y.ndim == 1
        assert X.shape[0] == len(y)

        print("y")
        print(y.shape)

        #itereate over all rounds
        for i in range(self._num_rounds):
            print("------------round---", i)
            #choose best split
            split = self.choose_split(i, X,y)

            #calculate error chosen best split for this round.
            error = self.calc_error(i, X[:,(abs(split)-1)], y)
            print("current error:" , error)

            #for the current split round, calculate trustworthiness alpha by means of the calculated error
            self._alpha[i] = np.log((1-error)/error)
            print("current alpha:", self._alpha[i])

            #update weights according to alpha
            self.update_weights(i, X[:,(abs(split)-1)], y)

            print("current weights", self._weights)

        result = sum(np.multiply(self._alpha, self._loss))

        return (result>0)

    def predict(self, X_test: np.array):

        result = sum(np.multiply(self._alpha,X_test))

        return (result > 0)


    def choose_split(self, k, X: np.array, y: np.array):
        minloss = 100 #high initial value
        #iterate over all features
        for i in range (X.shape[1]):
            #take all samples for a certain feature
            X_colomn = X[:,i]
            #check which samples match with the corresponding target y
                        
            factor = [1,-1] # 1 stands for >, -1 stands for <
            #j>0 means split is x_i > 0 , j<0 means split is x_i <0
            for j in factor:
                isequal = indicator_function(y, X_colomn)
                newloss = sum(np.multiply(self._weights,np.exp(-j*isequal)))
                #only update minloss if there was an unused split which resulted in a new minimum loss
                if (newloss < minloss and ((j*(i+1)) not in self._splits)):
                    minloss = newloss
                    split = j*(i+1) #we start at i=0 for x1

        print("Final loss: ")
        print(minloss)
        print("split at x", split)

        #save current loss and split
        self._loss[k] = minloss
        self._splits[k] = split

        return split


    def calc_error(self, k: int, X_col: np.array, y: np.array):
        assert len(X_col) == 4 and len(y) == 4
        #calculate weighted training error for the current split
        error =  sum(np.multiply(self._weights,(-1==indicator_function(y, X_col, (self._splits[k]>0)))))/sum(self._weights)

        return error

    def update_weights(self, k, X_col: np.array, y: np.array):
         assert len(X_col) == 4

         self._weights = np.multiply(self._weights, np.exp(self._alpha[k]*(-1==indicator_function(y, X_col,(self._splits[k]>0)))))
         print("new weights: ", self._weights)

ada = AdaboostClassifier()
#train classifier
result = ada.fit(X,y)
ada.print_info()

X_test = np.array([[+1, -1, -1],[-1,-1,-1],[1,-1,1]])

for i in range(X_test.shape[0]):
    X_test_col = X_test[:,i]
    print(X_test_col)
    prediction = ada.predict(X_test_col)
    print(prediction)
