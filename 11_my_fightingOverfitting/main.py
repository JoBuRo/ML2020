
from sklearn.datasets import load_breast_cancer
import numpy as np
import torch
from torch import nn
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import metrics
#from myNetwork import DropoutNet
import myNetwork




cancer = datasets.load_breast_cancer()

#print(X)
print(cancer.data)
print(cancer.data)
X = torch.tensor(preprocessing.normalize(cancer.data), dtype=torch.float)
y = torch.tensor(cancer.target.reshape(-1, 1), dtype=torch.float)
print(X.size())
print(y.size())

'''
# Sanity Check Training Set
num_ones = 100
num_zeros = 100
X = np.concatenate((np.ones((num_ones, 10)), np.zeros((num_zeros, 10)) - 1))
y = np.concatenate((np.ones(num_ones), np.zeros(num_zeros)))

X = torch.tensor(X, dtype= torch.float)
y = torch.tensor(y.reshape(-1,1),dtype=torch.float)
'''
print(X.size())
print(y.size())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

batch_size = 50
num_batches = 10

##################################With dropout ###################################
# Instantiate our model class and assign it to our model object
model = myNetwork.DropoutNet(num_inputs=30, num_epochs=100, use_dropout=True, drop_rate_input=0.9, drop_rate_hidden=0.7, is_training = True)

model.run_training(X_train, y_train)

pred = model.predict(X_test)

for i in range(len(X_test)):
    if i%5 == 0:
        print("Prediction: ", pred[i], "Truth:",y_test[i])

acc = metrics.accuracy_score(y_test, (pred>0.5))
print("Accuracy with dropout: ", acc)
