
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


''' Sanity Check Training Set
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
model = myNetwork.DropoutNet(num_inputs=30, num_epochs=100, use_dropout=True, drop_rate_input=0.9, drop_rate_hidden=0.5, is_training = True)

# # Loss list for plotting of loss behaviour
# loss_lst = []

# # Number of times we want our FNN to look at all 100 samples we have, 100 implies looking through 100x
# num_epochs = 100


# splittedx = torch.split(X_train, 50)
# splittedy =  torch.split(y_train, 50)
# num_batches = len(splittedx)

# X_store = X_train
# y_store = y_train

model.run_training(X_train, y_train)

# # Let's train our model with 100 epochs
# for epoch in range(num_epochs):

#     for minibatch in range(num_batches):
#         X_val = splittedx[minibatch]
#         y_val = splittedy[minibatch]
#         #print(minibatch)
#         X_train = torch.cat((X_store[:50*minibatch+1:,:], X_store[50*(minibatch+1)-1:,:]), axis = 0)
#         y_train = torch.cat((y_store[:50*minibatch+1:,:], y_store[50*(minibatch+1)-1:,:]), axis = 0)
#         #print("X_train shape", X_train.shape)
#         #print(y_train.shape)
#         #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

#         # (1) Forward propagation: to get our predictions to pass to our cross entropy loss function
#         # (2) Back propagation: get our partial derivatives w.r.t. parameters (gradients)
#         # (3) Gradient Descent: update our weights with our gradients
#         model.train(X_train, y_train)

#         # Get our predictions
#         y_hat = model.predict(X_val).double()

#         y_hat = torch.where(y_hat==1., 0.9999, y_hat)
#         y_hat = torch.where(y_hat==0., 0.0001, y_hat)
#         # Cross entropy loss, remember this can never be negative by nature of the equation
#         # But it does not mean the loss can't be negative for other loss functions
#         cross_entropy_loss = -(y_val * torch.log(y_hat) + (1 - y_val) * torch.log(1 - y_hat))

#         # We have to take cross entropy loss over all our samples, 100 in this 2-class iris dataset
#         mean_cross_entropy_loss = torch.mean(cross_entropy_loss).detach().item()

#     # Print our mean cross entropy loss
#     if epoch % 20 == 0:
#         print('Epoch {} | Loss: {}'.format(epoch, mean_cross_entropy_loss))
#     loss_lst.append(mean_cross_entropy_loss)


pred = model.predict(X_test)

for i in range(len(X_test)):
    if i%5 == 0:
        print("Prediction: ", pred[i], "Truth:",y_test[i])

acc = metrics.accuracy_score(y_test, (pred>0.5))
print("Accuracy with dropout: ", acc)
