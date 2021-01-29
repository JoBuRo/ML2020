import openml
import sklearn
from sklearn import svm
from sklearn import metrics
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
#from hyperopt import hp, fmin, rand, tpe, space_eval
from scipy.optimize import minimize
from scipy.stats import norm

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
#from bayesian_optimization_util import plot_approximation, plot_acquisition
from sklearn.preprocessing import LabelEncoder, scale

from array import array

def propose_lambda(pred):
    a = pred[0] +0.5 * pred[1]
    print(a)
    print(np.amax(a))

    index = np.where(a==np.amax(a))
    print(index)
    return index


#######################################################################################
#Load dataset
dataset = openml.datasets.get_dataset(31)

# X - An array where each row represents one example with the corresponding feature values.
# y - class vector
# categorical_indicator - Mask that indicate categorical features.
# attribute_names - List of attribute names
X, y, categorical_indicator, attribute_names = dataset.get_data(dataset_format="array", target=dataset.default_target_attribute)

for i in range(len(y)):
    if y[i] == 0:
        y[i] = -1

X  = scale(X)

#Split data
X_train = X[0:600]
y_train = y[0:600]
X_val = X[600:800]
y_val = y[600:800]
X_test = X[800:1000]
y_test = y[800:1000]


#C=1.0, gamma=1.0 --> how to set gamma properly?

#get 3 inital values by training with three different sets of hyperparameters
 #-->train some points in advance to have an initial surrogate 
d = {'C': [], 'gamma': []}
H_x = pd.DataFrame(data=d) #history for lambda possibilities
d = {'acc': []}
H_y = pd.DataFrame(data=d) #history for accuracy results

d={'C': np.tile(np.linspace(0.1, 2, 100), 100),
                         'gamma': np.repeat(np.linspace(0.1, 2, 100), 100)} # all possible hp combinations
hpoSpace = pd.DataFrame(data=d) # all possible hp combinations


for i in range(4):
    rbf_svc = svm.SVC(C=i/2.0+0.25, gamma = i/2.0+0.25)
    print("gamma = ",  i/2.0+0.25)
    parameters = rbf_svc.get_params()
    rbf_svc.fit(X_train, y_train)
    
    y_pred = rbf_svc.predict(X_val)
    acc = metrics.accuracy_score(y_val, y_pred)
    H_y.loc[i] = [acc]
    H_x.loc[i] = [i/2.0+0.25, i/2+0.25]

print(H_x)

print(H_x.loc[1][0])
print(H_x.loc[1][1])
print(H_x.loc[0][1])
print(H_y)
print(hpoSpace.loc[100][0])

#acquisition function takes the uncertainty of the surrogate model and tells me which region is promising.

#bounds for hyperparameter
bounds = np.array([[0.1, 2.0]])

#numbers of rounds
num_hpo = 40

gpr = GaussianProcessRegressor()

'''
in loop:

1. recommend the new lambda by aquisition function
2. train the model with the new lambda
3. evaluate the trained model on validation set.
4. Add result to your history 

aquisition function can e.g. be Lower confidence Bound acquisition: Look for regions where loss is small but region is uncertain
'''

for i in range(num_hpo):
    print("Round ", i)
    #fit gaussian model to given hyperparameters.
    gpr.fit(H_x, H_y)
    H_y_pred = gpr.predict(hpoSpace, return_std=True)

    #propose the next lambda by taking maximum value
    lambda_index_C, lambda_index_gamma = propose_lambda(H_y_pred)
    #lambda_next = propose_location(expected_improvement, H_x, H_y, gpr, bounds)
    #fit SVM model with current set of hyperparameters
    rbf_svc = svm.SVC(C=hpoSpace.loc[lambda_index_C[0]][0], gamma = hpoSpace.loc[lambda_index_gamma[0]][1])
    rbf_svc.fit(X_train, y_train)
    #calculate accuracy by evaluating on validation set.
    y_pred = rbf_svc.predict(X_val)
    acc = metrics.accuracy_score(y_val, y_pred)

    #add processed hyperparameters and result to history.
    H_x.loc[i+4] = [hpoSpace.loc[lambda_index_C[0]][0], hpoSpace.loc[lambda_index_gamma[0]][1]]
    H_y.loc[i+4] = [acc]

    print("New accuracy: ", acc)
    print(H_x)
    print(H_y)

print(min(H_y.loc[0]))
min_loss_index = np.where(np.amax(H_y.loc[0]))
print("total: accuracy ", min_loss_index)
print("for C = ", H_x.loc[min_loss_index])


