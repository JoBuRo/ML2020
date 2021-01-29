import openml
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

from array import array

def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.
    
    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample = gpr.predict(X_sample)

    sigma = sigma.reshape(-1, 1)
    
    # Needed for noise-based model,
    # otherwise use np.max(Y_sample).
    # See also section 2.4 in [1]
    mu_sample_opt = np.max(mu_sample)

    with np.errstate(divide='warn'):
        imp = mu - mu_sample_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def propose_location(acquisition, X_sample, Y_sample, gpr, bounds, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X_sample.shape[1]
    min_val = 1
    min_x = None
    
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_sample, Y_sample, gpr)
    
    print(bounds[:,0])
    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:,0], bounds[:,1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')        
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x           
            
    return min_x.reshape(-1, 1)


def propose_lambda(pred):
    a = pred[0] - 5 * pred[1]
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

#Split data
X_train = X[0:599]
y_train = y[0:599]
X_val = X[600:799]
y_val = y[600:799]
X_test = X[800:999]
y_test = y[800:999]


#C=1.0, gamma=1.0 --> how to set gamma properly?




#get 3 inital values by training with three different sets of hyperparameters
 #-->train some points in advance to have an initial surrogate 
d = {'C': [], 'gamma': []}
H_x = pd.DataFrame(data=d) #history for lambda possibilities
d = {'acc': []}
H_y = pd.DataFrame(data=d) #history for accuracy results

d={'C': np.linspace(0.1, 2, 200), 'gamma': np.linspace(0.1, 2, 200)}
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

#initial x-values (lambda configuations) for fitting the gaussian processer
#lambdas_sample = np.array([1.0, 1.5]).reshape(-1, 1)
#inital corresponding losses
#results_sample = np.array([acc, acc1])



#bounds for hyperparameter
bounds = np.array([[0.1, 2.0]])

#numbers of rounds
num_hpo = 10

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
    #H_y.append
    #results_sample = np.append(results_sample[:], acc).reshape(-1, 1)
    #lambdas_sample = np.append(lambdas_sample, lambda_next[0]).reshape(-1, 1)

    print("New accuracy: ", acc)
    print(H_x)
    print(H_y)

print(min(H_y.loc[0]))
min_loss_index = np.where(min(H_y.loc[0]))
print("total: accuracy ", min_loss_index)
print("for C = ", H_y.loc[min_loss_index])


