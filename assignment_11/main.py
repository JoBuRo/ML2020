import numpy as np
from network import NeuralNet
from sklearn.datasets import load_breast_cancer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, scale

# Jeez sklearn, SHUT UP ALREADY
# https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Hyper Hyper parameters
max_iterations = 30
inits = [
    (1, 1, 0.5),
    (0.8, 0.5, 0.02),
    (0.8, 0.5, 0.01),
    (0.5, 0.5, 0.05),
    (0.5, 0.2, 0.1)]
gauss_sample_len = 10
beta = 2
verbose = False

# Load dataset
print("Loading dataset...")
X_full, y_full = load_breast_cancer(return_X_y=True)
print("...done")

# Standardize X
X_full = scale(X_full)
data_len = len(X_full)

# Split dataset 60/20/20
X_train = X_full[:round(0.6 * data_len)]
y_train = y_full[:round(0.6 * data_len)]
X_test = X_full[round(0.6 * data_len):round(0.8 * data_len)]
y_test = y_full[round(0.6 * data_len):round(0.8 * data_len)]
X_eval = X_full[round(0.8 * data_len):]
y_eval = y_full[round(0.8 * data_len):]

# Get default accuracy
net = NeuralNet(num_features=len(X_full[0]), penalty=0, dropout=(0.8, 0.5), learning_rate=0.01, num_epochs=20, num_batches=20, batch_size=30)
net.train(np.concatenate((X_train, X_eval)), np.concatenate((y_train, y_eval)))
def_accuracy = accuracy_score(y_test, np.around(net.predict(X_test)))
print(f"Default accuracy: {def_accuracy:.5f}")

# Get intial config response
observations_lambda = np.zeros((len(inits), 3))
observations_response = np.zeros(len(inits))
for i, lambdas in enumerate(inits):
    net = NeuralNet(num_features=len(X_full[0]), penalty=0, dropout=(lambdas[0], lambdas[1]), learning_rate=lambdas[2], num_epochs=20, num_batches=20, batch_size=30)
    net.train(X_train, y_train)
    prediction = net.predict(X_eval)
    accuracy = accuracy_score(y_eval, np.around(net.predict(X_eval)))
    observations_lambda[i] = lambdas
    observations_response[i] = accuracy

# Run the optimization
print("Running optimization...")
for i in range(max_iterations):
    # Fit surrogate to observed response
    gauss = GaussianProcessRegressor()
    gauss.fit(observations_lambda, observations_response.reshape(-1, 1))

    # Get best hyper param, according to aquisition function
    # Get all combinations of C and gamma
    # https://stackoverflow.com/questions/27286537/numpy-efficient-way-to-generate-combinations-from-given-ranges
    # query_points = np.mgrid[step_size:2+step_size:step_size, step_size:2+step_size:step_size]
    # query_points = np.rollaxis(query_points, 0, 3)
    # query_points = query_points.reshape((query_points.size // 2, 2))
    # query space = qs
    qs = np.linspace(0, 1, gauss_sample_len)
    query_points = np.array([(a, b, c) for a in qs for b in qs for c in qs])

    pred = gauss.predict(query_points, return_std=True)
    pred = zip(pred[0], pred[1])
    # Lets use lower bound acquisition this time
    # acquisition = np.fromiter((mean + np.sqrt(beta) * std for mean, std in pred), dtype=np.float)
    acquisition = np.fromiter((-1 * mean + beta * std for mean, std in pred), dtype=np.float)
    lambda_new = query_points[np.argmax(acquisition)]

    # Get response to best hyper param
    net = NeuralNet(num_features=len(X_full[0]), penalty=0, dropout=(lambda_new[0], lambda_new[1]), learning_rate=lambda_new[2], num_epochs=20, num_batches=20, batch_size=30)
    net.train(X_train, y_train)
    prediction = np.around(net.predict(X_eval))
    accuracy = accuracy_score(y_eval, prediction)

    if verbose:
        print(f"Iteration: {i}")
        print("Prediction:")
        print(prediction)
        print(f"Accuracy: {accuracy}")

    # Add response to observations
    observations_lambda = np.concatenate((observations_lambda, np.array([lambda_new])))
    observations_response = np.concatenate((observations_response, np.array([accuracy])))

    # Rinse and repeat

print("...done")

# Get best lambda
best_lambda = observations_lambda[np.argmax(observations_response)]

# Train model on best lambda
best_net = NeuralNet(num_features=len(X_full[0]), penalty=0, dropout=(best_lambda[0], best_lambda[1]), learning_rate=best_lambda[2], num_epochs=20, num_batches=20, batch_size=30)
best_net.train(np.concatenate((X_train, X_eval)), np.concatenate((y_train, y_eval)))
prediction = best_net.predict(X_test)
best_accuracy = accuracy_score(y_test, np.around(net.predict(X_test)))
print(f"Best accuracy: {best_accuracy:.5f}")
print(f"Hyper parameters:")
print(f"input layer dropout = {best_lambda[0]}")
print(f"hidden layer dropout = {best_lambda[1]}")
print(f"learning rate = {best_lambda[2]}")