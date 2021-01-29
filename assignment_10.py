import numpy as np
import openml
from sklearn.svm import SVC
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
max_iterations = 50
C_init = [0.25, 0.5, 1.0, 1.5, 2.0]
gamma_init = [0.25, 0.5, 1.0, 1.5, 2.0]
initial_config = list(zip(C_init, gamma_init))
step_size = 0.01
beta = 2
verbose = False

# Load dataset
print("Loading dataset...")
full_data = openml.datasets.get_dataset(31)
print("...done")

# openML datasets are really weird to handle
df, _, indicator, _ = full_data.get_data()

# Encode the features
encoders = []
for i in range(len(indicator)):
    if indicator[i]:
        le = LabelEncoder()
        le.fit(df[df.columns[i]].values)
        df[df.columns[i]] = le.transform(df[df.columns[i]].values)
        encoders += [le]
    else:
        encoders += [None]

y_full = df["class"].values
X_full = df.drop("class", 1).values

for i in range(len(y_full)):
    if y_full[i] == 0:
        y_full[i] = -1

# Standardize X
X_full = scale(X_full)

# Split dataset 60/20/20
X_train = X_full[:600]
y_train = y_full[:600]
X_test = X_full[600:800]
y_test = y_full[600:800]
X_eval = X_full[800:]
y_eval = y_full[800:]

# Get default accuracy
machine = SVC()
machine.fit(np.concatenate((X_train, X_eval)), np.concatenate((y_train, y_eval)))
def_accuracy = accuracy_score(y_test, machine.predict(X_test))
print(f"Default accuracy: {def_accuracy:.5f}")
print(f"Default hyper parameters: C = 1.0, gamma = {1 / len(X_full[0])}")

# Get intial config response
observations_lambda = np.ndarray((0, 2))
observations_response = np.array([])
for C, gamma in initial_config:
    machine = SVC(C=C,gamma=gamma)
    machine.fit(X_train, y_train)
    machine_prediction = machine.predict(X_eval)
    accuracy = accuracy_score(y_eval, machine.predict(X_eval))
    observations_lambda = np.concatenate((observations_lambda, np.array([[C, gamma]])))
    observations_response = np.concatenate((observations_response, np.array([accuracy])))

# Run the optimization
print("Running optimization...")
for i in range(max_iterations):
    # Fit surrogate to observed response
    gauss = GaussianProcessRegressor()
    gauss.fit(observations_lambda, observations_response.reshape(-1, 1))

    # Get best hyper param, according to aquisition function
    # Get all combinations of C and gamma
    # https://stackoverflow.com/questions/27286537/numpy-efficient-way-to-generate-combinations-from-given-ranges
    query_points = np.mgrid[step_size:2+step_size:step_size, step_size:2+step_size:step_size]
    query_points = np.rollaxis(query_points, 0, 3)
    query_points = query_points.reshape((query_points.size // 2, 2))

    pred = gauss.predict(query_points, return_std=True)
    pred = zip(pred[0], pred[1])
    aquisition = np.fromiter((mean + np.sqrt(beta) * std for mean, std in pred), dtype=np.float)
    lambda_new = query_points[np.argmax(aquisition)]

    # Get response to best hyper param
    machine = SVC(C=lambda_new[0], gamma=lambda_new[1])
    machine.fit(X_train, y_train)
    prediction = machine.predict(X_eval)
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
best_machine = SVC(C=best_lambda[0], gamma=best_lambda[1])
best_machine.fit(np.concatenate((X_train, X_eval)), np.concatenate((y_train, y_eval)))
prediction = best_machine.predict(X_test)
best_accuracy = accuracy_score(y_test, best_machine.predict(X_test))
print(f"Best accuracy: {best_accuracy:.5f}")
print(f"Hyper parameters: C = {best_lambda[0]}, gamma = {best_lambda[1]}")