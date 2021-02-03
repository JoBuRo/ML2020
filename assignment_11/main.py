from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
import numpy as np
from network import NeuralNet


""" Sanity Check (because I lost mine while programming this dumpster fire)
num_ones = 50
num_zeros = 50
X = np.concatenate((np.ones((num_ones, 10)), np.zeros((num_zeros, 10)) - 1))
y = np.concatenate((np.ones(num_ones), np.zeros(num_zeros)))

nn = NeuralNet(num_features=10, penalty=0, learning_rate=0.05, num_epochs=5, num_batches=10, batch_size=5)
nn.train(X, y)
print(f"Predict one: {nn.predict(np.ones((1, 10)))}")
print(f"Predict zero: {nn.predict(np.zeros((1, 10)) - 1)}")
"""

X, y = load_breast_cancer(return_X_y=True)

X = scale(X)

test_size = 69
train_X = X[:-test_size]
train_y = y[:-test_size]
test_X = X[-test_size:]
test_y = y[-test_size:]

nn = NeuralNet(num_features=len(X[0]), penalty=0, learning_rate=0.05, num_epochs=10, num_batches=10, batch_size=50)
nn.train(train_X, train_y)
pred = nn.predict(test_X[:20])

for i, p in enumerate(pred):
    print(f"Prediction: {p:.5f}, Truth: {test_y[i]}")