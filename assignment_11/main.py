from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score
import numpy as np
from network import NeuralNet


""" Debuggy stuffs
num_ones = 50
num_zeros = 50
X = np.concatenate((np.ones((num_ones, 10)), np.zeros((num_zeros, 10)) - 1))
y = np.concatenate((np.ones(num_ones), np.zeros(num_zeros)))

nn = NeuralNet(num_features=10, penalty=0, dropout=(0.8, 0.5), learning_rate=0.05, num_epochs=20, num_batches=5, batch_size=20)
nn.train(X, y)
print(f"Predict one: {nn.predict(np.ones((1, 10)))[0]}")
print(f"Predict zero: {nn.predict(np.zeros((1, 10)) - 1)[0]}")

"""

X, y = load_breast_cancer(return_X_y=True)

X = scale(X)

test_size = 69
train_X = X[:-test_size]
train_y = y[:-test_size]
test_X = X[-test_size:]
test_y = y[-test_size:]

nn = NeuralNet(num_features=len(X[0]), penalty=0, dropout=(0.8, 0.5), learning_rate=0.02, num_epochs=20, num_batches=5, batch_size=100)
nn.train(train_X, train_y)
pred = nn.predict(test_X)

for i, p in enumerate(pred):
    if i == 10:
        break
    print(f"Prediction: {p:.5f}, Truth: {test_y[i]}")

print(f"Accuracy: {100 * accuracy_score(test_y, np.around(pred)):.5f}%")