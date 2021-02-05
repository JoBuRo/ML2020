""" A ff neural network for binary classification.
"""
import sys
import numpy as np

class NeuralNet():
    """ This neural network uses ReLU as activation functions
        and the logistic loss for the output.
    """
    def __init__(self, num_features, penalty, dropout, learning_rate, num_epochs, num_batches, batch_size):
        self.lr = learning_rate
        self.trained = False
        self.rng = np.random.default_rng()
        self.num_ep = num_epochs
        self.num_batch = num_batches
        self.batch_size = batch_size
        self.alpha = penalty
        # Number of hidden layers
        self.num_hl = 2
        # Nodes in hidden layer = nihl
        self.nihl = 30
        # Clip gradient, if abs(gradient) > clipvalue
        self.clipvalue = 2
        # The dropout parameter consists of a tuple: (input_dropout, hiddenlayer_dropout)
        # do_rate = droput_rate
        self.do_rate = dropout

        # Use He weight initialization, else ReLU will screw up the network
        self.weights = [self.rng.normal(0, np.sqrt(2 / num_features), (num_features, self.nihl))]
        self.weights += [self.rng.normal(0, np.sqrt(2 / self.nihl), (self.nihl, self.nihl)) for _ in range(self.num_hl - 1)]
        self.weights += [self.rng.normal(0, np.sqrt(2 / self.nihl), (self.nihl, 1))]

        self.bias = [self.rng.normal(0, np.sqrt(2 / self.nihl), (self.nihl, 1)) for _ in range(self.num_hl)]
        self.bias += [self.rng.normal(0, np.sqrt(2), (1, 1))]

        self.activation = [np.zeros(self.nihl) for _ in range(self.num_hl)]
        self.activation += [np.zeros(1)]

        self.dropout_mask = [np.ones((num_features, 1))]
        self.dropout_mask += [np.ones((self.nihl, 1)) for _ in range(self.num_hl)]

    def sigmoid(z):
        return 1 / (1 + np.exp(-1 * z))
    
    def log_loss(prediction, target):
        if prediction == 1.0: prediction -= sys.float_info.epsilon
        if prediction == 0.0: prediction = sys.float_info.epsilon
        return -target * np.log(prediction) - (1 - target) * np.log(1 - prediction)

    def update_dropout(self, prediction=False):
        """ Update the dropout mask using the probabilities given by the dropout rate
            If we're predicting, use the probability instead of a binary mask
        """
        if not prediction:
            self.dropout_mask[0] = self.rng.binomial(1, self.do_rate[0], self.dropout_mask[0].shape)
            for i in range(1, len(self.dropout_mask)):
                self.dropout_mask[i] = self.rng.binomial(1, self.do_rate[1], self.dropout_mask[i].shape)
        else:
            self.dropout_mask[0] = np.zeros(self.dropout_mask[0].shape) + self.do_rate[0]
            for i in range(1, len(self.dropout_mask)):
                self.dropout_mask[i] = np.zeros(self.dropout_mask[i].shape) + self.do_rate[1]

    def feed_forward(self, x, y=None):
        """ Feed the datapoint x forward through the network
            Return the loss
        """
        # Initially, activation = datapoints
        # neurac = neuron activation
        neuract = x.copy().reshape(-1, 1)
        neuract = neuract * self.dropout_mask[0]
        for i in range(len(self.weights)):
            # Apply weights
            neuract = self.weights[i].T.dot(neuract)

            # Add bias
            neuract += self.bias[i]

            # If not in the last layer, use activation function
            # and apply dropout
            if i < len(self.weights) - 1:
                neuract = np.maximum(neuract, np.zeros(neuract.shape))
                # Drop it like it's hot
                neuract = neuract * self.dropout_mask[i + 1]

            self.activation[i] = neuract
        # While training, output the loss so we can see what's going on
        # While testing, we have no target y
        loss = None
        if y is not None:
            prediction = NeuralNet.sigmoid(self.activation[-1][0][0])
            loss = NeuralNet.log_loss(prediction, y)
        return loss

    def backprop(self, x, y):
        """ Update the weights according to a single binary target y
            Uses the one and only backpropagation algorithm
        """
        weight_updates = [np.zeros((self.weights[0].shape[0], self.nihl))]
        weight_updates += [np.zeros((self.nihl, self.nihl)) for _ in range(self.num_hl - 1)]
        weight_updates += [np.zeros((self.nihl, 1))]

        bias_updates = [np.zeros((self.nihl, 1)) for _ in range(self.num_hl)]
        bias_updates += [np.zeros((1, 1))]

        # Compute the OG (G of last layer)
        G = NeuralNet.sigmoid(self.activation[-1][0]) - y
        weight_updates[-1] = G * self.activation[-2]
        bias_updates[-1][0][0] = G
        # Compute G for other layers
        for i in range(0, len(self.weights) - 1)[::-1]:
            # Get the derivation of the activation function (chain rule)
            # In the case of ReLU, this is essentially a mask
            actfun_deriv = (self.activation[i + 1] >= 0)
            # Drop muted neurons
            deriv = self.weights[i + 1].T
            deriv *= self.dropout_mask[i]# .copy().reshape(1, -1)
            deriv = deriv * actfun_deriv
            G_new = deriv.T.dot(G)

            # Compute the weight updates
            actfun_deriv = (self.activation[i] >= 0)
            prev_activation = self.activation[i - 1] if i > 0 else x.copy().reshape(-1, 1)
            # Drop it once more
            prev_activation = prev_activation * self.dropout_mask[i]
            gradient = prev_activation.dot(actfun_deriv.T)
            gradient = np.clip(gradient, (-1) * self.clipvalue, self.clipvalue)

            # NaN, begone!
            weight_updates[i] = np.clip(gradient * G_new, -1 * self.clipvalue, self.clipvalue)
            bias_updates[i] = np.clip(G_new.copy().reshape(-1, 1) * (self.activation[i] >= 0), -1 * self.clipvalue, self.clipvalue)
            G = G_new
        # Update weights and biases
        for i in range(len(self.weights)):
            # Use L2 regularization
            self.weights[i] -= self.lr * (weight_updates[i] + self.alpha * self.weights[i])
            self.bias[i] -= self.lr * (bias_updates[i] + self.alpha * self.bias[i])


    def train(self, X, y):
        """ Train the features on data points X and targets y
            X: N x M array
            Y: N array
        """
        assert(len(X) == len(y))
        for i in range(self.num_ep):
            # For each epoch, try to do a decent descent (stochastically)
            batches = self.rng.choice(np.arange(len(X)), size=(self.num_batch, self.batch_size), replace=True)
            epoch_loss = 0
            for batch in batches:
                batch_loss = 0
                self.update_dropout()
                for idx in batch:
                    loss = self.feed_forward(X[idx], y[idx])
                    batch_loss += loss
                    self.backprop(X[idx], y[idx])
                epoch_loss += batch_loss
            print(f"Epoch {i + 1}, Total epoch loss: {epoch_loss:.5f}")
        self.trained = True

    def predict(self, X):
        """ Predict class (between 0 and 1) for the dataset X
            X: N x M array

            returns: N predictions
        """
        if not self.trained:
            print("Hey! You gotta train me first")
        prediction = np.zeros(X.shape[0])

        # Get scaled dropouts for aggregation
        self.update_dropout(prediction=True)
        for i, x in enumerate(X):
            self.feed_forward(x)
            prediction[i] = NeuralNet.sigmoid(self.activation[-1][0])
        return prediction