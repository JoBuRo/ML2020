

from sklearn.datasets import load_breast_cancer
import torch
import sys
from torch import nn
from sklearn import datasets

import numpy as np



class DropoutNet(nn.Module):
    def __init__(self, num_inputs = 30, num_epochs=100, use_dropout=True, drop_rate_input=0.8, drop_rate_hidden=0.5, is_training = True):
        super().__init__()

        # Dimensions for input, hidden and output
        self.input_dim = num_inputs
        self.hidden_dim = 30
        self.output_dim = 1
        self.num_hidden_layers=2

        self.use_dropout = use_dropout
        self.drop_rate = [drop_rate_input, drop_rate_hidden, drop_rate_hidden]
        

        self.relu = nn.ReLU()

        self.is_training = is_training

        # Learning rate definition
        self.learning_rate = 0.002

        self.num_epochs = num_epochs

        self.b1 = torch.randn(self.input_dim)

        self.b2 = torch.randn(self.hidden_dim)
        self.b3 = torch.randn(self.output_dim)

        # Our parameters (weights)
        # w1: 30 x 30
        self.w1 = torch.randn(self.input_dim, self.hidden_dim)

        # w2: 30 x 30
        self.w2 = torch.randn(self.hidden_dim, self.hidden_dim)

        # w3: 30 x 1
        self.w3 = torch.randn(self.hidden_dim, self.output_dim)

    def update_dropout_mask(self,dropout):
        #create a mask with propability of 'dropout' for an element to be kept
        self.mask = [(torch.Tensor((self.input_dim)).uniform_(0, 1) < dropout[0]).float()]
        self.mask += [(torch.Tensor((self.hidden_dim)).uniform_(0, 1) < dropout[1]).float()]
        self.mask += [(torch.Tensor(self.hidden_dim).uniform_(0, 1) < dropout[2]).float()]
    
    def dropout_layer(self, X, layer):  #layer0:input Layer
        #scale before returning
        return self.mask[layer] * X / (1.0 - self.drop_rate[layer])

    def sigmoid(self, s):
        return 1 / (1 + torch.exp(-s))

    def sigmoid_first_order_derivative(self, s):
        return s * (1 - s)

    def relu_first_order_derivative(self, s):
        return torch.where(s>0, torch.tensor(1.), torch.tensor(0.))
    
    # Forward propagation
    def forward(self, X):
        # First linear layer
        if self.is_training == True & self.use_dropout == True:
            # Add a dropout layer after the second fully connected layer
            X = self.dropout_layer(X,0)
        elif self.is_training == False & self.use_dropout == True:
            self.h1 = self.X * self.drop_rate[0]

        self.y1 = torch.matmul(X, self.w1)
        #self.y1= self.b1.unsqueeze(0).expand_as(self.y1) # 3 X 3 ".dot" does not broadcast in PyTorch
        # First non-linearity
        self.h1 = self.relu(self.y1) #activation function

        if self.is_training == True & self.use_dropout == True:
            # Add a dropout layer after the second fully connected layer
            self.h1 = self.dropout_layer(self.h1, 1)
        elif self.is_training == False & self.use_dropout == True:
            self.h1 = self.h1 * self.drop_rate[1]

        # Second linear layer
        self.y2 = torch.matmul(self.h1, self.w2)
        #self.y2= self.b2.unsqueeze(0).expand_as(self.y2)

        # Second non-linearity
        self.h2 = self.relu(self.y2)

        if self.is_training == True & self.use_dropout == True:
            # Add a dropout layer after the second fully connected layer
            self.h2 = self.dropout_layer(self.h2, 2)
        elif self.is_training == False & self.use_dropout == True:
            self.h1 = self.h2 * self.drop_rate[2]

        # Third linear layer
        self.y3 = torch.matmul(self.h2, self.w3)
        #self.y3= self.b3.unsqueeze(0).expand_as(self.y3)
        # Second non-linearity
        self.h3 = self.sigmoid(self.y3)

        return self.h3

    # Backward propagation
    def backward(self, X, groundtruth, h3):
        # Derivative of binary cross entropy cost w.r.t. final output h2
        self.dC_dh3 = 2*(h3 - groundtruth)

        '''
        Gradients for w3: partial derivative of cost w.r.t. w3
        dC/dw3
        '''
        self.dh3_dy3 = self.sigmoid_first_order_derivative(h3)

        # h3 delta: dC_dh2 dh2_dy2
        self.h3_delta = self.dC_dh3 * self.dh3_dy3
        
        #dropout?
        if self.is_training == True & self.use_dropout == True:
            self.h2 = self.h2*self.mask[2]

        #                               // h3_delta \\  //h2\\
        # This is our gradients for w1: dC_dh2 dh2_dy2 dy2_dw2
        self.dC_dw3 = torch.matmul(torch.t(self.h2), self.h3_delta)

        '''
        Gradients for w2: partial derivative of cost w.r.t. w2
        dC/dw2
        '''
        self.dh2_dy2 = self.relu_first_order_derivative(self.y2)

                    #///h3_delta\\\  // w3\\ 
        # h2 delta: (dC_dh3 dh3_dy3) dy3_dh2 dh2_dy2 --> dC_dy2
        self.h2_delta = torch.matmul(self.h3_delta, torch.t(self.w3)) * self.dh2_dy2

        #dropout?
        if self.is_training == True & self.use_dropout == True:
            self.h1 = self.h1*self.mask[1]

        #                   //  h2_delta  \\ 
        # Gradients for w2: (dC_dh3 dh3_dy3) dy3_dh2 dh2_dy2 , dy2_dw2
        self.dC_dw2 = torch.matmul(torch.t(self.h1), self.h2_delta)

        '''
        Gradients for w1: partial derivative of cost w.r.t w1
        dC/dw1
        '''
        self.dh1_dy1 = self.relu_first_order_derivative(self.y1)
        self.dy2_dh1 = self.w2
        
        #           //  h2_delta  \\ // w2\\ //relu\\
        # h1 delta: (dC_dh2 dh2_dy2) dy2_dh1 dh1_dy1
        self.h1_delta = torch.matmul(self.h2_delta, torch.t(self.w2)) * self.dh1_dy1
        
        #dropout?
        if self.is_training == True & self.use_dropout == True:
            X = X*self.mask[0]
        
        #                   //   h1_delta \\ 
        # Gradients for w1: (dC_dh2 dh2_dy2) dy2_dh1 dh1_dy1 dy1_dw1
        self.dC_dw1 = torch.matmul(torch.t(X), self.h1_delta)

        # Gradient descent on the weights from our 2 linear layers
        self.w1 -= self.learning_rate * self.dC_dw1
        self.w2 -= self.learning_rate * self.dC_dw2
        self.w3 -= self.learning_rate * self.dC_dw3

    def run_training(self, X, y):
        
        split_x = torch.split(X, 50)
        split_y =  torch.split(y, 50)
        num_batches = len(split_x)

        loss_lst = []

        #For every minibatch, choose a different, random dropout rate.
        for minibatch in range(num_batches):

            if minibatch<(num_batches-1):
                X_val = split_x[minibatch+1]
                y_val = split_y[minibatch+1]
            else:
                X_val = split_x[0]
                y_val = split_y[0]

            #print(minibatch)
            #X_train = torch.cat((X_store[:50*minibatch+1:,:], X_store[50*(minibatch+1)-1:,:]), axis = 0)
            #y_train = torch.cat((y_store[:50*minibatch+1:,:], y_store[50*(minibatch+1)-1:,:]), axis = 0)

            X_train = split_x[minibatch]
            y_train = split_y[minibatch]
            
            #for each minibatch, update dropout masks
            print("Updating Dropout Mask, batch ", minibatch)
            self.update_dropout_mask(self.drop_rate)

            for epoch in range(self.num_epochs):
                # (1) Forward propagation: to get our predictions to pass to our cross entropy loss function
                # (2) Back propagation: get our partial derivatives w.r.t. parameters (gradients)
                # (3) Gradient Descent: update our weights with our gradients
                self.train(X_train, y_train)

                # Get our predictions
                y_hat = self.predict(X_val).double()

                y_hat = torch.where(y_hat==1., 0.9999, y_hat)
                y_hat = torch.where(y_hat==0., 0.0001, y_hat)
                # Cross entropy loss, remember this can never be negative by nature of the equation
                # But it does not mean the loss can't be negative for other loss functions
                cross_entropy_loss = -(y_val * torch.log(y_hat) + (1 - y_val) * torch.log(1 - y_hat))

                # We have to take cross entropy loss over all our samples, 100 in this 2-class iris dataset
                mean_cross_entropy_loss = torch.mean(cross_entropy_loss).detach().item()

                # Print our mean cross entropy loss
                if epoch % 10 == 0:
                    print('Epoch {} | Loss: {}'.format(epoch, mean_cross_entropy_loss))
                loss_lst.append(mean_cross_entropy_loss)


    def train(self, X, groundtruth):
        # Forward propagation
        self.is_training = True
        h2 = self.forward(X)

        # Backward propagation and gradient descent
        self.backward(X, groundtruth, h2)
        self.is_training = False

    def predict(self, X):
        return self.forward(X)




# cancer = datasets.load_breast_cancer()

# #print(X)
# print(cancer.data)
# print(cancer.data)
# X = torch.tensor(preprocessing.normalize(cancer.data), dtype=torch.float)
# y = torch.tensor(cancer.target.reshape(-1, 1), dtype=torch.float)
# print(X.size())
# print(y.size())


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# batch_size = 50
# num_batches = 10



# ##################################With dropout ###################################
# # Instantiate our model class and assign it to our model object
# model = DropoutNet()

# # Loss list for plotting of loss behaviour
# loss_lst = []

# # Number of times we want our FNN to look at all 100 samples we have, 100 implies looking through 100x
# num_epochs = 100


# splittedx = torch.split(X_train, 50)
# splittedy =  torch.split(y_train, 50)
# num_batches = len(splittedx)

# X_store = X_train
# y_store = y_train

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


# pred = model.predict(X_test)

# for i in range(len(X_test)):
#     if i%10 == 0:
#         print("Prediction: ", (pred[i]>0.5), "Truth:",y_test[i])

# acc = metrics.accuracy_score(y_test, (pred>0.5))
# print("Accuracy with dropout: ", acc)


# ##################################Without dropout ###################################
# # Instantiate our model class and assign it to our model object
# model = FNN(use_dropout=False)

# # Loss list for plotting of loss behaviour
# loss_lst = []


# # Let's train our model with 100 epochs
# for epoch in range(num_epochs):

#     for minibatch in range(num_batches):
#         X_val = splittedx[minibatch]
#         y_val = splittedy[minibatch]
#         #print(minibatch)
#         X_train = torch.cat((X_store[:50*minibatch+1:,:], X_store[50*(minibatch+1)-1:,:]), axis = 0)
#         y_train = torch.cat((y_store[:50*minibatch+1:,:], y_store[50*(minibatch+1)-1:,:]), axis = 0)

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


# pred = model.predict(X_test)

# for i in range(len(X_test)):
#     if i%10 == 0:
#         print("Prediction: ", (pred[i]>0.5), "Truth:",y_test[i])

# acc = metrics.accuracy_score(y_test, (pred>0.5).float())
# print("Accuracy Wihout dropout: ", acc)