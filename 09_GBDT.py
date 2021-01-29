import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.cm as cm
import pandas as pd

X = np.array([[1, 2],[1,3],[4,1],[2,1], [3,2],[4,3]])
y = np.array([1,1,1,0,0,0])

scatter_x = X[:,0]
scatter_y = X[:,1]
group = y
cdict = {1: 'red', 0: 'blue'}

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], label = g)
    plt.xlabel('x_1')
    plt.ylabel('x_2')
ax.legend()
plt.show()



y2 = ['Lost', 'Won' ]
colors = {'Won':'red', 'Lost':'green'}

#plt.scatter(X[:,0], X[:,1], c=y2) #2.map(colors))
#plt.xlabel('x_1')
#plt.xlabel('x_2')




#plt.show()

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class GBDT():

    def __init__(self, initial_prediction, num_splits: int=2, num_trees: int=2, gamma: int=1, lambd: int=1):
        self._num_splits = num_splits
        self._num_trees = num_trees
        self._unique_vals = []
        self._split_rules = []
        self._cur_predict = initial_prediction
        self._gamma = gamma
        self._lambda = lambd
        self._sample_in_leaf = []
        self._sample_in_leaf0 = []


    def print_info(self):
        print("__________________")
        print("Weights: ", self._weights)
        print("Alphas: ", self._alpha)
        print("Losses: ", self._loss)
        print("__________________")

    def fit(self, X_train, target):
        
        self._sample_in_leaf = np.ones(X_train.shape[0], dtype=bool)
        print("leave samples: ", self._sample_in_leaf)
        print("len", len(self._sample_in_leaf))
        split_rules = self.find_rules(X_train)
        max_gain, best_split = self.max_gain(target[self._sample_in_leaf])

        print("best split:" , best_split)
        split_data_rule = split_rules[best_split[0]][best_split[1]]

        print("Split data rule applied at depth 0: x_",best_split[0], " <= ", split_data_rule)
        print(split_data_rule)
        print(X_train[self._sample_in_leaf,best_split[0]])
        print(X_train[self._sample_in_leaf,best_split[0]]<=split_data_rule)
        left_data = X_train[X_train[self._sample_in_leaf,best_split[0]]<=split_data_rule,:]
        right_data = X_train[X_train[self._sample_in_leaf,best_split[0]]>split_data_rule,:]
        print(left_data)

        self._depth1left_data = left_data
        self._depth1right_data = right_data

        self._sample_in_leaf0 = self._sample_in_leaf

        #left
        self._sample_in_leaf = X_train[self._sample_in_leaf0,best_split[0]]<=split_data_rule
        print("new left leaf samples: ", self._sample_in_leaf)
        print("len", len(self._sample_in_leaf))
        split_rules = self.find_rules(X_train)
        max_gain2, best_split2 = self.max_gain(target[self._sample_in_leaf])

        print(best_split2)

        split_data_rule2 = split_rules[best_split2[0]][best_split2[1]]

        print("Split data rule applied at depth 1, left: x_",best_split2[0], " <= ", split_data_rule2)
        print(X_train[self._sample_in_leaf,best_split2[0]])
        print(X_train[self._sample_in_leaf,best_split2[0]]<=split_data_rule2)


        #right
        self._sample_in_leaf = X_train[self._sample_in_leaf0,best_split[0]]>split_data_rule
        print("new right leaf samples: ", self._sample_in_leaf)
        print("len", len(self._sample_in_leaf))
        split_rules = self.find_rules(X_train)
        max_gain2, best_split2 = self.max_gain(target[self._sample_in_leaf])

        print(best_split2)

        split_data_rule = split_rules[best_split2[0]][best_split2[1]]
        print("Split data rule applied at depth 1, right: x_",best_split2[0], " <= ", split_data_rule)
        print(split_data_rule)
        print(X_train[self._sample_in_leaf,best_split2[0]])
        print(X_train[self._sample_in_leaf,best_split2[0]]>split_data_rule)


    def calc_loss(self, target, pred):
        loss = -target*np.log(sigmoid( pred))-(1-target)*np.log(1-sigmoid(pred))
        return loss

    def calc_gradient(self, target, pred):
        gradient = sigmoid(pred)-target
        return gradient

    def calc_hessian(self, pred):
        hessian = sigmoid(pred)*(1-sigmoid(pred))
        return hessian

    def find_rules(self, X):
        unique_vals = []
        print("X shape=", X.shape[1])
        for m in range(X.shape[1]):
            print(m)
            #unique values for feature m
            unique_vals.append(np.unique(X[self._sample_in_leaf,m]))
        self._unique_vals = unique_vals
        print(unique_vals)
        split_rules=[]
        print(unique_vals[0].shape[0])
        #iterate over all features
        for m in range(len(unique_vals)):
            feat_splits = []
            for n in range(unique_vals[m].shape[0]-1):
                feat_splits.append((unique_vals[m][n]+unique_vals[m][n+1])/2)
                print(split_rules)
            split_rules.append(np.array(feat_splits))
        
        self._split_rules = split_rules
        print(split_rules)
        return split_rules

    def max_gain(self, y):
        split_rules = self._split_rules
        print("Split_rules = ", split_rules)
        gains = []
        best_split = []
        max_gain = -100
        #iterate over all features
        for m in range(len(split_rules)):
            print("m = ", m)
            feature_gains = []
            #iterate over all created split rules to divide samples up in left and right and calculate gain.
            for n in range(split_rules[m].shape[0]):
                print("n = ", n)
                #divide samples to left and right corresponding to current split rule
                print((X[:,m]<= split_rules[m][n]) & self._sample_in_leaf)
                samples_left = (X[:,m]<= split_rules[m][n]) & self._sample_in_leaf
                #samples_left = (X[:,m]<= split_rules[m][n]) # and self._sample_in_leaf
                #samples_right = (X[self._sample_in_leaf,m]> split_rules[m][n])
                samples_right = (X[:,m] > split_rules[m][n]) & self._sample_in_leaf
                print("split_rule:" , split_rules[m][n])
                print("samples left: = ", samples_left)
                print("X = ", X[self._sample_in_leaf,m])

                print(self._sample_in_leaf)
                self._samples_right = samples_right
                self._samples_left = samples_left

                #calc gain for this split
                gain = self.calc_gain(y)
                feature_gains.append(gain)
                if gain > max_gain:
                    print("new max gain!")
                    max_gain = gain
                    best_split = [m,n]
            gains.append(feature_gains)
        
        print("All gains:")
        print(gains)
        print(max_gain)

        print(best_split)

        return max_gain, best_split
        

    def calc_gain(self, targets):
        print("hi")
        obj_left_child = self.calc_objective(self._samples_left, y)
        obj_right_child = self.calc_objective(self._samples_right, y)
        obj_parent = self.calc_objective((self._samples_left | self._samples_right), y)
        gain = 0.5*(obj_left_child + obj_right_child - obj_parent)-self._gamma
        return gain

    def calc_objective(self, samples, y):
        leaf_sample_targets = y[samples]
        obj = (np.square(sum(self.calc_gradient(leaf_sample_targets, self._cur_predict[samples]))))/(self._lambda+sum(self.calc_hessian( self._cur_predict[samples])))
        return obj

pred = np.zeros(6)
gbdt = GBDT(pred)

loss = gbdt.calc_loss(y, pred)
grad = gbdt.calc_gradient(y, pred)
hess = gbdt.calc_hessian(pred)

print('grad')
print(grad)
print('hess')
print(hess)
print("loss")
print(loss)


gbdt.fit(X, y)
#gbdt.max_gain()