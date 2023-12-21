import numpy as np

class AdaBoost():
    def __init__(self):
        self.z = np.array([])
        # self.error = 0
        self.h = np.array([])
        self.M = None # number of boosting rounds 
        
    # here we create the base learner / decision trees
    # each decision tree will hae only one depth 
    # for each feature we create a stump
    # to choose which base learner to start with first in the sequence, the entropy/Gini index of the stumps is calculated
    # the stump with the lowest entropy/Gini index is selected
    def create_stump(x, y, weight, h):
        h = []

    def normalize(w):
        sum = np.sum(w)
        for i in range(len(w)):
            w[i] = w[i]/sum 

    def fit(self, x_train_input, y_train_input, M):
        feature_vector_length = x_train_input.shape[1]
        split_index = int(0.8 * x_train_input.shape[0])  # 80% for training, 20% for dev

        x_train, y_train = x_train_input[:split_index], y_train_input[:split_index]
        x_dev, y_dev= x_train_input[split_index:], y_train_input[split_index:]

        self.M = M
        weight = np.ones(len(x_train_input)) * 1 / len(x_train_input)
        for m in range (1, M):
            self.create_stump(x_train_input, y_train_input, weight, self.h)
            error = 0
            for i in range (len(x_train_input)):
                if self.h[m] != y[i]: 
                    error += weight[i]
            if error >= 0.5:
                m -= 1
                break
            for i in range (len(x_train_input)):
                if self.h[m] == y_train_input[i]:
                    weight[i] = weight[i]*error/(error-1)
            self.normalize(weight)
            self.z[m] = 1/2 * np.log((1-error)/error)
