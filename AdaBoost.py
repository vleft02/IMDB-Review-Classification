import numpy as np
import random as random 

class AdaBoost:
    def __init__(self):
        self.M = None
        self.models = []
    
    def fit (self, x_train_input, y_train_input, M):
        split_index = int(0.8 * x_train_input.shape[0])  # 80% for training, 20% for dev
        x_train, y_train = x_train_input[:split_index], y_train_input[:split_index]
        x_dev, y_dev= x_train_input[split_index:], y_train_input[split_index:]

        self.M = M
        # first all samples get the same weight, 1/total number of examples
        # that makes the samples all equally important
        sample_weights = np.ones(len(x_train[0])) * (1 / len(x_train[0]))
        for m in range(M):
            stump = CreateStump()
            stump.fit(stump, x_train, y_train, sample_weights)
            sample_weights = stump.weights
            self.models.append(stump)
            distribution = np.array([]) 
            distribution[0] = sample_weights[0]
            for i in range (len(sample_weights)):
                distribution[i] = distribution[i-1] + sample_weights[i]
            # make a new empty dataset the same size as the original and pick a random number [0..1]
            # and see where that number falls when the sample weights are use like a distribution
            # we fill the new dataset with those values 
            newDataset = np.array([])
            temp_y = np.shape(y_train)
            for i in range (len(x_train[0])):
                number = random.random()
                for k in range (len(x_train[0])):
                    if number <= distribution[k]:
                        for j in range (len(x_train[1])):
                            newDataset[i][j] = x_train[k][j]
                        sample_weights[i] = sample_weights[k]
                        temp_y[i] = y_train[k]
                        break 
            # normalize the weights so they all add up to 1
            sum = np.sum(sample_weights)
            for i in range(len(sample_weights)):
                self.weights[i] = sample_weights[i]/sum 
            x_train = np.copy(newDataset)
            y_train = np.copy(temp_y)

        # predict: after we create the forest we add the amount of say of all the stumps for positive 
        # and the amount of say of all the stumps for negative and decide whatever had the largest 
        # amount of say sum
            


class CreateStump:
    def __init__(self):
        self.amountOfSay = None
        self.word = None
        self.weights = np.array([])

    def fit(self, x, y, w):
        min = float('inf')
        for word_index in range(len(x[1])):           # placeholder - i have no idea how the data is stored
            total_error = 0         # the total error of a stump is the sum of the weights associated with the incorrectly classed samples
            cor_pos = 0             # correctly positive
            incor_pos = 0           # incorrectly positive
            cor_neg = 0             # correctly negative
            incor_neg = 0           # incorrectly negative
            for sample_index in range(len(x[0])):
                if x[word_index][sample_index] == 1 & y[sample_index] == 1:
                    cor_pos += 1
                elif x[word_index][sample_index] == 1 & y[sample_index] == 0:
                    incor_pos += 1
                    total_error += w[sample_index]
                elif x[word_index][sample_index] == 0 & y[sample_index] == 0:
                    cor_neg += 1
                else:
                    incor_neg += 1
                    total_error += w[sample_index]
            pos_total = cor_pos + incor_pos
            neg_total = cor_neg + incor_neg
            total = len(x[0])
            gini_index = ((pos_total)/total)*(1-(cor_pos/pos_total)^2-(incor_pos/pos_total)^2)+((neg_total)/total)*(1-(cor_neg/neg_total)^2-(incor_neg/neg_total)^2)
            if gini_index < min:
                self.amountOfSay = 1/2 * np.log((1-total_error)/total_error)        # might need to add a small error term to prevent division by 0
                self.word = word_index
        # now we need to decrease the sample weights for all the correctly classified samples
        # and increase the sample weights for all the incorrectly classifies samples
        for sample_index in range(len(x[1])):
            if (x[self.word][sample_index] == y[sample_index]):
                w[sample_index] = w[sample_index] * np.exp(-self.amountOfSay)
            else:   
                w[sample_index] = w[sample_index] * np.exp(self.amountOfSay)
        # normalize the weights so they all add up to 1
        sum = np.sum(w)
        for i in range(len(w)):
            self.weights[i] = w[i]/sum 