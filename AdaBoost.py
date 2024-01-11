import numpy as np
import random as random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AdaBoost:
    def __init__(self):
        self.M = None
        self.models = []
        self.weights = []
    
    def fit (self, x_train_input, y_train_input, M):
        split_index = int(0.8 * x_train_input.shape[0])  # 80% for training, 20% for dev
        x_train, y_train = x_train_input[:split_index], y_train_input[:split_index]
        x_dev, y_dev= x_train_input[split_index:], y_train_input[split_index:]
        x_train = x_train.toarray()
    
        self.M = M
        # first all samples get the same weight, 1/total number of examples
        # that makes the samples all equally important
        sample_weights = np.ones(x_train.shape[0]) * (1 / x_train.shape[0])
        for m in range(M):
            stump = CreateStump()
            stump.fit( x_train, y_train, sample_weights)
            sample_weights = stump.weights
            self.models.append(stump)
            distribution = list(np.shape(sample_weights) )
            distribution[0] = sample_weights[0]
            for i in range (1,len(sample_weights)):
                distribution.append(distribution[i-1] + sample_weights[i])
            # make a new empty dataset the same size as the original and pick a random number [0..1]
            # and see where that number falls when the sample weights are use like a distribution
            # we fill the new dataset with those values 
            newDataset = np.zeros_like(x_train)
            temp_y = np.zeros_like(y_train)
            for i in range (x_train.shape[0]):
                number = random.random()
                for k in range (x_train.shape[0]):
                    if number <= distribution[k]:
                        for j in range (x_train.shape[1]):
                            newDataset[i][j] = x_train[k][j]
                        sample_weights[i] = sample_weights[k]
                        temp_y += y_train[k]
                        break 
            # normalize the weights so they all add up to 1
            sum = np.sum(sample_weights)
            self.weights.extend([weight/sum for weight in sample_weights])
            x_train = np.copy(newDataset)
            y_train = np.copy(temp_y)
        y_dev_predicted = self.predict(x_dev)
        return self.evaluate(y_dev_predicted, y_dev)

    def predict(self, x_test):
        x_test = x_test.toarray()
        # Initialize arrays to store predictions and the amount of say for each stump
        predictions = np.zeros(x_test.shape[0])
        stump_says = np.zeros(len(self.models))

        for i, stump in enumerate(self.models):
            # Get predictions from each stump
            stump_pred = stump.predict(x_test)
            stump_says[i] = stump.amountOfSay

            # Update overall predictions based on the stump's amount of say
            predictions += stump_pred * stump_says[i]

        # Final prediction is based on the sign of the weighted sum
        final_predictions = np.sign(predictions)

        return final_predictions
    
    def evaluate(self, y_true, y_predicted):
        nan_indices1 = np.isnan(y_true)
        if np.any(nan_indices1):
            # Handle NaN values (e.g., remove or replace them)
            y_true = np.nan_to_num(y_true)
        nan_indices2 = np.isnan(y_predicted)
        if np.any(nan_indices2):
            # Handle NaN values (e.g., remove or replace them)
            y_predicted = np.nan_to_num(y_predicted)
        accuracy = accuracy_score(y_true, y_predicted)
        print("Accuracy:", accuracy)

        precision = precision_score(y_true, y_predicted)
        print("Precision:", precision)

        # Compute recall
        recall = recall_score(y_true, y_predicted)
        print("Recall:", recall)

        # Compute F1 score
        f1 = f1_score(y_true, y_predicted)
        print("F1 Score:", f1)


            


class CreateStump:
    def __init__(self):
        self.amountOfSay = None
        self.word = None
        self.weights = []

    def fit(self, x, y, w):
        min = float('inf')
        for word_index in range(x.shape[1]):    
            total_error = 0         # the total error of a stump is the sum of the weights associated with the incorrectly classed samples
            cor_pos = 0             # correctly positive
            incor_pos = 0           # incorrectly positive
            cor_neg = 0             # correctly negative
            incor_neg = 0           # incorrectly negative
            for sample_index in range(x.shape[0]):
                if x[sample_index][word_index] == 1 & y[sample_index] == 1:
                    cor_pos += 1
                elif x[sample_index][word_index] == 1 & y[sample_index] == 0:
                    incor_pos += 1
                    total_error += w[sample_index]
                elif x[sample_index][word_index] == 0 & y[sample_index] == 0:
                    cor_neg += 1
                else:
                    incor_neg += 1
                    total_error += w[sample_index]
            pos_total = cor_pos + incor_pos
            neg_total = cor_neg + incor_neg
            total = x.shape[0]
            epsilon = 1e-10  # Small epsilon value to avoid division by zero
            gini_index = ((pos_total)/total)*(1-(cor_pos/(pos_total + epsilon))**2+(incor_pos/(pos_total + epsilon))**2)+((neg_total)/total)*(1-(cor_neg/(neg_total + epsilon))**2+(incor_neg/(neg_total + epsilon))**2)
            if gini_index < min:
                self.amountOfSay = 1/2 * np.log((1-total_error)/(total_error+epsilon))      
                self.word = word_index
        # now we need to decrease the sample weights for all the correctly classified samples
        # and increase the sample weights for all the incorrectly classifies samples
        for sample_index in range(x.shape[0]):
            if (x[sample_index][self.word] == y[sample_index]):
                w[sample_index] = w[sample_index] * np.exp(-self.amountOfSay)
            else:   
                w[sample_index] = w[sample_index] * np.exp(self.amountOfSay)
        # normalize the weights so they all add up to 1
        sum = np.sum(w)
        for i in range(len(w)):
            self.weights.append(w[i]/sum) 

    def predict(self, x_test):
        # Make predictions using the chosen word index
        # Convert predictions to binary (0 or 1)
        stump_pred_binary = np.where(x_test[1][self.word] >= 0.5, 1, 0)

        return stump_pred_binary