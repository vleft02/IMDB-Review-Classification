import numpy as np
import random as random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import shuffle

class AdaBoost:
    def __init__(self, learning_rate=1.0):
        self.M = None
        self.models = []
        self.weights = []
        self.learning_rate = learning_rate
    
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
            x_train, y_train = shuffle(x_train, y_train, random_state=m)

            stump = CreateStump()
            x_train, y_train = stump.fit( x_train, y_train, sample_weights)
            sample_weights = stump.weights
            self.models.append(stump)
            self.weights.extend([self.learning_rate * w for w in sample_weights])  # Accumulate weights
           
        y_dev_predicted = self.predict(x_dev)
        return self.evaluate(y_dev,y_dev_predicted)

    def predict(self, x_test):
        x_test = x_test.toarray()
        
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
        # Handle NaN values, replace them with 0
        nan_indices1 = np.isnan(y_true)
        if np.any(nan_indices1):
            y_true = np.nan_to_num(y_true)
        nan_indices2 = np.isnan(y_predicted)
        if np.any(nan_indices2):
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
        n = x.shape[1]
        m = x.shape[0]

        best_amount_of_say = None
        best_word = None
        best_error = float('inf')
        for word_index in range(n):           
            # Vectorized calculations
            incorrect_mask = x[:, word_index] != y

            # Calculate weighted error
            total_error = np.sum(w[incorrect_mask])

            # Choose the word that minimizes the weighted error
            if total_error < best_error:
                best_amount_of_say = 0.5 * np.log((1 - total_error) / (total_error + 1e-10))
                best_word = word_index
                best_error = total_error
        self.amountOfSay = best_amount_of_say
        self.word = best_word
        # Update weights
        incorrect_mask = x[:, self.word] != y
        w[incorrect_mask] *= np.exp(self.amountOfSay)
        w[~incorrect_mask] *= np.exp(-self.amountOfSay)

        # normalize the weights so they all add up to 1
        self.weights = w / np.sum(w)
        indices = np.random.choice(np.arange(m), size=m, p=self.weights)
        return x[indices], y[indices] 

    def predict(self, x_test):
        # Make predictions using the chosen word index
        # Convert predictions to binary (0 or 1)
        stump_pred_binary = x_test[:, self.word] >= 0.5
        return stump_pred_binary