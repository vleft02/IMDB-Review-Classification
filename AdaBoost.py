import numpy as np
import random as random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AdaBoost:
    def __init__(self):
        self.M = None
        self.models = []
        self.weights = []
        self.amount_of_says = []
    
    def fit (self, x_train_input, y_train_input, M):
        split_index = int(0.8 * x_train_input.shape[0])  # 80% for training, 20% for dev
        x_train, y_train = x_train_input[:split_index], y_train_input[:split_index]
        x_dev, y_dev= x_train_input[split_index:], y_train_input[split_index:]
        x_train = x_train.toarray()
    
        self.M = M
        # initializing weights equally
        self.weights = np.ones(x_train.shape[0]) * (1 / x_train.shape[0])
        stump = CreateStump()
        m = 0 
        while m < M:
            # update the data according to the weights
            weighted_indices = np.random.choice(x_train.shape[0], size=x_train.shape[0], replace=True, p=self.weights)
            x_train_weighted, y_train_weighted = x_train[weighted_indices], y_train[weighted_indices]

            # create the base learner (decision stumps)
            best_stump = stump.fit( x_train_weighted, y_train_weighted, self.weights)
            self.change_weights(best_stump['predictions'], y_train, best_stump['amount_of_say'])
            if best_stump['error'] >= 0.55:
                m -= 1
            else:
                self.models.append(best_stump)
                self.amount_of_says.append(best_stump['amount_of_say'])
                m +=1
                
            print(f"Iteration {m}: amount_of_say = {best_stump['amount_of_say']}, Error = {best_stump['error']}")

        y_dev_predicted = self.predict(x_dev)
        return self.evaluate(y_dev,y_dev_predicted)

    # increase the weights of false predictions and decrease the others 
    def change_weights(self, pred, y, amount_of_say):
        for i in range(y.shape[0]):
            if ((pred[i]==y[i])):
                self.weights[i] *= np.exp(-amount_of_say)
            else:
                self.weights[i] *= np.exp(2*amount_of_say)
        # normalize the weights
        total_weight = np.sum(self.weights)
        self.weights /= total_weight if total_weight != 0 else np.ones_like(self.weights) / len(self.weights)

        
    def predict(self, x_test):
        x_test = x_test.toarray()
        predictions = np.zeros(x_test.shape[0])
        for i in range(len(self.models)):
            stump = self.models[i]
            word_values = x_test[:, stump['word']]
            stump_predictions = np.where(word_values == stump['threshold'], 1,0)
            predictions += self.amount_of_says[i] * stump_predictions

        # Final prediction is based on the sign of the weighted sum
        final_predictions = np.sign(predictions)
        return final_predictions
    
    def evaluate(self, y_true, y_predicted):
        accuracy = accuracy_score(y_true, y_predicted)
        print("Accuracy:", accuracy)

        precision = precision_score(y_true, y_predicted)
        print("Precision:", precision)

        recall = recall_score(y_true, y_predicted)
        print("Recall:", recall)

        f1 = f1_score(y_true, y_predicted)
        print("F1 Score:", f1)


class CreateStump:
    def __init__(self):
        self.weighted_errors = None

    def fit(self, x, y, w):
        best_stump = {
            'word': None,
            'threshold': None,
            'error': None,
            'amount_of_say': None,
            'predictions': None,
            'ig': float('-inf')
        }
        # we create a stump for each feature, in this case for each word
        for word in range(x.shape[1]):
            for threshold in [0,1]:
                predictions = (x[:, word] == threshold).astype(int)

                epsilon = 1e-10                                                     # Small epsilon value to avoid division by zero
                cor_pos = np.sum((predictions == 1) & (y == 1))                     # classified as positive correctly
                cor_neg = np.sum((predictions == 0) & (y == 0))                     # classified as negative correctly
                incor_pos = np.sum((predictions == 1) & (y == 0))                   # classified as positive incorrectly
                incor_neg = np.sum((predictions == 0) & (y == 1))                   # classified as negative incorrectly

                total = (cor_pos+cor_neg+incor_neg+incor_pos)                       # total sum of predictions

                p_cor_pos = cor_pos/(cor_pos+cor_neg)                               # probability of something being classified as positive correctly
                p_cor_neg = cor_neg/(cor_pos+cor_neg)                               # probability of something being classified as negative correctly
                child_entropy1 = - p_cor_neg*np.log2(p_cor_neg+epsilon) - p_cor_pos*np.log2(p_cor_pos+epsilon)          # entropy of correctly classified data

                p_incor_pos = incor_pos/(incor_pos+incor_neg)                       # probability of something being classified as positive incorrectly
                p_incor_neg = incor_neg/(incor_pos+incor_neg)                       # probability of something being classified as negative incorrectly
                child_entropy2 = - p_incor_neg*np.log2(p_incor_neg+epsilon) - p_incor_pos*np.log2(p_incor_pos+epsilon)  # entropy of incorrectly classified data

                parent_entropy = - ((cor_pos+incor_pos)/total*np.log2((cor_pos+incor_pos)/total+epsilon))- ((cor_neg+incor_neg)/total*np.log2((cor_neg+incor_neg)/total+epsilon))
                cor_pred = (cor_pos+cor_neg)/total                                  
                incor_pred = (incor_pos+incor_neg)/total
                average_child_entropy = (cor_pred*child_entropy1)+(incor_pred*child_entropy2)
                information_gain = parent_entropy - average_child_entropy

                # chose the stump with the highest information gain 
                if information_gain > best_stump['ig']:
                    self.weighted_errors = (sum(w*(np.not_equal(y, predictions)).astype(int)))/sum(w)
                    best_stump['word'] = word
                    best_stump['threshold'] = threshold
                    best_stump['error'] = self.weighted_errors
                    best_stump['predictions'] = predictions
                    best_stump['ig'] = information_gain

        best_stump['amount_of_say'] = (1/2) * np.log(((1 - self.weighted_errors )/self.weighted_errors+epsilon))
        return best_stump
