from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import tree

class AdaBoost:
    def __init__(self,amount_of_says=None, models = None):
        self.models = models if models is not None else []
        self.weights = []
        self.amount_of_says = amount_of_says if amount_of_says is not None else []
    
    def get_params(self, deep=True):
        return {'amount_of_says': self.amount_of_says,
                'models': self.models,
                }

    def fit (self, x_train_input, y_train_input, M,learning_curve=False):
        split_index = int(0.8 * x_train_input.shape[0])  # 80% for training, 20% for dev
        x_train, y_train = x_train_input[:split_index], y_train_input[:split_index]
        x_dev, y_dev= x_train_input[split_index:], y_train_input[split_index:]
        x_train = x_train.toarray()

        # initializing weights equally
        self.weights = np.ones(x_train.shape[0]) / x_train.shape[0]
        m = 0 
        epsilon = 1e-10                                                     # Small epsilon value to avoid division by zero
        while m < M:
            stump = tree.DecisionTreeClassifier(max_depth=1)
            # create the base learner (decision stumps)
            stump = stump.fit(x_train,y_train,self.weights)
            y_pred = stump.predict(x_train)
            error = self.calculate_error(y_train, y_pred, self.weights)
            alpha = np.log((1-error)/(error+epsilon))
            self.change_weights(y_pred, y_train, alpha)
            x_train, y_train = self.updateData(x_train, y_train)
            if error >= 0.5:
                m -= 1
            else:
                self.models.append(stump)
                self.amount_of_says.append(alpha)
                m +=1
            print(f"Iteration {m}: amount_of_say = {alpha}, Error = {error}")

        y_dev_predicted = self.predict(x_dev)
        return self.evaluate(y_dev,y_dev_predicted)

    def calculate_error(self, y, y_pred, w):
        return (sum(w*(np.not_equal(y, y_pred)).astype(int)))

    # make a new empty dataset the same size as the original and pick a random number [0..1)
    # and see where that number falls when the sample weights are used like a distribution
    # fill the new dataset with those values 
    def updateData(self, x, y):
        distribution = np.cumsum(self.weights)
        newDataset = np.zeros_like(x)
        temp_y = np.zeros_like(y)
        for i in range (x.shape[0]):
            number = np.random.rand()
            index = np.searchsorted(distribution, number)
            newDataset[i, :] = x[index, :]
            temp_y[i] = y[index]
        self.weights = np.ones(x.shape[0]) / x.shape[0]

        return newDataset, temp_y

    # increase the weights of false predictions and decrease the others 
    def change_weights(self, pred, y, amount_of_say):
        for i in range(y.shape[0]):
            if ((pred[i]==y[i])):
                self.weights[i] *= np.exp(-amount_of_say)
            else:
                self.weights[i] *= np.exp(amount_of_say)
        # normalize the weights
        # total_weight = np.sum(self.weights)
        self.weights = self.weights/np.sum(self.weights)

        
    def predict(self, x_test):
        x_test = x_test.toarray()
        predictions = np.zeros(x_test.shape[0])
        for i in range(len(self.models)):
            predictions += self.amount_of_says[i] * self.models[i].predict(x_test)

        # Final prediction is based on the sign of the weighted sum
        final_predictions = np.sign(predictions)
        return final_predictions
    
    def evaluate(self, y_true, y_predicted):
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